import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from lab.config import ModelConfig, DataConfig
from lab.model import TinyForecaster
from lab.data.pipeline import build_blind_test
from lab.plotting import plot_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark TinyForecaster with configurable lookahead")
    p.add_argument("--model", required=True, help="path to model.pt")
    p.add_argument("--run-dir", default=None, help="override run dir for data_cfg")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tickers", default=None,
                   help="Override tickers comma-separated (e.g. AAPL,NVDA)")
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--mode", default="blind",
                   choices=["blind", "nonblind", "partial"],
                   help="blind: never sees test prices.  "
                        "nonblind: one-step-ahead (sees real prices).  "
                        "partial: re-anchors every --partial-interval steps.")
    p.add_argument("--partial-interval", type=int, default=126,
                   help="Steps before a real-price anchor in partial mode (default 126 ≈ half year).")
    return p.parse_args()


MODE_LABELS = {
    "blind": "BLIND (never sees test prices)",
    "nonblind": "NON-BLIND (one-step-ahead, sees real prices)",
    "partial": "PARTIAL (re-anchor every N steps)",
}


def _normalize_window(window: np.ndarray) -> np.ndarray:
    return (window - window.mean()) / (window.std() + 1e-8)


def autoregressive_forecast(
    model, seed_raw: np.ndarray, n_steps: int, device: str,
    seq_len: int, mode: str, actuals_raw: np.ndarray | None = None,
    partial_interval: int = 126,
) -> np.ndarray:
    """Autoregressive rollout with per-step local normalization.

    The buffer holds **raw prices**.  At each step the last ``seq_len``
    prices are locally z-scored before being fed to the model.  The
    predicted normalised value is immediately de-normalised back to a raw
    price and appended to the buffer.
    """
    model.eval()
    buf = torch.tensor(seed_raw, dtype=torch.float32, device=device)
    preds: list[float] = []
    with torch.no_grad():
        for step in range(n_steps):
            window = buf[-seq_len:]
            w_mean = window.mean()
            w_std = window.std() + 1e-8
            x = ((window - w_mean) / w_std).view(1, seq_len, 1)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            p_norm = float(out.squeeze().item())
            p = p_norm * w_std.item() + w_mean.item()
            preds.append(p)

            if mode == "nonblind" and actuals_raw is not None:
                nxt = torch.tensor([actuals_raw[step]], device=device)
            elif mode == "partial" and actuals_raw is not None and (step + 1) % partial_interval == 0:
                nxt = torch.tensor([actuals_raw[step]], device=device)
            else:
                nxt = torch.tensor([p], device=device)

            buf = torch.cat([buf, nxt])

    return np.array(preds, dtype=np.float32)


def directional_accuracy_series(actual: np.ndarray, predicted: np.ndarray, last_seed_value: float) -> float:
    actual_prev = np.concatenate([[last_seed_value], actual[:-1]])
    pred_prev = np.concatenate([[last_seed_value], predicted[:-1]])
    actual_dir = np.sign(actual - actual_prev)
    pred_dir = np.sign(predicted - pred_prev)
    nz = actual_dir != 0
    if nz.sum() == 0:
        return float("nan")
    return float((actual_dir[nz] == pred_dir[nz]).mean())


def evaluate_forecast(
    model, seed_raw: np.ndarray, actual_raw: np.ndarray,
    seq_len: int, device: str, mode: str, partial_interval: int,
):
    pred_p = autoregressive_forecast(
        model, seed_raw, len(actual_raw), device, seq_len,
        mode=mode, actuals_raw=actual_raw if mode != "blind" else None,
        partial_interval=partial_interval,
    )

    last_seed_p = float(seed_raw[-1])

    mae = float(np.mean(np.abs(pred_p - actual_raw)))
    rmse = float(np.sqrt(np.mean((pred_p - actual_raw) ** 2)))
    mape = float(np.mean(np.abs((pred_p - actual_raw) / np.maximum(1e-8, np.abs(actual_raw)))) * 100.0)
    da = directional_accuracy_series(actual_raw, pred_p, last_seed_p)

    naive = np.full_like(actual_raw, last_seed_p)
    naive_mae = float(np.mean(np.abs(naive - actual_raw)))
    naive_rmse = float(np.sqrt(np.mean((naive - actual_raw) ** 2)))

    metrics = {
        "n_steps": int(len(pred_p)),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "directional_accuracy": da,
        "naive_flat_mae": naive_mae,
        "naive_flat_rmse": naive_rmse,
        "skill_vs_naive_rmse": float(1.0 - rmse / max(1e-8, naive_rmse)),
        "mode": mode,
        "partial_interval": partial_interval if mode == "partial" else None,
    }
    return metrics, pred_p, actual_raw


def main():
    args = parse_args()

    ckpt = torch.load(args.model, map_location=args.device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["cfg"])
    model = TinyForecaster(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model"])

    run_dir = args.run_dir or os.path.dirname(args.model)
    with open(os.path.join(run_dir, "data_cfg.json")) as f:
        data_dict = json.load(f)
    if args.tickers:
        data_dict["tickers"] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.test_start:
        data_dict["test_start"] = args.test_start
    if args.test_end:
        data_dict["test_end"] = args.test_end
    data_cfg = DataConfig(**data_dict)

    seeds, actuals, dates = build_blind_test(data_cfg)

    all_metrics: Dict[str, dict] = {}
    plot_dir = os.path.join(run_dir, "plots")
    label = MODE_LABELS[args.mode]

    print(f"[bench] {label}  (local-normalisation, raw prices)")
    print(f"[bench] seed = last {data_cfg.seq_len} raw training prices "
          f"({data_cfg.train_end}) → predict {data_cfg.test_start} .. {data_cfg.test_end}")

    for ticker in data_cfg.tickers:
        m, pred_p, actual_p = evaluate_forecast(
            model, seeds[ticker], actuals[ticker],
            data_cfg.seq_len, args.device, args.mode, args.partial_interval,
        )
        all_metrics[ticker] = m

        print(f"\n[bench] {ticker}  steps={m['n_steps']}  "
              f"MAPE={m['mape_pct']:.2f}%  DirAcc={m['directional_accuracy']:.3f}  "
              f"skill_vs_naive_rmse={m['skill_vs_naive_rmse']:+.3f}")

        if not args.no_plot:
            sfx = f"  ({data_cfg.test_start} → {data_cfg.test_end}, {args.mode.upper()})"
            png = plot_predictions(
                dates=dates[ticker],
                actual=actual_p,
                predicted=pred_p,
                ticker=ticker,
                out_path=os.path.join(plot_dir, f"{ticker}_{args.mode}.png"),
                title_suffix=sfx,
            )
            print(f"[bench] {ticker} plot -> {png}")

    out_path = os.path.join(run_dir, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[bench] full metrics -> {out_path}")


if __name__ == "__main__":
    main()
