import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from lab.config import ModelConfig, DataConfig
from lab.model import TinyForecaster
from lab.data.pipeline import build_blind_test, Normalizer
from lab.plotting import plot_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark TinyForecaster with configurable lookahead")
    p.add_argument("--model", required=True, help="path to model.pt")
    p.add_argument("--run-dir", default=None, help="override run dir for norms/data_cfg")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tickers", default=None,
                   help="Override tickers, comma-separated (e.g. AAPL,NVDA)")
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--mode", default="blind",
                   choices=["blind", "nonblind", "partial"],
                   help="blind: never sees test prices.  "
                        "nonblind: one-step-ahead (sees real prices).  "
                        "partial: re-anchors every --partial-interval steps.")
    p.add_argument("--partial-interval", type=int, default=126,
                   help="Trading steps before a real-price anchor in 'partial' mode "
                        "(default 126 ≈ half year).")
    return p.parse_args()


MODE_LABELS = {
    "blind": "BLIND (never sees test prices)",
    "nonblind": "NON-BLIND (one-step-ahead, sees real prices)",
    "partial": "PARTIAL (re-anchor every N steps)",
}


def autoregressive_forecast(
    model, seed: np.ndarray, n_steps: int, device: str,
    seq_len: int, mode: str, actuals: np.ndarray | None = None,
    partial_interval: int = 126,
) -> np.ndarray:
    model.eval()
    buf = torch.tensor(seed, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for step in range(n_steps):
            x = buf[-seq_len:].view(1, seq_len, 1)
            p = float(model(x).squeeze().item())
            preds.append(p)

            if mode == "nonblind" and actuals is not None:
                nxt = torch.tensor([actuals[step]], device=device)
            elif mode == "partial" and actuals is not None and (step + 1) % partial_interval == 0:
                nxt = torch.tensor([actuals[step]], device=device)
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


def _diffs_to_prices(diffs: np.ndarray, start_price: float) -> np.ndarray:
    prices = np.empty(len(diffs), dtype=np.float32)
    p = start_price
    for i, d in enumerate(diffs):
        p = p + d
        prices[i] = p
    return prices


def evaluate_forecast(
    model, seed_n: np.ndarray, actual_n: np.ndarray, norm: Normalizer,
    seq_len: int, device: str, mode: str, partial_interval: int,
    diff_mode: bool = False, last_train_price: float | None = None,
    actual_raw_prices: np.ndarray | None = None,
):
    pred_n = autoregressive_forecast(
        model, seed_n, len(actual_n), device, seq_len,
        mode=mode, actuals=actual_n, partial_interval=partial_interval,
    )

    if diff_mode:
        pred_diffs = norm.inverse(pred_n)
        actual_diffs = norm.inverse(actual_n)
        pred_p = _diffs_to_prices(pred_diffs, float(last_train_price))
        actual_p = actual_raw_prices if actual_raw_prices is not None else _diffs_to_prices(actual_diffs, float(last_train_price))
        last_seed_p = float(last_train_price)
    else:
        pred_p = norm.inverse(pred_n)
        actual_p = norm.inverse(actual_n)
        last_seed_p = float(norm.inverse(seed_n[-1:])[0])

    mae = float(np.mean(np.abs(pred_p - actual_p)))
    rmse = float(np.sqrt(np.mean((pred_p - actual_p) ** 2)))
    mape = float(np.mean(np.abs((pred_p - actual_p) / np.maximum(1e-8, np.abs(actual_p)))) * 100.0)
    da = directional_accuracy_series(actual_p, pred_p, last_seed_p)

    naive = np.full_like(actual_p, last_seed_p)
    naive_mae = float(np.mean(np.abs(naive - actual_p)))
    naive_rmse = float(np.sqrt(np.mean((naive - actual_p) ** 2)))

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
    return metrics, pred_p, actual_p


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

    seeds, actuals_norm, norms, dates, last_train_prices, actual_raw_prices = (
        build_blind_test(data_cfg)
    )

    all_metrics: Dict[str, dict] = {}
    plot_dir = os.path.join(run_dir, "plots")
    label = MODE_LABELS[args.mode]
    dim = "diffs" if data_cfg.diff_mode else "prices"

    print(f"[bench] {label}  (model trained on {dim})")
    print(f"[bench] seed = last {data_cfg.seq_len} training {dim} "
          f"({data_cfg.train_end}) → predict {data_cfg.test_start} .. {data_cfg.test_end}")

    for ticker in data_cfg.tickers:
        m, pred_p, actual_p = evaluate_forecast(
            model, seeds[ticker], actuals_norm[ticker], norms[ticker],
            data_cfg.seq_len, args.device, args.mode, args.partial_interval,
            diff_mode=data_cfg.diff_mode,
            last_train_price=last_train_prices[ticker],
            actual_raw_prices=actual_raw_prices[ticker],
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
