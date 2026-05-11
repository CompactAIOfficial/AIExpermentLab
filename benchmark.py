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
    p = argparse.ArgumentParser(description="Blind autoregressive benchmark for TinyForecaster")
    p.add_argument("--model", required=True, help="path to model.pt")
    p.add_argument("--run-dir", default=None, help="override run dir for norms/data_cfg")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tickers", default=None,
                   help="Override tickers, comma-separated (e.g. AAPL,NVDA)")
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def autoregressive_forecast(model, seed: np.ndarray, n_steps: int, device: str, seq_len: int) -> np.ndarray:
    model.eval()
    buf = torch.tensor(seed, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = buf[-seq_len:].view(1, seq_len, 1)
            p = float(model(x).squeeze().item())
            preds.append(p)
            buf = torch.cat([buf, torch.tensor([p], device=device)])
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


def evaluate_blind(model, seed_n: np.ndarray, actual_n: np.ndarray, norm: Normalizer,
                   seq_len: int, device: str):
    pred_n = autoregressive_forecast(model, seed_n, len(actual_n), device, seq_len)

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
        "blind": True,
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

    seeds, actuals, norms, dates = build_blind_test(data_cfg)

    all_metrics: Dict[str, dict] = {}
    plot_dir = os.path.join(run_dir, "plots")

    print(f"[bench] BLIND autoregressive forecast")
    print(f"[bench] seed = last {data_cfg.seq_len} days of training; "
          f"model never sees test prices.")

    for ticker in data_cfg.tickers:
        m, pred_p, actual_p = evaluate_blind(
            model, seeds[ticker], actuals[ticker], norms[ticker],
            data_cfg.seq_len, args.device,
        )
        all_metrics[ticker] = m

        print(f"\n[bench] {ticker}  steps={m['n_steps']}  "
              f"MAPE={m['mape_pct']:.2f}%  DirAcc={m['directional_accuracy']:.3f}  "
              f"skill_vs_naive_rmse={m['skill_vs_naive_rmse']:+.3f}")

        if not args.no_plot:
            png = plot_predictions(
                dates=dates[ticker],
                actual=actual_p,
                predicted=pred_p,
                ticker=ticker,
                out_path=os.path.join(plot_dir, f"{ticker}_pred_vs_actual.png"),
                title_suffix=f"  ({data_cfg.test_start} → {data_cfg.test_end}, BLIND)",
            )
            print(f"[bench] {ticker} plot -> {png}")

    out_path = os.path.join(run_dir, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[bench] full metrics -> {out_path}")


if __name__ == "__main__":
    main()
