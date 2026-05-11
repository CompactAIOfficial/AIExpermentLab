import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from lab.config import ModelConfig, DataConfig
from lab.model import TinyForecaster
from lab.data.pipeline import build_datasets, Normalizer


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark a trained TinyForecaster")
    p.add_argument("--model", required=True, help="path to model.pt")
    p.add_argument("--run-dir", default=None, help="override run dir for norm/data_cfg")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ticker", default=None)
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    return p.parse_args()


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, last_in_window: np.ndarray) -> float:
    true_dir = np.sign(y_true - last_in_window)
    pred_dir = np.sign(y_pred - last_in_window)
    nz = true_dir != 0
    if nz.sum() == 0:
        return float("nan")
    return float((true_dir[nz] == pred_dir[nz]).mean())


def evaluate(model, dataset, device, norm: Normalizer):
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    preds, trues, lasts = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().squeeze(-1)
            preds.append(out)
            trues.append(yb.numpy().squeeze(-1))
            lasts.append(xb[:, -1, 0].cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    lasts = np.concatenate(lasts)

    preds_p = norm.inverse(preds)
    trues_p = norm.inverse(trues)
    lasts_p = norm.inverse(lasts)

    mae = float(np.mean(np.abs(preds_p - trues_p)))
    rmse = float(np.sqrt(np.mean((preds_p - trues_p) ** 2)))
    mape = float(np.mean(np.abs((preds_p - trues_p) / np.maximum(1e-8, np.abs(trues_p)))) * 100.0)
    da = directional_accuracy(trues_p, preds_p, lasts_p)

    naive = lasts_p
    naive_mae = float(np.mean(np.abs(naive - trues_p)))
    naive_rmse = float(np.sqrt(np.mean((naive - trues_p) ** 2)))

    return {
        "n_samples": int(len(preds)),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "directional_accuracy": da,
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "skill_vs_naive_rmse": float(1.0 - rmse / max(1e-8, naive_rmse)),
    }


def main():
    args = parse_args()
    ckpt = torch.load(args.model, map_location=args.device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["cfg"])
    model = TinyForecaster(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model"])

    run_dir = args.run_dir or os.path.dirname(args.model)
    with open(os.path.join(run_dir, "data_cfg.json")) as f:
        data_dict = json.load(f)
    if args.ticker:
        data_dict["ticker"] = args.ticker
    if args.test_start:
        data_dict["test_start"] = args.test_start
    if args.test_end:
        data_dict["test_end"] = args.test_end
    data_cfg = DataConfig(**data_dict)

    _, test_ds, norm = build_datasets(data_cfg)
    print(f"[bench] ticker={data_cfg.ticker}  test={data_cfg.test_start}..{data_cfg.test_end}  n={len(test_ds)}")

    metrics = evaluate(model, test_ds, args.device, norm)
    out_path = os.path.join(run_dir, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[bench] saved to {out_path}")


if __name__ == "__main__":
    main()
