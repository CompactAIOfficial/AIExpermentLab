import argparse
import os
import json

from lab.config import SERIES, TrainConfig, DataConfig
from lab.data.pipeline import build_datasets
from lab.training import train_model


def parse_args():
    p = argparse.ArgumentParser(description="Train a TinyForecaster on stock prices")
    p.add_argument("--series", default="Glint", choices=list(SERIES.keys()))
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2026-01-01")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    model_cfg = SERIES[args.series]
    model_cfg.seq_len = args.seq_len

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )

    data_cfg = DataConfig(
        ticker=args.ticker,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )

    out = args.output or os.path.join("runs", f"{args.series}_{args.ticker}")
    os.makedirs(out, exist_ok=True)

    print(f"[train] series={args.series} ticker={args.ticker} device={args.device}")
    print(f"[train] window={data_cfg.train_start}..{data_cfg.train_end}  horizon={args.horizon}")

    train_ds, _, norm = build_datasets(data_cfg)
    print(f"[train] train_examples={len(train_ds)}")

    with open(os.path.join(out, "norm.json"), "w") as f:
        json.dump({"mean": norm.mean, "std": norm.std}, f)
    with open(os.path.join(out, "data_cfg.json"), "w") as f:
        json.dump(data_cfg.__dict__, f, indent=2)

    model, _ = train_model(model_cfg, train_cfg, train_ds, out)
    print(f"[train] params={model.num_params():,}  saved to {out}/model.pt")


if __name__ == "__main__":
    main()
