import argparse
import json
import os

from lab.config import SERIES, TrainConfig, DataConfig
from lab.data.pipeline import build_datasets
from lab.training import train_model
from lab.plotting import plot_loss_curve


def parse_tickers(s: str):
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Train a TinyForecaster on stock prices")
    p.add_argument("--series", default="Glint", choices=list(SERIES.keys()))
    p.add_argument("--tickers", default="AAPL",
                   help="Comma-separated symbols, e.g. AAPL,NVDA")
    p.add_argument("--train-start", default="2022-01-01")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2026-01-01")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    tickers = parse_tickers(args.tickers)

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
        tickers=tickers,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )

    tag = "_".join(tickers)
    out = args.output or os.path.join("runs", f"{args.series}_{tag}")
    os.makedirs(out, exist_ok=True)

    print(f"[train] series={args.series}  tickers={tickers}  device={args.device}")
    print(f"[train] window={data_cfg.train_start}..{data_cfg.train_end}  horizon={args.horizon}")

    train_ds, _, _ = build_datasets(data_cfg)
    print(f"[train] train_examples={len(train_ds)}  (across {len(tickers)} ticker(s))")

    with open(os.path.join(out, "data_cfg.json"), "w") as f:
        json.dump(data_cfg.__dict__, f, indent=2)

    model, history = train_model(model_cfg, train_cfg, train_ds, out)
    plot_loss_curve(history, os.path.join(out, "loss_curve.png"))

    print(f"[train] params={model.num_params():,}  saved to {out}/model.pt")
    print(f"[train] loss curve: {out}/loss_curve.png")


if __name__ == "__main__":
    main()
