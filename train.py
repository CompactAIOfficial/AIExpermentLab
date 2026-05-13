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

    # experiment flags
    p.add_argument("--mtp-horizons", type=str, default=None,
                   help="Comma-separated auxiliary horizons, e.g. '2,4,8'")
    p.add_argument("--input-dropout", type=float, default=0.0,
                   help="Replace fraction of input tokens with zero during training")
    p.add_argument("--output-reg", type=float, default=0.0,
                   help="L2 penalty weight on output predictions")
    p.add_argument("--crowfeather", action="store_true",
                   help="Crowfeather AdamW: eps=1e-20, beta2 ramps 0.95->0.97 post-warmup")
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "wsd"],
                   help="LR schedule: cosine or warmup-stable-decay (WSD) with sqrt cooldown")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA decay rate for model weight averaging (0=disabled, typical 0.999-0.9999)")
    p.add_argument("--ohem-fraction", type=float, default=0.0,
                   help="Fraction of hardest examples to train on (0=disabled, 1=all). Online Hard Example Mining.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Smoothing epsilon for regression targets (0=disabled). Pulls targets toward batch mean.")
    return p.parse_args()


def main():
    args = parse_args()
    tickers = parse_tickers(args.tickers)

    model_cfg = SERIES[args.series]
    model_cfg.seq_len = args.seq_len

    if args.mtp_horizons:
        model_cfg.mtp_horizons = [int(h) for h in args.mtp_horizons.split(",")]
    else:
        model_cfg.mtp_horizons = []

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
        mtp_horizons=model_cfg.mtp_horizons,
    )

    tag = "_".join(tickers)
    out = args.output or os.path.join("runs", f"{args.series}_{tag}")
    os.makedirs(out, exist_ok=True)

    flags = []
    if args.mtp_horizons:
        flags.append(f"mtp={args.mtp_horizons}")
    if args.input_dropout > 0:
        flags.append(f"drop={args.input_dropout}")
    if args.output_reg > 0:
        flags.append(f"oreg={args.output_reg}")
    if args.crowfeather:
        flags.append("crow")
    if args.lr_schedule == "wsd":
        flags.append("wsd")
    if args.ema_decay > 0:
        flags.append(f"ema={args.ema_decay}")
    if args.ohem_fraction > 0:
        flags.append(f"ohem={args.ohem_fraction}")
    if args.label_smoothing > 0:
        flags.append(f"lsmooth={args.label_smoothing}")
    suffix = "_" + "_".join(flags) if flags else ""
    out = out + suffix

    print(f"[train] series={args.series}  tickers={tickers}  device={args.device}")
    print(f"[train] window={data_cfg.train_start}..{data_cfg.train_end}  horizon={args.horizon}")
    if flags:
        print(f"[train] flags: {' '.join(flags)}")
    print(f"[train] output={out}")

    train_ds, _, _ = build_datasets(data_cfg)
    print(f"[train] train_examples={len(train_ds)}  (across {len(tickers)} ticker(s))")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "data_cfg.json"), "w") as f:
        json.dump(data_cfg.__dict__, f, indent=2)

    model, history = train_model(
        model_cfg, train_cfg, train_ds, out,
        input_dropout=args.input_dropout,
        output_reg=args.output_reg,
        crowfeather=args.crowfeather,
        lr_schedule=args.lr_schedule,
        ema_decay=args.ema_decay,
        ohem_fraction=args.ohem_fraction,
        label_smoothing=args.label_smoothing,
    )
    plot_loss_curve(history, os.path.join(out, "loss_curve.png"))

    print(f"[train] params={model.num_params():,}  saved to {out}/model.pt")
    print(f"[train] loss curve: {out}/loss_curve.png")


if __name__ == "__main__":
    main()
