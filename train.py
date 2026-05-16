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

    # existing experiment flags
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

    # Day 2 experiment flags
    p.add_argument("--muon-lr", type=float, default=0.0,
                   help="Muon optimizer LR for 2D params (0=disabled). Uses Newton-Schulz orthogonalization.")
    p.add_argument("--mhc-streams", type=int, default=0,
                   help="Number of hyper-connection residual streams (0=disabled). Implements Manifold Hyper-Connections.")
    p.add_argument("--fim-rate", type=float, default=0.0,
                   help="Fill-In-the-Middle augmentation rate (0=disabled). Masks random spans during training.")
    p.add_argument("--gqa-kv-heads", type=int, default=0,
                   help="Number of KV heads for Grouped Query Attention (0=use full n_heads, standard MHA)")
    p.add_argument("--qk-norm", action="store_true",
                   help="Apply RMSNorm to Q and K before attention")
    p.add_argument("--slerp-t", type=float, default=0.0,
                   help="SLERP merge ratio (0=disabled). Marks run for checkpoint merging; actual merge is post-training.")
    p.add_argument("--lora-rank", type=int, default=0,
                   help="Depth LoRA rank (0=disabled). Adds low-rank adapters to all linear layers in each block.")
    p.add_argument("--lora-alpha", type=float, default=1.0,
                   help="LoRA scaling factor (default 1.0). Scaling = alpha / rank.")
    p.add_argument("--act-max-steps", type=int, default=0,
                   help="Adaptive Halting max ponder steps (0=disabled). Per-position dynamic computation.")
    p.add_argument("--act-epsilon", type=float, default=0.01,
                   help="ACT halting threshold (default 0.01). Position halts when accumulated prob > 1-epsilon.")
    p.add_argument("--act-time-penalty", type=float, default=0.001,
                   help="ACT ponder cost weight (default 0.001). Multiplied by avg steps used.")

    # Day 3 experiment flags
    p.add_argument("--latent-steps", type=int, default=0,
                   help="Number of latent thinking steps (0=disabled). COCONUT-style reasoning in continuous latent space.")
    p.add_argument("--ssm-decay", type=float, default=0.0,
                   help="SSM recurrent injection decay (0=disabled). Adds state-space model path through blocks.")
    p.add_argument("--curriculum-epochs", type=int, default=0,
                   help="Ramp latent steps from 0 to full over this many epochs (0=instant).")
    p.add_argument("--ple", action="store_true",
                   help="Enable Per-Layer Embeddings (PLE). Adds learned per-layer bias vectors to hidden states.")
    p.add_argument("--anti-pattern-weight", type=float, default=0.0,
                   help="Weight for anti-pattern unlikelihood loss (0=disabled). Penalizes confident wrong-direction predictions.")
    p.add_argument("--nce-weight", type=float, default=0.0,
                   help="Weight for NCE context contrastive loss (0=disabled). Pulls nearby positions together, pushes distant apart.")

    # Day 3 Round 2 experiment flags
    p.add_argument("--engram", action="store_true",
                   help="Enable Engram conditional memory. Hashes price n-grams into learned embedding lookup table.")
    p.add_argument("--engram-n", type=int, default=4,
                   help="N-gram length for Engram pattern hashing (default 4).")
    p.add_argument("--engram-table", type=int, default=512,
                   help="Engram embedding table size (default 512).")
    p.add_argument("--sleep-gate", action="store_true",
                   help="Enable SleepGate memory consolidation. Scores and consolidates hidden states across positions.")
    p.add_argument("--trim-kv", action="store_true",
                   help="Enable TRIM-KV learned retention gate. Per-key importance scoring in self-attention.")
    p.add_argument("--gadw", action="store_true",
                   help="Enable Gradient Aware Dynamic Weighting for multi-loss balancing. Learns adaptive loss weights.")
    p.add_argument("--recurrent-depth", type=int, default=0,
                   help="Number of recurrent core iterations for Recurrent-Depth Transformer (0=disabled). Mythos-style Prelude→Core→Coda.")
    p.add_argument("--think-depth-weight", type=float, default=0.0,
                   help="Weight for think depth loss on latent reasoning steps (0=disabled). Penalizes lazy COCONUT steps with high cosine similarity.")

    # Day 4 experiment flags
    p.add_argument("--stoch-depth", type=float, default=0.0,
                   help="Survival probability for stochastic depth (0=disabled, 1=always active). Randomly drops entire blocks during training.")
    p.add_argument("--aux-direction", type=float, default=0.0,
                   help="Weight for auxiliary direction classification loss (0=disabled). Binary BCE on up/down price direction.")
    p.add_argument("--loop-reg", type=float, default=0.0,
                   help="L2 penalty weight on hidden state norm during loop operations (0=disabled). Prevents hidden state collapse/explosion.")
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

    if args.mhc_streams > 0:
        model_cfg.mhc_streams = args.mhc_streams
    if args.gqa_kv_heads > 0:
        model_cfg.gqa_kv_heads = args.gqa_kv_heads
    if args.qk_norm:
        model_cfg.qk_norm = True
    if args.lora_rank > 0:
        model_cfg.lora_rank = args.lora_rank
        model_cfg.lora_alpha = args.lora_alpha
    if args.act_max_steps > 0:
        model_cfg.act_max_steps = args.act_max_steps
        model_cfg.act_epsilon = args.act_epsilon
        model_cfg.act_time_penalty = args.act_time_penalty

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
    if args.muon_lr > 0:
        flags.append(f"muon={args.muon_lr}")
    if args.mhc_streams > 0:
        flags.append(f"mhc={args.mhc_streams}")
    if args.fim_rate > 0:
        flags.append(f"fim={args.fim_rate}")
    if args.gqa_kv_heads > 0:
        flags.append(f"gqa={args.gqa_kv_heads}")
    if args.qk_norm:
        flags.append("qknorm")
    if args.slerp_t > 0:
        flags.append(f"slerp={args.slerp_t}")
    if args.lora_rank > 0:
        flags.append(f"lora={args.lora_rank}")
    if args.act_max_steps > 0:
        flags.append(f"act={args.act_max_steps}")
    if args.latent_steps > 0:
        flags.append(f"latent={args.latent_steps}")
    if args.ssm_decay > 0:
        flags.append(f"ssm={args.ssm_decay}")
    if args.curriculum_epochs > 0:
        flags.append(f"curric={args.curriculum_epochs}")
    if args.ple:
        flags.append("ple")
    if args.anti_pattern_weight > 0:
        flags.append(f"anti={args.anti_pattern_weight}")
    if args.nce_weight > 0:
        flags.append(f"nce={args.nce_weight}")
    if args.engram:
        flags.append(f"engram")
    if args.sleep_gate:
        flags.append("sleep")
    if args.trim_kv:
        flags.append("trimkv")
    if args.gadw:
        flags.append("gadw")
    if args.recurrent_depth > 0:
        flags.append(f"rdepth={args.recurrent_depth}")
    if args.think_depth_weight > 0:
        flags.append(f"thinkd={args.think_depth_weight}")
    if args.stoch_depth > 0:
        flags.append(f"stoch={args.stoch_depth}")
    if args.aux_direction > 0:
        flags.append(f"auxdir={args.aux_direction}")
    if args.loop_reg > 0:
        flags.append(f"loopreg={args.loop_reg}")
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
        muon_lr=args.muon_lr,
        fim_rate=args.fim_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        latent_steps=args.latent_steps,
        ssm_decay=args.ssm_decay,
        curriculum_epochs=args.curriculum_epochs,
        ple=args.ple,
        anti_pattern_weight=args.anti_pattern_weight,
        nce_weight=args.nce_weight,
        engram=args.engram,
        sleep_gate=args.sleep_gate,
        trim_kv=args.trim_kv,
        gadw=args.gadw,
        recurrent_depth=args.recurrent_depth,
        think_depth_weight=args.think_depth_weight,
        stoch_depth=args.stoch_depth,
        aux_direction=args.aux_direction,
        loop_reg=args.loop_reg,
    )
    plot_loss_curve(history, os.path.join(out, "loss_curve.png"))

    print(f"[train] params={model.num_params():,}  saved to {out}/model.pt")
    print(f"[train] loss curve: {out}/loss_curve.png")


if __name__ == "__main__":
    main()
