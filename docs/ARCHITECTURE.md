# Architecture

How the code is organized and where to graft new experiments.

## File map

```
AIExpermentLab/
├── train.py                  CLI entrypoint, no logic of its own.
├── benchmark.py              CLI entrypoint, runs evaluate() on a saved checkpoint.
├── lab/
│   ├── config.py             ModelConfig, TrainConfig, DataConfig (multi-ticker), SERIES presets.
│   ├── model.py              TinyForecaster + RMSNorm, CausalSelfAttention, SwiGLU, Block.
│   ├── tokenizer.py          ValueBucketTokenizer (placeholder for future text/symbolic work).
│   ├── training.py           train_model() loop, MSE + AdamW + grad clip.
│   ├── plotting.py           plot_predictions() + plot_loss_curve() (matplotlib, Agg backend).
│   ├── data/
│   │   ├── pipeline.py       fetch_prices(), Normalizer, WindowedSeries, build_datasets().
│   │   └── formats.py        log-return / sliding-window helpers used by experiments.
│   └── experiments/          One file per grafted feature (empty for now).
├── runs/                     Per-run output: model.pt, history.json, benchmark.json.
└── docs/
    ├── USAGE.md              How to install and run.
    └── ARCHITECTURE.md       This file.
```

## Data flow

```diagram
 per ticker
╭──────────────╮   ╭────────────────╮   ╭────────────────╮
│ yfinance     │──▶│ Normalizer     │──▶│ WindowedSeries │──╮
│ (cached CSV) │   │ (per-ticker μσ)│   │ (seq_len, h)   │  │
╰──────────────╯   ╰────────────────╯   ╰────────────────╯  │
                                                            ▼
                                              ╭──────────────────────╮
                                              │ ConcatDataset        │
                                              │ (joint training set) │
                                              ╰──────────┬───────────╯
                                                         │
                                                         ▼
                       ╭──────────────────────────────────────────────╮
                       │ TinyForecaster                               │
                       │   input_proj → +pos → N×Block → norm → head  │
                       ╰──────────────────────────────────────────────╯
                                                         │
                                ╭────────────────────────┴────────╮
                                │ Train: MSE → AdamW → clip       │
                                │ Bench: per-ticker invert norm → │
                                │        MAE/RMSE/MAPE/DirAcc +   │
                                │        PNG plots                │
                                ╰─────────────────────────────────╯
```

Each ticker is normalized with its **own** mean/std. The model sees a single
mixed stream during training, but each evaluation ticker is de-normalized with
the matching `Normalizer` so dollar metrics and plots stay on the right scale.

## Model

A textbook decoder-only transformer block:

* **RMSNorm** (pre-norm).
* **Causal self-attention** via `F.scaled_dot_product_attention(is_causal=True)`.
* **SwiGLU** FFN.
* **Sinusoidal positional encoding** added once after the input projection.

The only "stock-specific" piece is `nn.Linear(input_dim, d_model)` at the front and
`nn.Linear(d_model, output_dim)` at the back. To retarget at text:

1. Replace `input_proj` with an `nn.Embedding(vocab_size, d_model)`.
2. Replace `head` with `nn.Linear(d_model, vocab_size)`.
3. Swap `MSELoss` for `CrossEntropyLoss` over a shifted target.

Everything in between is the same generic causal transformer.

## Where new experiments go

Each backlog item from the README lands as its own file under `lab/experiments/`,
exposing an opt-in flag in `train.py`. Examples (not yet implemented):

```
lab/experiments/sleep_gate.py        --sleep-gate-cap N
lab/experiments/recurrent_depth.py   --recurrent-loops N --recurrent-lora-rank R
lab/experiments/mtp.py               --mtp-horizons 1,2,4
```

The `Block` class in `lab/model.py` is the integration point: an experiment either
wraps it, replaces it, or injects an extra step before/after it.

## Conventions

* Configs are dataclasses. Pass them around, don't smuggle globals.
* Every run produces `model.pt`, `history.json`, `data_cfg.json`, `norm.json`,
  and (after benchmarking) `benchmark.json`. Anything reproducible from the run
  directory alone.
* CPU-first. If a feature needs a GPU, gate it on `--device cuda` and keep the
  CPU path working.
