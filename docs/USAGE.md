# Usage

Step-by-step guide for running the baseline stock forecaster end to end on CPU.

## 1. Install

```bash
git clone https://github.com/compactaiofficial/aiexpermentlab.git
cd aiexpermentlab
pip install -r requirements.txt
```

Tested on Python 3.12 with PyTorch 2.10. CPU is enough for the default `Glint` size.

## 2. Train

By default the model trains on **3 years of data** (2022–2024) before
predicting 2025–2026, so it has enough market history to learn from.

Single ticker:

```bash
python train.py --series Glint --tickers AAPL \
    --train-start 2022-01-01 --train-end 2024-12-31 \
    --test-start 2025-01-01 --test-end 2026-01-01 \
    --epochs 30 --device cpu
```

Multiple tickers (one model trained jointly):

```bash
python train.py --series Glint --tickers AAPL,NVDA \
    --train-start 2022-01-01 --train-end 2024-12-31 \
    --epochs 30
```

Output goes to `runs/<series>_<TICKERS>/`:

```
runs/Glint_AAPL_NVDA/
├── model.pt          # best validation checkpoint
├── history.json      # per-epoch loss curve
├── data_cfg.json     # data config (so benchmark.py can rehydrate it)
├── norms.json        # per-ticker mean/std used during training
└── loss_curve.png    # train/val loss plot
```

`yfinance` data is cached to `lab/data/_cache/` after the first download.

## 3. Benchmark

Three forecast modes control how much real data the model sees during evaluation:

| Mode | What happens |
|------|-------------|
| `blind` (default) | Seed with last `seq_len` training days, then autoregressive — the model never sees test prices. |
| `nonblind` | One-step-ahead: at every step the model receives the *real* previous price, so errors don't compound. |
| `partial` | Like blind, but every `--partial-interval` steps (default 126 ≈ half a trading year) the model gets one real price to re-anchor. |

```bash
# Blind (default) — same as omitting --mode
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt --mode blind

# Non-blind one-step-ahead
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt --mode nonblind

# Partial: re-anchor every half year
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt --mode partial

# Custom re-anchor interval (every 63 steps ≈ 3 months)
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt --mode partial --partial-interval 63
```

Each mode produces per-ticker metrics and a PNG:

```bash
runs/Glint_AAPL_NVDA/
├── benchmark.json
└── plots/
    ├── AAPL_blind.png
    ├── AAPL_nonblind.png
    ├── AAPL_partial.png
    ├── NVDA_blind.png
    ├── NVDA_nonblind.png
    └── NVDA_partial.png
```

Override the test window or evaluate a different set of tickers (must have been seen at
training time, since each needs a fitted normalizer):

```bash
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt \
    --tickers AAPL --test-start 2025-06-01 --test-end 2026-01-01

# Skip plotting entirely
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt --no-plot
```

Reuses the data config saved during training. For each ticker it prints metrics and writes
a `Predicted vs Actual` PNG:

```
runs/Glint_AAPL_NVDA/
├── benchmark.json
└── plots/
    ├── AAPL_pred_vs_actual.png
    └── NVDA_pred_vs_actual.png
```

Override the test window or evaluate a different set of tickers (must have been seen at
training time, since each needs a fitted normalizer):

```bash
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt \
    --tickers AAPL --test-start 2025-06-01 --test-end 2026-01-01

# Skip plotting entirely
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt --no-plot
```

## 4. Reading the metrics

| Metric | Meaning |
|--------|---------|
| `mae`, `rmse` | Absolute / squared error in dollars on the de-normalized close price. |
| `mape_pct` | Mean absolute percentage error. Lower is better. |
| `directional_accuracy` | Fraction of days where the predicted direction (up/down vs the last input value) matches the true direction. 0.5 = random. |
| `naive_mae`, `naive_rmse` | The "predict yesterday's close" baseline. Beating this on RMSE is hard. |
| `skill_vs_naive_rmse` | `1 - rmse / naive_rmse`. Positive means the model beats the naive baseline. |

For a tiny model on a daily close series, `mape_pct` in the low single digits and
`directional_accuracy > 0.5` is a useful sanity check that training did something.

## 5. CLI reference

### `train.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--series` | `Glint` | Picks a model size from `lab/config.py:SERIES`. |
| `--tickers` | `AAPL` | Comma-separated `yfinance` symbols. |
| `--train-start` / `--train-end` | `2022-01-01` / `2024-12-31` | Training window (3 years by default). |
| `--test-start` / `--test-end` | `2025-01-01` / `2026-01-01` | Saved into the run for later benchmarking. |
| `--seq-len` | `32` | Number of past days fed into the model. |
| `--horizon` | `1` | Days ahead to predict. |
| `--epochs` | `30` | Training epochs. |
| `--batch-size` | `32` | |
| `--lr` | `1e-3` | AdamW learning rate. |
| `--device` | `cpu` | Use `cuda` if you have it. |
| `--output` | `runs/<series>_<tickers>` | Run directory. |
| `--seed` | `42` | RNG seed. |
| `--diff-mode` | off | Train on price diffs instead of absolute prices. **Eliminates autoregressive drift.** |

### `benchmark.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | _required_ | Path to `model.pt`. |
| `--run-dir` | dirname of `--model` | Where to read `data_cfg.json` and `norms.json` from. |
| `--device` | `cpu` | |
| `--tickers` | from run config | Comma-separated override; each must have a normalizer in `norms.json`. |
| `--test-start` / `--test-end` | from run config | Override the test window. |
| `--no-plot` | off | Skip writing per-ticker PNGs. |
| `--mode` | `blind` | Forecast mode: `blind`, `nonblind`, or `partial`. |
| `--partial-interval` | `126` | Steps before a real-price anchor in `partial` mode (~126 trading days = half year). |

## 6. Quick sanity test

```bash
# Short train window so it finishes fast
python train.py --tickers AAPL --epochs 2 --train-start 2024-06-01 --train-end 2024-12-31 \
    --output runs/_smoke

# Test all three modes
python benchmark.py --model runs/_smoke/model.pt --mode blind --no-plot
python benchmark.py --model runs/_smoke/model.pt --mode nonblind --no-plot
python benchmark.py --model runs/_smoke/model.pt --mode partial --no-plot
rm -rf runs/_smoke
```

If all three exit zero, `benchmark.json` contains numeric metrics, and the install is good.
