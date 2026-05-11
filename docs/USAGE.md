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

```bash
python train.py --series Glint --ticker AAPL \
    --train-start 2024-01-01 --train-end 2024-12-31 \
    --test-start 2025-01-01 --test-end 2026-01-01 \
    --epochs 30 --device cpu \
    --output runs/Glint_AAPL
```

Output goes to `runs/Glint_AAPL/`:

```
runs/Glint_AAPL/
├── model.pt        # best validation checkpoint
├── history.json    # per-epoch loss curve
├── data_cfg.json   # data config (so benchmark.py can rehydrate it)
└── norm.json       # mean/std used to normalize the training series
```

`yfinance` data is cached to `lab/data/_cache/` after the first download.

## 3. Benchmark

```bash
python benchmark.py --model runs/Glint_AAPL/model.pt
```

Reuses the data config saved during training. Override the test window or ticker if you want
to evaluate on something different:

```bash
python benchmark.py --model runs/Glint_AAPL/model.pt \
    --ticker MSFT --test-start 2025-06-01 --test-end 2026-01-01
```

The benchmark writes `runs/Glint_AAPL/benchmark.json` and prints a summary.

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
| `--ticker` | `AAPL` | Any symbol `yfinance` understands. |
| `--train-start` / `--train-end` | `2024-01-01` / `2024-12-31` | Training window. |
| `--test-start` / `--test-end` | `2025-01-01` / `2026-01-01` | Saved into the run for later benchmarking. |
| `--seq-len` | `32` | Number of past days fed into the model. |
| `--horizon` | `1` | Days ahead to predict. |
| `--epochs` | `20` | Training epochs. |
| `--batch-size` | `32` | |
| `--lr` | `1e-3` | AdamW learning rate. |
| `--device` | `cpu` | Use `cuda` if you have it. |
| `--output` | `runs/<series>_<ticker>` | Run directory. |
| `--seed` | `42` | RNG seed. |

### `benchmark.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | _required_ | Path to `model.pt`. |
| `--run-dir` | dirname of `--model` | Where to read `data_cfg.json` and `norm.json` from. |
| `--device` | `cpu` | |
| `--ticker` | from run config | Override the symbol. |
| `--test-start` / `--test-end` | from run config | Override the test window. |

## 6. Quick sanity test

```bash
python train.py --epochs 2 --output runs/_smoke
python benchmark.py --model runs/_smoke/model.pt
rm -rf runs/_smoke
```

If both commands exit zero and `benchmark.json` contains numeric metrics, the install is good.
