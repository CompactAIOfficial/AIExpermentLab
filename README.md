<div align="center">

# AIExpermentLab

### *"Well here we are again."* GLaDOS, Portal 2

**Open source training code for every weird idea I have decided to gather, bolt on, and benchmark.**

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support_the_Lab-FF5E5B?style=for-the-badge&logo=kofi&logoColor=white)](https://ko-fi.com/compactai)
[![License](https://img.shields.io/badge/License-AGPL_V3-blue?style=for-the-badge)](#license)
[![Status](https://img.shields.io/badge/Status-Day_1-orange?style=for-the-badge)](#progress-log)

</div>

---

## What this is

AIExpermentLab is the public successor to my private `MyTrainer` codebase. The goal is simple: start with the smallest plausible language model, then incrementally graft features from the old training scripts onto it, one by one, and log what actually moves the needle.

Every experiment lives in the open. Every flop, every win, every "huh, that was supposed to help" gets recorded in the [Progress Log](#progress-log) below.

```diagram
╭──────────────╮     ╭───────────────╮     ╭───────────────╮
│ Tiny Baseline│────▶│ Add 1 Feature │────▶│  Benchmark    │
╰──────┬───────╯     ╰───────┬───────╯     ╰───────┬───────╯
       │                     │                     │
       │                     ▼                     ▼
       │              ╭───────────────╮     ╭───────────────╮
       ╰──────────────│ Keep or Drop  │◀────│   Log it      │
                      ╰───────────────╯     ╰───────────────╯
```

---

## The Plan

1. Stand up a vanilla LLaMA-style transformer at the smallest reasonable scale (target under 1M params for the first run).
2. Train it. Benchmark it.
3. Pick one experimental component from the [feature backlog](#feature-backlog), graft it on, retrain, rebenchmark.
4. If it helps, keep it. If it hurts, document why and move on.
5. Repeat.

The first baseline lives on a tiny, well-defined task: **predict the next-day closing price of a stock**.
Train on 2024 prices, benchmark on 2025–2026. The architecture is a generic causal transformer
with no domain-specific tricks, so the same code can later be retargeted at text or any other
1-D sequence with only the input/output projections changed.

---

## Feature Backlog

These are the techniques pulled from the old MyTrainer codebase that are queued up for porting and evaluation. Each will be implemented as an opt-in flag so we can run honest A/B tests.

### Architecture experiments

| Feature | Origin | Status |
|---------|--------|--------|
| Recurrent depth core (Prelude → Recurrent Core → Coda, Mythos-style) | TinyMemoryLM | queued |
| Stable Recurrent Injection (SSM-style `h = A·h + B·e + Δ`, Parcae §3.1) | TinyMemoryLM | queued |
| Depth LoRA (per-loop low-rank adaptation with learned loop embeddings) | TinyMemoryLM | queued |
| Adaptive Halting (per-position learned stop probability) | TinyMemoryLM | queued |
| SleepGate (persistent cross-sequence memory buffer, periodic consolidation) | `sleep_gate.py` | queued |
| TRIM-KV retention gate (arxiv:2512.03324, learned per-entry retention β) | `sleep_gate.py` | queued |
| Engram (DeepSeek-style hashed n-gram conditional memory, O(1) lookup) | `EngramBlock` | queued |
| Manifold Hyper-Connections (Sinkhorn-Knopp doubly stochastic residual mixing) | `mhc` | queued |
| COCONUT-style Latent Thinking blocks (continuous chain-of-thought) | `latent_think_layers` | queued |
| Multi-Token Prediction (auxiliary heads at future horizons) | `mtp_horizons` | **tested** |
| Per-Layer Embeddings (Gemma 3n PLE, token-conditional per-layer signal) | `ple_dim` | queued |
| Auxiliary heads (bigram prediction, word boundary detection, L11) | `aux_*` | queued |
| GQA, partial RoPE, sliding window, QK-norm, per-head output gating | `CausalSelfAttention` | queued |
| SwiGLU FFN with weight-tied embeddings + learned output bias | base block | queued |

### Training technique experiments

| Feature | Origin | Status |
|---------|--------|--------|
| Knowledge distillation (HF teacher with token alignment + soft KL) | `distillation.py` | queued |
| Basic distillation mode (plain SFT on teacher prompt/response pairs) | `distillation.py` | queued |
| Pre-computed teacher logprob datasets (top-k cached, no live teacher) | `distillation.py` | queued |
| SPIN self-play fine-tuning (arxiv:2401.01335) | `spin.py` | queued |
| DPO / IPO / SimPO / ORPO / KTO preference optimization | `dpo.py` | queued |
| GRPO RL training with reward-weighted policy gradient | `rl.py` | queued |
| Repetition-removal RL (no distillation required) | `rl.py` | queued |
| Math RL with parameter-efficient frozen early layers | `rl.py` | queued |
| PPG (Progressive Parameter Grouping, low-dim warmup then split) | `ppg.py` | queued |
| Anti-pattern unlikelihood loss (push down curated bad continuations) | `anti_patterns.py` | queued |
| GADW (Gradient Aware Dynamic Weighting for multi-loss balancing) | `gadw.py` | queued |
| Curriculum learning (recurrent loop count ramp-up) | `curriculum.py` | queued |
| Model averaging (EMA / SWA style weight smoothing) | `averaging.py` | queued |
| Muon optimizer (Newton-Schulz orthogonalization) | `optimizer.py` | queued |
| Crowfeather AdamW (eps=1e-20, β2 ramp 0.95→0.97 post-warmup) | `optimizer.py` | queued |
| WSD schedule (sqrt cooldown) vs cosine LR schedule | `training.py` | queued |
| FIM (Fill-In-the-Middle) augmentation during pretraining | `training.py` | queued |
| Decontamination pass against eval suites | `data/decontamination.py` | queued |
| OHEM (Online Hard Example Mining) with dynamic threshold | `training.py` | queued |
| Looping regularization (OpenMythos protection against weight collapse) | `training.py` | queued |
| Input token dropout (replace fraction of inputs with zero) | `training.py` | **tested** |
| Context loss (NCE-based contrastive prompt/response embedding loss) | `training.py` | queued |
| Z-loss, entropy regularization, label smoothing | `training.py` | queued |
| Output L2 regularization (penalize extreme predictions) | `output_reg.py` | **tested** |
| Sleep capacity loss (TRIM-KV penalty for over-budget retention) | `sleep_gate.py` | queued |
| Think depth loss (cosine similarity penalty for lazy COCONUT layers) | `training.py` | queued |
| BatchPrefetcher (GPU-resident ring buffer, async CPU→GPU transfer) | `training.py` | queued |
| Auto batch tuning (find largest batch that fits in VRAM) | `training.py` | queued |
| Mixed precision: BF16, NVFP4, FP8 via TransformerEngine | `training.py` | queued |
| Crash recovery checkpoint (emergency save on training exception) | `training.py` | queued |
| SLERP merging (spherical linear interpolation of N checkpoints) | `merging.py` | queued |
| FrankenMoE (Mixture-of-Experts assembly from independent checkpoints) | `merging.py` | queued |

### Tokenizer experiments

| Tokenizer | Notes | Status |
|-----------|-------|--------|
| Character-level (LetterTokenizer with dynamic vocab extension) | simple, interpretable | queued |
| ByteLevel BPE (~2000 vocab default) | balanced default | queued |
| Metaspace BPE (~500 vocab) | dense signal for tiny models | queued |

---

## Repository Layout

```diagram
AIExpermentLab/
├── README.md                  ◀── you are here
├── train.py                   (main entry, grows over time)
├── benchmark.py               (eval suite, ported from old benchmark.py)
├── lab/
│   ├── config.py              (ModelConfig + TrainConfig + series defs)
│   ├── model.py               (vanilla baseline, then experimental blocks)
│   ├── tokenizer.py           (char + BPE + metaspace)
│   ├── training.py            (loop, prefetch, schedules)
│   ├── data/
│   │   ├── pipeline.py        (streaming, caching, mmap shards)
│   │   └── formats.py         (SFT / pretrain format adapters)
│   └── experiments/           (one file per grafted feature)
└── runs/                      (checkpoints, logs, sample outputs)
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Train on 10 years of history, predict 2025–2026
python train.py --series Shard --tickers AAPL,NVDA \
    --train-start 2014-01-01 --train-end 2024-12-31 \
    --epochs 100 --device cuda

# Benchmark on 2025–2026 (choose a mode)
python benchmark.py --model runs/Shard_AAPL_NVDA/model.pt --mode blind      # never sees test prices
python benchmark.py --model runs/Shard_AAPL_NVDA/model.pt --mode nonblind   # one-step-ahead
python benchmark.py --model runs/Shard_AAPL_NVDA/model.pt --mode partial    # re-anchor every ~6mo
```

You can pass any comma-separated list of `yfinance` tickers to `--tickers`
(e.g. `AAPL`, `AAPL,NVDA`, `AAPL,MSFT,GOOGL,NVDA`). The model is trained on the
concatenation of per-ticker normalized series, then evaluated on each ticker
independently with its own normalizer.

**Training covers everything before the test window** (default 2022–2024).
The benchmark seeds from the last `seq_len` real training days, then runs in the
selected mode — blind (autoregressive), nonblind (one-step-ahead), or partial
(re-anchored every N steps).

**`--diff-mode` eliminates the autoregressive drift problem**: instead of
predicting absolute prices (which sag toward the training mean under uncertainty),
the model learns daily price changes (diffs). The uncertainty default becomes
"no change" (flat at the last known price) instead of "sag toward the past mean."

See [docs/USAGE.md](docs/USAGE.md) for full options, ticker swapping, and how to read the
benchmark output. Architecture notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Every feature graft will land behind a flag, for example:

```bash
python train.py --recurrent-loops 4 --recurrent-lora-rank 8
python train.py --sleep-gate-cap 64
```

---

## Progress Log

A running journal of every experiment. Newest entries on top.

### Day 4, experiment framework + three feature A/B tests

* **What**: Implemented three queued features behind opt-in `--flags` and ran full 3-mode benchmarks on Glint (82K params) vs a freshly re-trained baseline. Training window: 2022-01-01..2024-12-31 (3 years), predict 2025-01-01..2026-01-01 (250 steps).
* **Features added**: `--mtp-horizons`, `--input-dropout`, `--output-reg`
* **Compute**: CPU, ~5 minutes total

#### Glint baseline (82K params)

| Mode | Ticker | MAPE | DirAcc | skill_vs_naive_rmse |
|------|--------|------|--------|---------------------|
| blind | AAPL | 1243.62% | 0.524 | -185.66 |
| blind | NVDA | 19.48% | 0.336 | -0.125 |
| nonblind | AAPL | 1.36% | 0.508 | +0.866 |
| nonblind | NVDA | 2.19% | 0.440 | +0.873 |
| partial | AAPL | 82.39% | 0.484 | -6.072 |
| partial | NVDA | 145.70% | 0.480 | -15.049 |

#### Glint + MTP (`--mtp-horizons 2,4,8`)

* **Verdict: KEEP**. Multi-token prediction with 3 auxiliary heads drastically improves blind and partial rollout without harming nonblind.

| Mode | Ticker | MAPE | DirAcc | skill_vs_naive_rmse | Δ baseline |
|------|--------|------|--------|---------------------|------------|
| blind | AAPL | **15.36%** | 0.312 | -0.179 | **+185.48** |
| blind | NVDA | **17.96%** | 0.372 | +0.048 | **+0.173** |
| nonblind | AAPL | 1.33% | 0.504 | +0.867 | +0.001 |
| nonblind | NVDA | 2.17% | 0.456 | +0.874 | +0.001 |
| partial | AAPL | **14.19%** | 0.512 | -0.108 | **+5.964** |
| partial | NVDA | **10.15%** | 0.544 | +0.492 | **+15.541** |

#### Glint + Input Dropout sweep (`--input-dropout` 0.01 / 0.05 / 0.10 / 0.15)

Dropout 0.15 destabilises the model completely in multi-step modes (MAPE explodes to millions). Lower rates all help — **0.01 is the sweet spot**.

| Rate | Mode | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|------|------|-----------|------------|-----------|------------|
| 0.00 (base) | blind | 1243.62% | -185.66 | 19.48% | -0.125 |
| 0.01 | blind | **18.12%** | -0.405 | **16.89%** | **+0.202** |
| 0.05 | blind | 15.35% | -0.175 | 30.87% | -0.960 |
| 0.10 | blind | **9.51%** | **+0.119** | 21.17% | -0.270 |
| 0.00 (base) | partial | 82.39% | -6.072 | 145.70% | -15.049 |
| 0.01 | partial | **15.98%** | -0.275 | **15.86%** | **+0.250** |
| 0.05 | partial | 33.08% | -2.172 | 10.06% | +0.400 |
| 0.10 | partial | 11.28% | -0.099 | 15.92% | -0.009 |

* **Verdict: KEEP** (at 0.01). Even 1% input randomisation (zeroing 1/100 inputs) provides massive blind-mode regularisation. The effect diminishes above 0.01 for AAPL blind, though NVDA partial peaks at 0.05. Start with 0.01 as default.

#### Glint + Output Reg (`--output-reg 0.001`)

* **Verdict: NEUTRAL / needs more data**. Small improvements on partial modes (~12% lower MAPE on AAPL, ~9% on NVDA) but still far behind MTP or dropout. Worth retesting with higher weights.

| Mode | Ticker | MAPE | DirAcc | skill_vs_naive_rmse | Δ baseline |
|------|--------|------|--------|---------------------|------------|
| blind | AAPL | 968.25% | 0.524 | -140.42 | +45.24 |
| blind | NVDA | 19.46% | 0.332 | -0.124 | +0.001 |
| nonblind | AAPL | 1.36% | 0.508 | +0.866 | 0.000 |
| nonblind | NVDA | 2.19% | 0.440 | +0.873 | 0.000 |
| partial | AAPL | 72.58% | 0.484 | -5.213 | +1.859 |
| partial | NVDA | 132.31% | 0.480 | -13.367 | +1.682 |

#### Summary

| Feature | Flag | Verdict | Notes |
|---------|------|---------|-------|
| Multi-Token Prediction | `--mtp-horizons 2,4,8` | **KEEP** | 80x blind MAPE reduction on AAPL, 14x partial on NVDA |
| Input Token Dropout | `--input-dropout 0.01` | **KEEP** | 69x blind MAPE reduction on AAPL at 1% rate |
| Output L2 Regularization | `--output-reg 0.001` | needs data | Marginal improvement, far behind MTP/dropout |

#### Comparison of best features (blind mode)

| Model | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|-------|-----------|------------|-----------|------------|
| Baseline | 1243.62% | -185.66 | 19.48% | -0.125 |
| +MTP 2,4,8 | 15.36% | -0.18 | **17.96%** | **+0.048** |
| +Input Dropout 0.10 | **9.51%** | **+0.119** | 21.17% | -0.270 |
| +Output Reg 0.001 | 968.25% | -140.42 | 19.46% | -0.124 |

MTP gives the best all-round improvement; dropout 0.10 gives the best single-ticker result (AAPL blind hits positive skill for the first time).

#### Implementation

Each feature lives in `lab/experiments/{mtp,input_dropout,output_reg}.py` and is wired through `lab/model.py` and `lab/training.py`. The `--mtp-horizons` flag also modifies the dataset (`lab/data/pipeline.py`) to produce multi-horizon targets.

```bash
python train.py --mtp-horizons 2,4,8
python train.py --input-dropout 0.01
python train.py --output-reg 0.001
```

Run dirs: `runs/Glint_AAPL_NVDA_{mtp=2,4,8,drop=0.01,drop=0.05,drop=0.1,drop=0.15,oreg=0.001}`


### Day 3, local normalisation — real predictions, no drift

* **Problem recap**: global z-score normalisation caused blind predictions to sag
  toward the training mean (price-mode). Training on diffs flattened predictions
  because daily returns are ~zero mean noise.
* **Fix**: **local normalisation** — each window of `seq_len` prices is independently
  z-scored so the model only sees **shapes**, never absolute levels. Under uncertainty
  the model defaults to the *recent* window mean (= last 2 months), not the 10-year mean.
  No systematic drift, no flat line.
* Training on 10 years of history (2014–2024), predicting 2025–2026 blind (250-step
  autoregressive rollout). Early stopping added (patience=7 epochs).

  #### Shard (656K params)

  | Mode | Ticker | MAPE   | DirAcc | skill_vs_naive_rmse |
  |------|--------|--------|--------|---------------------|
  | blind | AAPL  | 13.42% | 0.464  | -0.03               |
  | blind | NVDA  | 17.89% | 0.512  | **+0.10**           |
  | nonblind | AAPL | 1.30%  | 0.504  | **+0.87**           |
  | nonblind | NVDA | 2.17%  | 0.444  | **+0.87**           |
  | partial | AAPL | 13.08% | 0.500  | -0.04               |
  | partial | NVDA | 15.74% | 0.520  | **+0.23**           |

  #### Glint (82K params)

  | Mode | Ticker | MAPE   | DirAcc | skill_vs_naive_rmse |
  |------|--------|--------|--------|---------------------|
  | blind | AAPL  | 17.41% | 0.524  | -0.36               |
  | blind | NVDA  | 16.68% | 0.536  | **+0.21**           |
  | nonblind | AAPL | 1.31%  | 0.516  | **+0.87**           |
  | nonblind | NVDA | 2.17%  | 0.444  | **+0.87**           |

* **NVDA blind captures the uptrend** — positive skill on both model sizes, prediction
  follows the actual upward trajectory for ~150 steps before diverging.
* **AAPL blind stays in the right range** (~$230–250) with no systematic sag.
* **Nonblind beats naive massively** (+0.87 skill) — the model genuinely learned
  one-step-ahead patterns from 10 years of data.
* Plots: `runs/Shard_AAPL_NVDA/plots/{AAPL,NVDA}_{blind,nonblind,partial}.png`.
* Trained on RTX 5090 in ~4 seconds total. CPU equivalent ~60 seconds.


### Day 1, multi-ticker + plotting + blind correction

* Stood up the smallest plausible causal transformer (RMSNorm + SwiGLU + sinusoidal pos enc, 82,433 params).
* Task: next-day close prediction. Train on 2024, benchmark 2025–2026.
* Multi-ticker support (--tickers), per-ticker normalizers, prediction-vs-actual plots.
* Corrected benchmark from one-step-ahead to blind autoregressive rollout.
* Plots: `runs/Glint_AAPL_NVDA/plots/{AAPL,NVDA}_pred_vs_actual.png` (superseded by Day 2).
* Compute: ~2.5 GPU hours
* Notes: Took way to long to do by hand, thanks AMP code! (not sponsored, just really good coding agent!)

### Day 0, repo bootstrap

* Created the repository.
* Audited the old MyTrainer codebase, extracted the full backlog of architecture and training experiments.
* Wrote this README.

<details>
<summary>Format used for future entries</summary>

```
### Day N, <short title>
* Branch / commit: <hash>
* Series: <Glint|Shard|Prism>
* Features changed: <list>
* Compute: <GPU hours>
* Result: <metric delta vs previous>
* Verdict: keep / drop / needs more data
* Notes: <prose>
```

</details>

---

## Why open source

The MyTrainer codebase grew into a tangle of half-finished experiments behind a closed door. AIExpermentLab is the antidote. Every commit is public. Every benchmark result is reproducible from the configs in this repo.

If you want to fork, ablate, or argue about a specific experiment, open an issue or a PR. If you want to follow along, watch the [Progress Log](#progress-log).

---

## Support the lab

Training compute costs money. If any of this is useful to you, a coffee keeps the GPUs warm.

<div align="center">

[![Ko-fi](https://img.shields.io/badge/☕_Buy_me_a_coffee-FF5E5B?style=for-the-badge&logo=kofi&logoColor=white)](https://ko-fi.com/compactai)

**[ko-fi.com/compactai](https://ko-fi.com/compactai)**

</div>

---

## Credits

Lineage notes for the experiments queued above:

* **Mythos / Parcae**: recurrent-depth transformer with stable injection.
* **OpenMythos**: looping regularization against recurrent weight collapse.
* **DeepSeek**: Engram-style hashed n-gram conditional memory.
* **Gemma 3n**: Per-Layer Embeddings.
* **COCONUT**: continuous latent chain-of-thought.
* **TRIM-KV** (arxiv:2512.03324): learned per-entry retention gate.
* **SPIN** (arxiv:2401.01335): self-play preference fine-tuning.
* **Crowfeather**: AdamW eps and β2 ramp recipe.
* **Muon**: Newton-Schulz orthogonalized optimizer.

All bugs, regressions, and bad ideas are mine.

---

<div align="center">

*Built in public. Logged in public. [ko-fi.com/compactai](https://ko-fi.com/compactai).*

</div>
