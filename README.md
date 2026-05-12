<div align="center">

# AIExpermentLab

### *"Well, here we are again."* GLaDOS, Portal 2

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
| Multi-Token Prediction (auxiliary heads at future horizons) | `mtp_horizons` | queued |
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
| Input token dropout (replace fraction of inputs with `<UNK>`) | `training.py` | queued |
| Context loss (NCE-based contrastive prompt/response embedding loss) | `training.py` | queued |
| Z-loss, entropy regularization, label smoothing | `training.py` | queued |
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

# Train on 3 years of history (2022–2024), then predict 2025–2026
# Use --diff-mode to prevent blind autoregressive drift toward the training mean
python train.py --series Glint --tickers AAPL,NVDA \
    --train-start 2022-01-01 --train-end 2024-12-31 \
    --diff-mode --epochs 30

# Benchmark on 2025–2026 (choose a mode)
python benchmark.py --model runs/Glint_AAPL_NVDA_diff/model.pt --mode blind      # never sees test prices
python benchmark.py --model runs/Glint_AAPL_NVDA_diff/model.pt --mode nonblind   # one-step-ahead
python benchmark.py --model runs/Glint_AAPL_NVDA_diff/model.pt --mode partial    # re-anchor every ~6mo
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

### Day 3, diff-mode: autoregressive drift killed

* The model was systematically sagging toward the 2022–24 mean in long blind rollouts
  because MSE loss on z-scored **prices** defaults to "predict the training mean" under
  uncertainty.
* Fix: train on price **diffs** (daily changes) instead of absolute prices. Diffs are
  stationary around zero, so the model's uncertainty default is "no change" (flat
  prediction), not "sag toward the past mean".
* Added `--diff-mode` flag to `train.py`. The model architecture is unchanged; only
  the data pipeline converts prices → diffs on the fly.
* Retrained Glint on AAPL + NVDA (2022–24 training, 2025–26 test):

  | Mode | Ticker | MAPE   | DirAcc | skill_vs_naive_rmse |
  |------|--------|--------|--------|---------------------|
  | blind | AAPL  | 15.04% | 0.524  | -0.16               |
  | blind | NVDA  | 15.22% | 0.532  | **+0.26**           |
  | nonblind | AAPL | 17.64% | 0.492 | -0.35               |
  | nonblind | NVDA | 12.77% | 0.512 | **+0.38**           |
  | partial | AAPL | 15.04% | 0.528  | -0.16               |
  | partial | NVDA | 15.24% | 0.532  | **+0.26**           |

* NVDA beats the naive baseline (positive skill) — the model learned real signal.
* AAPL blind prediction is a flat line at ~$240 (last training price).
  **No downward drift whatsoever.**
* Plots: `runs/Glint_AAPL_NVDA_diff/plots/{AAPL,NVDA}_{blind,nonblind,partial}.png`.

### Day 2, three benchmark modes + 3 years training

* Training window widened to 3 years (2022–2024) so the model has enough
  market history before predicting 2025–2026.
* Three benchmark modes added to `benchmark.py --mode`:
  - **blind**: seed from last `seq_len` training days, autoregressive rollout.
  - **nonblind**: one-step-ahead (receives real previous price each step).
  - **partial**: re-anchors with one real price every N steps (default 126 ≈ half year).
* Same 82,433-param Glint model trained on AAPL + NVDA jointly.
* Results:

  | Mode | Ticker | n   | MAPE   | DirAcc | skill_vs_naive_rmse |
  |------|--------|-----|--------|--------|---------------------|
  | blind | AAPL  | 250 | 12.51% | 0.476  | -0.35               |
  | blind | NVDA  | 250 | 37.14% | 0.464  | -1.34               |
  | nonblind | AAPL | 250 | 1.57%  | 0.512  | +0.85               |
  | nonblind | NVDA | 250 | 2.64%  | 0.440  | +0.85               |
  | partial | AAPL | 250 | 11.95% | 0.480  | -0.30               |
  | partial | NVDA | 250 | 26.37% | 0.464  | -0.61               |

* Partial re-anchor helps NVDA noticeably (26% vs 37% blind) but barely moves
  AAPL — the higher-volatility stock benefits more from a reality reset.
* Nonblind one-step-ahead beats naive (positive skill) and shows the model *can*
  learn something; the problem is pure autoregressive drift.
* Plots: `runs/Glint_AAPL_NVDA/plots/{AAPL,NVDA}_{blind,nonblind,partial}.png`.

### Day 1, multi-ticker + plotting + blind correction

* Stood up the smallest plausible causal transformer (RMSNorm + SwiGLU + sinusoidal pos enc, 82,433 params).
* Task: next-day close prediction. Train on 2024, benchmark 2025–2026.
* Multi-ticker support (--tickers), per-ticker normalizers, prediction-vs-actual plots.
* Corrected benchmark from one-step-ahead to blind autoregressive rollout.
* Plots: `runs/Glint_AAPL_NVDA/plots/{AAPL,NVDA}_pred_vs_actual.png` (superseded by Day 2).

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
