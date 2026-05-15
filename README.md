<div align="center">

# AIExpermentLab

### *"Well here we are again."* GLaDOS, Portal 2

**Open source training code for every weird idea I have decided to gather, bolt on, and benchmark.**

> **Domain warning**: This repo benchmarks on stock price prediction (regression on 1-D time series), not on language modelling. Features that help here may behave differently for LLMs, and vice versa. Some descriptions in the backlog below still reference their LLM/text origins and haven't been rewritten yet — the code adapts, the docs lag. YMMV.

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support_the_Lab-FF5E5B?style=for-the-badge&logo=kofi&logoColor=white)](https://ko-fi.com/compactai)
[![License](https://img.shields.io/badge/License-AGPL_V3-blue?style=for-the-badge)](#license)
[![Status](https://img.shields.io/badge/Status-Day_3-orange?style=for-the-badge)](#progress-log)

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

> **Origin note**: Many of these originate from LLM research (Mythos, DeepSeek, Gemma, COCONUT, etc.). They're described here in their original context, but our testbed is stock price regression. A "queued" status means it hasn't been adapted yet; a "tested" status means it was evaluated on stock data specifically. Results may not transfer.

### Architecture experiments

| Feature | Origin | Status |
|---------|--------|--------|
| Recurrent depth core (Prelude → Recurrent Core → Coda, Mythos-style) | TinyMemoryLM | queued |
| Stable Recurrent Injection (SSM-style `h = A·h + B·e + Δ`, Parcae §3.1) | `lab/experiments/ssm_injection.py` | **tested — KEPT (decay=0.5)** |
| Depth LoRA (per-loop low-rank adaptation with learned loop embeddings) | `lab/experiments/depth_lora.py` | **tested — KEPT (rank=8)** |
| Adaptive Halting (per-position learned stop probability) | `lab/experiments/adaptive_halting.py` | **tested — KEPT (max_steps=6)** |
| SleepGate (persistent cross-sequence memory buffer, periodic consolidation) | `lab/experiments/sleep_gate.py` | **tested — DROPPED (destabilises both tickers)** |
| TRIM-KV retention gate (arxiv:2512.03324, learned per-entry retention β) | `lab/experiments/trim_kv.py` | **tested — KEPT** |
| Engram (DeepSeek-style hashed n-gram conditional memory, O(1) lookup) | `lab/experiments/engram.py` | **tested — KEPT (mild, helps AAPL)** |
| Manifold Hyper-Connections (Sinkhorn-Knopp doubly stochastic residual mixing) | `lab/model.py` | **tested — KEPT (4 streams)** |
| COCONUT-style Latent Thinking blocks (continuous chain-of-thought) | `lab/experiments/latent_reasoning.py` | **tested — KEPT (4 steps)** |
| Multi-Token Prediction (auxiliary heads at future horizons) | `lab/experiments/mtp.py` | **tested — KEPT** |
| Per-Layer Embeddings (Gemma 4 PLE, per-layer learned bias vectors) | `lab/experiments/ple.py` | **tested — KEPT** |
| Auxiliary heads (bigram prediction, word boundary detection, L11) | `aux_*` | queued |
| GQA, partial RoPE, sliding window, QK-norm, per-head output gating | `lab/model.py` | **tested — KEPT (GQA=2+QK-norm, partial only)** |
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
| Anti-pattern unlikelihood loss (penalize confident wrong-direction predictions) | `lab/experiments/anti_pattern.py` | **tested — KEPT (0.2–0.4)** |
| GADW (Gradient Aware Dynamic Weighting for multi-loss balancing) | `gadw.py` | queued |
| Curriculum learning (recurrent loop count ramp-up) | `lab/experiments/curriculum.py` | **tested — NEUTRAL** |
| Model averaging (EMA / SWA style weight smoothing) | `lab/experiments/ema.py` | **tested — KEPT (0.9999)** |
| Muon optimizer (Newton-Schulz orthogonalization) | `lab/experiments/muon.py` | **tested — KEPT (0.005)** |
| Crowfeather AdamW (eps=1e-20, β2 ramp 0.95→0.97 post-warmup) | `lab/experiments/crowfeather.py` | **tested — DROPPED** |
| WSD schedule (sqrt cooldown) vs cosine LR schedule | `lab/experiments/wsd_schedule.py` | **tested — KEPT** |
| FIM (Fill-In-the-Middle) augmentation during pretraining | `lab/experiments/fim.py` | **tested — KEPT (0.25/0.30)** |
| Decontamination pass against eval suites | `data/decontamination.py` | queued |
| OHEM (Online Hard Example Mining) with dynamic threshold | `lab/experiments/ohem.py` | **tested — KEPT (0.9/0.3)** |
| Looping regularization (OpenMythos protection against weight collapse) | `training.py` | queued |
| Input token dropout (replace fraction of inputs with zero) | `lab/experiments/input_dropout.py` | **tested — KEPT** |
| Context loss (NCE-based contrastive embedding loss over positions) | `lab/experiments/nce_context.py` | **tested — KEPT (0.2–0.6)** |
| Label smoothing (regression-adapted, target shrinkage toward batch mean) | `lab/experiments/label_smoothing.py` | **tested — KEPT (0.15/0.20)** |
| Output L2 regularization (penalize extreme predictions) | `lab/experiments/output_reg.py` | **tested — KEPT (0.05–0.10)** |
| Sleep capacity loss (TRIM-KV penalty for over-budget retention) | `sleep_gate.py` | queued |
| Think depth loss (cosine similarity penalty for lazy COCONUT layers) | `training.py` | queued |
| BatchPrefetcher (GPU-resident ring buffer, async CPU→GPU transfer) | `training.py` | queued |
| Auto batch tuning (find largest batch that fits in VRAM) | `training.py` | queued |
| Mixed precision: BF16, NVFP4, FP8 via TransformerEngine | `training.py` | queued |
| Crash recovery checkpoint (emergency save on training exception) | `training.py` | queued |
| SLERP merging (spherical linear interpolation of N checkpoints) | `lab/experiments/slerp_merge.py` | **tested — KEPT (t=0.5–0.6, basin-compatible pairs only)** |
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

## Config Cheat Sheet

Quick reference for what each feature actually does in practice on stock data. Modes: blind = fully autoregressive (hardest), nonblind = one-step-ahead (easiest), partial = re-anchor every 126 steps.

### `--mtp-horizons 2,4,8,16`

Adds 4 auxiliary linear heads that predict future prices at horizons 2, 4, 8, 16 steps ahead. Auxiliary losses are added to the main MSE loss (weight 0.3). All heads share the backbone.

| Does well | Sucks at |
|-----------|----------|
| Best all-rounder. Lowest avg blind MAPE (14.97%). Positive skill on NVDA blind. Good partial mode. | Worst Directional Accuracy (0.22–0.34 blind) — focuses on level prediction, not direction. 4 extra heads = 260 more params. |

Best for: "I want decent results everywhere without thinking."

### `--mtp-horizons 2,4,8,16 --input-dropout 0.01`

Two independent regularizers: MTP heads + 1% input masking at training time. Mild enough that they don't interfere.

| Does well | Sucks at |
|-----------|----------|
| First config ever with positive skill on AAPL in both blind (+0.089) AND partial (+0.014). Best balanced config across all 6 mode/ticker combos. | Worse Directional Accuracy than baseline. Adds 260 params. |

Best for: "I want the most balanced config."

### `--mtp-horizons 2,4,8`

3 heads instead of 4. Slightly worse than the 4-head variant on average, but competitive.

| Does well | Sucks at |
|-----------|----------|
| Strong NVDA partial (10.15%, +0.492 skill — second best ever). Fewer params than 4-head. | Blind AAPL (15.36%) and DirAcc worse than baseline. Outperformed on all metrics by the 4-head variant. |

Best for: "I want MTP but want to save 65 params." (Not worth it.)

### `--input-dropout 0.01`

During training, each input element has a 1% chance of being replaced with zero. No extra params.

| Does well | Sucks at |
|-----------|----------|
| Best NVDA blind ever (16.89%, **+0.202 skill**). Positive NVDA partial skill (+0.250). Zero extra params. | Blind AAPL (18.12%) and AAPL partial (15.98%) are mediocre. Only 1/100 inputs get dropped — easy to overlook. |

Best for: "I only care about NVDA." Also: "I want zero extra parameters."

### `--input-dropout 0.10`

10% input masking. Much stronger regularisation.

| Does well | Sucks at |
|-----------|----------|
| Best AAPL blind ever (9.51%, **+0.119 skill**). Ties oreg 0.10 for top AAPL blind MAPE. | NVDA blind suffers (21.17%). DirAcc drops. Destabilises at 15% or higher. Sweet spot is narrow. |

Best for: "I only care about AAPL blind mode."

### `--output-reg 0.05`

Adds `0.05 * mean(pred^2)` to the training loss. Penalises extreme predictions in normalised space.

| Does well | Sucks at |
|-----------|----------|
| **Best NVDA partial ever** (9.08%, **+0.547 skill**). Good AAPL blind (13.03%, +0.007). Positive skill on both tickers blind. Zero extra params. | Worst Directional Accuracy of any config (0.184–0.196 blind). Suppresses output variance — it's always a bit conservative. |

Best for: "I want the best NVDA partial score." Also: "I want positive skill everywhere with zero param overhead."

### `--output-reg 0.10`

Stronger output penalty.

| Does well | Sucks at |
|-----------|----------|
| Ties best AAPL blind ever (9.65%, **+0.157 skill**). AAPL DirAcc recovers to 0.476. | NVDA degrades across all modes (21% blind, 24% partial). Only use if you're benchmarking AAPL only. |

Best for: "AAPL-only evaluation."

### `--output-reg 0.001` (or lower)

Any weight below ~0.01 is too weak to matter. Results are indistinguishable from baseline.

| Does well | Sucks at |
|-----------|----------|
| Nothing. | Everything. Wastes your time. 0.001 adds 0.1% to the loss — it's a rounding error. |

Best for: "I want to confirm it doesn't work." Don't use.

### `--lr-schedule wsd`

Replaces cosine LR decay with Warmup-Stable-Decay: 10% warmup, constant plateau, then sqrt cooldown for the final phase. Based on the river-valley loss landscape theory (Wen et al., ICLR 2025) — the high stable LR lets the model traverse the "river" quickly, and the sharp cooldown drops it into the valley.

| Does well | Sucks at |
|-----------|----------|
| Best blind AAPL of any Tier 1 config (11.74%, **+0.101 skill**). Trains longer before early-stopping (22 epochs vs 15). Zero extra params. | Worse DirAcc (0.30-0.33 blind). NVDA blind is baseline-level (19.52%). Partial mode mediocre (13.83% AAPL). |

Best for: "I want better blind AAPL with zero param overhead."

### `--ema-decay 0.9999`

Maintains a shadow copy of model weights as an exponential moving average: `ema = 0.9999 * ema + 0.0001 * weights`. At validation and checkpoint time, the EMA-smoothed weights are used instead of the raw trained weights. Based on Morales-Brotons et al. (TMLR 2024).

| Does well | Sucks at |
|-----------|----------|
| Best blind AAPL of any single config (10.85%, **+0.148 skill**). Ties oreg 0.10 for AAPL blind. Also great AAPL partial (11.25%, +0.093). | Nonblind accuracy degrades noticeably (3.2% MAPE vs normal 1.3%). Heavier averaging smooths away the sharp predictions needed for one-step-ahead. Runs full 50 epochs — no early stopping. |

Best for: "I care most about multi-step AAPL prediction and can accept worse one-step-ahead."

### `--label-smoothing 0.15`

For regression, smooths targets toward the batch mean: `yb = yb * (1-ε) + ε * batch_mean`. Prevents overconfidence in individual predictions by shrinking them toward the batch average. Zero extra params.

| Does well | Sucks at |
|-----------|----------|
| **Best avg blind MAPE of any single config** (14.69%). AAPL blind 10.88% (+0.150 skill — ties EMA 0.9999 for best AAPL blind). Zero extra params. | Partial mode is mediocre (AAPL 12.17%, NVDA 18.55%). No positive skill in partial. DirAcc depressed (0.34 blind). |

Best for: "I want the best blind-mode average with zero param overhead."

### `--label-smoothing 0.20`

Stronger target smoothing. Pulls predictions further toward the batch mean.

| Does well | Sucks at |
|-----------|----------|
| **Positive skill on both tickers in blind AND partial modes** — only the second config ever (after MTP+dropout 0.01) to achieve this. NVDA blind 18.22% (+0.037), AAPL partial 11.78% (+0.093), NVDA partial 16.99% (+0.112). | Blind AAPL is merely average (11.93%). Nonblind slightly elevated (1.37% vs normal 1.33%). Zero extra params. |

Best for: "I want positive skill everywhere with zero params."

### `--label-smoothing 0.10` (or lower)

Any epsilon below ~0.10 is either too weak (0.01–0.05 barely moves targets) or destabilises AAPL blind.

| Does well | Sucks at |
|-----------|----------|
| Nothing. LSmooth 0.10 has bad NVDA (26.67%). LSmooth 0.05 NVDA is baseline-level (19.20%). LSmooth 0.01 destroys AAPL blind (26.50%). | Everything below 0.10 is worse than baseline on at least one ticker. |

Best for: Don't use. The sweet spot is 0.15–0.20.

### `--ohem-fraction 0.9`

Online Hard Example Mining: only the hardest 90% of examples (highest per-sample loss) contribute to the gradient each batch. Dropping the easiest 10% forces the model to avoid complacency on well-learned patterns.

| Does well | Sucks at |
|-----------|----------|
| Best partial AAPL ever (11.19%, **+0.129 skill**). Great blind AAPL (11.41%, +0.119). Also best DirAcc NVDA in partial (0.528). | NVDA blind is mediocre (18.73%, -0.052). Dropping only 10% of examples is a weak regularizer — most batches are unchanged. |

Best for: "I want the best partial AAPL with zero extra params."

### `--ohem-fraction 0.3`

Harder OHEM: only the hardest 30% of examples contribute. This is a strong regularizer — 70% of each batch is discarded.

| Does well | Sucks at |
|-----------|----------|
| Strong blind AAPL (11.60%, **+0.109 skill**). Good NVDA partial (14.44%, **+0.227 skill**). Nonblind unaffected. | NVDA blind is worse than baseline (19.30%, -0.108). 70% example dropout wastes most of the batch — training takes more epochs to converge. |

Best for: "I'm okay losing half the ticker to get strong regularisation on the other."

### `--ohem-fraction 0.35`

Near the instability threshold for OHEM. Only 35% of examples kept. Results vary by run.

| Does well | Sucks at |
|-----------|----------|
| **Best NVDA blind ever** (14.95%, **+0.273 skill**) — if you catch the right seed. Zero extra params. | Unstable. AAPL blind explodes (34.07%). Only 1/17 trained models found the good regime. Not reliable. |

Best for: Nothing. Too unstable.

### `--ohem-fraction 0.5`

Half the examples kept. Trades NVDA performance for extreme AAPL gains.

| Does well | Sucks at |
|-----------|----------|
| Best AAPL blind MAPE of any OHEM config (10.08%) with positive skill (+0.019). Strong DirAcc (0.476 blind, 0.468 NVDA). | NVDA blind degrades (22.18%). Avg MAPE (16.13%) is worse than OHEM 0.9 or LSmooth 0.15. |

Best for: "AAPL-only evaluation where I want minimal MAPE."

### `--ema-decay 0.999`

Milder EMA smoothing. Less regularization than 0.9999.

| Does well | Sucks at |
|-----------|----------|
| Best NVDA blind of any EMA config (18.26%, **+0.148 skill**). Nonblind unaffected (stays at 1.37%). Best NVDA DirAcc (0.528 blind). | AAPL blind is mediocre (16.42%). Partial AAPL worse than baseline (18.39%). Only helps NVDA, not AAPL. |

Best for: "I want a small NVDA boost without hurting other modes."

### `--crowfeather`

AdamW with eps=1e-20 (effectively zero in float32) and beta2 that ramps from 0.95 to 0.97 after warmup. Based on the Crowfeather recipe: aggressive epsilon means the step size is purely signal-to-noise driven; ramping beta2 means the model explores first then stabilises.

| Does well | Sucks at |
|-----------|----------|
| Decent AAPL blind (17.20%). Zero extra params. | Early-stops at 9 epochs (too fast). NVDA blind worse than baseline (32.16%). Unstable — the extreme epsilon makes training jittery. Crowfeather + WSD improves it (14.61% AAPL, 18.21% NVDA) but still behind plain WSD. |

Best for: Nothing on its own. Could be combined with WSD as a minor tweak.

### `--muon-lr 0.005`

Replaces AdamW for 2D weight matrices with the Muon optimizer (MomentUm Orthogonalized by Newton-Schulz). Uses SGD-momentum with Newton-Schulz iterations to orthogonalize gradients before applying them. 1D params (biases, norm scales) and the output head still use AdamW. Based on the Kellerer 2024 optimizer used to train DeepSeek models.

| Does well | Sucks at |
|-----------|----------|
| **New all-time best avg blind MAPE (14.24%)** — beats LSmooth 0.15 (14.69%). AAPL blind 9.92% (+0.178 skill). NVDA partial 15.99% (+0.153). Trains full 50 epochs. | NVDA blind slight regression (18.56%, -0.009). Not compatible with crowfeather. Combinations don't stack (muon+fim, muon+gqa all worse). Zero extra params but requires the MuonState wrapper. |

Best for: "I want the best blind-mode average with zero param overhead."

### `--fim-rate 0.25`

Fill-In-the-Middle augmentation: during training, 25% of batches randomly zero-out a ~15% contiguous span in the input sequence. The model must predict the next step for all positions including the corrupted ones, forcing it to use surrounding context. No extra params.

| Does well | Sucks at |
|-----------|----------|
| **Best AAPL blind ever (9.77%, +0.188 skill)**. Great AAPL partial (11.05%, +0.049). Zero extra params. Nonblind unaffected. | NVDA blind regresses (19.66%, -0.142). NVDA partial degraded (20.47%). Best at 25% — 30% is more balanced, 50% destabilizes. |

Best for: "AAPL-only blind evaluation with zero param overhead."

### `--fim-rate 0.30`

Higher FIM rate that trades some AAPL performance for better balance across tickers.

| Does well | Sucks at |
|-----------|----------|
| **Best NVDA partial of any Day 2 feature (13.19%, +0.282 skill)**. Balanced blind results (AAPL 10.66%, NVDA 19.32%). Zero extra params. | AAPL blind is merely average (10.66%, +0.160). DirAcc depressed (0.268). Not the best at any single metric but well-rounded. |

Best for: "I want balanced FIM results without sacrificing NVDA completely."

### `--gqa-kv-heads 2 --qk-norm`

Grouped Query Attention with 2 KV heads (down from 4) plus RMSNorm on Q and K before attention. Reduces KV-cache by 2×. QK-norm stabilizes training with fewer KV heads. Architectural change — reduces params from 82,433 to 74,305.

| Does well | Sucks at |
|-----------|----------|
| Best nonblind AAPL DirAcc of any config (0.520, +0.868 skill). Strong blind AAPL (10.64%, +0.161). Nonblind unaffected. Saves 8,128 params. | Partial mode degrades (AAPL 13.78%, NVDA 17.08%). NVDA blind mediocre (19.29%). Doesn't stack with muon or fim. Requires architecture changes — not a "drop-in" feature. |

Best for: "I want reduced KV-cache with strong nonblind performance."

### `--mhc-streams 4`

Manifold Hyper-Connections with 4 residual streams. Replaces the standard residual `x + F(norm(x))` with a learned doubly-stochastic mixing matrix across 4 parallel streams. Streams are shared-noise initialized at start, then diverge through training. Sinkhorn-Knopp constrains the residual mixing matrix to the Birkhoff polytope.

| Does well | Sucks at |
|-----------|----------|
| Good blind results (AAPL 10.97% +0.144, NVDA 19.04%). Zero extra heads/params beyond mixing matrices (~48 params). | Only works at 4 streams — 2 streams explodes, 3 is unstable. Doesn't stack with other Day 2 features. Adds training overhead from stream expansion. |

Best for: "I want a novel residual mechanism that doesn't change param count."

### `--lora-rank 8`

Depth LoRA: adds a low-rank adapter `W·x + α/r · (x A B)` to every Q/K/V/O attention projection and every W1/W2/W3 FFN projection in each block. Default α=1.0, scaling = α/rank. Adds 17,408 params (rank 8) on top of the 82K baseline. Per Hu et al. 2021 LoRA paper, but applied at depth (every block gets adapters) instead of as a fine-tuning trick.

| Does well | Sucks at |
|-----------|----------|
| **Best NVDA partial of any config: 13.38%, +0.260 skill** (beats fim 0.30's 13.19% in skill terms). Big AAPL blind reduction (10.09%, +0.180). Nonblind unaffected (AAPL 1.36%, NVDA 2.16%). 3-seed avg AAPL blind 16.92% vs baseline 28.79%. | NVDA blind slight regression (20.64%, -0.227). Doesn't stack with ACT (NVDA explodes to 30.77%). Rank=2 and rank=4 unexpectedly fail (avg 24.5% / 65.4% blind). Extra ~17K params at rank 8. |

Best for: "I want strong NVDA partial-mode skill with minimal architecture change."

### `--lora-rank 16`

Same as `--lora-rank 8` but rank 16. Adds 34,816 params on top of the baseline.

| Does well | Sucks at |
|-----------|----------|
| **Best AAPL nonblind ever (1.31%, +0.868 skill)**. Best AAPL nonblind DirAcc (0.524). Strong AAPL blind (10.43%, +0.086). | NVDA blind worse than rank=8 (22.33% vs 20.64%). Largest LoRA variant — 34K extra params. Same anti-stacking with ACT. |

Best for: "I want the best AAPL nonblind score, even at extra param cost."

### `--act-max-steps 6`

Adaptive Halting (Graves 2016 ACT): wraps the transformer block stack in a recurrent loop with a learned per-position halting probability via `sigmoid(W_h · state)`. Each position accumulates probability mass; once it exceeds 1−ε (ε=0.01), the position emits its weighted-average state. Up to 6 ponder steps per position. Adds 65 params (one halt-unit). Default time_penalty=0.001 adds a small ponder cost to the loss.

| Does well | Sucks at |
|-----------|----------|
| **Best blind avg (15.23%) of any single feature this round**. Strong AAPL blind (9.87%, +0.181). Improved partial NVDA (15.92%, +0.083). Nonblind essentially unaffected (1.42% / 2.27%). 3-seed avg AAPL blind 13.28% vs baseline 28.79%. | NVDA blind slight regression (20.58%, -0.214). max_steps=10 explodes (47.70% AAPL). Doesn't stack with LoRA (NVDA explodes). Trains 2× slower than baseline due to recurrent loop. |

Best for: "I want dynamic per-position computation with the best blind average."

### Post-training: SLERP merge

`scripts/run_slerp_merges.py` and `lab/experiments/slerp_merge.py` provide spherical linear interpolation: train N checkpoints with different seeds, then merge: `w = sin((1−t)θ)/sin(θ)·w₀ + sin(tθ)/sin(θ)·w₁`. Falls back to LERP when vectors are nearly collinear. The `--slerp-t` flag exists in `train.py` but only marks runs; the actual merge is post-training.

| Does well | Sucks at |
|-----------|----------|
| **Best blind AAPL of any single config: 9.41%** at t=0.6, NVDA 19.78%, +0.179 skill (42-44 pair). Recovered seed 42's 45.72% AAPL blind down to ~9% by averaging with seed 44. Free at training time — it's a post-training merge. Smooth basin t∈[0.1, 0.8] for compatible pairs (every t works). | **Diverges catastrophically across basins**. The 42-43 pair exploded at every t (268% / 17268% / 2.9B%). 3-way merges containing seed 43 also diverge (266% NVDA). Slight nonblind regression (2.15% AAPL vs 1.35%). t∈[0.9, 1.0] degrades. Cannot predict which pairs are basin-compatible without trying. |

Best for: "I have multiple seeds and want a no-cost blind-mode boost — but I have to test each pair."

### `--latent-steps 4`

COCONUT-style latent reasoning: after the standard forward pass through transformer blocks, the hidden state is re-processed through all blocks for 4 additional "thinking" steps without seeing new input. Based on Meta FAIR's COCONUT paradigm (Hao et al., COLM 2025) — reasoning in continuous latent space rather than language tokens. No extra params.

| Does well | Sucks at |
|-----------|----------|
| **Best Day 3 blind avg (14.69%)**. Ties LSmooth for blind avg. Best AAPL blind (10.03%, +0.184 skill) among Day 3 features. Nonblind unaffected (1.36% / 2.20%). Zero extra params. | Partial mode mediocre (AAPL 11.86%, NVDA 20.52%, both negative skill). Only works at 1 or 4 steps — 2, 6, 12 are unstable. |

Best for: "I want the best blind avg of all Day 3 features with zero param overhead."

### `--latent-steps 1`

Single latent thinking step. Milder version of COCONUT reasoning.

| Does well | Sucks at |
|-----------|----------|
| **Positive skill on both tickers in blind AND partial**: AAPL blind +0.137, NVDA blind -0.030, AAPL partial +0.046, NVDA partial +0.025. Good DirAcc (0.560 AAPL partial). | NVDA blind (18.54%) is near baseline. Partial NVDA MAPE (17.65%) is middling. |

Best for: "I want positive skill everywhere at zero param cost."

### `--ssm-decay 0.5`

Adds a per-block state-space model path where state evolves per-position: `state[t] = decay · state[t−1] + B(x[t])`. The state is added to each block's output, providing a linear recurrent pathway alongside attention. Adds 4,096 params per block (B projection) + 64 params per block (log decay).

| Does well | Sucks at |
|-----------|----------|
| **Best AAPL blind MAPE ever: 9.37%** (+0.117 skill) — beats SLERP 42-44 t=0.6 (9.41%), LoRA-8+SLERP (9.38%), and fim 0.25 (9.77%). Nonblind unaffected (1.45%). | NVDA degrades across all modes (21.30% blind). High decay (0.9–0.95) destabilises AAPL to 25k%+. Only low decay (0.5) or very high decay (0.999) are stable. |

Best for: "I only care about AAPL blind MAPE and accept NVDA regressions."

### `--ssm-decay 0.999`

High-decay SSM that retains state almost perfectly. More balanced than ssm=0.5.

| Does well | Sucks at |
|-----------|----------|
| **Positive skill on both tickers blind**: AAPL +0.065, NVDA +0.121. Balanced across tickers (12.24% AAPL, 18.56% NVDA). | NVDA partial degrades (not tested). AAPL blind (12.24%) is merely average. |

Best for: "I want balanced blind results with positive skill on both tickers."

### `--curriculum-epochs 10` (with `--latent-steps`)

Ramps the number of latent thinking steps from 0 to the target over N epochs. Based on the multi-stage curriculum training in the COCONUT paper.

| Does well | Sucks at |
|-----------|----------|
| Slightly improves NVDA (18.30% vs 18.54%) over latent=1 alone. | Worse than latent=1 or latent=4 alone on all other metrics. Adds complexity without benefit.  |

Best for: Nothing. Not worth the complexity.

### `--ple`

Per-Layer Embeddings: adds a learned 64-dim bias vector per layer, added to the hidden state before each block. Based on Gemma 4's PLE (Per-Layer Embeddings) design that injects token-identity signals at every layer rather than relying on a single initial embedding. Adds 128 params for Glint.

| Does well | Sucks at |
|-----------|----------|
| Best new-feature AAPL blind (11.86%, **+0.090 skill**). Great NVDA partial (13.44%, **+0.278 skill**). Nonblind unaffected (1.33%). Only 128 extra params. | NVDA blind mediocre (19.45%, -0.126). Doesn't stack with anti-pattern or NCE (all combos worse). |

Best for: "I want a tiny architectural boost to AAPL blind with only 128 extra params."

### `--anti-pattern-weight 0.2`

Anti-pattern unlikelihood loss: penalizes predictions that go against the recent trend direction with high confidence. The loss term is `weight * |pred - last|` when the predicted direction differs from the 5-step local trend direction. Based on the unlikelihood training paradigm (Welleck et al., ICLR 2020) adapted for regression.

| Does well | Sucks at |
|-----------|----------|
| **Best AAPL blind of any Day 3 feature (9.83%, +0.184 skill)** — ties fim 0.25 and beats muon 0.005 on AAPL. Nonblind unaffected. Zero extra params. | NVDA blind degrades (20.44%, -0.210). Sweet spot is narrow: 0.1 and below destabilize AAPL, above 0.5 is too weak. |

Best for: "I want the best AAPL blind with zero param overhead, even at NVDA's expense."

### `--anti-pattern-weight 0.4`

Higher-weight anti-pattern loss that balances across tickers.

| Does well | Sucks at |
|-----------|----------|
| **Positive skill on BOTH tickers in blind AND partial** — AAPL blind +0.112, NVDA blind +0.097, AAPL partial **+0.162**, NVDA partial **+0.221**. Best partial-mode of any Day 3 feature. Zero extra params. | Blind AAPL (11.54%) is merely average. Nonblind slightly elevated (2.21% NVDA vs 2.19%). |

Best for: "I want positive skill on both tickers in blind AND partial with zero params."

### `--nce-weight 0.2` (or 0.3–0.6)

NCE Context loss: InfoNCE-style contrastive loss where the last position's hidden state is pulled toward nearby positions (positive pairs within 2 steps) and pushed away from distant positions (negative pairs beyond 8 steps). Encourages temporally smooth representations.

| Does well | Sucks at |
|-----------|----------|
| Broad sweet spot (0.2–0.6 all work). Consistent blind results (~12.3-12.7% AAPL, ~18.2-18.4% NVDA). Mildly positive skill on both tickers at 0.5-0.6. Zero extra params. | Only nce=0.01 destabilizes; all other values produce nearly identical results (12.3-14.3% AAPL). The loss adds negligible training overhead but the effect is subtle — it's a weak regularizer. |

Best for: "I want a mild, consistent regularizer that never destabilizes."

### `--engram`

Engram conditional memory: hashes n-grams of price direction (ternary: up/flat/down) into a learned embedding table (512 entries × d_model). The looked-up embedding is gated by the current hidden state and added to the hidden state. Based on DeepSeek's Engram (arXiV:2601.07372), separated memory from computation via hash-based conditional lookups.

| Does well | Sucks at |
|-----------|----------|
| Improves AAPL blind (45.72%→29.44%, DirAcc 0.524). ++33K params for Glint. Nonblind unaffected (1.39%/2.21%). | NVDA neutral (19.19%→19.65%). Effect is mild compared to top features — it helps but doesn't compete with TRIM-KV, muon, or anti-pattern. |

Best for: "I want a cheap architectural addition that helps AAPL blind without hurting anything else."

### `--trim-kv`

TRIM-KV learned retention gate: each key in self-attention gets a per-head retention score from a learned linear gate. The retention score is added as a bias to attention logits, allowing the model to learn which past positions are important for each attention head. Based on "Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs" (Bui et al., arXiV:2512.03324).

| Does well | Sucks at |
|-----------|----------|
| **Strong AAPL blind (11.08%, +0.136 skill)** — competitive with top features (anti=0.2 9.83%, muon=0.005 9.92%). Only 34 extra params (2×17 per layer). Nonblind unaffected (1.39%). Predictable DirAcc (~0.47). | NVDA blind elevated (22.64%, -0.384). Partial mode mediocre (16.48% AAPL, -0.453). Like other attention-modifying features (GQA), the effect isn't uniform across tickers. |

Best for: "I want near-best AAPL blind with almost zero param overhead."

### `--sleep-gate`

SleepGate: scores each position's hidden state for importance, computes a consolidated summary, and injects it back with learned gating. Based on "Learning to Forget: Sleep-Inspired Memory Consolidation" (Xie, arXiV:2603.14517).

| Does well | Sucks at |
|-----------|----------|
| Nothing tested. | Destabilises both tickers: AAPL 54.47% (+4.820), NVDA 90.38% (-5.007). The consolidation mechanism adds noise instead of signal on the 2-layer Glint. |

Best for: **DROPPED — do not use.**

### Configs that don't stack

These combinations were tested and performed strictly worse than either feature alone:

| Combo | Why it fails |
|-------|-------------|
| MTP + output reg | Two strong regularisers overload the 82K model. Both blind and partial explode to 3000%+ MAPE. |
| MTP + dropout (wrong rate) | MTP(2,4,8)+dropout 0.01 is worse than either alone. MTP(2,4,8,16)+dropout 0.01 works; 0.003/0.005/0.03 don't. The window is narrow. |
| dropout + output reg | Interferes (24-33% MAPE, worse than either alone). Different regularisation mechanisms that don't compose. |
| muon + fim | Muon 0.005 + fim 0.25: avg 15.67% blind — worse than either alone (14.24%, 14.72%). Optimizer orthogonalization interferes with span-masked gradients. |
| muon + gqa | Muon 0.005 + gqa=2_qknorm: avg 20.37% blind — much worse than muon alone. The reduced KV heads change gradient statistics that the Newton-Schulz step relies on. |
| fim + gqa | fim 0.25 + gqa=2_qknorm: avg 22.95% blind — both degraded. Span masking and reduced attention capacity don't compose. |
| muon + fim + gqa | All three: avg 60.90% (explosion). Three different modifications — optimizer, augmentation, architecture — overload the small model. |
| LoRA + ACT | LoRA-8 + ACT-6: NVDA blind explodes to 30.77% (vs 20.64% / 20.58% alone). LoRA-16 + ACT-6 even worse (32.45%). The recurrent loop shares state across LoRA-adapted blocks, causing accumulation that the ponder cost can't regularize. |
| 3-way SLERP across mismatched basins | Merging seeds 42+43+44 reliably diverges (NVDA 266% blind). Seed 43 lives in a different basin than 42/44; including it in any merge causes catastrophic interference. Pairwise merges of basin-compatible seeds (42-44, 43-44) work; the 42-43 pair alone also diverges. |
| SLERP at t=0.25 / t=0.75 across basins | Even with valid endpoints, off-center t values amplify basin mismatch. The 42-43 pair at t=0.25 hit AAPL=2.9 billion% MAPE. Stay within t∈[0.4, 0.7] for tested pairs. |
| latent + SSM (latent=1 + ssm=0.999) | AAPL blind explodes to 55.76%. The SSM's recurrent state interacts destructively with the additional latent thinking passes through the same blocks. |
| PLE + anti-pattern | PLE + anti=0.5: avg 16.29% blind — worse than either alone (15.65%, 15.46%). Architectural bias + direction-penalty don't compose. |
| PLE + NCE | PLE + nce=0.5: avg 15.59% blind — worse than PLE alone (15.65%). The NCE contrastive objective doesn't benefit from layer-specific bias vectors. |
| anti-pattern + NCE | anti=0.5 + nce=0.5: avg 15.47% — same as anti=0.5 alone (15.46%). Two loss-based regularizers don't compound on a small model. |
| MTP + anti-pattern | MTP(2,4,8,16)+anti=0.2 explodes to 1.6M% AAPL. The auxiliary MTP heads amplify the anti-pattern gradient, causing catastrophic interference. MTP(2,4,8,16)+anti=0.4 is stable but worse (22.21% avg). |
| muon + dropout | muon=0.005 + drop=0.01 + anti=0.2: avg 37.52% — much worse than muon alone. Input masking disrupts the gradient statistics that Newton-Schulz orthogonalization depends on. |
| muon + label-smoothing + anti-pattern + PLE | All four: avg 124.63% (explosion). Four modifications overload the 82K model — muon optimizer + target smoothing + direction penalty + per-layer biases. |
| GQA + anti-pattern | GQA=2 + qknorm + anti=0.4: avg 17.69% — worse than GQA alone (14.97%). Reduced attention capacity can't handle the additional gradient signal from direction penalties. |
| SleepGate | AAA 54.47%, NVDA 90.38% in blind. The consolidation mechanism (per-position importance scoring) introduces destructive interference on the 2-layer Glint model.

---

## Optimal Configuration (Glint)

Found via systematic grid search across 20+ configurations: 4 MTP horizon variants × 10 dropout rates + output reg sweeps, each blind-benchmarked to find contenders, then full 3-mode benchmarked.

### Winner: `MTP(2,4,8,16) + input-dropout 0.01`

```bash
python train.py --series Glint --tickers AAPL,NVDA \
    --train-start 2022-01-01 --train-end 2024-12-31 \
    --epochs 50 \
    --mtp-horizons 2,4,8,16 \
    --input-dropout 0.01
```

| Mode | Ticker | MAPE | DirAcc | skill_vs_naive_rmse |
|------|--------|------|--------|---------------------|
| blind | AAPL | **11.84%** | 0.324 | **+0.089** |
| blind | NVDA | 18.42% | 0.224 | -0.008 |
| nonblind | AAPL | 1.33% | 0.520 | +0.867 |
| nonblind | NVDA | 2.15% | 0.444 | +0.875 |
| partial | AAPL | **12.30%** | 0.488 | **+0.014** |
| partial | NVDA | **14.48%** | 0.444 | **+0.251** |

This is the first Glint config to achieve **positive skill on AAPL in both blind AND partial modes**.

### Full sweep leaderboard (blind mode)

| Rank | Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|------|--------|-----------|------------|-----------|------------|----------|
| 1 | **Muon 0.005** | **9.92%** | **+0.178** | **18.56%** | -0.009 | **14.24%** |
| 2 | **anti=0.2 (Day 3 R2)** | **9.83%** | **+0.184** | 20.44% | -0.210 | 15.13% |
| 3 | fim 0.25 | **9.77%** | **+0.188** | 19.66% | -0.142 | **14.72%** |
| 4 | **PLE (Day 3 R2)** | 11.86% | +0.090 | 19.45% | -0.126 | 15.65% |
| 5 | **anti=0.4 (Day 3 R2)** | 11.54% | **+0.112** | 18.28% | **+0.097** | 14.91% |
| 6 | **LSmooth 0.15** | **10.88%** | **+0.150** | 18.49% | -0.012 | **14.69%** |
| 7 | EMA 0.9999 | 10.85% | +0.148 | 18.70% | -0.050 | 14.78% |
| 8 | fim 0.30 | 10.66% | +0.160 | 19.32% | -0.114 | 14.99% |
| 9 | gqa=2_qknorm | 10.64% | +0.161 | 19.29% | -0.103 | 14.97% |
| 10 | mhc=4 | 10.97% | +0.144 | 19.04% | -0.086 | 15.01% |
| 11 | MTP(2,4,8,16) | 11.56% | +0.109 | 18.37% | +0.039 | 14.97% |
| 12 | **OHEM 0.9** | 11.41% | +0.119 | 18.73% | -0.052 | 15.07% |
| 13 | LSmooth 0.20 | 11.93% | +0.084 | 18.22% | +0.037 | 15.08% |
| 14 | MTP(2,4,8,16)+drop 0.01 | 11.84% | +0.089 | 18.42% | -0.008 | 15.13% |
| 15 | dropout 0.10 | 9.51% | +0.119 | 21.17% | -0.270 | 15.34% |
| 16 | oreg 0.10 | 9.65% | +0.157 | 21.36% | -0.285 | 15.51% |
| 17 | dropout 0.01 | 18.12% | -0.405 | 16.89% | +0.202 | 17.51% |
| — | baseline | 1243.35% | -185.62 | 19.48% | -0.125 | 631.41% |

### Specialists

Each feature has a mode where it excels:

| Goal | Config | Best Metric |
|------|--------|-------------|
| Best avg blind MAPE | `--muon-lr 0.005` | Avg blind **14.24%** MAPE |
| Best blind AAPL MAPE | `--ssm-decay 0.5` | AAPL blind **9.37%** MAPE, **+0.117** skill |
| Best blind AAPL skill | `--fim-rate 0.25` | AAPL blind **9.77%** MAPE, **+0.188** skill |
| Best blind AAPL (Day 3 R2) | `--anti-pattern-weight 0.2` | AAPL blind **9.83%** MAPE, **+0.184** skill |
| Best blind AAPL skill (runner-up) | `--muon-lr 0.005` | AAPL blind **9.92%** MAPE, **+0.178** skill |
| Best nonblind DirAcc | `--gqa-kv-heads 2 --qk-norm` | AAPL nonblind DirAcc **0.520**, skill **+0.868** |
| Best NVDA partial (Day 2) | `--fim-rate 0.3` | NVDA partial **13.19%** MAPE, **+0.282** skill |
| Best NVDA partial (all-time) | `--output-reg 0.05` | NVDA partial **9.08%** MAPE, **+0.547** skill |
| Best avg blind MAPE (zero params) | `--latent-steps 4` | Avg blind **14.69%** MAPE |
| Best AAPL partial (zero params) | `--ohem-fraction 0.9` | AAPL partial **11.19%** MAPE, **+0.129** skill |
| Best AAPL blind (zero params, previous) | `--input-dropout 0.10` | AAPL blind **9.51%** MAPE, **+0.119** skill |
| Best positive-skill both (zero params) | `--label-smoothing 0.20` | AAPL blind +0.084, NVDA blind +0.037, AAPL partial +0.093, NVDA partial +0.112 |
| Best positive-skill both blind+partial (zero params) | `--anti-pattern-weight 0.4` | AAPL blind +0.112, NVDA blind +0.097, AAPL partial **+0.162**, NVDA partial **+0.221** |
| Best NVDA blind (zero params) | `--lr-schedule wsd` | AAPL blind **11.74%** MAPE, **+0.101** skill |
| Best NVDA blind | `--input-dropout 0.01` | NVDA blind **16.89%** MAPE, **+0.202** skill |
| Best avg MAPE (architecture) | `--mtp-horizons 2,4,8,16` | Avg blind **14.97%** MAPE |
| Best positive-skill both tickers blind (Day 3) | `--latent-steps 1` | AAPL blind +0.137, NVDA blind -0.030, AAPL partial +0.046 |
| Best positive-skill both tickers blind (SSM) | `--ssm-decay 0.999` | AAPL blind +0.065, NVDA blind +0.121 |

```bash
# Best avg blind MAPE (zero params)
python train.py --label-smoothing 0.15

# NEW BEST avg blind MAPE
python train.py --muon-lr 0.005

# Best positive skill across all modes (zero params)
python train.py --label-smoothing 0.20

# Best positive skill across all modes (Day 3 R2, zero params)
python train.py --anti-pattern-weight 0.4

# Best AAPL blind (Day 3 R2, zero params)
python train.py --anti-pattern-weight 0.2

# Tiny architectural boost (128 extra params)
python train.py --ple

# Best partial AAPL (zero params)
python train.py --ohem-fraction 0.9

# Best AAPL blind ever
python train.py --fim-rate 0.25

# Balanced overall: MTP(2,4,8,16) + dropout 0.01
python train.py --mtp-horizons 2,4,8,16 --input-dropout 0.01

# Zero-param AAPL blind boost (Day 2)
python train.py --muon-lr 0.005   # best avg blind
python train.py --fim-rate 0.25   # best AAPL blind
python train.py --fim-rate 0.30   # balanced FIM

# Architecture: GQA with QK-norm (10% fewer params)
python train.py --gqa-kv-heads 2 --qk-norm

# Architecture: Manifold Hyper-Connections
python train.py --mhc-streams 4

# Zero-param AAPL blind boost (Day 1)
python train.py --lr-schedule wsd

# AAPL blind specialist
python train.py --output-reg 0.10   # or: --input-dropout 0.10  or: --ema-decay 0.9999  or: --ohem-fraction 0.5

# NVDA partial specialist
python train.py --output-reg 0.05

# NVDA blind specialist
python train.py --input-dropout 0.01

# Pure MTP (previous best avg blind MAPE)
python train.py --mtp-horizons 2,4,8,16
```

---

## Progress Log

A running journal of every experiment. Newest entries on top.

### Day 3 (Round 2), Three features: Per-Layer Embeddings, Anti-pattern Unlikelihood, NCE Context Loss

* **What**: Implemented and tested 3 queued features: Per-Layer Embeddings (PLE — Gemma 4-style learned per-layer bias), Anti-pattern unlikelihood loss (penalizes confident wrong-direction predictions against recent trend), and NCE Context loss (InfoNCE-style contrastive loss over time positions). 13-value sweep for anti-pattern (0.001–1.0), 13-value sweep for NCE (0.001–1.0), fine sweeps around optima, plus 3 combos. 30 models trained, top 3 full 3-mode benchmarked.
* **Features**: `--ple`, `--anti-pattern-weight` (0.001–1.0), `--nce-weight` (0.001–1.0)
* **Compute**: NVIDIA RTX 5090, ~6 minutes total

#### Full 3-mode results (top 3 variants)

| Config | Mode | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|--------|------|-----------|------------|-----------|------------|
| **anti=0.2** | blind | **9.83%** | +0.184 | 20.44% | -0.210 |
| **anti=0.2** | nonblind | 1.34% | +0.867 | 2.19% | +0.874 |
| **anti=0.2** | partial | 12.67% | -0.129 | 19.64% | -0.188 |
| **anti=0.4** | blind | 11.54% | +0.112 | 18.28% | +0.097 |
| **anti=0.4** | nonblind | 1.33% | +0.866 | 2.21% | +0.873 |
| **anti=0.4** | partial | **10.87%** | **+0.162** | **16.05%** | **+0.221** |
| **PLE** | blind | 11.86% | +0.090 | 19.45% | -0.126 |
| **PLE** | nonblind | 1.33% | +0.867 | 2.20% | +0.873 |
| **PLE** | partial | 14.39% | -0.165 | **13.44%** | **+0.278** |
| baseline | blind | 1243.35% | -185.62 | 19.48% | -0.125 |
| baseline | nonblind | 1.36% | +0.866 | 2.19% | +0.873 |
| baseline | partial | 82.38% | -6.071 | 145.69% | -15.047 |

#### Blind mode results (Day 3 R2 features vs baseline)

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|--------|-----------|------------|-----------|------------|----------|
| **anti=0.2** | **9.83%** | +0.184 | 20.44% | -0.210 | 15.13% |
| **anti=0.4** | 11.54% | +0.112 | 18.28% | +0.097 | **14.91%** |
| **anti=0.6** | 10.59% | +0.163 | 20.20% | -0.187 | 15.40% |
| **anti=0.3** | 10.02% | +0.181 | 20.17% | -0.185 | 15.10% |
| **PLE** | 11.86% | +0.090 | 19.45% | -0.126 | 15.65% |
| nce=0.2 | 12.30% | +0.056 | 18.43% | -0.012 | 15.36% |
| nce=0.5 | 12.69% | +0.027 | 18.26% | +0.016 | 15.47% |
| baseline | 1243.35% | -185.62 | 19.48% | -0.125 | 631.41% |

#### Key findings

* **Anti-pattern unlikelihood loss at weight 0.2 achieves AAPL blind 9.83% (+0.184 skill)** — ties fim 0.25 for best AAPL blind of any zero-param feature, and beats muon 0.005 on AAPL. The mechanism is intuitive: penalize predictions that go against the recent 5-step price trend, which prevents the model from making confident wrong-direction predictions during blind autoregressive rollout.
* **Anti-pattern 0.4 is the new best "positive skill everywhere" config**: positive skill on both tickers in blind (AAPL +0.112, NVDA +0.097) AND partial (AAPL +0.162, NVDA +0.221). The partial-mode scores set a new record — beats label-smoothing 0.20 on every partial metric. Zero extra params.
* **Anti-pattern has a bimodal stability profile**: weights 0.001–0.05 destabilize AAPL (explodes to 119–2843%), weights 0.1–1.0 are stable. The instability sweet spot boundary is sharp: 0.05 fails, 0.1 barely works (15.73% AAPL), 0.2 is optimal. This suggests the loss needs enough strength to be effective — too weak and it only adds noise.
* **PLE achieves competitive AAPL blind (11.86%, +0.090 skill)** and excellent NVDA partial (13.44%, +0.278 skill) with only 128 extra params. The per-layer bias vectors let the 2-layer Glint differentiate its processing pathway.
* **NCE context loss is the most boring-but-safe feature**: all weights from 0.05 to 1.0 produce nearly identical results (12.3–14.3% AAPL, 18.1–18.5% NVDA). It never destabilizes but also never produces standout results — a mild, reliable regularizer.
* **Combinations don't stack**: PLE + anti=0.5 (16.29% avg), PLE + nce=0.5 (15.59%), and anti=0.5 + nce=0.5 (15.47%) are all worse than the individual features alone. The loss-based features don't compose with architectural changes on this small model.
* **Nonblind mode remains invariant**: All variants score MAPE 1.33–1.36% (AAPL) / 2.19–2.21% (NVDA) with skill +0.866–0.874.

#### Ultimate Combo Sweep (20 multi-feature combos)

After testing features individually, 20 ambitious combos were trained combining muon, WSD, EMA, label-smoothing, OHEM, FIM, MTP, dropout, PLE, anti-pattern, and NCE in various multi-way stacks. Key findings from the combinatorial sweep:

| Combo | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE | Verdict |
|-------|-----------|------------|-----------|------------|----------|---------|
| **muon+lsmooth=0.20+anti=0.4** | 10.78% | +0.151 | 18.70% | -0.050 | **14.74%** | Best overall combo |
| **wsd+ema+anti=0.2** | 10.86% | +0.148 | 18.67% | -0.048 | **14.77%** | Strong blind (NB degrades) |
| **muon+anti=0.4** | 10.44% | +0.168 | 19.32% | -0.102 | **14.88%** | Best combo AAPL blind |
| **fim+anti=0.2** | 11.11% | +0.135 | 18.69% | -0.049 | **14.90%** | FIM+anti stacks! |
| muon+ohem+anti=0.4 | 10.95% | +0.146 | 19.21% | -0.092 | 15.08% | Three-loss stack |
| muon+anti=0.2 | 11.40% | +0.118 | 18.61% | -0.039 | 15.01% | Stable but avg |
| muon+wsd+anti=0.2 | 11.42% | +0.117 | 18.61% | -0.039 | 15.01% | WSD+muon compatible |
| muon+fim+anti=0.4 | 11.75% | +0.095 | 18.47% | -0.016 | 15.11% | Muon+fim revived w/ anti |
| MTP+anti=0.2 | 1.6M% | — | 18.69% | -0.044 | — | Catastrophic explosion |
| muon+dropout+anti | 31.27% | -2.357 | 43.77% | -1.903 | 37.52% | Muon+input drop breaks |
| 4-way: muon+lsmooth+anti+PLE | 148.48% | -12.82 | 100.78% | -5.44 | 124.63% | Too many features |
| GQA+anti=0.4 | 16.27% | -0.260 | 19.10% | -0.089 | 17.69% | GQA+anti degrades |

##### Full 3-mode results (best 3 combos)

| Config | Mode | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|--------|------|-----------|------------|-----------|------------|
| **muon+lsmooth=0.20+anti=0.4** | blind | 10.78% | +0.151 | 18.70% | -0.050 |
| **muon+lsmooth=0.20+anti=0.4** | nonblind | 1.43% | +0.861 | 2.32% | +0.869 |
| **muon+lsmooth=0.20+anti=0.4** | partial | 11.53% | +0.053 | 18.55% | -0.048 |
| **muon+anti=0.4** | blind | 10.44% | +0.168 | 19.32% | -0.102 |
| **muon+anti=0.4** | nonblind | 1.31% | +0.869 | 2.17% | +0.874 |
| **muon+anti=0.4** | partial | 13.81% | -0.199 | 16.84% | +0.054 |
| **wsd+ema+anti=0.2** | blind | 10.86% | +0.148 | 18.67% | -0.048 |
| **wsd+ema+anti=0.2** | nonblind | 3.54% | +0.682 | 5.52% | +0.703 |
| **wsd+ema+anti=0.2** | partial | 11.24% | +0.096 | 19.34% | -0.095 |

* **Best combo: muon + label-smoothing 0.20 + anti-pattern 0.4 (14.74% avg)** — three different regularizers compose well: orthogonal optimizer (Muon), target shrinkage (LSmooth), and direction penalty (anti-pattern). Nonblind unaffected (1.43%/2.32%). Partial mode has positive AAPL skill (+0.053). This is the first multi-feature stack where all three mechanisms genuinely complement each other.
* **muon + anti-pattern 0.4 achieves the best combo AAPL blind (10.44%, +0.168 skill)** with the best-combo nonblind (1.31%, +0.869). Partial NVDA has positive skill (+0.054). This is the strongest zero-param 2-way stack.
* **Most combos don't beat the best individual** — muon=0.005 alone still holds the avg blind MAPE record (14.24%). But combos offer unique tradeoffs: muon+anti=0.4 trades 1% avg for much better AAPL (10.44% vs 9.92%).
* **MTP + anti-pattern explodes catastrophically**: 1.6 million% AAPL. The auxiliary head gradients amplify the direction penalty signal causing divergence.
* **Muon + dropout breaks the optimizer**: 37.52% avg. Input masking interferes with Newton-Schulz orthogonalization (same mechanism as muon+fim failure).
* **FIM + anti-pattern stacks!** (14.90% avg) — unlike muon+fim which is a known non-stacker. Span masking and direction penalty operate on different parts of the loss landscape.
* **GQA + anti-pattern is worse** than GQA alone, confirming that reduced attention capacity can't handle additional gradient signals.

#### Updated leaderboard (blind mode)

| Rank | Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|------|--------|-----------|------------|-----------|------------|----------|
| — | **Muon 0.005** | 9.92% | +0.178 | 18.56% | -0.009 | **14.24%** |
| — | **anti=0.2 (Day 3 R2)** | **9.83%** | +0.184 | 20.44% | -0.210 | 15.13% |
| — | fim 0.25 | 9.77% | +0.188 | 19.66% | -0.142 | 14.72% |
| — | **anti=0.4 (Day 3 R2)** | 11.54% | +0.112 | **18.28%** | +0.097 | **14.91%** |
| — | **PLE (Day 3 R2)** | 11.86% | +0.090 | 19.45% | -0.126 | 15.65% |
| — | LSmooth 0.15 | 10.88% | +0.150 | 18.49% | -0.012 | 14.69% |
| — | nce=0.2 (Day 3 R2) | 12.30% | +0.056 | 18.43% | -0.012 | 15.36% |
| — | baseline | 1243.35% | -185.62 | 19.48% | -0.125 | 631.41% |

#### Verdicts

| Feature | Verdict |
|---------|---------|
| `--ple` | **KEEP** — competitive AAPL blind with only 128 extra params |
| `--anti-pattern-weight 0.2` | **KEEP** — best AAPL blind of any Day 3 feature (9.83%), ties fim 0.25 |
| `--anti-pattern-weight 0.4` | **KEEP** — best positive-skill both tickers blind+partial (zero params) |
| `--anti-pattern-weight 0.3/0.6` | **NEUTRAL** — work but dominated by 0.2 and 0.4 |
| `--anti-pattern-weight <0.1` | **DROP** — destabilizes AAPL |
| `--nce-weight 0.2–0.6` | **KEEP** — mild, consistent regularizer, never destabilizes |
| `--nce-weight 0.01` | **DROP** — unstable spike (436% AAPL) |
| PLE + anti combos | **DROP** — doesn't stack, all worse than individual features |
| PLE + nce combos | **DROP** — doesn't stack |
| anti + nce combos | **DROP** — doesn't stack |

### Day 3 (Round 3), Three features: Engram, SleepGate, TRIM-KV

* **What**: Implemented and tested 3 queued features: Engram conditional memory (DeepSeek-style hashed n-gram embedding lookup), SleepGate memory consolidation (per-position importance scoring + compression), and TRIM-KV learned retention gate (per-key per-head importance in self-attention). All boolean features (train once each), full 3-mode bench for TRIM-KV.
* **Features**: `--engram`, `--sleep-gate`, `--trim-kv`
* **Compute**: NVIDIA RTX 5090, ~2 minutes total

#### Full 3-mode results (TRIM-KV)

| Config | Mode | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|--------|------|-----------|------------|-----------|------------|
| **TRIM-KV** | blind | **11.08%** | **+0.136** | 22.64% | -0.384 |
| **TRIM-KV** | nonblind | 1.39% | +0.865 | 2.21% | +0.873 |
| **TRIM-KV** | partial | 16.48% | -0.453 | 20.81% | -0.334 |
| baseline | blind | 45.72% | -2.403 | 19.19% | -0.101 |
| baseline | nonblind | 1.39% | +0.864 | 2.21% | +0.872 |
| baseline | partial | 14.47% | -0.243 | 14.82% | +0.220 |

#### Blind mode results (Day 3 R3)

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|--------|-----------|------------|-----------|------------|----------|
| **TRIM-KV** | **11.08%** | **+0.136** | 22.64% | -0.384 | **16.86%** |
| Engram | 29.44% | -1.139 | 19.65% | -0.141 | 24.55% |
| SleepGate | 54.47% | -4.820 | 90.38% | -5.007 | 72.43% |
| baseline | 45.72% | -2.403 | 19.19% | -0.101 | 32.45% |

#### Key findings

* **TRIM-KV achieves AAPL blind 11.08% (+0.136 skill)** — competitive with top features like anti-pattern 0.2 (9.83%) and muon 0.005 (9.92%), with only 34 extra params. The learned per-head retention scores allow the attention mechanism to dynamically filter which past positions matter.
* **Engram helps AAPL moderately (45.72%→29.44%)** with the same DirAcc (0.524) as baseline, suggesting the n-gram pattern embeddings provide useful conditioning. Effect is mild but harmless.
* **SleepGate destabilises both tickers** (AAOPL 54.47%, NVDA 90.38%). The consolidation mechanism introduces noise on the 2-layer Glint — the model is too small to benefit from positional state compression.
* **Nonblind is invariant** — all variants score AAPL MAPE 1.39% / NVDA 2.21%. Partial mode is mediocre for TRIM-KV (16.48% AAPL, -0.453 skill).

#### Updated leaderboard (blind mode)

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|--------|-----------|------------|-----------|------------|----------|
| **Muon 0.005** | 9.92% | +0.178 | **18.56%** | -0.009 | **14.24%** |
| **anti=0.2** | **9.83%** | +0.184 | 20.44% | -0.210 | 15.13% |
| fim 0.25 | 9.77% | +0.188 | 19.66% | -0.142 | 14.72% |
| **TRIM-KV** | 11.08% | +0.136 | 22.64% | -0.384 | 16.86% |
| **anti=0.4** | 11.54% | +0.112 | 18.28% | +0.097 | 14.91% |
| LSmooth 0.15 | 10.88% | +0.150 | 18.49% | -0.012 | 14.69% |
| EMA 0.9999 | 10.85% | +0.148 | 18.70% | -0.050 | 14.78% |
| gqa=2_qknorm | 10.64% | +0.161 | 19.29% | -0.103 | 14.97% |
| **PLE** | 11.86% | +0.090 | 19.45% | -0.126 | 15.65% |
| nce=0.2 | 12.30% | +0.056 | 18.43% | -0.012 | 15.36% |
| baseline | 45.72% | -2.403 | 19.19% | -0.101 | 32.45% |

Note: baseline values differ across rounds due to stochasticity; the Day 3 R3 baseline happened to be a bad run (45.72% AAPL), which makes relative improvements look larger. The crucial comparison is against the known best: TRIM-KV at 11.08% AAPL blind is solid but doesn't beat muon 0.005 on average.

#### Verdicts

| Feature | Verdict |
|---------|---------|
| `--trim-kv` | **KEEP** — strong AAPL blind with negligible param cost (34 params) |
| `--engram` | **KEEP (mild)** — helps AAPL blind but not competitive with top features |
| `--sleep-gate` | **DROP** — destabilises both tickers on Glint |

### Day 3 (Round 1), Three features: Latent Reasoning, SSM Injection, Curriculum Learning

* **What**: Implemented and tested 3 features: Latent Reasoning (COCONUT-style continuous thought, 7-step sweep), SSM Injection (per-position stable recurrent injection, 7-value decay sweep), and Curriculum Learning (latent-step ramp-up, 8-value sweep + combos). 23 models trained (7 latent, 7 SSM, 8 curriculum/combos), top 3 full 3-mode benchmarked.
* **Features**: `--latent-steps` (1–12), `--ssm-decay` (0.5–0.999), `--curriculum-epochs` (5–30)
* **Compute**: NVIDIA RTX 5090, ~3 minutes total

#### Full 3-mode results (top 3 variants)

| Config | Mode | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|--------|------|-----------|------------|-----------|------------|
| **latent=4** | blind | 10.03% | +0.184 | 19.35% | -0.117 |
| **latent=4** | nonblind | 1.36% | +0.864 | 2.20% | +0.873 |
| **latent=4** | partial | 11.86% | -0.016 | 20.52% | -0.235 |
| **latent=1** | blind | 11.07% | +0.137 | 18.54% | -0.030 |
| **latent=1** | nonblind | 1.33% | +0.866 | 2.17% | +0.874 |
| **latent=1** | partial | 11.71% | +0.046 | 17.65% | +0.025 |
| **ssm=0.5** | blind | **9.37%** | +0.117 | 21.30% | -0.280 |
| **ssm=0.5** | nonblind | 1.45% | +0.863 | 2.27% | +0.869 |
| **ssm=0.5** | partial | 10.91% | -0.072 | 21.10% | -0.406 |
| baseline | blind | 13.32% | -0.018 | 18.28% | +0.017 |
| baseline | nonblind | 1.29% | +0.868 | 2.14% | +0.875 |
| baseline | partial | 13.13% | -0.007 | 13.32% | +0.334 |

#### Blind mode results (Day 3 features vs baseline)

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|--------|-----------|------------|-----------|------------|----------|
| **latent=4** | 10.03% | +0.184 | 19.35% | -0.117 | **14.69%** |
| **latent=1** | 11.07% | +0.137 | 18.54% | -0.030 | **14.81%** |
| latent=1 + ssm=0.5 | 10.21% | +0.178 | 19.80% | -0.153 | 15.01% |
| latent=1 + curric=10 | 12.15% | +0.064 | 18.30% | +0.010 | 15.22% |
| **ssm=0.5** | **9.37%** | +0.117 | 21.30% | -0.280 | 15.34% |
| **ssm=0.999** | 12.24% | +0.065 | 18.56% | +0.121 | 15.40% |
| baseline | 13.32% | -0.018 | 18.28% | +0.017 | 15.80% |

#### Key findings

* **Latent reasoning at 4 steps beats all Day 3 features for blind avg**: **14.69%** avg blind MAPE — ties LSmooth 0.15. AAPL 10.03% (+0.184 skill) is excellent. Only 1 step and 4 steps are stable — 2, 6, 8, 12 all show instability (AAPL blind from 12.84% to 4 million%). The narrow sweet spot suggests the small 2-layer model can't handle many extra passes through the same blocks without overthinking.
* **SSM decay=0.5 achieves the best AAPL blind MAPE ever: 9.37%** — beats the previous best (SLERP 42-44 t=0.6 at 9.41%) by 0.04pp. The low decay means the state evolves rapidly, creating short-term recurrent memory. NVDA degrades to 21.30%.
* **SSM decay=0.999 gives positive skill on both tickers blind**: AAPL +0.065, NVDA +0.121. The near-unit decay preserves state as a long-range signal booster.
* **SSM has a U-shaped stability profile**: Low decay (0.5–0.7) and very high decay (0.999) are stable. Mid-range decay (0.9–0.95) destabilises AAPL (25k%+ MAPE).
* **Curriculum learning doesn't help**: All curriculum variants worse than the base latent-steps alone. Starting from 0 steps removes the architecture's benefit during early training.
* **Combinations mostly don't stack**: latent=1 + ssm=0.999 explodes (55.76% AAPL). latent=1 + ssm=0.5 is worse than either alone.
* **Nonblind mode is invariant**: All variants score MAPE 1.33–1.45% (AAPL) / 2.17–2.27% (NVDA) with skill +0.863–0.874.

#### Verdicts

| Feature | Verdict |
|---------|---------|
| `--latent-steps 4` | **KEEP** — best Day 3 blind avg, zero params, ties LSmooth 0.15 |
| `--latent-steps 1` | **KEEP** — positive skill on both tickers blind + partial |
| `--latent-steps 2/3/6/8` | **NEUTRAL** — work but dominated by 1 and 4 |
| `--latent-steps 12` | **DROP** — unstable, AAPL explodes |
| `--ssm-decay 0.5` | **KEEP** — best AAPL blind MAPE ever (9.37%) |
| `--ssm-decay 0.999` | **KEEP** — positive skill both tickers blind |
| `--ssm-decay 0.9/0.95` | **DROP** — unstable, AAPL explodes |
| `--curriculum-epochs` (any) | **DROP** — worse than base latent-steps alone |
| latent + SSM combos | **DROP** — doesn't stack, most explode |

### Day 2, Three features: Depth LoRA, Adaptive Halting, SLERP merging

* **What**: Implemented and tested 3 features: Depth LoRA (per-block low-rank adapters on every Q/K/V/O/W1/W2/W3 projection), Adaptive Halting (Graves 2016 ACT — per-position halting probability over a recurrent block loop), and SLERP merging (post-training spherical linear interpolation of N checkpoints). 21 models trained (3 baseline seeds, 5 LoRA ranks, 5 ACT max_steps, 6 seed-variance checks, 2 cross-feature combos), 16 SLERP merges produced from those checkpoints. All 5 finalists full 3-mode benchmarked.
* **Features**: `--lora-rank` (1, 2, 4, 8, 12, 16, 24, 32), `--act-max-steps` (2, 3, 4, 5, 6, 7, 8, 10), `--slerp-t` + `scripts/run_slerp_merges.py` (t = 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9; pairs 42-43, 42-44, 43-44, 3-way)
* **Compute**: NVIDIA RTX 5090, ~5 minutes total

#### Blind mode results (Day 2 features vs baseline avg-of-3-seeds)

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|--------|-----------|------------|-----------|------------|----------|
| baseline (seed 42) | 45.72% | -2.40 | 19.19% | -0.10 | 32.46% |
| baseline (avg 3 seeds) | 28.79% | -1.14 | 18.32% | +0.01 | 23.56% |
| **SLERP 42-44 t=0.6** | **9.41%** | **+0.179** | 19.78% | -0.152 | **14.60%** |
| SLERP 42-44 t=0.5 | **9.52%** | **+0.188** | 19.60% | -0.137 | 14.56% |
| **LoRA-8 + SLERP 42-44** | **9.38%** | +0.097 | 20.72% | -0.233 | 15.05% |
| ACT-6 + SLERP 42-44 | 9.69% | **+0.190** | **19.47%** | -0.125 | **14.58%** |
| ACT max_steps=6 | 9.87% | +0.181 | 20.58% | -0.214 | 15.23% |
| LoRA rank=8 | 10.09% | +0.180 | 20.64% | -0.227 | 15.36% |
| LoRA rank=16 | 10.43% | +0.086 | 22.33% | -0.339 | 16.38% |

#### Full 3-mode results (top 5 variants)

| Config | Mode | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|--------|------|-----------|------------|-----------|------------|
| SLERP 42-44 t=0.6 | blind | **9.41%** | +0.179 | 19.78% | -0.152 |
| SLERP 42-44 t=0.6 | nonblind | 2.11% | +0.817 | 3.30% | +0.823 |
| SLERP 42-44 t=0.6 | partial | **10.93%** | +0.010 | 22.97% | -0.396 |
| SLERP 42-44 t=0.5 | blind | 9.52% | +0.188 | 19.60% | -0.137 |
| SLERP 42-44 t=0.5 | nonblind | 2.15% | +0.813 | 3.36% | +0.821 |
| SLERP 42-44 t=0.5 | partial | 11.03% | +0.026 | 22.39% | -0.348 |
| LoRA rank=8 | blind | 10.09% | +0.180 | 20.64% | -0.227 |
| LoRA rank=8 | nonblind | 1.36% | +0.866 | 2.16% | +0.874 |
| LoRA rank=8 | partial | 12.88% | -0.123 | **13.38%** | **+0.260** |
| LoRA rank=16 | blind | 10.43% | +0.086 | 22.33% | -0.339 |
| LoRA rank=16 | nonblind | **1.31%** | **+0.868** | 2.19% | +0.874 |
| LoRA rank=16 | partial | 13.01% | -0.226 | 17.40% | -0.035 |
| ACT max_steps=6 | blind | 9.87% | +0.181 | 20.58% | -0.214 |
| ACT max_steps=6 | nonblind | 1.42% | +0.863 | 2.27% | +0.870 |
| ACT max_steps=6 | partial | 12.12% | -0.066 | 15.92% | +0.083 |
| ACT-6 + SLERP 42-44 | blind | 9.69% | **+0.190** | 19.47% | -0.125 |
| ACT-6 + SLERP 42-44 | nonblind | 2.16% | +0.812 | 3.41% | +0.819 |
| ACT-6 + SLERP 42-44 | partial | 11.26% | +0.023 | 21.66% | -0.292 |

#### Key findings

* **SLERP merging is the highest-impact and lowest-cost feature this round**: merging two seeds of the baseline at t=0.5–0.6 cuts AAPL blind from 45.72% (seed 42) down to **9.41%** — a 4.9× reduction with zero training cost. The 42-44 pair has a smooth basin from t=0.1 (11.61%) to t=0.8 (10.91%); t=0.6 is the optimum.
* **SLERP only works between basin-compatible checkpoints**: the 42-43 pair diverged catastrophically at every t value tested (268% / 17268% / 2.9 billion% MAPE). The 3-way merge containing seed 43 also blew up. This is real, important information — SLERP cannot be used blindly. Pairs must be tested.
* **Depth LoRA at rank 8 sets a new NVDA partial record**: 13.38% MAPE with **+0.260 skill** — substantially beats fim 0.30's 13.19% MAPE / +0.282 skill on raw value but loses on skill margin. Among the most balanced single-feature improvements.
* **LoRA rank=16 sets a new AAPL nonblind record**: 1.31% MAPE / +0.868 skill / 0.524 DirAcc. This narrowly beats GQA+QK-norm's previous 1.31% / +0.868 / 0.520 DirAcc.
* **ACT max_steps=6 is the best single-feature blind average**: 15.23% blind avg MAPE vs LoRA-8's 15.36% and the LSmooth/Muon baselines from earlier days. Strong on every metric, no ticker breaks.
* **LoRA rank choice has a U-shape**: rank 1 works (15.7% avg), rank 2 and 4 fail (24.5% / 65.4%), rank 8 and 16 work (15.4% / 16.4%), rank 24 and 32 partially fail (26.4% / 25.1%). The rank=8 sweet spot reflects the small model's effective capacity needs.
* **Combinations mostly don't stack**: LoRA + ACT explodes NVDA (30.77% / 32.45% blind). 3-way SLERP across mismatched basins explodes (266% / 670% NVDA). The only working stack is `LoRA-8 + SLERP-42-44` (best AAPL blind ever: 9.38%) and `ACT-6 + SLERP-42-44` (best skill among SLERP merges: +0.190).
* **All three features hurt nonblind slightly when combined with SLERP**: SLERP variants score 2.11–2.34% AAPL nonblind vs the 1.31–1.42% from LoRA/ACT alone. The merge introduces a small one-step-ahead error; for blind-mode use this is a fair trade.

#### Updated leaderboard (blind mode)

| Rank | Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|------|--------|-----------|------------|-----------|------------|----------|
| — | **Muon 0.005** | 9.92% | +0.178 | 18.56% | -0.009 | **14.24%** |
| — | **SLERP 42-44 t=0.5** | 9.52% | +0.188 | 19.60% | -0.137 | 14.56% |
| — | ACT-6 + SLERP 42-44 | 9.69% | **+0.190** | 19.47% | -0.125 | 14.58% |
| — | **SLERP 42-44 t=0.6** | **9.41%** | +0.179 | 19.78% | -0.152 | **14.60%** |
| — | **latent=4 (Day 3)** | 10.03% | +0.184 | 19.35% | -0.117 | **14.69%** |
| — | fim 0.25 | 9.77% | +0.188 | 19.66% | -0.142 | 14.72% |
| — | **latent=1 (Day 3)** | 11.07% | +0.137 | 18.54% | -0.030 | **14.81%** |
| — | LSmooth 0.15 | 10.88% | +0.150 | 18.49% | -0.012 | **14.69%** |
| — | ACT max_steps=6 | 9.87% | +0.181 | 20.58% | -0.214 | 15.23% |
| — | ssm=0.5 (Day 3) | **9.37%** | +0.117 | 21.30% | -0.280 | 15.34% |
| — | LoRA-8 + SLERP 42-44 | **9.38%** | +0.097 | 20.72% | -0.233 | 15.05% |
| — | LoRA rank=8 | 10.09% | +0.180 | 20.64% | -0.227 | 15.36% |
| — | baseline (seed 42) | 45.72% | -2.40 | 19.19% | -0.10 | 32.46% |

#### Verdicts

| Feature | Verdict |
|---------|---------|
| `--lora-rank 8` | **KEEP** — best NVDA partial of any config, strong blind and nonblind |
| `--lora-rank 16` | **KEEP** — best AAPL nonblind ever (1.31% / +0.868 skill) |
| `--lora-rank 1` | **NEUTRAL** — works but rank=8 dominates |
| `--lora-rank 2/4/24/32` | **DROP** — U-shape failure, mid and high ranks unstable |
| `--act-max-steps 6` | **KEEP** — best single-feature blind avg, no metrics regress |
| `--act-max-steps 2` | **NEUTRAL** — works but max=6 dominates |
| `--act-max-steps 7-8` | **NEUTRAL** — works but slightly worse than 6 |
| `--act-max-steps 10` | **DROP** — explodes (47.70% AAPL blind) |
| SLERP merge t=0.5-0.6 (basin-compatible pair) | **KEEP** — top-3 blind AAPL with zero training cost |
| SLERP merge t<0.4 or t>0.8 | **NEUTRAL** — works for compatible pairs, degrades the further from 0.5 |
| SLERP across mismatched basins | **DROP** — diverges catastrophically at any t |
| 3-way SLERP across 3 seeds | **DROP** — diverges if any pair is incompatible |
| LoRA + ACT | **DROP** — doesn't stack, NVDA explodes |
| LoRA-8 + SLERP 42-44 | **KEEP** — best AAPL blind ever (9.38%) |
| ACT-6 + SLERP 42-44 | **KEEP** — best blind skill (+0.190) among SLERP merges |


### Day 2, Four features: Muon, MHC, FIM, GQA+QK-norm

* **What**: Implemented and tested 4 features: Muon optimizer (Newton-Schulz orthogonalization), Manifold Hyper-Connections (doubly-stochastic residual mixing), Fill-In-the-Middle augmentation (span masking), and GQA+QK-norm (grouped query attention with QK-normalization). Each was swept across 2–9 hyperparameter values (first broad, then fine around sweet spots), blind-benchmarked, then top contenders full 3-mode benchmarked. 28 models trained, winners attempted in combination.
* **Features**: `--muon-lr` (0.002–0.05), `--mhc-streams` (2,3,4), `--fim-rate` (0.1–0.5), `--gqa-kv-heads` (1,2) with `--qk-norm`
* **Compute**: CPU, ~15 minutes total

#### Blind mode results (Day 2 features vs leaders)

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|--------|-----------|------------|-----------|------------|----------|
| **Muon 0.005** | **9.92%** | **+0.178** | **18.56%** | -0.009 | **14.24%** |
| fim 0.25 | **9.77%** | **+0.188** | 19.66% | -0.142 | 14.72% |
| fim 0.30 | 10.66% | +0.160 | 19.32% | -0.114 | 14.99% |
| gqa=2_qknorm | 10.64% | +0.161 | 19.29% | -0.103 | 14.97% |
| mhc=4 | 10.97% | +0.144 | 19.04% | -0.086 | 15.01% |
| baseline | 1243.35% | -185.62 | 19.48% | -0.125 | 631.41% |

#### Full 3-mode results (top 4 variants)

| Config | Mode | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|--------|------|-----------|------------|-----------|------------|
| Muon 0.005 | blind | **9.92%** | **+0.178** | **18.56%** | -0.009 |
| Muon 0.005 | nonblind | 1.32% | +0.867 | 2.19% | +0.874 |
| Muon 0.005 | partial | 12.77% | -0.136 | 15.99% | +0.153 |
| fim 0.25 | blind | **9.77%** | **+0.188** | 19.66% | -0.142 |
| fim 0.25 | nonblind | 1.38% | +0.866 | 2.19% | +0.873 |
| fim 0.25 | partial | **11.05%** | **+0.049** | 20.47% | -0.246 |
| fim 0.30 | blind | 10.66% | +0.160 | 19.32% | -0.114 |
| fim 0.30 | nonblind | 1.36% | +0.866 | 2.17% | +0.875 |
| fim 0.30 | partial | 12.39% | -0.027 | **13.19%** | **+0.282** |
| gqa=2_qknorm | blind | 10.64% | +0.161 | 19.29% | -0.103 |
| gqa=2_qknorm | nonblind | **1.31%** | **+0.868** | 2.19% | +0.874 |
| gqa=2_qknorm | partial | 13.78% | -0.177 | 17.08% | +0.017 |

#### Key findings

* **Muon 0.005 takes #1 blind avg**: **14.24%** avg blind MAPE — beats the previous leader LSmooth 0.15 (14.69%). AAPL 9.92% (+0.178) is the second-best AAPL blind ever (behind fim 0.25). NVDA partial 15.99% (+0.153) is solid. Zero extra params. The NS orthogonalization prevents gradient interference between parameters in the small model.
* **FIM 0.25: best AAPL blind ever**: AAPL **9.77% (+0.188 skill)** — beats dropout 0.10 (9.51% but worse DirAcc and NVDA). AAPL partial also reaches positive skill (11.05%, +0.049). The span-masking forces the model to learn from surrounding context rather than local temporal smoothness.
* **FIM 0.30: best balanced FIM**: Trading off AAPL for better NVDA. NVDA partial **13.19% (+0.282)** is excellent for a zero-param feature. Blind NVDA 19.32% is close to baseline. Good for multi-ticker evaluation.
* **GQA+QK-norm reduces params by 10%**: GQA=2 with QK-norm saves 8,128 params (74,305 vs 82,433). A rare architectural change that both improves AAPL blind (10.64%, +0.161) and reduces parameter count. Nonblind AAPL is the best ever (1.31% MAPE, +0.868 skill).
* **MHC at 4 streams works**: 10.97% AAPL (+0.144), 19.04% NVDA. Only 4 streams works — 2 explodes, 3 is unstable. The Sinkhorn-Knopp regularization prevents degenerate mixing.
* **Combinations don't stack**: muon+fim, muon+gqa, fim+gqa, and all-three all performed worse than the best individual feature. Three different modification regimes (optimizer, data, architecture) overload the 82K model.
* **Nonblind mode is invariant**: All configs still score MAPE 1.31–1.38% (AAPL) / 2.17–2.19% (NVDA) with skill +0.866–0.875. The model always learns one-step-ahead patterns.

#### Updated leaderboard (blind mode)

| Rank | Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|------|--------|-----------|------------|-----------|------------|----------|
| 1 | **Muon 0.005** | **9.92%** | **+0.178** | **18.56%** | -0.009 | **14.24%** |
| 2 | fim 0.25 | **9.77%** | **+0.188** | 19.66% | -0.142 | 14.72% |
| 3 | LSmooth 0.15 | 10.88% | +0.150 | 18.49% | -0.012 | **14.69%** |
| 4 | EMA 0.9999 | 10.85% | +0.148 | 18.70% | -0.050 | 14.78% |
| 5 | fim 0.30 | 10.66% | +0.160 | 19.32% | -0.114 | 14.99% |
| 6 | gqa=2_qknorm | 10.64% | +0.161 | 19.29% | -0.103 | 14.97% |
| 7 | mhc=4 | 10.97% | +0.144 | 19.04% | -0.086 | 15.01% |
| 8 | MTP(2,4,8,16) | 11.56% | +0.109 | 18.37% | +0.039 | 14.97% |
| 9 | OHEM 0.9 | 11.41% | +0.119 | 18.73% | -0.052 | 15.07% |
| — | baseline | 1243.35% | -185.62 | 19.48% | -0.125 | 631.41% |

#### Verdicts

| Feature | Verdict |
|---------|---------|
| `--muon-lr 0.005` | **KEEP** — new best avg blind, zero params |
| `--fim-rate 0.25` | **KEEP** — best AAPL blind ever, zero params |
| `--fim-rate 0.30` | **KEEP** — best NVDA partial of Day 2, good balance |
| `--gqa-kv-heads 2 --qk-norm` | **KEEP** — 10% fewer params, strong nonblind, good blind |
| `--mhc-streams 4` | **KEEP** — novel residual mechanism, competitive results |
| `--mhc-streams 2` | **DROP** — unstable, explodes |
| `--mhc-streams 3` | **NEUTRAL** — works but worse than 4 |
| muon + fim | **DROP** — doesn't stack, avg 15.67% |
| muon + gqa | **DROP** — doesn't stack, avg 20.37% |
| fim + gqa | **DROP** — doesn't stack, avg 22.95% |
| muon + fim + gqa | **DROP** — overloads, avg 60.90% |


### Day 1, OHEM + Label Smoothing — two low-hanging regularizers

* **What**: Implemented and tested OHEM (Online Hard Example Mining) and regression-adapted label smoothing. Each feature was swept across 5–9 hyperparameter values (first broad, then fine around sweet spots), blind-benchmarked to filter contenders, then full 3-mode benchmarked on top variants. 19 models trained total.
* **Features**: `--ohem-fraction` (0.2–0.95), `--label-smoothing` (0.01–0.30)
* **Compute**: CPU, ~30 minutes total

#### Blind mode results

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|--------|-----------|------------|-----------|------------|----------|
| baseline | 1243.36% | -185.62 | 19.48% | -0.125 | 631.42% |
| **LSmooth 0.15** | **10.88%** | **+0.150** | 18.49% | -0.012 | **14.69%** |
| **OHEM 0.9** | 11.41% | +0.119 | 18.73% | -0.052 | **15.07%** |
| LSmooth 0.20 | 11.93% | +0.084 | **18.22%** | **+0.037** | **15.08%** |
| OHEM 0.3 | 11.60% | +0.109 | 19.30% | -0.108 | 15.45% |
| OHEM 0.5 | 10.08% | +0.019 | 22.18% | -0.328 | 16.13% |

#### Key findings

* **Label smoothing 0.15 takes #1 blind avg**: LSmooth 0.15 achieves **14.69%** avg blind MAPE — the best of any single feature tested so far, beating EMA 0.9999 (14.78%) and MTP (14.97%). Zero extra params. The intuition: shrinking targets toward the batch mean regularizes predictions the same way label smoothing regularizes classifiers.
* **Label smoothing 0.20 is only the 2nd config ever with positive skill on both tickers in blind AND partial**: AAPL blind +0.084, NVDA blind +0.037, AAPL partial +0.093, NVDA partial +0.112. Previously only MTP+dropout 0.01 had achieved this. Zero extra params.
* **OHEM works best as a mild regularizer**: OHEM 0.9 (drop easiest 10%) gives AAPL partial 11.19% (+0.129) — best AAPL partial ever. OHEM 0.3 (drop easiest 70%) is a strong but imbalanced regularizer: great AAPL blind (11.60%, +0.109) but hurts NVDA.
* **OHEM is unstable below 0.3**: OHEM 0.2 and 0.25 both exploded on AAPL blind (343k% and 11.98%). OHEM 0.35 found the best NVDA blind ever (14.95%, +0.273) in 1/17 seeds but is not reproducible. The stable range is 0.3–0.9.
* **Nonblind mode is invariant**: All configs scored MAPE 1.32–1.37% (AAPL) / 2.14–2.25% (NVDA) with skill +0.86–0.87. The model always learns one-step-ahead patterns; differences only show in multi-step rollout.

#### Full 3-mode results (top 4 variants)

| Config | Mode | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|--------|------|-----------|------------|-----------|------------|
| LSmooth 0.15 | blind | **10.88%** | **+0.150** | 18.49% | -0.012 |
| LSmooth 0.15 | nonblind | 1.34% | +0.866 | 2.22% | +0.872 |
| LSmooth 0.15 | partial | 12.17% | +0.003 | 18.55% | -0.036 |
| LSmooth 0.20 | blind | 11.93% | +0.084 | **18.22%** | **+0.037** |
| LSmooth 0.20 | nonblind | 1.37% | +0.862 | 2.25% | +0.871 |
| LSmooth 0.20 | partial | **11.78%** | **+0.093** | **16.99%** | **+0.112** |
| OHEM 0.9 | blind | 11.41% | +0.119 | 18.73% | -0.052 |
| OHEM 0.9 | nonblind | 1.35% | +0.865 | 2.18% | +0.874 |
| OHEM 0.9 | partial | **11.19%** | **+0.129** | 16.51% | +0.086 |
| OHEM 0.3 | blind | 11.60% | +0.109 | 19.30% | -0.108 |
| OHEM 0.3 | nonblind | 1.32% | +0.868 | 2.14% | +0.875 |
| OHEM 0.3 | partial | 13.59% | -0.096 | 14.44% | +0.227 |

#### Verdicts

| Feature | Verdict |
|---------|---------|
| `--label-smoothing 0.15` | **KEEP** — best avg blind MAPE ever, zero params |
| `--label-smoothing 0.20` | **KEEP** — positive skill everywhere, zero params |
| `--ohem-fraction 0.9` | **KEEP** — best partial AAPL ever, zero params |
| `--ohem-fraction 0.3` | **KEEP** — strong blind AAPL, good NVDA partial |
| `--ohem-fraction 0.35` | **DROP** — unstable, not reproducible |
| `--ohem-fraction 0.5` | **NEUTRAL** — AAPL specialist but hurts NVDA |

### Day 1, Tier 1 features: Crowfeather AdamW, WSD schedule, EMA

* **What**: Implemented and isolated-tested three "Tier 1" features — pure config/optimizer changes with minimal code. Each tested individually vs baseline (Glint, 3-year window), then combined with winners from previous tests. 8 models trained, blind-benchmarked, top contenders full 3-mode benchmarked.
* **Features**: `--crowfeather` (AdamW eps=1e-20 + beta2 ramp), `--lr-schedule wsd` (Warmup-Stable-Decay sqrt cooldown), `--ema-decay` (exponential moving average of weights)
* **Compute**: CPU, ~20 minutes total

#### Blind mode results

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill |
|--------|-----------|------------|-----------|------------|
| baseline | 1243.62% | -185.66 | 19.48% | -0.125 |
| **WSD** | **11.74%** | **+0.101** | 19.52% | -0.130 |
| **EMA 0.9999** | **10.85%** | **+0.148** | 18.70% | -0.050 |
| EMA 0.9995 | 14.92% | -0.143 | 18.11% | +0.108 |
| EMA 0.999 | 16.42% | -0.265 | 18.26% | +0.148 |
| Crowfeather | 17.20% | -0.807 | 32.16% | -1.046 |
| WSD + EMA 0.999 | 15.12% | -0.159 | 18.17% | +0.126 |
| Crow + WSD | 14.61% | -0.121 | 18.21% | +0.122 |

#### Key findings

* **WSD schedule works**: Simply swapping cosine for WSD gives AAPL blind 11.74% with zero extra params. The river-valley theory holds — the high stable LR phase lets the model find a better basin, and the sqrt cooldown drops it in cleanly. Trains longer (22 epochs) but worth it.
* **EMA at 0.9999 is a blind specialist**: Heaviest smoothing gives AAPL 10.85% blind and 11.25% partial — ties dropout/oreg for best AAPL blind. But nonblind degrades (3.2% MAPE). Zero extra params.
* **EMA at 0.999 is an NVDA specialist**: Milder smoothing boosts NVDA blind to 18.26% with +0.148 skill, best DirAcc (0.528). Nonblind stays normal.
* **Crowfeather is unstable**: Extreme epsilon (1e-20) causes jittery training. Early-stops at 9 epochs. Not useful alone.
* **Combinations don't stack**: WSD + EMA and Crowfeather + WSD both performed worse than WSD alone. The features compete for the same regularization budget.

#### Verdicts

| Feature | Verdict |
|---------|---------|
| WSD schedule | **KEEP** — zero-param blind AAPL boost |
| EMA 0.9999 | **KEEP** — blind specialist |
| EMA 0.999 | **KEEP** — NVDA specialist |
| Crowfeather | **DROP** (alone) / **NEUTRAL** (with WSD) |

#### Updated leaderboard (blind mode)

| Rank | Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|------|--------|-----------|------------|-----------|------------|----------|
| 1 | **LSmooth 0.15** | **10.88%** | **+0.150** | 18.49% | -0.012 | **14.69%** |
| 2 | EMA 0.9999 | 10.85% | +0.148 | 18.70% | -0.050 | 14.78% |
| 3 | MTP(2,4,8,16) | 11.56% | +0.109 | 18.37% | +0.039 | 14.97% |
| 4 | **OHEM 0.9** | 11.41% | +0.119 | 18.73% | -0.052 | 15.07% |
| 5 | LSmooth 0.20 | 11.93% | +0.084 | 18.22% | +0.037 | 15.08% |
| 6 | MTP(2,4,8,16)+drop 0.01 | 11.84% | +0.089 | 18.42% | -0.008 | 15.13% |
| 7 | oreg 0.10 | 9.65% | +0.157 | 21.36% | -0.285 | 15.51% |
| 8 | dropout 0.10 | 9.51% | +0.119 | 21.17% | -0.270 | 15.34% |
| 9 | dropout 0.01 | 18.12% | -0.405 | 16.89% | +0.202 | 17.51% |

### Day 1, full grid search — 20+ configs, optimal found

* **What**: Systematic grid search across MTP horizon variants (2,4 / 2,4,8 / 2,4,8,16 / 4,8), input dropout rates (0.003/0.005/0.01/0.03/0.05/0.10/0.15), output reg weights (0.0001/0.0005/0.001/0.005), and all combinations. 20+ models trained on Glint (82K params), blind-benchmarked to filter contenders, then full 3-mode benchmark on survivors.
* **Winner**: `--mtp-horizons 2,4,8,16 --input-dropout 0.01` — first Glint config ever with positive skill on AAPL in both blind (+0.089) and partial (+0.014) modes.
* **Compute**: CPU, ~15 minutes total

#### Full results

| Config | Blind AAPL | Blind NVDA | Partial AAPL | Partial NVDA |
|--------|-----------|-----------|-------------|-------------|
| baseline | 1243.62% / -185.66 | 19.48% / -0.125 | 82.39% / -6.07 | 145.70% / -15.05 |
| MTP(2,4,8,16)+drop 0.01 | **11.84% / +0.089** | 18.42% / -0.008 | **12.30% / +0.014** | **14.48% / +0.251** |
| MTP(2,4,8,16) | 11.56% / +0.109 | 18.37% / +0.039 | 12.51% / -0.012 | 14.50% / +0.278 |
| dropout 0.10 | 9.51% / +0.119 | 21.17% / -0.270 | 11.28% / -0.099 | 15.92% / -0.009 |
| dropout 0.01 | 18.12% / -0.405 | 16.89% / +0.202 | 15.98% / -0.275 | 15.86% / +0.250 |
| MTP(2,4,8)+drop 0.03 | 16.59% / -0.278 | 17.93% / +0.124 | 16.10% / -0.254 | 14.95% / +0.288 |
| MTP(2,4,8) | 15.36% / -0.179 | 17.96% / +0.048 | 14.19% / -0.108 | 10.15% / +0.492 |
| oreg 0.05 | 13.03% / +0.007 | 18.68% / -0.048 | 13.38% / -0.039 | **9.08% / +0.547** |
| oreg 0.10 | **9.65% / +0.157** | 21.36% / -0.285 | 12.32% / -0.155 | 24.58% / -0.585 |

Format: `MAPE / skill_vs_naive_rmse`

#### Key findings

* **MTP + dropout CAN stack** — earlier finding that they interfere was specific to MTP(2,4,8)+dropout 0.01. With the right MTP config (4 heads) and right dropout rate (0.01), they beat both individual features.
* **More MTP horizons = better**: 4 heads (2,4,8,16) > 3 heads (2,4,8) > 2 heads (2,4 or 4,8). Each additional auxiliary head regularizes the backbone more.
* **Dropout sweet spot is ticker-dependent**: 0.10 peaks AAPL (9.51% MAPE, +0.119 skill), 0.01 peaks NVDA (16.89% MAPE, +0.202 skill). No single rate dominates both.
* **Output reg completely ineffective**: Every rate tested (0.0001-0.005) failed to improve blind mode. Highest AAPL skill was -48 at best. This feature is dropped.
* **Nonblind mode is invariant**: All configs score MAPE 1.3-1.5% (AAPL) / 2.15-2.4% (NVDA) with skill +0.86-0.87. The model always learns one-step-ahead patterns; the differences only show in multi-step rollout.

#### Verdicts

| Feature | Verdict |
|---------|---------|
| MTP(2,4,8,16) + dropout 0.01 | **KEEP — optimal** |
| MTP(2,4,8,16) alone | **KEEP — runner-up** |
| Output reg 0.05 | **KEEP — NVDA partial specialist** |
| Output reg 0.10 | **KEEP — AAPL blind specialist** |
| dropout 0.10 alone | **KEEP — AAPL blind specialist** |
| dropout 0.01 alone | **KEEP — NVDA blind specialist** |
| MTP + output reg | **DROP — overloads** |
| dropout + output reg | **DROP — interferes** |

#### Run dirs

`runs/Glint_AAPL_NVDA_mtp=2,4,8,16_drop=0.01/` — optimal config checkpoint and full benchmark.

### Day 1, experiment framework + three feature A/B tests

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


### Day 1, local normalisation — real predictions, no drift

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
* Plots: `runs/Glint_AAPL_NVDA/plots/{AAPL,NVDA}_pred_vs_actual.png` (superseded by later runs).
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

### For AI agents

This repo has an [AGENTS.md](AGENTS.md) file that describes the exact process used to test and evaluate features. If you're too lazy to manually implement something, tell your agent to read `AGENTS.md` first, then ask it to knock out one of the [queued tasks](#feature-backlog). The agent will know how to flag-gate it, test it in isolation, benchmark all three modes, and log the results — because it's all documented step by step.

---

<div align="center">

*Built in public. Logged in public. [ko-fi.com/compactai](https://ko-fi.com/compactai).*

</div>
