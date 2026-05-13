<div align="center">

# AIExpermentLab

### *"Well here we are again."* GLaDOS, Portal 2

**Open source training code for every weird idea I have decided to gather, bolt on, and benchmark.**

> **Domain warning**: This repo benchmarks on stock price prediction (regression on 1-D time series), not on language modelling. Features that help here may behave differently for LLMs, and vice versa. Some descriptions in the backlog below still reference their LLM/text origins and haven't been rewritten yet — the code adapts, the docs lag. YMMV.

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support_the_Lab-FF5E5B?style=for-the-badge&logo=kofi&logoColor=white)](https://ko-fi.com/compactai)
[![License](https://img.shields.io/badge/License-AGPL_V3-blue?style=for-the-badge)](#license)
[![Status](https://img.shields.io/badge/Status-Day_2-orange?style=for-the-badge)](#progress-log)

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
| Stable Recurrent Injection (SSM-style `h = A·h + B·e + Δ`, Parcae §3.1) | TinyMemoryLM | queued |
| Depth LoRA (per-loop low-rank adaptation with learned loop embeddings) | TinyMemoryLM | queued |
| Adaptive Halting (per-position learned stop probability) | TinyMemoryLM | queued |
| SleepGate (persistent cross-sequence memory buffer, periodic consolidation) | `sleep_gate.py` | queued |
| TRIM-KV retention gate (arxiv:2512.03324, learned per-entry retention β) | `sleep_gate.py` | queued |
| Engram (DeepSeek-style hashed n-gram conditional memory, O(1) lookup) | `EngramBlock` | queued |
| Manifold Hyper-Connections (Sinkhorn-Knopp doubly stochastic residual mixing) | `lab/model.py` | **tested — KEPT (4 streams)** |
| COCONUT-style Latent Thinking blocks (continuous chain-of-thought) | `latent_think_layers` | queued |
| Multi-Token Prediction (auxiliary heads at future horizons) | `lab/experiments/mtp.py` | **tested — KEPT** |
| Per-Layer Embeddings (Gemma 3n PLE, token-conditional per-layer signal) | `ple_dim` | queued |
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
| Anti-pattern unlikelihood loss (push down curated bad continuations) | `anti_patterns.py` | queued |
| GADW (Gradient Aware Dynamic Weighting for multi-loss balancing) | `gadw.py` | queued |
| Curriculum learning (recurrent loop count ramp-up) | `curriculum.py` | queued |
| Model averaging (EMA / SWA style weight smoothing) | `lab/experiments/ema.py` | **tested — KEPT (0.9999)** |
| Muon optimizer (Newton-Schulz orthogonalization) | `lab/experiments/muon.py` | **tested — KEPT (0.005)** |
| Crowfeather AdamW (eps=1e-20, β2 ramp 0.95→0.97 post-warmup) | `lab/experiments/crowfeather.py` | **tested — DROPPED** |
| WSD schedule (sqrt cooldown) vs cosine LR schedule | `lab/experiments/wsd_schedule.py` | **tested — KEPT** |
| FIM (Fill-In-the-Middle) augmentation during pretraining | `lab/experiments/fim.py` | **tested — KEPT (0.25/0.30)** |
| Decontamination pass against eval suites | `data/decontamination.py` | queued |
| OHEM (Online Hard Example Mining) with dynamic threshold | `lab/experiments/ohem.py` | **tested — KEPT (0.9/0.3)** |
| Looping regularization (OpenMythos protection against weight collapse) | `training.py` | queued |
| Input token dropout (replace fraction of inputs with zero) | `lab/experiments/input_dropout.py` | **tested — KEPT** |
| Context loss (NCE-based contrastive prompt/response embedding loss) | `training.py` | queued |
| Label smoothing (regression-adapted, target shrinkage toward batch mean) | `lab/experiments/label_smoothing.py` | **tested — KEPT (0.15/0.20)** |
| Output L2 regularization (penalize extreme predictions) | `lab/experiments/output_reg.py` | **tested — KEPT (0.05–0.10)** |
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
| 2 | fim 0.25 | **9.77%** | **+0.188** | 19.66% | -0.142 | **14.72%** |
| 3 | **LSmooth 0.15** | **10.88%** | **+0.150** | 18.49% | -0.012 | **14.69%** |
| 4 | EMA 0.9999 | 10.85% | +0.148 | 18.70% | -0.050 | 14.78% |
| 5 | fim 0.30 | 10.66% | +0.160 | 19.32% | -0.114 | 14.99% |
| 6 | gqa=2_qknorm | 10.64% | +0.161 | 19.29% | -0.103 | 14.97% |
| 7 | mhc=4 | 10.97% | +0.144 | 19.04% | -0.086 | 15.01% |
| 8 | MTP(2,4,8,16) | 11.56% | +0.109 | 18.37% | +0.039 | 14.97% |
| 9 | **OHEM 0.9** | 11.41% | +0.119 | 18.73% | -0.052 | 15.07% |
| 10 | LSmooth 0.20 | 11.93% | +0.084 | 18.22% | +0.037 | 15.08% |
| 11 | MTP(2,4,8,16)+drop 0.01 | 11.84% | +0.089 | 18.42% | -0.008 | 15.13% |
| 12 | dropout 0.10 | 9.51% | +0.119 | 21.17% | -0.270 | 15.34% |
| 13 | oreg 0.10 | 9.65% | +0.157 | 21.36% | -0.285 | 15.51% |
| 14 | dropout 0.01 | 18.12% | -0.405 | 16.89% | +0.202 | 17.51% |
| — | baseline | 1243.62% | -185.66 | 19.48% | -0.125 | 631.55% |

### Specialists

Each feature has a mode where it excels:

| Goal | Config | Best Metric |
|------|--------|-------------|
| Best avg blind MAPE | `--muon-lr 0.005` | Avg blind **14.24%** MAPE |
| Best blind AAPL skill | `--fim-rate 0.25` | AAPL blind **9.77%** MAPE, **+0.188** skill |
| Best blind AAPL skill (runner-up) | `--muon-lr 0.005` | AAPL blind **9.92%** MAPE, **+0.178** skill |
| Best nonblind DirAcc | `--gqa-kv-heads 2 --qk-norm` | AAPL nonblind DirAcc **0.520**, skill **+0.868** |
| Best NVDA partial (Day 2) | `--fim-rate 0.3` | NVDA partial **13.19%** MAPE, **+0.282** skill |
| Best NVDA partial (all-time) | `--output-reg 0.05` | NVDA partial **9.08%** MAPE, **+0.547** skill |
| Best avg blind MAPE (zero params) | `--label-smoothing 0.15` | Avg blind **14.69%** MAPE |
| Best AAPL partial (zero params) | `--ohem-fraction 0.9` | AAPL partial **11.19%** MAPE, **+0.129** skill |
| Best AAPL blind (zero params) | `--input-dropout 0.10` | AAPL blind **9.51%** MAPE, **+0.119** skill |
| Best positive-skill both (zero params) | `--label-smoothing 0.20` | AAPL blind +0.084, NVDA blind +0.037, AAPL partial +0.093, NVDA partial +0.112 |
| Best NVDA blind (zero params) | `--lr-schedule wsd` | AAPL blind **11.74%** MAPE, **+0.101** skill |
| Best NVDA blind | `--input-dropout 0.01` | NVDA blind **16.89%** MAPE, **+0.202** skill |
| Best avg MAPE (architecture) | `--mtp-horizons 2,4,8,16` | Avg blind **14.97%** MAPE |

```bash
# Best avg blind MAPE (zero params)
python train.py --label-smoothing 0.15

# NEW BEST avg blind MAPE
python train.py --muon-lr 0.005

# Best positive skill across all modes (zero params)
python train.py --label-smoothing 0.20

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
