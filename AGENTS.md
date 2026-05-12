# Agent Workflow for AIExpermentLab

This document describes the exact process used to evaluate features in this repo. Follow it step by step. Deviate only when you have a strong reason and document why.

## Core Principles

1. **Feature behind a flag.** Every new technique lands as an opt-in `--flag` in `train.py`. The experiment code lives in `lab/experiments/<name>.py`. No silent changes to the baseline.
2. **Isolate first.** Test the feature alone vs a freshly trained baseline before combining it with anything.
3. **Sweep wide, then zoom.** Start with a broad range (2+ orders of magnitude for hyperparameters). Once you find the rough optimum, do a finer sweep around it.
4. **Don't give up early.** If 3 settings don't work, try 10 more before declaring something dead. The output reg story is the cautionary tale: first 5 values (1e-6 to 0.005) looked useless; the 6th (0.05) was a record-breaker.
5. **Blind mode is the filter.** Always bench in blind mode first. It's the hardest mode and fastest to run. Only full 3-mode bench the contenders.
6. **Log honestly.** Every result goes in the Config Cheat Sheet with "Does well" and "Sucks at" columns. If it failed, say why. If it only helps one ticker, say that.
7. **Commit frequently.** One commit per experiment phase (implementation, results, adjustments).

## Step-by-Step Process

### Phase 1: Research

Before writing any code, search the web to understand the technique:

- What is the core idea? (1-2 sentences)
- What hyperparameters matter? What are typical values?
- Has it been applied to time series / regression before?
- Is there a river-valley / signal-to-noise / regularization explanation for why it works?

Do this even if you think you know the technique. Your training data may be stale, and papers get updated.

### Phase 2: Implementation

**Files to create or modify:**

1. `lab/experiments/<name>.py` — The experiment code. Keep it self-contained. Export functions, not classes (unless a class is needed like ModelEMA).
2. `lab/training.py` — Wire the feature into the training loop. Add parameters to `train_model()`. Keep the signature clean with defaults that match baseline behavior.
3. `train.py` — Add the CLI flag in `parse_args()`. Add it to the flags list for the output dir suffix. Pass it to `train_model()`.
4. `lab/model.py` — Only modify if the feature changes the model architecture (e.g. MTP heads). Otherwise keep the model clean.

**Implementation rules:**

- The baseline must produce identical results with and without the flag at default values.
- New experiment files import from PyTorch/stdlib only. No new dependencies.
- If the feature has a hyperparameter, make it a CLI argument with a sensible default.
- If the feature can be combined with others, make sure the flags compose correctly.

### Phase 3: Train the Baseline

After implementing, train a fresh baseline. Run `runs/` has been deleted.

```bash
python train.py --series Glint --tickers AAPL,NVDA \
    --train-start 2022-01-01 --train-end 2024-12-31 \
    --epochs 50 --device cpu
```

Note: the baseline typically early-stops around epoch 15. This is normal. Save the checkpoint path.

Blind-benchmark the baseline immediately to get the reference numbers:

```bash
python benchmark.py --model runs/Glint_AAPL_NVDA/model.pt --mode blind --no-plot
```

### Phase 4: Isolated Test — Blind Mode Sweep

Train the feature alone with a range of hyperparameter values. Start broad:

- For a **rate/weight** (dropout, output reg, EMA decay): sweep at least 5-7 values across 2-3 orders of magnitude. e.g. [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
- For a **boolean** (crowfeather): train once.
- For a **multi-choice** (MTP horizons, schedule type): train each variant.

**Batch the training calls in parallel** — they're independent.

After training, blind-benchmark every variant:

```bash
for tag in "..."; do
  python benchmark.py --model runs/Glint_AAPL_NVDA_${tag}/model.pt --mode blind --no-plot
done
```

Save the blind benchmark JSONs: `cp benchmark.json benchmark_blind.json`

**Analyze the results.** Build a table like:

| Config | AAPL MAPE | AAPL Skill | NVDA MAPE | NVDA Skill | Avg MAPE |
|--------|-----------|------------|-----------|------------|----------|

Look for:
- Clear trends (monotonic improvement with hyperparameter)
- Plateaus (diminishing returns)
- Explosions (instability — note the threshold)
- Ticker asymmetries (helps AAPL but hurts NVDA, or vice versa)

### Phase 5: Determine the Sweep Window

**If you see a clear optimum within your range:** do a finer sweep around it. e.g. if 0.05 is best, also try 0.03, 0.07, 0.08, 0.09.

**If you see a monotonic trend toward one end of your range:** extend the range further in that direction. e.g. if 0.1 is the best and 0.01 is worse, try 0.5, 1.0.

**If you see no effect:** the feature is dead. Skip to Phase 7 (log it as DROPPED).

**If you see mixed results** (helps one ticker, hurts another): note the asymmetry. This is common and useful information.

### Phase 6: Full 3-Mode Benchmark

Only the top 1-3 variants of this feature deserve a full benchmark. Run all three modes:

```bash
for mode in blind nonblind partial; do
  python benchmark.py --model ... --mode "$mode" --no-plot
  cp benchmark.json benchmark_${mode}.json
done
```

**When to full-bench:**
- The variant clearly beats baseline in blind mode
- The variant shows a unique strength (e.g. best NVDA partial ever)
- You need the partial/nonblind numbers for the Config Cheat Sheet

**When to skip full-bench:**
- It's clearly worse than baseline
- It's dominated by another variant of the same feature
- It exploded (MAPE > 1000%)

### Phase 7: Combine with Winners

After finding the best individual configs, try combining them:

1. Pick the top 1-2 configs from each feature family
2. Train + blind-benchmark each combination
3. If the combination beats both individuals, full-bench it
4. If the combination is worse than either individual, log it as "doesn't stack"

**Common failure patterns for combinations:**
- Two strong regularizers overload a small model (MTP + output reg)
- Both features target the same mechanism (WSd + EMA both smooth weights)
- The interaction is non-linear (dropout 0.01 interferes with MTP at 3 heads but works at 4 heads)

### Phase 8: Update the Config Cheat Sheet

Add an entry to `## Config Cheat Sheet` in README.md. Every entry must have:

```
### `--flag-name value`

(1-2 sentence explanation of what it does, referencing the research paper if applicable)

| Does well | Sucks at |
|-----------|----------|
| Concrete metric. Best MAPE/skill for a specific mode/ticker. | Concrete tradeoff. What gets worse. When not to use it. |

Best for: "One-liner describing the ideal use case."
```

**If the feature was tested and kept:** add it to the Specialist table.
**If the feature was tested and dropped:** add a short note explaining why, grouped under "Configs that don't stack" or in its own entry with "Nothing / Everything" in the table.

### Phase 9: Update the Feature Backlog

Change the Status column in `## Feature Backlog`:
- `queued` → `**tested — KEPT**` (with notes on optimal settings)
- `queued` → `**tested — DROPPED**` (with reason)

Update the Origin column to point to `lab/experiments/<name>.py` (not a vague reference to `training.py` or the paper name).

### Phase 10: Update the Progress Log

Add a new entry at the top of `## Progress Log` with:

```markdown
### Day 1, <short title>

* **What**: <what you did, what was tested, how many models>
* **Features**: <list of flags tested>
* **Compute**: <CPU/GPU, approximate time>

<brief results table showing blind-mode numbers vs baseline>

#### Key findings

* <bullet point per insight>

#### Verdicts

| Feature | Verdict |
|---------|---------|
| <flag> | **KEEP** or **DROP** or **NEUTRAL** |
```

### Phase 11: Day Numbering

The day number in commit messages and the README Progress Log stays as-is unless the user explicitly says "it is now Day X".

**Rules:**
- If the current day is Day 1, all new commits say `Day 1:` and new log entries say `### Day 1,`.
- Do NOT increment the day number on your own. Ever.
- If the user says "it is now Day 2", then:
  - New commits use `Day 2:`
  - New Progress Log entries use `### Day 2,`
  - Update the `[![Status](https://img.shields.io/badge/Status-Day_1-orange)]` badge in the README header to point to the new day number
  - Existing commits and entries stay as-is — do not rebase old ones
- If the user says "rename everything to Day X", use `git filter-branch` (Phase 12) to rename all past commits AND update the README content.

### Phase 12: Commit

One commit per logical phase. Use descriptive messages:

```
Day 1: <feature> implementation behind --flag
Day 1: <feature> isolated test — result (MAPE/Skill)
Day 1: <feature> combined with <other> — stacks/doesn't stack
```

Good example:
```
Day 1: Tier 1 features: Crowfeather AdamW, WSD LR schedule, EMA weight averaging
```

Bad example:
```
update stuff
fix things
```

### Phase 13: When to Git Rebase

Rename commit messages if the user asks. Use `git filter-branch` for batch renaming:

```bash
git filter-branch --msg-filter 'sed "s/Old Text/New Text/g"' -- <parent>..HEAD
```

Only do this if the commits haven't been pushed to remote.

## Specific Patterns Learned

### How to test a rate/weight parameter

1. Pick 7+ values spanning 2+ orders of magnitude
2. Train all in parallel
3. Blind-benchmark all
4. Find the trend
5. If there's a clear optimum, fine-sweep 3-5 values around it
6. Full-bench the best 1-2

Example (output reg):
- First sweep: [1e-6, 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005] — found monotonic improvement
- Second sweep: [0.01, 0.03, 0.05, 0.07, 0.08, 0.09, 0.1] — found optimum at 0.05–0.10
- Total: 14 models trained, not 3

### How to test a multi-choice parameter

1. Train every distinct value
2. Blind-benchmark to rank them
3. Full-bench the top 2-3
4. Note the pattern (e.g. "4 heads > 3 heads > 2 heads" for MTP)

### When a combination doesn't stack

This is valuable information. Log it in "Configs that don't stack". The mechanism often reveals something about how both features work.

### Output dir naming convention

`runs/<Series>_<TICKERS>_<flag1>=<val1>_<flag2>=<val2>`

Examples:
- `runs/Glint_AAPL_NVDA_mtp=2,4,8,16_drop=0.01`
- `runs/Glint_AAPL_NVDA_wsd_ema=0.999`
- `runs/Glint_AAPL_NVDA_crow`

Flags appear in the order they tested/composable, not alphabetical.

### Benchmark notes

- Blind mode (250-step autoregressive rollout without seeing real prices) is the hardest test and the best differentiator.
- Nonblind mode is essentially invariant — every competent config scores MAPE 1.3-1.5% AAPL / 2.15-2.4% NVDA with skill +0.86-0.87. If a feature breaks nonblind, that's a red flag.
- Partial mode (re-anchor every 126 steps) lies between blind and nonblind. It often reveals mid-range regularisation effects.

### When to stop testing a feature

Stop when you have:
- A clear verdict (KEEP / DROP / NEUTRAL)
- The optimal hyperparameter value(s) identified
- An understanding of what it helps and what it hurts
- A note on whether it stacks with other features

Do NOT stop just because the first 3 values didn't work. Keep sweeping.

Do NOT stop because you're tired of training. Batch the jobs and let them run.

## Checklist Before Declaring a Feature "DROPPED"

- [ ] Tested at least 7 different hyperparameter values across 2+ orders of magnitude
- [ ] Tested alone (not just in combination with other features)
- [ ] Blind-benchmarked all variants
- [ ] If any variant showed promise, did a finer sweep around it
- [ ] If it helped in blind but hurt nonblind, that's a tradeoff, not a drop
- [ ] Checked if it stacks with the current best config
- [ ] Checked if a different MTP/dropout/oreg setting makes it work
- [ ] Documented why it fails

## CLI Quick Reference

```bash
# Train with a single feature
python train.py --mtp-horizons 2,4,8,16 --input-dropout 0.01

# Train with WSD schedule
python train.py --lr-schedule wsd

# Train with EMA
python train.py --ema-decay 0.9999

# Benchmark all modes
python benchmark.py --model runs/.../model.pt --mode blind --no-plot
python benchmark.py --model runs/.../model.pt --mode nonblind --no-plot
python benchmark.py --model runs/.../model.pt --mode partial --no-plot
```

## Model Config Reference

```
Glint:  d_model=64,  n_heads=4, n_layers=2, d_ff=128  (82K params)
Shard:  d_model=128, n_heads=4, n_layers=4, d_ff=256  (656K params)
Prism:  d_model=256, n_heads=8, n_layers=6, d_ff=512  (2.6M params)
```

Always use Glint (82K params) for quick experiments. It's fast enough on CPU that you can batch 4 training runs in parallel without waiting.
