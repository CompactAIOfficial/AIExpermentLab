import math


def get_wsd_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 0.0):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    decay_frac = (step - warmup_steps) / max(1, decay_steps)
    return min_lr + (max_lr - min_lr) * math.sqrt(max(0.0, 1.0 - decay_frac))
