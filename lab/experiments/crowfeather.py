import torch


EPS = 1e-20
BETA2_START = 0.95
BETA2_END = 0.97


def apply_crowfeather(opt: torch.optim.Optimizer, warmup_steps: int, current_step: int):
    pg = opt.param_groups[0]
    pg["eps"] = EPS
    if current_step <= warmup_steps:
        beta2 = BETA2_START
    else:
        frac = (current_step - warmup_steps) / max(1, warmup_steps)
        beta2 = BETA2_START + (BETA2_END - BETA2_START) * min(1.0, frac)
    pg["betas"] = (pg["betas"][0], beta2)
