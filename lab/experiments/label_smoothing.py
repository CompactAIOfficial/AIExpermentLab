import torch


def smooth_targets(yb: torch.Tensor, epsilon: float) -> torch.Tensor:
    if epsilon <= 0.0:
        return yb
    mean = yb.mean()
    return yb * (1.0 - epsilon) + mean * epsilon
