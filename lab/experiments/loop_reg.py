import torch
import torch.nn as nn


def loop_regularization(
    model: nn.Module,
    weight: float,
) -> torch.Tensor:
    if weight == 0.0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    h = model._last_hidden
    return weight * h.pow(2).mean()
