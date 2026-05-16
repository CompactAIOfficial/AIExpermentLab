import torch
import torch.nn as nn


class DirectionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(h))


def add_direction_head(model, weight: float, device):
    if weight == 0.0:
        return model
    model.dir_head = DirectionHead(model.cfg.d_model).to(device)
    return model


def direction_aux_loss(model, yb: torch.Tensor, weight: float) -> torch.Tensor:
    if weight == 0.0 or not hasattr(model, 'dir_head'):
        return torch.tensor(0.0, device=yb.device)
    h = model._last_hidden[:, -1]
    dir_pred = model.dir_head(h)
    dir_target = (yb > 0.0).float()
    return weight * nn.functional.binary_cross_entropy(dir_pred, dir_target)
