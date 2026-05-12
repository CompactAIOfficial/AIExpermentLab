import torch
import torch.nn as nn


def build_mtp_heads(d_model: int, horizons: list[int], output_dim: int = 1):
    """Build auxiliary prediction heads for each horizon."""
    return nn.ModuleDict({
        str(h): nn.Linear(d_model, output_dim) for h in horizons
    })


def mtp_loss(mtp_preds: dict, mtp_targets: dict, loss_fn, weights: dict | None = None):
    total = 0.0
    for h_str, pred in mtp_preds.items():
        h = int(h_str)
        w = weights.get(h, 1.0) if weights else 1.0
        total += w * loss_fn(pred, mtp_targets[h])
    return total
