import torch


def ohem_loss(losses: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction <= 0.0 or fraction >= 1.0:
        return losses.mean()
    k = max(1, int(losses.size(0) * fraction))
    topk, _ = torch.topk(losses, k)
    return topk.mean()
