import torch


def output_regularization(pred: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.tensor(0.0, device=pred.device)
    return weight * pred.square().mean()
