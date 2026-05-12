import torch


def apply_input_dropout(x: torch.Tensor, rate: float, training: bool):
    if not training or rate <= 0.0:
        return x
    mask = torch.rand_like(x[:, :, 0:1]) > rate
    return x * mask.float()
