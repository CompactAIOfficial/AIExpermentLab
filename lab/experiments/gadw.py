import torch
import torch.nn as nn


class GADW(nn.Module):
    def __init__(self, n_losses: int = 6):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(losses):
            if i < len(self.log_vars):
                precision = torch.exp(-self.log_vars[i])
                total += precision * loss + self.log_vars[i] / 2
            else:
                total += loss
        return total
