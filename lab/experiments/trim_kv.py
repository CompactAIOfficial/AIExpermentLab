import torch
import torch.nn as nn


class TrimKVGate(nn.Module):
    def __init__(self, dh: int, n_heads: int):
        super().__init__()
        self.gate = nn.Linear(dh, 1)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.gate(k))
        return scores.squeeze(-1)
