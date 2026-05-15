import torch
import torch.nn as nn


class SleepGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.importance = nn.Linear(d_model, 1)
        self.consolidate = nn.Linear(d_model * 2, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, C = h.shape
        scores = torch.sigmoid(self.importance(h))
        scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
        summary = (scores * h).sum(dim=1, keepdim=True)
        gate = torch.sigmoid(self.importance(h))
        return h + gate * self.consolidate(torch.cat([h, summary.expand(-1, T, -1)], dim=-1))
