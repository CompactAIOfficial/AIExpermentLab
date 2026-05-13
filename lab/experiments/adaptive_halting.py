import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHalting(nn.Module):
    def __init__(self, blocks: nn.ModuleList, d_model: int,
                 max_steps: int = 5, epsilon: float = 0.01,
                 time_penalty: float = 0.001):
        super().__init__()
        self.blocks = blocks
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.time_penalty = time_penalty
        self.halt = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = h.shape
        device = h.device
        state = h
        halting = torch.zeros(B, T, device=device)
        output = torch.zeros_like(h)
        total_ponder = 0.0

        for step in range(self.max_steps):
            for blk in self.blocks:
                state = blk(state)

            p = torch.sigmoid(self.halt(state)).squeeze(-1)
            remaining = 1.0 - halting
            p = torch.where(remaining > self.epsilon, p, torch.zeros_like(p))
            p = torch.min(p, remaining)

            output = output + p.unsqueeze(-1) * state
            halting = halting + p
            total_ponder = total_ponder + p.sum()

            if (remaining > self.epsilon).sum() == 0:
                break

        remainder = (1.0 - halting).clamp(min=0)
        output = output + remainder.unsqueeze(-1) * state
        total_ponder = total_ponder + remainder.sum()

        avg_steps = total_ponder / max(B * T, 1)
        ponder_cost = avg_steps * self.time_penalty

        return output, ponder_cost
