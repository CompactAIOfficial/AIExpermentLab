import torch
import torch.nn as nn


class LatentReasoning(nn.Module):
    def __init__(self, blocks: nn.ModuleList, d_model: int, latent_steps: int = 3,
                 think_penalty: float = 0.0):
        super().__init__()
        self.blocks = blocks
        self.latent_steps = latent_steps
        self.think_penalty = think_penalty
        self.think_proj = nn.Linear(d_model, d_model, bias=False) if think_penalty > 0 else None

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = h.shape
        state = h

        for step in range(self.latent_steps):
            for blk in self.blocks:
                state = blk(state)

        think_loss = torch.tensor(0.0, device=h.device)
        if self.think_penalty > 0 and self.training:
            diff = state - h.detach()
            think_loss = self.think_penalty * diff.square().mean()

        return state, think_loss


def get_latent_steps_for_epoch(epoch: int, total_latent_steps: int,
                               curriculum_epochs: int) -> int:
    if curriculum_epochs <= 0:
        return total_latent_steps
    if epoch >= curriculum_epochs:
        return total_latent_steps
    frac = (epoch + 1) / curriculum_epochs
    return max(0, min(total_latent_steps, int(round(total_latent_steps * frac))))
