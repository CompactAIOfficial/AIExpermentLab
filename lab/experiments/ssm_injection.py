import torch
import torch.nn as nn


class SSMBlock(nn.Module):
    def __init__(self, dim: int, decay_init: float = 0.9):
        super().__init__()
        self.log_decay = nn.Parameter(torch.full((dim,), torch.log(torch.tensor(decay_init)).item()))
        self.B = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decay = torch.exp(self.log_decay).clamp(max=1.0)
        new_state = decay * state + self.B(x).mean(dim=1)
        return x + new_state.unsqueeze(1), new_state


class SSMInjection(nn.Module):
    def __init__(self, dim: int, n_layers: int, decay_init: float = 0.9):
        super().__init__()
        self.ssm_blocks = nn.ModuleList([SSMBlock(dim, decay_init) for _ in range(n_layers)])
        self.state = None
        self.n_layers = n_layers

    def forward(self, h: torch.Tensor, block_idx: int) -> torch.Tensor:
        if self.state is None:
            B, T, C = h.shape
            self.state = [torch.zeros(B, C, device=h.device) for _ in range(self.n_layers)]
        out, self.state[block_idx] = self.ssm_blocks[block_idx](h, self.state[block_idx])
        return out

    def reset_state(self, batch_size: int, device: torch.device):
        self.state = [torch.zeros(batch_size, ssm_block.B.in_features, device=device)
                      for ssm_block in self.ssm_blocks]
