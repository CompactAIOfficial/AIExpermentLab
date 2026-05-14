import torch
import torch.nn as nn


class SSMBlock(nn.Module):
    def __init__(self, dim: int, decay_init: float = 0.9):
        super().__init__()
        self.log_decay = nn.Parameter(torch.full((dim,), torch.log(torch.tensor(decay_init)).item()))
        self.B = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        decay = torch.exp(self.log_decay).clamp(max=1.0)
        inp = self.B(x)
        state = torch.zeros(B, 1, C, device=x.device)
        outputs = []
        for t in range(T):
            state = decay * state + inp[:, t:t+1, :]
            outputs.append(state)
        ssm_out = torch.cat(outputs, dim=1)
        return x + ssm_out


class SSMInjection(nn.Module):
    def __init__(self, dim: int, n_layers: int, decay_init: float = 0.9):
        super().__init__()
        self.blocks = nn.ModuleList([SSMBlock(dim, decay_init) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, h: torch.Tensor, block_idx: int) -> torch.Tensor:
        return self.blocks[block_idx](h)

    def reset_state(self, batch_size: int, device: torch.device):
        pass
