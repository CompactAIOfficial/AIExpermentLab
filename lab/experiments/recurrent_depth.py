import torch
import torch.nn as nn


class RecurrentDepthCore(nn.Module):
    def __init__(self, blocks: nn.ModuleList, n_recurrent: int):
        super().__init__()
        self.n_recurrent = n_recurrent
        n = len(blocks)
        mid = max(1, n // 2)
        self.prelude = nn.ModuleList(blocks[:mid])
        self.core = blocks[min(mid, n - 1)]
        self.coda = nn.ModuleList(blocks[mid + 1:]) if mid + 1 < n else nn.ModuleList()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for blk in self.prelude:
            h = blk(h)
        for _ in range(self.n_recurrent):
            h = self.core(h)
        for blk in self.coda:
            h = blk(h)
        return h
