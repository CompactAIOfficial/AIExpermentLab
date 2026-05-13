import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, layer: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.lora_A = nn.Parameter(torch.zeros(layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, layer.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.layer(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def apply_depth_lora(model: nn.Module, rank: int, alpha: float = 1.0):
    for block in model.blocks:
        for name in ['q', 'k', 'v', 'proj']:
            setattr(block.attn, name, LoRALinear(getattr(block.attn, name), rank, alpha))
        for name in ['w1', 'w2', 'w3']:
            setattr(block.ffn, name, LoRALinear(getattr(block.ffn, name), rank, alpha))
    return model
