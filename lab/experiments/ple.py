import torch
import torch.nn as nn


class PerLayerEmbeddings(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.zeros(n_layers, d_model))
        nn.init.normal_(self.embeddings, std=0.02)

    def forward(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return h + self.embeddings[layer_idx].unsqueeze(0).unsqueeze(0)
