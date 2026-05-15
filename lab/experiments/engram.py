import torch
import torch.nn as nn
import torch.nn.functional as F


class EngramMemory(nn.Module):
    def __init__(self, d_model: int, ngram_n: int = 4, table_size: int = 512):
        super().__init__()
        self.ngram_n = ngram_n
        self.table = nn.Embedding(table_size, d_model)
        self.gate = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def _hash_ngrams(self, ngrams: torch.Tensor) -> torch.Tensor:
        B, T, N = ngrams.shape
        h = torch.zeros(B, T, device=ngrams.device, dtype=torch.long)
        for i in range(N):
            h = h * 3 + ngrams[:, :, i].long()
        return h % self.table.num_embeddings

    def forward(self, h: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        B, T = prices.shape[:2]
        dx = prices.diff(dim=1)
        ternary = torch.zeros(B, T - 1, 1, device=dx.device, dtype=torch.long)
        ternary[dx > 1e-4] = 1
        ternary[dx < -1e-4] = 2
        ternary = ternary.squeeze(-1)
        padded = F.pad(ternary, (self.ngram_n - 1, 0), value=0)
        ngrams = padded.unfold(1, self.ngram_n, 1)
        indices = self._hash_ngrams(ngrams)
        indices = F.pad(indices, (1, 0), value=0)
        lookup = self.table(indices)
        gate = torch.sigmoid(self.gate(h))
        return h + gate * self.norm(lookup)
