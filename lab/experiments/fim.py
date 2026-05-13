import torch


def apply_fim(x: torch.Tensor, fim_rate: float, span_frac: float = 0.15) -> torch.Tensor:
    if not fim_rate > 0:
        return x
    B, T, C = x.shape
    x = x.clone()
    span_len = max(1, int(T * span_frac))
    for b in range(B):
        if torch.rand(1).item() > fim_rate:
            continue
        start = torch.randint(0, T - span_len, (1,)).item()
        x[b, start:start + span_len] = 0.0
    return x
