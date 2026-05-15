import torch
import torch.nn.functional as F


def nce_context_loss(
    h: torch.Tensor,
    weight: float,
    temperature: float = 0.5,
    pos_window: int = 2,
    neg_window: int = 8,
) -> torch.Tensor:
    if weight == 0.0:
        return torch.tensor(0.0, device=h.device)
    B, T, D = h.shape
    if T < neg_window + 2:
        return torch.tensor(0.0, device=h.device)
    anchor = h[:, -1:, :]  # (B, 1, D) last position
    pos = h[:, -pos_window-1:-1, :]  # (B, pos_window, D) nearby positions
    neg = h[:, :T-neg_window, :]  # (B, T-neg_window, D) distant positions
    pos_sim = torch.einsum('bld,bkd->blk', anchor, pos) / temperature
    neg_sim = torch.einsum('bld,bkd->blk', anchor, neg) / temperature
    pos_sim = pos_sim.sum(dim=-1)  # (B, 1) aggregate positive similarities
    neg_sim_max = neg_sim.logsumexp(dim=-1)  # (B, 1) softmax-denominator for negatives
    loss_per_sample = -pos_sim.squeeze(-1) + neg_sim_max.squeeze(-1)
    return weight * loss_per_sample.mean()
