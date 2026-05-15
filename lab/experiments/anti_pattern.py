import torch


def anti_pattern_loss(
    pred: torch.Tensor,
    xb: torch.Tensor,
    weight: float,
    lookback: int = 5,
) -> torch.Tensor:
    if weight == 0.0:
        return torch.tensor(0.0, device=pred.device)
    if xb.size(1) < lookback:
        return torch.tensor(0.0, device=pred.device)
    last = xb[:, -1:]  # (B, 1)
    trend = xb[:, -1] - xb[:, -lookback]  # (B,) recent direction
    pred_dir = pred - last  # (B, 1) positive = predicting up
    trend_dir = trend.sign().unsqueeze(-1)  # (B, 1)
    wrong = (pred_dir * trend_dir) < 0
    magnitude = pred_dir.abs()
    penalty = (wrong.float() * magnitude).mean()
    return weight * penalty
