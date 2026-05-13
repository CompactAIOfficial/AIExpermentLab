import argparse
import torch
import numpy as np


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if t <= 0.0:
        return v0.clone()
    if t >= 1.0:
        return v1.clone()
    dot = (v0 * v1).sum()
    dot = dot.clamp(-1, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    if sin_theta < eps:
        return (1 - t) * v0 + t * v1
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    return s0 * v0 + s1 * v1


def merge_state_dicts(state_dicts: list[dict], t: float = 0.5) -> dict:
    if len(state_dicts) == 1:
        return state_dicts[0]
    if len(state_dicts) == 2:
        return {k: slerp(t, state_dicts[0][k], state_dicts[1][k]) for k in state_dicts[0]}
    result = state_dicts[0]
    for i in range(1, len(state_dicts)):
        result = {k: slerp(t, result[k], state_dicts[i][k]) for k in result}
    return result


def merge_checkpoints(paths: list[str], t: float = 0.5, device: str = "cpu") -> dict:
    state_dicts = []
    for p in paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        state_dicts.append(ckpt["model"])
    merged_sd = merge_state_dicts(state_dicts, t)
    cfg = torch.load(paths[0], map_location=device, weights_only=False)["cfg"]
    return {"model": merged_sd, "cfg": cfg}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SLERP-merge multiple checkpoints")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to model.pt checkpoints to merge")
    p.add_argument("--t", type=float, default=0.5,
                   help="SLERP interpolation ratio (0.0 = first model, 1.0 = last model)")
    p.add_argument("--output", default="merged.pt",
                   help="Output path for merged checkpoint")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    merged = merge_checkpoints(args.checkpoints, args.t, args.device)
    torch.save(merged, args.output)
    print(f"[slerp] merged {len(args.checkpoints)} checkpoints with t={args.t} -> {args.output}")
