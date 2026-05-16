import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep_prob * mask


def apply_stochastic_depth(model, survival_prob: float):
    drop_prob = 1.0 - survival_prob
    if drop_prob <= 0.0:
        return model
    drop_path = DropPath(drop_prob)
    for block in model.blocks:
        orig_attn_norm = block.norm1
        orig_attn = block.attn
        orig_ffn_norm = block.norm2
        orig_ffn = block.ffn

        def make_forward(a_norm, attn, f_norm, ffn, dp):
            def new_forward(x):
                attn_out = attn(a_norm(x))
                ffn_out = ffn(f_norm(x + attn_out))
                branch = attn_out + ffn_out
                return x + dp(branch)
            return new_forward

        block.forward = make_forward(orig_attn_norm, orig_attn, orig_ffn_norm, orig_ffn, drop_path)
    return model
