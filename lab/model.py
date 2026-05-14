import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def sinkhorn_log(logits: torch.Tensor, num_iters: int = 10, tau: float = 0.05):
    logits = logits / tau
    for _ in range(num_iters):
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        logits = logits - logits.logsumexp(dim=-2, keepdim=True)
    return logits.exp()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        if cfg.gqa_kv_heads and cfg.gqa_kv_heads > 0 and cfg.gqa_kv_heads < cfg.n_heads:
            self.n_kv_heads = cfg.gqa_kv_heads
        else:
            self.n_kv_heads = cfg.n_heads

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, self.head_dim * self.n_kv_heads, bias=False)
        self.v = nn.Linear(cfg.d_model, self.head_dim * self.n_kv_heads, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.n_kv_heads != self.n_heads:
            n_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HyperConnection(nn.Module):
    def __init__(self, dim: int, num_streams: int):
        super().__init__()
        self.num_streams = num_streams
        self.res_logits = nn.Parameter(torch.zeros(num_streams, num_streams))
        self.pre_logits = nn.Parameter(torch.zeros(num_streams))
        self.post_logits = nn.Parameter(torch.zeros(num_streams))

    def forward(self, streams: torch.Tensor, branch_fn):
        B, N, T, C = streams.shape
        H_res = sinkhorn_log(self.res_logits, num_iters=10, tau=0.05)
        pre = F.softmax(self.pre_logits, dim=0)
        post = F.softmax(self.post_logits, dim=0)
        branch_in = torch.einsum('n,bntc->btc', pre, streams)
        branch_out = branch_fn(branch_in)
        branch_scatter = torch.einsum('n,btc->bntc', post, branch_out)
        residual = torch.einsum('ij,bjtc->bitc', H_res, streams)
        return residual + branch_scatter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TinyForecaster(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, max_len=cfg.seq_len * 4)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.output_dim)

        if cfg.mhc_streams > 0:
            self.hc_layers = nn.ModuleList([
                HyperConnection(cfg.d_model, cfg.mhc_streams) for _ in range(cfg.n_layers)
            ])

        if cfg.mtp_horizons:
            from .experiments.mtp import build_mtp_heads
            self.mtp_heads = build_mtp_heads(cfg.d_model, cfg.mtp_horizons, cfg.output_dim)
        else:
            self.mtp_heads = None

        if cfg.act_max_steps > 0:
            from .experiments.adaptive_halting import AdaptiveHalting
            self.act = AdaptiveHalting(
                self.blocks, cfg.d_model,
                max_steps=cfg.act_max_steps,
                epsilon=cfg.act_epsilon,
                time_penalty=cfg.act_time_penalty,
            )
        else:
            self.act = None

        if cfg.latent_steps > 0:
            from .experiments.latent_reasoning import LatentReasoning
            self.latent = LatentReasoning(
                self.blocks, cfg.d_model,
                latent_steps=cfg.latent_steps,
                think_penalty=0.0,
            )
        else:
            self.latent = None

        if cfg.ssm_decay > 0:
            from .experiments.ssm_injection import SSMInjection
            self.ssm = SSMInjection(cfg.d_model, cfg.n_layers, decay_init=cfg.ssm_decay)
        else:
            self.ssm = None

        self._ponder_cost = None
        self._think_cost = None

    def _process_blocks(self, h: torch.Tensor) -> torch.Tensor:
        if self.ssm is not None:
            self.ssm.reset_state(h.size(0), h.device)
        if self.cfg.mhc_streams > 0:
            streams = h.unsqueeze(1).expand(-1, self.cfg.mhc_streams, -1, -1).clone()
            streams = streams + torch.randn_like(streams) * 0.01
            for blk, hc in zip(self.blocks, self.hc_layers):
                streams = hc(streams, lambda x, b=blk: b(x))
            h = streams.mean(dim=1)
        elif self.act is not None:
            h, ponder_cost = self.act(h)
            self._ponder_cost = ponder_cost
        else:
            for i, blk in enumerate(self.blocks):
                h = blk(h)
                if self.ssm is not None:
                    h = self.ssm(h, i)
        return h

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos(h)

        h = self._process_blocks(h)

        if self.latent is not None:
            h, think_loss = self.latent(h)
            self._think_cost = think_loss

        h = self.norm(h)
        main = self.head(h[:, -1])
        if self.mtp_heads is not None:
            mtp = {}
            for h_str, head in self.mtp_heads.items():
                mtp[int(h_str)] = head(h[:, -1])
            return main, mtp
        return main

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
