import torch


def orthogonalize(g: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    if g.ndim < 2:
        return g
    scale = g.norm()
    g = g / (scale + 1e-8)
    for _ in range(ns_steps):
        g = g @ (1.5 * torch.eye(g.size(1), device=g.device) - 0.5 * g.T @ g)
    return g * scale


class MuonState:
    def __init__(self, model: torch.nn.Module, muon_lr: float, adamw_lr: float,
                 momentum: float = 0.95, ns_steps: int = 5, nesterov: bool = True,
                 weight_decay: float = 0.0):
        self.muon_lr = muon_lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.muon_bufs: dict[str, torch.Tensor] = {}
        self.muon_params: list[tuple[str, torch.Tensor]] = []
        self.other_params: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                self.muon_params.append((name, p))
                self.muon_bufs[name] = torch.zeros_like(p)
            else:
                self.other_params.append(p)
        self.adamw = torch.optim.AdamW(self.other_params, lr=adamw_lr,
                                        betas=(0.9, 0.95), weight_decay=weight_decay)

    def step(self):
        for name, p in self.muon_params:
            if p.grad is None:
                continue
            g = p.grad
            buf = self.muon_bufs[name]
            buf.mul_(self.momentum).add_(g)
            if self.nesterov:
                update = g.add(buf, alpha=self.momentum)
            else:
                update = buf
            update = orthogonalize(update, self.ns_steps)
            if self.weight_decay > 0:
                update = update + self.weight_decay * p.data
            p.data.add_(update, alpha=-self.muon_lr)
        self.adamw.step()

    def zero_grad(self):
        for _, p in self.muon_params:
            if p.grad is not None:
                p.grad = None
        self.adamw.zero_grad()

    def state_dict(self):
        return {
            "muon_bufs": self.muon_bufs,
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, sd):
        self.muon_bufs = sd["muon_bufs"]
        self.adamw.load_state_dict(sd["adamw"])
