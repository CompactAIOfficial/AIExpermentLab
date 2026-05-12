import copy
import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        for p in self.shadow.values():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                self.shadow[name].lerp_(param.to(self.shadow[name].device), 1.0 - self.decay)

    def swap(self, model: nn.Module):
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return saved

    def state_dict(self):
        return self.shadow
