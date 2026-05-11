import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import ModelConfig, TrainConfig
from .model import TinyForecaster


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(train_ds, cfg: TrainConfig):
    n_val = max(1, int(len(train_ds) * cfg.val_frac))
    n_train = len(train_ds) - n_val
    g = torch.Generator().manual_seed(cfg.seed)
    tr, va = random_split(train_ds, [n_train, n_val], generator=g)
    return (
        DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False),
        DataLoader(va, batch_size=cfg.batch_size, shuffle=False, drop_last=False),
    )


def train_model(model_cfg: ModelConfig, train_cfg: TrainConfig, train_ds, run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    set_seed(train_cfg.seed)

    device = torch.device(train_cfg.device)
    model = TinyForecaster(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader, val_loader = make_loaders(train_ds, train_cfg)

    history = []
    best_val = float("inf")

    for epoch in range(train_cfg.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= max(1, n)

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
                nv += xb.size(0)
        val_loss /= max(1, nv)

        dt = time.time() - t0
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "sec": dt})
        print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  ({dt:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": model_cfg.__dict__}, os.path.join(run_dir, "model.pt"))

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model, history
