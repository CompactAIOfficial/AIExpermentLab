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


def train_model(
    model_cfg: ModelConfig, train_cfg: TrainConfig, train_ds, run_dir: str,
    input_dropout: float = 0.0, output_reg: float = 0.0, mtp_weight: float = 0.3,
    crowfeather: bool = False, lr_schedule: str = "cosine", ema_decay: float = 0.0,
    ohem_fraction: float = 0.0, label_smoothing: float = 0.0,
):
    os.makedirs(run_dir, exist_ok=True)
    set_seed(train_cfg.seed)

    device = torch.device(train_cfg.device)
    model = TinyForecaster(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = nn.MSELoss()
    loss_fn_per_sample = nn.MSELoss(reduction="none") if ohem_fraction > 0 else None

    from .experiments.input_dropout import apply_input_dropout
    from .experiments.output_reg import output_regularization

    has_mtp = bool(model_cfg.mtp_horizons)

    if crowfeather:
        opt.param_groups[0]["eps"] = 1e-20

    ema = None
    if ema_decay > 0:
        from .experiments.ema import ModelEMA
        ema = ModelEMA(model, decay=ema_decay)

    train_loader, val_loader = make_loaders(train_ds, train_cfg)
    batches_per_epoch = len(train_loader)
    warmup_steps = int(0.1 * train_cfg.epochs * batches_per_epoch)
    total_steps = train_cfg.epochs * batches_per_epoch

    history = []
    best_val = float("inf")
    stalled = 0
    global_step = 0

    for epoch in range(train_cfg.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            if has_mtp:
                xb, yb, aux_targets = batch
            else:
                xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)

            xb = apply_input_dropout(xb, input_dropout, training=True)

            if lr_schedule == "wsd":
                from .experiments.wsd_schedule import get_wsd_lr
                opt.param_groups[0]["lr"] = get_wsd_lr(
                    global_step, warmup_steps, total_steps, train_cfg.lr
                )

            opt.zero_grad()
            out = model(xb)

            if has_mtp:
                pred, mtp_preds = out
            else:
                pred = out
                mtp_preds = None

            if label_smoothing > 0:
                from .experiments.label_smoothing import smooth_targets
                yb = smooth_targets(yb, label_smoothing)

            if ohem_fraction > 0:
                from .experiments.ohem import ohem_loss
                per_sample = loss_fn_per_sample(pred, yb).squeeze(-1)
                loss = ohem_loss(per_sample, ohem_fraction)
            else:
                loss = loss_fn(pred, yb)

            if mtp_preds is not None and model_cfg.mtp_horizons:
                for h, aux_pred in mtp_preds.items():
                    aux_target = aux_targets[h].to(device)
                    loss += mtp_weight * loss_fn(aux_pred, aux_target)

            loss = loss + output_regularization(pred, output_reg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

            if crowfeather:
                from .experiments.crowfeather import apply_crowfeather
                apply_crowfeather(opt, warmup_steps, global_step)

            if ema is not None:
                ema.update(model)

            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            global_step += 1
        train_loss /= max(1, n)

        if ema is not None:
            saved = ema.swap(model)

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                if has_mtp:
                    xb, yb, _ = batch
                else:
                    xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                if has_mtp:
                    pred, _ = out
                else:
                    pred = out
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
                nv += xb.size(0)
        val_loss /= max(1, nv)

        if ema is not None:
            model.load_state_dict(saved)

        dt = time.time() - t0
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "sec": dt})
        print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  ({dt:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            stalled = 0
            save_state = model.state_dict()
            if ema is not None:
                save_state = ema.state_dict()
            torch.save({"model": save_state, "cfg": model_cfg.__dict__}, os.path.join(run_dir, "model.pt"))
        else:
            stalled += 1
            if stalled >= train_cfg.patience:
                print(f"[train] early stop after {epoch+1} epochs (val no improvement for {train_cfg.patience})")
                break

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model, history
