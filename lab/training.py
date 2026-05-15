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
    muon_lr: float = 0.0, fim_rate: float = 0.0,
    lora_rank: int = 0, lora_alpha: float = 1.0,
    latent_steps: int = 0, ssm_decay: float = 0.0,
    curriculum_epochs: int = 0,
    ple: bool = False, anti_pattern_weight: float = 0.0,
    nce_weight: float = 0.0,
):
    os.makedirs(run_dir, exist_ok=True)
    set_seed(train_cfg.seed)

    device = torch.device(train_cfg.device)
    model = TinyForecaster(model_cfg).to(device)

    if lora_rank > 0:
        from .experiments.depth_lora import apply_depth_lora
        model = apply_depth_lora(model, lora_rank, lora_alpha)
        model = model.to(device)

    if latent_steps > 0:
        model_cfg.latent_steps = latent_steps
        from .experiments.latent_reasoning import LatentReasoning
        model.latent = LatentReasoning(
            model.blocks, model_cfg.d_model,
            latent_steps=latent_steps, think_penalty=0.0,
        ).to(device)

    if ssm_decay > 0:
        model_cfg.ssm_decay = ssm_decay
        from .experiments.ssm_injection import SSMInjection
        model.ssm = SSMInjection(model_cfg.d_model, model_cfg.n_layers, decay_init=ssm_decay).to(device)

    if ple:
        model_cfg.ple = True
        from .experiments.ple import PerLayerEmbeddings
        model.ple = PerLayerEmbeddings(model_cfg.d_model, model_cfg.n_layers).to(device)

    from .experiments.input_dropout import apply_input_dropout
    from .experiments.output_reg import output_regularization

    has_mtp = bool(model_cfg.mtp_horizons)

    muon = None
    if muon_lr > 0:
        from .experiments.muon import MuonState
        muon = MuonState(model, muon_lr=muon_lr, adamw_lr=train_cfg.lr,
                         weight_decay=train_cfg.weight_decay)
        opt = muon
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                weight_decay=train_cfg.weight_decay)

    loss_fn = nn.MSELoss()
    loss_fn_per_sample = nn.MSELoss(reduction="none") if ohem_fraction > 0 else None

    if crowfeather and not muon:
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
        if latent_steps > 0 and curriculum_epochs > 0 and hasattr(model, 'latent') and model.latent is not None:
            from .experiments.latent_reasoning import get_latent_steps_for_epoch
            model.latent.latent_steps = get_latent_steps_for_epoch(
                epoch, latent_steps, curriculum_epochs
            )
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

            if fim_rate > 0:
                from .experiments.fim import apply_fim
                xb = apply_fim(xb, fim_rate)

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

            if hasattr(model, '_ponder_cost') and model._ponder_cost is not None:
                loss = loss + model._ponder_cost
            if hasattr(model, '_think_cost') and model._think_cost is not None:
                loss = loss + model._think_cost
            if anti_pattern_weight > 0:
                from .experiments.anti_pattern import anti_pattern_loss
                loss = loss + anti_pattern_loss(pred, xb, anti_pattern_weight)
            if nce_weight > 0 and hasattr(model, '_last_hidden') and model._last_hidden is not None:
                from .experiments.nce_context import nce_context_loss
                loss = loss + nce_context_loss(model._last_hidden, nce_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            if muon is not None:
                muon.step()
            else:
                opt.step()

            if crowfeather and not muon:
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
