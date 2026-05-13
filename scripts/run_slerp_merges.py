import os
import shutil
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab.experiments.slerp_merge import merge_state_dicts


def merge(out_dir: str, src_dirs: list[str], t: float = 0.5):
    os.makedirs(out_dir, exist_ok=True)
    state_dicts = []
    for d in src_dirs:
        ckpt = torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=False)
        state_dicts.append(ckpt["model"])
    merged_sd = merge_state_dicts(state_dicts, t)
    template = torch.load(os.path.join(src_dirs[0], "model.pt"), map_location="cpu", weights_only=False)
    template["model"] = merged_sd
    torch.save(template, os.path.join(out_dir, "model.pt"))
    shutil.copy(os.path.join(src_dirs[0], "data_cfg.json"), os.path.join(out_dir, "data_cfg.json"))
    print(f"saved {out_dir} (merged {len(src_dirs)} ckpts at t={t})")


if __name__ == "__main__":
    s42 = "runs/Glint_AAPL_NVDA"
    s43 = "runs/Glint_AAPL_NVDA_seed43"
    s44 = "runs/Glint_AAPL_NVDA_seed44"

    merge("runs/Glint_AAPL_NVDA_slerp=42-43_t0.5", [s42, s43], t=0.5)
    merge("runs/Glint_AAPL_NVDA_slerp=42-44_t0.5", [s42, s44], t=0.5)
    merge("runs/Glint_AAPL_NVDA_slerp=43-44_t0.5", [s43, s44], t=0.5)
    merge("runs/Glint_AAPL_NVDA_slerp=3way_t0.5", [s42, s43, s44], t=0.5)
    merge("runs/Glint_AAPL_NVDA_slerp=42-43_t0.25", [s42, s43], t=0.25)
    merge("runs/Glint_AAPL_NVDA_slerp=42-43_t0.75", [s42, s43], t=0.75)
