import os
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions(
    dates: Sequence,
    actual: np.ndarray,
    predicted: np.ndarray,
    ticker: str,
    out_path: str,
    title_suffix: str = "",
):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual, label="Actual", linewidth=1.6, color="#1f77b4")
    ax.plot(dates, predicted, label="Predicted", linewidth=1.2, color="#ff7f0e", alpha=0.85)
    ax.fill_between(dates, actual, predicted, color="grey", alpha=0.15, label="Error")

    ax.set_title(f"{ticker} — Predicted vs Actual close{title_suffix}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_loss_curve(history, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train"] for h in history], label="train", color="#1f77b4")
    ax.plot(epochs, [h["val"] for h in history], label="val", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized)")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
