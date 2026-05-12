from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    input_dim: int = 1
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    seq_len: int = 32
    dropout: float = 0.1
    output_dim: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 20
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 42
    val_frac: float = 0.1


@dataclass
class DataConfig:
    tickers: List[str] = field(default_factory=lambda: ["AAPL"])
    train_start: str = "2020-01-01"
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2026-01-01"
    seq_len: int = 32
    horizon: int = 1
    features: List[str] = field(default_factory=lambda: ["Close"])
    diff_mode: bool = False


SERIES = {
    "Glint": ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128),
    "Shard": ModelConfig(d_model=128, n_heads=4, n_layers=4, d_ff=256),
    "Prism": ModelConfig(d_model=256, n_heads=8, n_layers=6, d_ff=512),
}
