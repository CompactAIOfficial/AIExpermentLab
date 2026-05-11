import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from ..config import DataConfig


CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} {start}..{end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.to_csv(cache_path)
    return df


class WindowedSeries(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, horizon: int = 1):
        self.x = series.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return max(0, len(self.x) - self.seq_len - self.horizon + 1)

    def __getitem__(self, i):
        window = self.x[i : i + self.seq_len]
        target = self.x[i + self.seq_len + self.horizon - 1]
        return (
            torch.from_numpy(window).unsqueeze(-1),
            torch.tensor([target], dtype=torch.float32),
        )


class Normalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, x: np.ndarray):
        self.mean = float(x.mean())
        self.std = float(x.std() + 1e-8)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        n = cls()
        n.mean = float(d["mean"])
        n.std = float(d["std"])
        return n


def _load_series(ticker: str, start: str, end: str, feature: str) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    df = fetch_prices(ticker, start, end)
    return df[feature].values.astype(np.float32), df.index


def build_datasets(cfg: DataConfig):
    feat = cfg.features[0]
    train_sets: List[Dataset] = []
    test_sets: Dict[str, WindowedSeries] = {}
    test_dates: Dict[str, pd.DatetimeIndex] = {}
    norms: Dict[str, Normalizer] = {}

    for ticker in cfg.tickers:
        train_raw, _ = _load_series(ticker, cfg.train_start, cfg.train_end, feat)
        test_raw, test_idx = _load_series(ticker, cfg.test_start, cfg.test_end, feat)

        norm = Normalizer().fit(train_raw)
        norms[ticker] = norm

        train_sets.append(WindowedSeries(norm.transform(train_raw), cfg.seq_len, cfg.horizon))
        test_sets[ticker] = WindowedSeries(norm.transform(test_raw), cfg.seq_len, cfg.horizon)
        test_dates[ticker] = test_idx

    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    return train_ds, test_sets, norms, test_dates


def build_blind_test(cfg: DataConfig):
    feat = cfg.features[0]
    seeds: Dict[str, np.ndarray] = {}
    actuals: Dict[str, np.ndarray] = {}
    dates: Dict[str, pd.DatetimeIndex] = {}
    norms: Dict[str, Normalizer] = {}

    for ticker in cfg.tickers:
        train_raw, _ = _load_series(ticker, cfg.train_start, cfg.train_end, feat)
        test_raw, test_idx = _load_series(ticker, cfg.test_start, cfg.test_end, feat)

        if len(train_raw) < cfg.seq_len:
            raise RuntimeError(
                f"{ticker}: training window only has {len(train_raw)} rows, "
                f"need at least seq_len={cfg.seq_len} for the seed."
            )

        norm = Normalizer().fit(train_raw)
        norms[ticker] = norm

        seeds[ticker] = norm.transform(train_raw)[-cfg.seq_len:]
        actuals[ticker] = norm.transform(test_raw)
        dates[ticker] = test_idx

    return seeds, actuals, norms, dates
