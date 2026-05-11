import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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


def build_datasets(cfg: DataConfig):
    train_df = fetch_prices(cfg.ticker, cfg.train_start, cfg.train_end)
    test_df = fetch_prices(cfg.ticker, cfg.test_start, cfg.test_end)

    feat = cfg.features[0]
    train_raw = train_df[feat].values.astype(np.float32)
    test_raw = test_df[feat].values.astype(np.float32)

    norm = Normalizer().fit(train_raw)
    train_n = norm.transform(train_raw)
    test_n = norm.transform(test_raw)

    train_ds = WindowedSeries(train_n, cfg.seq_len, cfg.horizon)
    test_ds = WindowedSeries(test_n, cfg.seq_len, cfg.horizon)
    return train_ds, test_ds, norm
