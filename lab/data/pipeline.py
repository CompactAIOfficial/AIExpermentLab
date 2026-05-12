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


class LocallyNormalizedSeries(Dataset):
    """Each window is independently z-scored so the model learns *shapes*, not levels.

    Under uncertainty the model defaults to predicting the local window mean
    (= recent average price) instead of some distant training mean.  This
    eliminates autoregressive drift while preserving the model's ability to
    learn trends and patterns.
    """

    def __init__(self, prices: np.ndarray, seq_len: int, horizon: int = 1,
                 aux_horizons: list[int] | None = None):
        self.prices = prices.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.aux_horizons = aux_horizons or []
        self._max_h = max([horizon] + self.aux_horizons)

    def __len__(self):
        return max(0, len(self.prices) - self.seq_len - self._max_h + 1)

    def _norm_target(self, target_idx: int, w_mean: float, w_std: float):
        return torch.tensor([(self.prices[target_idx] - w_mean) / w_std], dtype=torch.float32)

    def __getitem__(self, i):
        window = self.prices[i : i + self.seq_len]
        w_mean = window.mean()
        w_std = window.std() + 1e-8
        x = torch.from_numpy(((window - w_mean) / w_std)).unsqueeze(-1)
        y_main = self._norm_target(i + self.seq_len + self.horizon - 1, w_mean, w_std)
        if self.aux_horizons:
            aux = {}
            for h in self.aux_horizons:
                aux[h] = self._norm_target(i + self.seq_len + h - 1, w_mean, w_std)
            return x, y_main, aux
        return x, y_main


def _load_raw(ticker: str, start: str, end: str, feature: str) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    df = fetch_prices(ticker, start, end)
    return df[feature].values.astype(np.float32), df.index


def build_datasets(cfg: DataConfig):
    feat = cfg.features[0]
    aux = cfg.mtp_horizons or None
    train_sets: List[Dataset] = []
    test_sets: Dict[str, LocallyNormalizedSeries] = {}
    test_dates: Dict[str, pd.DatetimeIndex] = {}

    for ticker in cfg.tickers:
        train_raw, _ = _load_raw(ticker, cfg.train_start, cfg.train_end, feat)
        test_raw, test_idx = _load_raw(ticker, cfg.test_start, cfg.test_end, feat)

        train_sets.append(LocallyNormalizedSeries(train_raw, cfg.seq_len, cfg.horizon, aux))
        test_sets[ticker] = LocallyNormalizedSeries(test_raw, cfg.seq_len, cfg.horizon, aux)
        test_dates[ticker] = test_idx

    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    return train_ds, test_sets, test_dates


def build_blind_test(cfg: DataConfig):
    """Return raw seed prices + raw test prices.  Benchmark normalizes on the fly."""
    feat = cfg.features[0]
    seeds: Dict[str, np.ndarray] = {}
    actuals: Dict[str, np.ndarray] = {}
    dates: Dict[str, pd.DatetimeIndex] = {}

    for ticker in cfg.tickers:
        train_raw, _ = _load_raw(ticker, cfg.train_start, cfg.train_end, feat)
        test_raw, test_idx = _load_raw(ticker, cfg.test_start, cfg.test_end, feat)

        if len(train_raw) < cfg.seq_len:
            raise RuntimeError(
                f"{ticker}: training window only has {len(train_raw)} rows, "
                f"need at least seq_len={cfg.seq_len} for the seed."
            )

        seeds[ticker] = train_raw[-cfg.seq_len:]
        actuals[ticker] = test_raw
        dates[ticker] = test_idx

    return seeds, actuals, dates
