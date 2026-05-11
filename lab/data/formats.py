import numpy as np


def to_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices.astype(np.float64))).astype(np.float32)


def from_returns(returns: np.ndarray, p0: float) -> np.ndarray:
    return p0 * np.exp(np.cumsum(returns.astype(np.float64))).astype(np.float32)


def sliding_windows(arr: np.ndarray, win: int, step: int = 1) -> np.ndarray:
    n = (len(arr) - win) // step + 1
    out = np.zeros((n, win), dtype=arr.dtype)
    for i in range(n):
        out[i] = arr[i * step : i * step + win]
    return out
