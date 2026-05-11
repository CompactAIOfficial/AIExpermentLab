import numpy as np


class ValueBucketTokenizer:
    def __init__(self, n_buckets: int = 256):
        self.n_buckets = n_buckets
        self.edges = None

    def fit(self, x: np.ndarray):
        self.edges = np.quantile(x, np.linspace(0, 1, self.n_buckets + 1))
        self.edges[0] = -np.inf
        self.edges[-1] = np.inf
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges, x) - 1, 0, self.n_buckets - 1)

    def decode(self, ids: np.ndarray) -> np.ndarray:
        centers = (self.edges[:-1] + self.edges[1:]) / 2
        centers[0] = self.edges[1]
        centers[-1] = self.edges[-2]
        return centers[ids]
