from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .kmeans import KMeansParams, kmeans_fit
from .utils import ensure_float32_1d, ensure_float32_2d


@dataclass
class PqParams:
    M: int = 16
    nbits: int = 8
    seed: int = 1
    train_size: int = 5000
    max_iter: int = 25
    encode_chunk: int = 5000


class ProductQuantizer:
    """Product Quantizer with Asymmetric Distance Computation (ADC)."""

    def __init__(self, params: PqParams) -> None:
        self.params = params
        self._rng = np.random.default_rng(params.seed)
        self._dsub: int | None = None
        self._centroids: list[np.ndarray] | None = None  # list of (k, dsub)

    @property
    def dsub(self) -> int:
        if self._dsub is None:
            raise RuntimeError("PQ is not fitted")
        return self._dsub

    @property
    def k(self) -> int:
        return 1 << int(self.params.nbits)

    def fit(self, X: np.ndarray) -> None:
        X = ensure_float32_2d(X)
        n, d = X.shape
        p = self.params
        if p.M <= 0:
            raise ValueError("M must be positive")
        if d % p.M != 0:
            raise ValueError(f"Vector dimension {d} must be divisible by M={p.M}")
        if p.nbits <= 0 or p.nbits > 16:
            raise ValueError("nbits must be in [1,16]")

        dsub = d // p.M
        self._dsub = dsub

        k = 1 << int(p.nbits)
        if k > n:
            raise ValueError(f"PQ k=2^nbits={k} cannot exceed training points {n}")

        # Train on a subset for speed.
        train_n = min(int(p.train_size), n)
        if train_n < n:
            idx = self._rng.choice(n, size=train_n, replace=False)
            train = X[idx]
        else:
            train = X

        cents: list[np.ndarray] = []
        for m in range(p.M):
            sub = train[:, m * dsub : (m + 1) * dsub]
            km = kmeans_fit(
                sub,
                KMeansParams(k=k, max_iter=p.max_iter, seed=p.seed + m),
            )
            cents.append(km.centroids.astype(np.float32, copy=False))

        self._centroids = cents

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode vectors to codes array of shape (n, M)."""
        if self._centroids is None or self._dsub is None:
            raise RuntimeError("PQ is not fitted")

        X = ensure_float32_2d(X)
        n, d = X.shape
        p = self.params
        dsub = self._dsub
        if d != dsub * p.M:
            raise ValueError("Unexpected dimension for PQ encoding")

        k = self.k
        code_dtype = np.uint8 if k <= 256 else np.uint16
        codes = np.empty((n, p.M), dtype=code_dtype)

        chunk = max(1, int(p.encode_chunk))
        for m in range(p.M):
            C = self._centroids[m]  # (k, dsub)
            Xm = X[:, m * dsub : (m + 1) * dsub]
            # Chunked assignment to limit peak memory.
            for i0 in range(0, n, chunk):
                i1 = min(n, i0 + chunk)
                block = Xm[i0:i1]
                # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
                x_norm = np.sum(block * block, axis=1, keepdims=True)
                c_norm = np.sum(C * C, axis=1)[None, :]
                dot = block @ C.T
                dist_sq = x_norm + c_norm - 2.0 * dot
                codes[i0:i1, m] = np.argmin(dist_sq, axis=1).astype(code_dtype, copy=False)

        return codes

    def distance_table(self, q: np.ndarray) -> np.ndarray:
        """Compute per-subspace squared-distance table for a query.

        Returns shape (M, k) float32.
        """
        if self._centroids is None or self._dsub is None:
            raise RuntimeError("PQ is not fitted")

        qv = ensure_float32_1d(q)
        p = self.params
        dsub = self._dsub
        if qv.shape[0] != dsub * p.M:
            raise ValueError("Unexpected query dimension for PQ")

        table = np.empty((p.M, self.k), dtype=np.float32)
        for m in range(p.M):
            C = self._centroids[m]
            qs = qv[m * dsub : (m + 1) * dsub]
            diff = C - qs[None, :]
            table[m] = np.sum(diff * diff, axis=1)
        return table

    def adc_distances(self, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Asymmetric distances (squared) for many encoded vectors.

        table: (M,k) from distance_table
        codes: (n,M)
        returns: (n,) float32 squared distances
        """
        p = self.params
        if table.shape[0] != p.M:
            raise ValueError("Invalid table shape")
        if codes.shape[1] != p.M:
            raise ValueError("Invalid codes shape")

        dist = np.zeros((codes.shape[0],), dtype=np.float32)
        for m in range(p.M):
            dist += table[m, codes[:, m].astype(np.int64)]
        return dist

