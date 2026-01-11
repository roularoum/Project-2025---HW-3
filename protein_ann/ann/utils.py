from __future__ import annotations

from typing import Tuple

import numpy as np


def ensure_float32_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array (n, d)")
    return arr


def ensure_float32_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Expected 1D array (d,)")
    return arr


def l2_sq_to_many(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Squared L2 distances from q to each row of X."""
    q = ensure_float32_1d(q)
    X = ensure_float32_2d(X)
    diff = X - q[None, :]
    return np.einsum("ij,ij->i", diff, diff).astype(np.float32, copy=False)


def topk_smallest(dist_sq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) for the k smallest distances."""
    if k <= 0:
        raise ValueError("k must be positive")
    n = int(dist_sq.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)
    k_eff = min(k, n)
    part = np.argpartition(dist_sq, k_eff - 1)[:k_eff]
    order = np.argsort(dist_sq[part])
    idx = part[order].astype(np.int32, copy=False)
    d = np.sqrt(dist_sq[idx]).astype(np.float32, copy=False)
    return idx, d

