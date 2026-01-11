from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .utils import ensure_float32_2d


@dataclass
class KMeansParams:
    k: int
    max_iter: int = 30
    tol: float = 1e-4
    seed: int = 1


@dataclass(frozen=True)
class KMeansResult:
    centroids: np.ndarray  # (k, d), float32
    labels: np.ndarray  # (n,), int32


def kmeans_fit(X: np.ndarray, params: KMeansParams) -> KMeansResult:
    X = ensure_float32_2d(X)
    n, d = X.shape
    if n == 0:
        raise ValueError("Empty dataset")
    if params.k <= 0:
        raise ValueError("k must be positive")
    if params.k > n:
        raise ValueError("k cannot exceed number of points")

    rng = np.random.default_rng(params.seed)
    C = _init_kmeans_pp(X, params.k, rng)

    last_shift = np.inf
    labels = np.zeros((n,), dtype=np.int32)
    for _ in range(params.max_iter):
        oldC = C.copy()

        labels = predict_labels(X, C)
        C = _recompute_centroids(X, labels, params.k, rng, oldC)

        shift = float(np.max(np.linalg.norm(C - oldC, axis=1)))
        last_shift = shift
        if shift <= params.tol:
            break

    return KMeansResult(centroids=C, labels=labels)


def predict_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each row of X to the nearest centroid (L2)."""
    X = ensure_float32_2d(X)
    C = ensure_float32_2d(centroids)
    # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
    x_norm = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    c_norm = np.sum(C * C, axis=1)[None, :]  # (1,k)
    dot = X @ C.T  # (n,k)
    dist_sq = x_norm + c_norm - 2.0 * dot
    return np.argmin(dist_sq, axis=1).astype(np.int32, copy=False)


def _init_kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    C = np.empty((k, d), dtype=np.float32)
    idx0 = int(rng.integers(0, n))
    C[0] = X[idx0]

    # Distance to nearest centroid for each point.
    dist_sq = np.sum((X - C[0][None, :]) ** 2, axis=1)
    for i in range(1, k):
        total = float(dist_sq.sum())
        if total <= 0.0:
            # All points identical; pick random.
            idx = int(rng.integers(0, n))
            C[i] = X[idx]
            continue
        probs = dist_sq / total
        idx = int(rng.choice(n, p=probs))
        C[i] = X[idx]
        new_dist = np.sum((X - C[i][None, :]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist)
    return C


def _recompute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    rng: np.random.Generator,
    fallback: np.ndarray,
) -> np.ndarray:
    n, d = X.shape
    sums = np.zeros((k, d), dtype=np.float32)
    counts = np.bincount(labels, minlength=k).astype(np.int32)
    np.add.at(sums, labels, X)

    C = fallback.copy()
    for j in range(k):
        if counts[j] > 0:
            C[j] = sums[j] / float(counts[j])
        else:
            # Reinitialize empty cluster to a random point.
            idx = int(rng.integers(0, n))
            C[j] = X[idx]
    return C

