from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import AnnIndex, AnnQueryResult
from .kmeans import KMeansParams, kmeans_fit, predict_labels
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest


@dataclass
class IvfFlatParams:
    kclusters: int = 50
    nprobe: int = 5
    max_iter: int = 30
    seed: int = 1
    train_size: int | None = None  # if set, sample this many points to train KMeans


class IvfFlatIndex(AnnIndex):
    """IVF-Flat index: coarse KMeans clusters + exact distances within probed lists."""

    def __init__(self, params: IvfFlatParams) -> None:
        self.params = params
        self._rng = np.random.default_rng(params.seed)
        self._X: np.ndarray | None = None
        self._centroids: np.ndarray | None = None  # (kclusters, d)
        self._lists: list[np.ndarray] | None = None  # inverted lists (arrays of indices)

    def build(self, vectors: np.ndarray) -> None:
        X = ensure_float32_2d(vectors)
        n, _d = X.shape
        p = self.params
        if p.kclusters <= 0:
            raise ValueError("kclusters must be positive")
        if p.nprobe <= 0:
            raise ValueError("nprobe must be positive")
        if p.kclusters > n:
            raise ValueError("kclusters cannot exceed number of points")

        train = X
        if p.train_size is not None and p.train_size < n:
            idx = self._rng.choice(n, size=int(p.train_size), replace=False)
            train = X[idx]

        km = kmeans_fit(train, KMeansParams(k=p.kclusters, max_iter=p.max_iter, seed=p.seed))
        centroids = km.centroids
        labels = predict_labels(X, centroids)

        lists: list[list[int]] = [[] for _ in range(p.kclusters)]
        for i, lab in enumerate(labels.tolist()):
            lists[int(lab)].append(i)
        lists_np = [np.asarray(lst, dtype=np.int32) for lst in lists]

        self._X = X
        self._centroids = centroids
        self._lists = lists_np

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:
        if self._X is None or self._centroids is None or self._lists is None:
            raise RuntimeError("Index is not built")
        qv = ensure_float32_1d(q)
        p = self.params

        # Find nearest clusters
        dist_c = l2_sq_to_many(qv, self._centroids)
        clusters_idx, _ = topk_smallest(dist_c, min(p.nprobe, p.kclusters))
        cand: list[int] = []
        for c in clusters_idx.tolist():
            cand.extend(self._lists[int(c)].tolist())
        if not cand:
            return AnnQueryResult(
                indices=np.empty((0,), dtype=np.int32),
                distances=np.empty((0,), dtype=np.float32),
            )
        cand_idx = np.asarray(cand, dtype=np.int32)
        dist_sq = l2_sq_to_many(qv, self._X[cand_idx])
        local_idx, dists = topk_smallest(dist_sq, k)
        return AnnQueryResult(indices=cand_idx[local_idx], distances=dists)

