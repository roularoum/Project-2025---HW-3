from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AnnIndex, AnnQueryResult
from .ivf_flat import IvfFlatParams
from .kmeans import KMeansParams, kmeans_fit, predict_labels
from .pq import PqParams, ProductQuantizer
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest


@dataclass
class IvfPqParams:
    kclusters: int = 50
    nprobe: int = 5
    seed: int = 1
    max_iter: int = 30
    train_size: int | None = None
    pq_M: int = 16
    pq_nbits: int = 8
    pq_train_size: int = 5000
    pq_max_iter: int = 25
    pq_encode_chunk: int = 5000


class IvfPqIndex(AnnIndex):
    """IVFPQ index: IVF coarse quantizer + PQ codes in each list (ADC)."""

    def __init__(self, params: IvfPqParams) -> None:
        self.params = params
        self._rng = np.random.default_rng(params.seed)

        self._X: np.ndarray | None = None
        self._centroids: np.ndarray | None = None  # coarse centroids (kclusters, d)
        self._lists: list[np.ndarray] | None = None  # inverted lists (arrays of indices)
        self._pq: ProductQuantizer | None = None
        self._codes: np.ndarray | None = None  # (n, M) uint8/uint16

    def build(self, vectors: np.ndarray) -> None:
        X = ensure_float32_2d(vectors)
        n, d = X.shape
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

        pq = ProductQuantizer(
            PqParams(
                M=p.pq_M,
                nbits=p.pq_nbits,
                seed=p.seed,
                train_size=p.pq_train_size,
                max_iter=p.pq_max_iter,
                encode_chunk=p.pq_encode_chunk,
            )
        )
        pq.fit(train)
        codes = pq.encode(X)

        self._X = X
        self._centroids = centroids
        self._lists = lists_np
        self._pq = pq
        self._codes = codes

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:
        if self._X is None or self._centroids is None or self._lists is None or self._pq is None or self._codes is None:
            raise RuntimeError("Index is not built")
        qv = ensure_float32_1d(q)
        p = self.params

        # Coarse cluster selection
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
        codes_c = self._codes[cand_idx]
        table = self._pq.distance_table(qv)
        dist_sq = self._pq.adc_distances(table, codes_c)

        # Select by PQ distances, then re-rank the selected neighbors by TRUE L2
        # so the reported "L2 Dist" is exact in embedding space (as required).
        local_idx, _ = topk_smallest(dist_sq, k)
        chosen = cand_idx[local_idx]
        true_sq = l2_sq_to_many(qv, self._X[chosen])
        order = np.argsort(true_sq)
        chosen = chosen[order]
        dists = np.sqrt(true_sq[order]).astype(np.float32, copy=False)
        return AnnQueryResult(indices=chosen, distances=dists)

