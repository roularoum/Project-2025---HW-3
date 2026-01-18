from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .base import AnnIndex, AnnQueryResult
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest


@dataclass
class LshParams:
    k: int = 4
    L: int = 5
    w: float = 4.0
    seed: int = 1


class LshIndex(AnnIndex):
    """Euclidean LSH (as in Assignment 1/2 codebase), adapted to embeddings."""

    def __init__(self, params: LshParams) -> None:
        self.params = params
        self._rng = np.random.default_rng(params.seed)

        self._X: np.ndarray | None = None
        self._A: np.ndarray | None = None  # (L, k, d)
        self._B: np.ndarray | None = None  # (L, k)
        self._tables: list[Dict[int, np.ndarray]] | None = None

    def build(self, vectors: np.ndarray) -> None:
        X = ensure_float32_2d(vectors)
        n, d = X.shape

        p = self.params
        if p.k <= 0 or p.L <= 0:
            raise ValueError("k and L must be positive")
        if p.w <= 0:
            raise ValueError("w must be positive")

        A = self._rng.standard_normal(size=(p.L, p.k, d), dtype=np.float32)
        B = self._rng.random(size=(p.L, p.k), dtype=np.float32) * float(p.w)

        # Precompute hash keys for each table.
        prime = np.int64(4294967291)
        tables: list[Dict[int, list[int]]] = []
        for l in range(p.L):
            proj = X @ A[l].T  # (n, k)
            h = np.floor((proj + B[l]) / float(p.w)).astype(np.int64)
            key = np.zeros((n,), dtype=np.int64)
            for i in range(p.k):
                key = (key * prime + h[:, i]) % prime

            bucket: Dict[int, list[int]] = {}
            for idx, kv in enumerate(key.tolist()):
                bucket.setdefault(int(kv), []).append(idx)
            tables.append(bucket)

        # Convert lists to compact numpy arrays.
        tables_np: list[Dict[int, np.ndarray]] = []
        for bucket in tables:
            tables_np.append({k: np.asarray(v, dtype=np.int32) for k, v in bucket.items()})

        self._X = X
        self._A = A
        self._B = B
        self._tables = tables_np

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:
        if self._X is None or self._A is None or self._B is None or self._tables is None:
            raise RuntimeError("Index is not built")
        qv = ensure_float32_1d(q)
        X = self._X
        p = self.params

        # Use Python ints here to avoid NumPy int64 overflow warnings.
        prime = 4294967291
        cand: list[int] = []
        seen = set()
        for l in range(p.L):
            proj = self._A[l] @ qv  # (k,)
            h = np.floor((proj + self._B[l]) / float(p.w)).astype(np.int64)
            key = 0
            for i in range(p.k):
                key = (key * prime + int(h[i])) % prime
            bucket = self._tables[l].get(key)
            if bucket is None:
                continue
            # Deduplicate candidates across tables.
            for idx in bucket.tolist():
                if idx not in seen:
                    seen.add(idx)
                    cand.append(idx)

        if not cand:
            return AnnQueryResult(
                indices=np.empty((0,), dtype=np.int32),
                distances=np.empty((0,), dtype=np.float32),
            )

        cand_idx = np.asarray(cand, dtype=np.int32)
        dist_sq = l2_sq_to_many(qv, X[cand_idx])
        local_idx, dists = topk_smallest(dist_sq, k)
        # Map back to original indices
        return AnnQueryResult(indices=cand_idx[local_idx], distances=dists)

