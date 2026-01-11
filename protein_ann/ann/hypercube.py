from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List

import numpy as np

from .base import AnnIndex, AnnQueryResult
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest


@dataclass
class HypercubeParams:
    kproj: int = 14
    w: float = 4.0
    M: int = 10
    probes: int = 2
    seed: int = 1


class HypercubeIndex(AnnIndex):
    """Hypercube projection index (random projection + binary cube buckets)."""

    def __init__(self, params: HypercubeParams) -> None:
        self.params = params
        self._rng = np.random.default_rng(params.seed)

        self._X: np.ndarray | None = None
        self._A: np.ndarray | None = None  # (kproj, d)
        self._B: np.ndarray | None = None  # (kproj,)
        self._f_maps: list[Dict[int, int]] | None = None  # per-projection hash->bit
        self._cube: Dict[int, np.ndarray] | None = None  # vertex -> indices

    def build(self, vectors: np.ndarray) -> None:
        X = ensure_float32_2d(vectors)
        n, d = X.shape
        p = self.params
        if p.kproj <= 0:
            raise ValueError("kproj must be positive")
        if p.w <= 0:
            raise ValueError("w must be positive")
        if p.M <= 0:
            raise ValueError("M must be positive")
        if p.probes <= 0:
            raise ValueError("probes must be positive")

        A = self._rng.standard_normal(size=(p.kproj, d), dtype=np.float32)
        B = self._rng.random(size=(p.kproj,), dtype=np.float32) * float(p.w)
        H = np.floor((X @ A.T + B[None, :]) / float(p.w)).astype(np.int64)  # (n, kproj)

        f_maps: list[Dict[int, int]] = [dict() for _ in range(p.kproj)]
        cube_lists: Dict[int, List[int]] = {}

        for i in range(n):
            vertex = 0
            for j in range(p.kproj):
                hv = int(H[i, j])
                mp = f_maps[j]
                bit = mp.get(hv)
                if bit is None:
                    bit = int(self._rng.integers(0, 2))
                    mp[hv] = bit
                vertex = (vertex << 1) | bit
            cube_lists.setdefault(vertex, []).append(i)

        cube = {v: np.asarray(idxs, dtype=np.int32) for v, idxs in cube_lists.items()}

        self._X = X
        self._A = A
        self._B = B
        self._f_maps = f_maps
        self._cube = cube

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:
        if self._X is None or self._A is None or self._B is None or self._f_maps is None or self._cube is None:
            raise RuntimeError("Index is not built")
        qv = ensure_float32_1d(q)
        p = self.params

        h = np.floor((self._A @ qv + self._B) / float(p.w)).astype(np.int64)  # (kproj,)
        vertex = 0
        for j in range(p.kproj):
            hv = int(h[j])
            mp = self._f_maps[j]
            bit = mp.get(hv)
            if bit is None:
                bit = int(self._rng.integers(0, 2))
                mp[hv] = bit
            vertex = (vertex << 1) | bit

        probes = _probe_sequence(vertex, p.kproj, p.probes)
        cand: list[int] = []
        for v in probes:
            arr = self._cube.get(v)
            if arr is None:
                continue
            for idx in arr.tolist():
                cand.append(idx)
                if len(cand) >= p.M:
                    break
            if len(cand) >= p.M:
                break

        if not cand:
            return AnnQueryResult(
                indices=np.empty((0,), dtype=np.int32),
                distances=np.empty((0,), dtype=np.float32),
            )

        cand_idx = np.asarray(cand, dtype=np.int32)
        dist_sq = l2_sq_to_many(qv, self._X[cand_idx])
        local_idx, dists = topk_smallest(dist_sq, k)
        return AnnQueryResult(indices=cand_idx[local_idx], distances=dists)


def _probe_sequence(vertex: int, kproj: int, max_probes: int) -> list[int]:
    """Vertices to probe in increasing Hamming distance, capped to max_probes."""
    seq: list[int] = [vertex]
    if max_probes <= 1:
        return seq

    # dist = 1..kproj
    for dist in range(1, kproj + 1):
        for bits in combinations(range(kproj), dist):
            v = vertex
            for b in bits:
                v ^= 1 << b
            seq.append(v)
            if len(seq) >= max_probes:
                return seq
    return seq

