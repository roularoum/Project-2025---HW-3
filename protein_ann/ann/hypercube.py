from __future__ import annotations  # epitrepei type hints me forward refs

from dataclasses import dataclass  # gia dataclass 
from itertools import combinations  # gia na vgalei combos bits 
from typing import Dict, List  # type hints (Dict/List)

import numpy as np  # numpy gia vectors kai arithmitikes prakseis

from .base import AnnIndex, AnnQueryResult  # koino interface gia ANN indexes + return type
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest  #types kai distances


@dataclass  # dataclass gia hypercube hyperparameters
class HypercubeParams:  # rythmiseis tou Hypercube index
    kproj: int = 14  # diastasi tou cube
    w: float = 4.0  # bucket width
    M: int = 10  # megistos arithmos candidates pou tha elegxoume sto query
    probes: int = 2  # posa vertices tha psaksoume me hamming neighbors
    seed: int = 1  # seed gia reproducibility


class HypercubeIndex(AnnIndex):  # ylopoiei AnnIndex me hypercube hashing
    """Hypercube projection index (random projection + binary cube buckets)."""  

    def __init__(self, params: HypercubeParams) -> None:  # constructor me params
        self.params = params  # apothikeuei params
        self._rng = np.random.default_rng(params.seed)  # random generator me seed

        self._X: np.ndarray | None = None  # tha kratisei ta database vectors
        self._A: np.ndarray | None = None  # matrix projections (kproj x d)
        self._B: np.ndarray | None = None  # random offsets
        self._f_maps: list[Dict[int, int]] | None = None  # gia kathe projection: map hv me bit 
        self._cube: Dict[int, np.ndarray] | None = None  # indices twn vectors pou pesan ekei

    def build(self, vectors: np.ndarray) -> None:  # xtizei to hypercube index panw sto dataset
        X = ensure_float32_2d(vectors)  # sigourevei oti einai 2D float32 array
        n, d = X.shape  # n=plithos vectors, d=diastasi
        p = self.params  # alias sta params
        if p.kproj <= 0:  # elegxos egkyrotitas
            raise ValueError("kproj must be positive")  # kproj>0
        if p.w <= 0:  # elegxos w
            raise ValueError("w must be positive")  # w>0
        if p.M <= 0:  # elegxos M
            raise ValueError("M must be positive")  # M>0
        if p.probes <= 0:  # elegxos probes
            raise ValueError("probes must be positive")  # probes>0

        A = self._rng.standard_normal(size=(p.kproj, d), dtype=np.float32)  # random projection vectors
        B = self._rng.random(size=(p.kproj,), dtype=np.float32) * float(p.w)  # random offsets
        H = np.floor((X @ A.T + B[None, :]) / float(p.w)).astype(np.int64)  # quantized hashes

        f_maps: list[Dict[int, int]] = [dict() for _ in range(p.kproj)]  # ena dict ana projection
        cube_lists: Dict[int, List[int]] = {}  # vertex -> lista indices

        for i in range(n):  # gia kathe database vector
            vertex = 0  # tha ftiaksei to vertex bits se integer
            for j in range(p.kproj):  # gia kathe projection
                hv = int(H[i, j])  # hash value 
                mp = f_maps[j]  # to dict gia ayto to projection
                bit = mp.get(hv)  # psaxnei an exei idi antistoixisi hv->bit
                if bit is None:  # an einai kainourgio hv
                    bit = int(self._rng.integers(0, 2))  # dinei tuxaia bit 0/1
                    mp[hv] = bit  # apothikeuei tin antistoixisi
                vertex = (vertex << 1) | bit  # shift kai bazei to bit gia na xtisei to vertex integer
            cube_lists.setdefault(vertex, []).append(i)  # bazei to index i sto bucket tou vertex

        cube = {v: np.asarray(idxs, dtype=np.int32) for v, idxs in cube_lists.items()}  # metatrepei listes se numpy arrays

        self._X = X  # apothikeuei dataset vectors
        self._A = A  # apothikeuei projections
        self._B = B  # apothikeuei offsets
        self._f_maps = f_maps  # apothikeuei maps hv->bit
        self._cube = cube  # apothikeuei to cube vertex->indices

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:  # kanei query kai gyrnaei top-k
        if self._X is None or self._A is None or self._B is None or self._f_maps is None or self._cube is None:  # an den exei build
            raise RuntimeError("Index is not built")  # error an den xtistike
        qv = ensure_float32_1d(q)  # sigourevei oti to query einai 1D float32
        p = self.params  # alias params

        h = np.floor((self._A @ qv + self._B) / float(p.w)).astype(np.int64)  # quantized hashes gia to query
        vertex = 0  # tha xtisei to query vertex
        for j in range(p.kproj):  # gia kathe projection
            hv = int(h[j])  # hash value gia ayto to projection
            mp = self._f_maps[j]  # map hv->bit
            bit = mp.get(hv)  # pare to bit an yparxei
            if bit is None:  # an den yparxei 
                bit = int(self._rng.integers(0, 2))  # orizei tuxaia bit 0/1
                mp[hv] = bit  # apothikeuei gia na einai consistent meta
            vertex = (vertex << 1) | bit  # xtizei to vertex integer

        probes = _probe_sequence(vertex, p.kproj, p.probes)  # lista vertices pou tha elegksoume 
        cand: list[int] = []  # lista candidate indices apo to cube
        for v in probes:  # gia kathe vertex 
            arr = self._cube.get(v)  # pare ta indices pou exei ayto to vertex
            if arr is None:  # an den exei kanena
                continue  # pigene sto epomeno vertex
            for idx in arr.tolist():  # perna kathe index mesa sto bucket
                cand.append(idx)  # prosthese candidate
                if len(cand) >= p.M:  # an ftasame ta M candidates
                    break  # stamata
            if len(cand) >= p.M:  
                break

        if not cand:  # an den vrethike kanenas candidate
            return AnnQueryResult(  # epistrefei adeia apotelesmata
                indices=np.empty((0,), dtype=np.int32),  # adeio indices array
                distances=np.empty((0,), dtype=np.float32),  # adeio distances array
            )

        cand_idx = np.asarray(cand, dtype=np.int32)  # metatrepei candidates se numpy array
        dist_sq = l2_sq_to_many(qv, self._X[cand_idx])  # ypologizei L2^2 tou query me kathe candidate vector
        local_idx, dists = topk_smallest(dist_sq, k)  # pairnei ta top-k mikrottera distances
        return AnnQueryResult(indices=cand_idx[local_idx], distances=dists)  # metatrepei se global indices kai epistrefei


def _probe_sequence(vertex: int, kproj: int, max_probes: int) -> list[int]:  # vgazei seira vertices pou tha psaksoume
    """Vertices to probe in increasing Hamming distance, capped to max_probes."""  # seira me auxousa Hamming apostasi
    seq: list[int] = [vertex]  # ksekiname apo to idio vertex
    if max_probes <= 1:  # an theloume mono 1 probe
        return seq  # gyrnaei mono to arxiko vertex

    # dist = 1..kproj  # proxwraei se Hamming distance
    for dist in range(1, kproj + 1):  # gia kathe apostasi
        for bits in combinations(range(kproj), dist):  # epilegei poia bits tha allaksoun
            v = vertex  # ksekiname apo arxiko vertex
            for b in bits:  # gia kathe bit
                v ^= 1 << b  # kanei flip to bit b
            seq.append(v)  # prosthetei to kainourgio vertex
            if len(seq) >= max_probes:  # an ftasame to max probes
                return seq  # stamata kai epistrefei
    return seq  # an den ftasei epistrefei oti exei
