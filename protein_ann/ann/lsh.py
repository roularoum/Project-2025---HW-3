from __future__ import annotations  # epitrepei type hints me forward refs

from dataclasses import dataclass  # gia dataclass
from typing import Dict, List  # type hints

import numpy as np  # numpy gia arrays kai arithmitika

from .base import AnnIndex, AnnQueryResult  # koino interface index kai return type
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest  # types, distances, top-k


@dataclass  # dataclass gia LSH hyperparameters
class LshParams:  # rythmiseis LSH
    k: int = 4  # hash functions per table
    L: int = 5  # posa hash tables
    w: float = 4.0  # bucket width
    seed: int = 1  # seed gia reproducibility


class LshIndex(AnnIndex):  # LSH index gia L2
    """Euclidean LSH (as in Assignment 1/2 codebase), adapted to embeddings."""  # LSH gia embeddings se Euclidean xwro

    def __init__(self, params: LshParams) -> None:  # constructor
        self.params = params  # apothikeuei params
        self._rng = np.random.default_rng(params.seed)  # RNG me seed

        self._X: np.ndarray | None = None  # tha kratisei ta database vectors
        self._A: np.ndarray | None = None  # random projection matrices
        self._B: np.ndarray | None = None  # random offsets 
        self._tables: list[Dict[int, np.ndarray]] | None = None  # kathe table: key -> array indices

    def build(self, vectors: np.ndarray) -> None:  # xtizei LSH tables gia olo to dataset
        X = ensure_float32_2d(vectors)  # 2D float32 array
        n, d = X.shape  # n=plithos points, d=diastasi

        p = self.params  # alias params
        if p.k <= 0 or p.L <= 0:  # elegxos k kai L
            raise ValueError("k and L must be positive")  # prepei > 0
        if p.w <= 0:  # elegxos w
            raise ValueError("w must be positive")  # prepei > 0

        A = self._rng.standard_normal(size=(p.L, p.k, d), dtype=np.float32)  # random projections gia kathe table
        B = self._rng.random(size=(p.L, p.k), dtype=np.float32) * float(p.w)  # random offsets sto

        # Precompute hash keys for each table.  # ypologizei ta hash keys gia kathe table kai kathe point
        prime = np.int64(4294967291)  # megalos prime gia na kanei combine ta k hashes se ena key
        tables: list[Dict[int, list[int]]] = []  # kathe table einai dict key->lista indices
        for l in range(p.L):  # gia kathe table
            proj = X @ A[l].T  # projections
            h = np.floor((proj + B[l]) / float(p.w)).astype(np.int64)  # quantized hashes 
            key = np.zeros((n,), dtype=np.int64)  # tha kratisei to teliko key ana point
            for i in range(p.k):  # combine ta k hashes se ena integer key
                key = (key * prime + h[:, i]) % prime  # rolling hash mod prime

            bucket: Dict[int, list[int]] = {}  # key -> lista indices pou pesane sto idio bucket
            for idx, kv in enumerate(key.tolist()):  # gia kathe point index kai key value
                bucket.setdefault(int(kv), []).append(idx)  # vale to idx sto bucket tou key
            tables.append(bucket)  # prosthese to bucket dict san ena table

        # metatrepei tis listes indices se numpy arrays gia pio grigoro compact
        tables_np: list[Dict[int, np.ndarray]] = []  # key -> np.array indices
        for bucket in tables:  # gia kathe table dict
            tables_np.append({k: np.asarray(v, dtype=np.int32) for k, v in bucket.items()})  # list->np.array

        self._X = X  # apothikeuei dataset
        self._A = A  # apothikeuei projections
        self._B = B  # apothikeuei offsets
        self._tables = tables_np  # apothikeuei tables

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:  # vriskei candidates apo L tables kai meta exact top-k
        if self._X is None or self._A is None or self._B is None or self._tables is None:  # an den exei build
            raise RuntimeError("Index is not built")  # error
        qv = ensure_float32_1d(q)  # 1D float32 query vector
        X = self._X  # alias dataset vectors
        p = self.params  # alias params
        prime = 4294967291  # idio prime san python int
        cand: list[int] = []  # candidates indices apo ola ta tables
        seen = set()  # set gia na min diplo-mpei to idio candidate
        for l in range(p.L):  # gia kathe table
            proj = self._A[l] @ qv  # projections (k,)
            h = np.floor((proj + self._B[l]) / float(p.w)).astype(np.int64)  # quantized hashes (k,)
            key = 0  # teliko key gia ayto to table
            for i in range(p.k):  # combine k hashes se ena key
                key = (key * prime + int(h[i])) % prime  # rolling hash mod prime
            bucket = self._tables[l].get(key)  # pare to bucket pou antistoixei sto key
            if bucket is None:  # an den yparxei bucket
                continue  # pame sto epomeno table
            # afairei diplo-andidates apo diaforetika tables
            for idx in bucket.tolist():  # loop sta indices tou bucket
                if idx not in seen:  # an den to exoume vrei idi
                    seen.add(idx)  # mark as seen
                    cand.append(idx)  # prosthese stous candidates

        if not cand:  # an den vrike tipota
            return AnnQueryResult(  # epistrefei adeio result
                indices=np.empty((0,), dtype=np.int32),  # adeio indices array
                distances=np.empty((0,), dtype=np.float32),  # adeio distances array
            )

        cand_idx = np.asarray(cand, dtype=np.int32)  # metatrepei candidates se numpy array
        dist_sq = l2_sq_to_many(qv, X[cand_idx])  # ypologizei L2^2 query me kathe candidate
        local_idx, dists = topk_smallest(dist_sq, k)  # pairnei top-k mikroteres distances mesa stous candidates
        # metatrepei local indices se arxika dataset indices
        return AnnQueryResult(indices=cand_idx[local_idx], distances=dists)  # epistrefei indices kai distances
