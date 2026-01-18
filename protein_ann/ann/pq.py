"""
pq.py
=====
To arxeio ylopoiei Product Quantization (PQ) me Asymmetric Distance Computation (ADC) gia grigoro ANN search.

- Kathe embedding vector diastasis d spaei se M isomera kommatia (subspaces), opou dsub = d/M.
- Gia kathe subspace, ekpaideuoume ena "codebook" me k=2^nbits centroids mesw k-means.
- Encoding: gia kathe vector krataμε MONO ton index tou kontinioterou centroid se kathe subspace (ara (n,M) codes),
  anti na apothikeuoume olo to float vector.
- Query: ftiaxnoume ena distance table (M,k) me tis apostaseis tou query subvector pros ola ta centroids.
- ADC: gia kathe encoded vector, ypologizoume approximate apostasi me lookup + athroisma apo to table.
"""
from __future__ import annotations  

from dataclasses import dataclass 
from typing import List            

import numpy as np                # NumPy gia arrays, math, random

from .kmeans import KMeansParams, kmeans_fit          
from .utils import ensure_float32_1d, ensure_float32_2d  # sigoureuoun swsto shape + dtype float32


@dataclass
class PqParams:
    M: int = 16              # posa "kommatia" (subspaces) tha spaei to vector (d -> M κομμάτια)
    nbits: int = 8           # posa bits ana code -> k = 2^nbits centroids ana subspace
    seed: int = 1            # seed gia reproducibility sto sampling/training
    train_size: int = 5000   # posa vectors tha xrhsimopoihsoume gia training (subset gia taxytita)
    max_iter: int = 25       # max iterations tou kmeans se kathe subspace
    encode_chunk: int = 5000 # posa vectors ana batch sto encode (gia na min "skasei" i mnhmh)


class ProductQuantizer:
    # Product Quantization (PQ): compress vectors se M codes (indexes centroids)
    # ADC: sto query ypologizoume distances me lookup table (oxi full vectors)

    def __init__(self, params: PqParams) -> None:
        self.params = params  # apothikeuoume tis rythmiseis PQ
        self._rng = np.random.default_rng(params.seed)  # random generator me seed 
        self._dsub: int | None = None  # diastasi kathe subvector (d/M). ginetai set sto fit()
        self._centroids: list[np.ndarray] | None = None  # M codebooks: lista me M pinakes (k, dsub)

    @property
    def dsub(self) -> int:
        # dsub = diastasi kathe subvector. an den exei trexei fit(), den yparxei.
        if self._dsub is None:
            raise RuntimeError("PQ is not fitted")  # den mporeis na kaneis encode/ADC xwris training
        return self._dsub

    @property
    def k(self) -> int:
        # k = 2^nbits centroids ana subspace (px nbits=8 -> k=256)
        return 1 << int(self.params.nbits)

    def fit(self, X: np.ndarray) -> None:
        # Training PQ:
        # 1) spame kathe vector se M sub-vectors
        # 2) gia kathe subspace trexoume kmeans me k=2^nbits
        # 3) ta centroids ginetai "codebook" pou tha xrhsimopoihsoume sto encode
        X = ensure_float32_2d(X)  # kanonikopoiisi input: 2D array float32
        n, d = X.shape            # n = posa training vectors, d = diastasi embedding
        p = self.params           # shortcut (anti self.params synexeia)

        # validations gia na min exoume lathos rythmiseis
        if p.M <= 0:
            raise ValueError("M must be positive")  # prepei na exoume toulaxiston 1 subspace
        if d % p.M != 0:
            # prepei d na xwrizei akribws se M koμμάτια, alliws den mporoume na spame isotima
            raise ValueError(f"Vector dimension {d} must be divisible by M={p.M}")
        if p.nbits <= 0 or p.nbits > 16:
            # poly megalo nbits -> terastio k=2^nbits kai den einai praktikο
            raise ValueError("nbits must be in [1,16]")

        dsub = d // p.M      # dsub = diastasi kathe subvector (px 320/16 = 20)
        self._dsub = dsub    # apothikeuoume to dsub gia na to xrhsimopoihsoume meta

        k = 1 << int(p.nbits)  # k = 2^nbits (px 256)
        if k > n:
            # den ginetai na ekpaideuseis k centroids me ligotera apo k points (kmeans thelei toulaxiston k points)
            raise ValueError(f"PQ k=2^nbits={k} cannot exceed training points {n}")

        # se megalo dataset, den theloume na trexoume kmeans se ola (einai argo),
        # opote pairnoume tyxaio subset train_n.
        train_n = min(int(p.train_size), n)  # posa tha xrhsimopoihsoume telika
        if train_n < n:
            idx = self._rng.choice(n, size=train_n, replace=False)  # tyxaia epilogi indices xwris repeats
            train = X[idx]  # to subset pou tha ekpaideusei ta codebooks
        else:
            train = X  # an eimaste hdh mikro dataset, xrhsimopoioume ola

        cents: list[np.ndarray] = []  # edw tha mazepstei ta centroid  gia kathe subspace m
        for m in range(p.M):
            # pare to m-th kommati twn vectors:
            # columns [m*dsub : (m+1)*dsub]
            sub = train[:, m * dsub : (m + 1) * dsub]

            # trexoume kmeans sto sugkekrimeno subspace gia na ftiaxoume k centroids
            km = kmeans_fit(
                sub,
                KMeansParams(k=k, max_iter=p.max_iter, seed=p.seed + m),  # seed+m: diaforetiko init ana subspace
            )

            # apothikeuoume ta centroids (k,dsub) ws float32 (codebook gia auto to subspace)
            cents.append(km.centroids.astype(np.float32, copy=False))

        self._centroids = cents  # telika exoume M codebooks: centroids[0..M-1]

    def encode(self, X: np.ndarray) -> np.ndarray:
        # Gia kathe vector x:
        # - to spame se M subvectors
        # - gia kathe subvector vriskei to kontiniotero centroid sto antistoixo codebook
        # - apothikeuei MONO ton index tou centroid 
        if self._centroids is None or self._dsub is None:
            raise RuntimeError("PQ is not fitted")  # prepei na exei ginei fit() prin kaneis encode

        X = ensure_float32_2d(X)  # sigoureuoume 2D float32
        n, d = X.shape            # n vectors, d dimensions
        p = self.params
        dsub = self._dsub         # diastasi subvector

        if d != dsub * p.M:
            # an to X exei diaforetiko d apo auto pou ekpaideutike to PQ, tote ta slices den vgainoun
            raise ValueError("Unexpected dimension for PQ encoding")

        k = self.k  # posa centroids ana subspace
        # epilegoume mikrotero dtype gia na exoume ligotero storage:
        # an k<=256 xwraei se 1 byte (uint8), alliws thelei 2 bytes (uint16)
        code_dtype = np.uint8 if k <= 256 else np.uint16
        codes = np.empty((n, p.M), dtype=code_dtype)  # gia kathe vector, M codes

        chunk = max(1, int(p.encode_chunk))  # posa vectors ana batch gia na min kanei megalo (b,k)
        for m in range(p.M):
            C = self._centroids[m]  # (k, dsub) codebook gia subspace m
            Xm = X[:, m * dsub : (m + 1) * dsub]  # (n, dsub) ola ta subvectors tou subspace m

            # Spame ta subvectors se batches gia na min ftiaxtei terastios pinakas apostasewn (n,k) sti RAM
            for i0 in range(0, n, chunk):
                i1 = min(n, i0 + chunk)     # oria tou batch
                block = Xm[i0:i1]           # (b, dsub) to batch subvectors

                # dist^2 = ||x||^2 + ||c||^2 - 2 x·c  (grigoros ypologismos L2^2 xwris loops)
                x_norm = np.sum(block * block, axis=1, keepdims=True)  # (b,1)
                c_norm = np.sum(C * C, axis=1)[None, :]                # (1,k)
                dot = block @ C.T                                      # (b,k)
                dist_sq = x_norm + c_norm - 2.0 * dot                  # (b,k)

                # Gia kathe subvector, vriskoume to pio kontino centroid -> code (index)
                codes[i0:i1, m] = np.argmin(dist_sq, axis=1).astype(code_dtype, copy=False)

        return codes

    def distance_table(self, q: np.ndarray) -> np.ndarray:
        # Gia ena query q, theloume precompute:
        # table[m,j] = ||q_sub(m) - centroid(m,j)||^2
        # auto mas epitrepei sto ADC na kanei mono lookups (xwris na ksanaypologizei apostaseis)
        if self._centroids is None or self._dsub is None:
            raise RuntimeError("PQ is not fitted")  # prepei na exei ginei fit()

        qv = ensure_float32_1d(q)  # query se 1D float32
        p = self.params
        dsub = self._dsub

        if qv.shape[0] != dsub * p.M:
            # query prepei na exei idia diastasi me ta vectors
            raise ValueError("Unexpected query dimension for PQ")

        table = np.empty((p.M, self.k), dtype=np.float32)  # (M,k) lookup distances
        for m in range(p.M):
            C = self._centroids[m]  # (k,dsub) codebook
            qs = qv[m * dsub : (m + 1) * dsub]  # query subvector gia subspace m: (dsub,)
            diff = C - qs[None, :]              # (k,dsub) diafores me kathe centroid
            table[m] = np.sum(diff * diff, axis=1)  # (k,) squared distances gia ola ta centroids
        return table

    def adc_distances(self, table: np.ndarray, codes: np.ndarray) -> np.ndarray:
        # ADC (Asymmetric Distance Computation):
        # - Den ginetai "decode" kanonika ta vectors.
        # - Gia kathe vector exoume M codes (indexes centroids).
        # - H approximate apostasi einai:
        #   dist(q,x) ≈ sum_m table[m, code_m(x)]
        #   dhladi lookup kai athroisma.
        p = self.params

        if table.shape[0] != p.M:
            raise ValueError("Invalid table shape")  # prepei table na exei M grammές
        if codes.shape[1] != p.M:
            raise ValueError("Invalid codes shape")  # prepei codes na exei M stiles

        dist = np.zeros((codes.shape[0],), dtype=np.float32)  # dist gia kathe encoded vector (n,)
        for m in range(p.M):
            # codes[:,m] einai o code index gia to subspace m gia ola ta vectors
            # table[m, ...] kanei lookup distances apo to query subspace m pros to centroid pou antistoixei sto code
            dist += table[m, codes[:, m].astype(np.int64)]  # a8roizoume ana subspace
        return dist  # (n,) approximate squared distances q->ola ta vectors
