"""
ivf_pq.py
=========
Ylopoiisi ANN methodou IVFPQ (Inverted File + Product Quantization).
Idea:
- IVF: spame to dataset se K coarse clusters (me k-means). Gia query psaxnoume mono se ligoystous kontinoous clusters (nprobe).
- PQ: anti na krataμε olokliro to vector, to "kwdikopoioume" se M kommatia (sub-vectors) kai to kathena pairnei ena code (quantization).
  Έτσι kanoume grigora approximate distances (ADC) xwris na ypologizoume L2 me ola ta vectors.
- Telos: pairnoume top-k me PQ distance kai meta "re-rank" me to pragmatiko L2 για na vgoun swstes apodoseissto output.
"""

from __future__ import annotations  # type hints pio eukola

from dataclasses import dataclass  # dataclass gia params

import numpy as np  # arrays/math

from .base import AnnIndex, AnnQueryResult  # base interface + apotelesma query
from .ivf_flat import IvfFlatParams  #params gia IVF-Flat
from .kmeans import KMeansParams, kmeans_fit, predict_labels  # k-means fit + assign labels
from .pq import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest  # helpers + distance + topk 
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest

@dataclass
class IvfPqParams:
    # IVF params
    kclusters: int = 50        # posa coarse clusters (K) tha ftiaxoume me kmeans
    nprobe: int = 5            # se posa clusters tha psaxnoume ana query (ligotera = pio grigoro)
    seed: int = 1              # seed gia random επιλογes/train subset
    max_iter: int = 30         # max iterations gia kmeans
    train_size: int | None = None  # an theloume na kanoume kmeans se subset (gia taxytita)

    # PQ params
    pq_M: int = 16             # posa sub-vectors (M) spame to vector (d prepei na "xwrizei" se M)
    pq_nbits: int = 8          # posa bits ana sub-quantizer (8 bits => 256 centroids ana subspace)
    pq_train_size: int = 5000  # posa samples gia training tis PQ
    pq_max_iter: int = 25      # iterations gia PQ training (kmeans se kathe subspace)
    pq_encode_chunk: int = 5000  # se posa vectors ana batch tha ginete encoding (gia RAM)


class IvfPqIndex(AnnIndex):
    """IVFPQ index: IVF coarse quantizer + PQ codes in each list (ADC)."""  

    def __init__(self, params: IvfPqParams) -> None:
        self.params = params  # apothikeuoume tis rythmiseis
        self._rng = np.random.default_rng(params.seed)  # RNG gia sampling

        # tha gemisoun sto build()
        self._X: np.ndarray | None = None  # ta original vectors (float32) gia final re-rank
        self._centroids: np.ndarray | None = None  # coarse centroids (K, d)
        self._lists: list[np.ndarray] | None = None  # inverted lists: gia kathe cluster lista me indices
        self._pq: ProductQuantizer | None = None  # PQ object  (orismeno sto pq.py)
        self._codes: np.ndarray | None = None  # PQ codes gia kathe vector: shape (n, M), uint8/uint16

    def build(self, vectors: np.ndarray) -> None:
        X = ensure_float32_2d(vectors)  # ensure 2D float32
        n, d = X.shape  # n points, d dims
        p = self.params  # shortcut gia params

       # elegxoi parametron gia na min trexei o algorithm me paraloges times
        if p.kclusters <= 0:
            raise ValueError("kclusters must be positive")
        if p.nprobe <= 0:
            raise ValueError("nprobe must be positive")
        if p.kclusters > n:
            raise ValueError("kclusters cannot exceed number of points")  # den mporoume na exoume perissotera clusters apo points

        # dialegoume training set gia kmeans 
        train = X  # default: olo to dataset
        if p.train_size is not None and p.train_size < n:  # an theloume subset
            idx = self._rng.choice(n, size=int(p.train_size), replace=False)  # random sample xwris replacement
            train = X[idx]  # training subset

        # 1) KMeans gia coarse centroids (IVF part)
        km = kmeans_fit(train, KMeansParams(k=p.kclusters, max_iter=p.max_iter, seed=p.seed))  # fit kmeans
        centroids = km.centroids  # (K,d)
        labels = predict_labels(X, centroids)  # assign kathe point sto kontini centroid -> label 0..K-1

        # 2) Ftiaxnoume inverted lists: gia kathe cluster, poioi indices tou dataset peftoun ekei
        lists: list[list[int]] = [[] for _ in range(p.kclusters)]  # arxika adeies listes
        for i, lab in enumerate(labels.tolist()):  # loop se kathe point
            lists[int(lab)].append(i)  # prosthese index i sto list tou cluster lab
        lists_np = [np.asarray(lst, dtype=np.int32) for lst in lists]  # metatrepoume se numpy arrays gia grigori prosvasi

        # 3) Train PQ (Product Quantization)
        pq = ProductQuantizer(
            PqParams(
                M=p.pq_M,                 # posa subspaces
                nbits=p.pq_nbits,         # bits ana code
                seed=p.seed,              # seed
                train_size=p.pq_train_size,  # training samples gia PQ
                max_iter=p.pq_max_iter,   # iterations gia sub-kmeans
                encode_chunk=p.pq_encode_chunk,  # chunk size gia encode
            )
        )
        qv = ensure_float32_1d(q)  # metatrepei to query se float32 1D vector, giati oloi oi upologismoi (L2/PQ) perimenoun statheri dtype/shape
        p = self.params            # apothikeuei ta params se p gia na min grafoume synexeia self.params (kanei ton kwdika pio anagnwsimo)


        # apothikeuoume ta panta sto index
        self._X = X
        self._centroids = centroids
        self._lists = lists_np
        self._pq = pq
        self._codes = codes

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:
        # elegxos oti exei ginei build()
        if self._X is None or self._centroids is None or self._lists is None or self._pq is None or self._codes is None:
            raise RuntimeError("Index is not built")

        qv = ensure_float32_1d(q)  # query vector se float32 1D
        p = self.params  # params shortcut

        # vriskoume ta pio kontina centroids (nprobe clusters)
        dist_c = l2_sq_to_many(qv, self._centroids)  # L2^2 apo query pros kathe centroid (K,)
        clusters_idx, _ = topk_smallest(dist_c, min(p.nprobe, p.kclusters))  # pare top-nprobe clusters me mikroteri dist

        # mazeuoume olous tous points apo auta ta clusters
        cand: list[int] = []
        for c in clusters_idx.tolist():
            cand.extend(self._lists[int(c)].tolist())  # prosthese ola ta indices apo inverted list tou cluster c

        if not cand:  # an gia kapoio logo den vrikame candidates 
            return AnnQueryResult(
                indices=np.empty((0,), dtype=np.int32),
                distances=np.empty((0,), dtype=np.float32),
            )

        cand_idx = np.asarray(cand, dtype=np.int32)  
        # metatrepoume tin lista me kontinous geitones se NumPy array (int32) gia grigoro indexing

        codes_c = self._codes[cand_idx]  
        # pairnoume ta PQ codes MONO gia tous candidates (oxi gia olo to dataset), wste na ypologisoume PQ distances grigora


        # (C) PQ distance computation (ADC):
        table = self._pq.distance_table(qv)          # precompute distances query->subcentroids (lookup table)
        dist_sq = self._pq.adc_distances(table, codes_c)  # approximate distances gia kathe candidate (me codes)

        # Pairnoume top-k me base to approximate PQ dist kai meta ypologizoume akrivi L2 gia auta.
        local_idx, _ = topk_smallest(dist_sq, k)  # epilegei top-k candidates (local positions mesa sto cand_idx)
        chosen = cand_idx[local_idx]              # metatrepei se global dataset indices

        true_sq = l2_sq_to_many(qv, self._X[chosen])  # akrivi L2^2 mono gia tous chosen
        order = np.argsort(true_sq)                   # sort me base tin akrivi dist
        chosen = chosen[order]                        # re-order indices
        dists = np.sqrt(true_sq[order]).astype(np.float32, copy=False)
        # true_sq einai L2^2 (athroisma tetragwnwn diaforwn xwris riza).
        # Pairnoume sqrt gia na paragoume pragmatiko L2 distance opws thelei to output.

        return AnnQueryResult(indices=chosen, distances=dists)  # epistrefoume top-k indices + exact distances
