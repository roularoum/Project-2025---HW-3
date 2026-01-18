from __future__ import annotations  # epitrepei type hints me forward refs

from dataclasses import dataclass  # gia dataclass 
from typing import List  # type hint 

import numpy as np  # numpy gia arrays kai arithmitikes prakseis

from .base import AnnIndex, AnnQueryResult  # koino interface index kai return type
from .kmeans import KMeansParams, kmeans_fit, predict_labels  # KMeans training kai labels gia IVF clusters
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest  # types, distances, top-k


@dataclass  # dataclass gia IVF-Flat hyperparameters
class IvfFlatParams:  # rythmiseis IVF-Flat
    kclusters: int = 50  # posa clusters inverted lists
    nprobe: int = 5  # posa clusters psaxnoume sto query
    max_iter: int = 30  # posa iterations gia KMeans training
    seed: int = 1  # seed gia reproducibility
    train_size: int | None = None  # an mpei pairnei sample points gia training KMeans


class IvfFlatIndex(AnnIndex):  # IVF-Flat ANN index
    """IVF-Flat index: coarse KMeans clusters + exact distances within probed lists."""  # KMeans clusters kai akriveis apostaseis mesa sta lists

    def __init__(self, params: IvfFlatParams) -> None:  # constructor
        self.params = params  # apothikeuei params
        self._rng = np.random.default_rng(params.seed)  # RNG me seed
        self._X: np.ndarray | None = None  # tha kratisei ta database vectors
        self._centroids: np.ndarray | None = None  # centroids
        self._lists: list[np.ndarray] | None = None  # inverted lists kathe cluster -> array indices

    def build(self, vectors: np.ndarray) -> None:  # xtizei to index train KMeans kai ftiaxnei inverted lists
        X = ensure_float32_2d(vectors)  # 2D float32 array
        n, _d = X.shape  # plithos points
        p = self.params  # alias params
        if p.kclusters <= 0:  # elegxos kclusters
            raise ValueError("kclusters must be positive")  # prepei > 0
        if p.nprobe <= 0:  # elegxos nprobe
            raise ValueError("nprobe must be positive")  # prepei > 0
        if p.kclusters > n:  # den mporoun na einai perissotera clusters apo points
            raise ValueError("kclusters cannot exceed number of points")  # error

        train = X  # train KMeans se ola ta points
        if p.train_size is not None and p.train_size < n:  # an exei oristei train_size
            idx = self._rng.choice(n, size=int(p.train_size), replace=False)  # sample tuxaia points xwris replacement
            train = X[idx]  # training subset

        km = kmeans_fit(train, KMeansParams(k=p.kclusters, max_iter=p.max_iter, seed=p.seed))  # trexei KMeans gia centroids
        centroids = km.centroids  # pairnei ta centroids
        labels = predict_labels(X, centroids)  # vriskei se poio centroid anikei kathe point

        lists: list[list[int]] = [[] for _ in range(p.kclusters)]  # proswrines listes cluster -> lista indices
        for i, lab in enumerate(labels.tolist()):  # gia kathe point i pare to label cluster
            lists[int(lab)].append(i)  # vazei to index sto antistoixo cluster
        lists_np = [np.asarray(lst, dtype=np.int32) for lst in lists]  # metatrepei tis listes se numpy arrays

        self._X = X  # apothikeuei dataset vectors
        self._centroids = centroids  # apothikeuei centroids
        self._lists = lists_np  # apothikeuei inverted lists

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:  # psaxnei se merika clusters kai meta kanei exact top-k
        if self._X is None or self._centroids is None or self._lists is None:  # an den exei xtistei
            raise RuntimeError("Index is not built")  # error
        qv = ensure_float32_1d(q)  # 1D float32 query vector
        p = self.params  # alias params

        # vriskei poia clusters einai pio konta sto query
        dist_c = l2_sq_to_many(qv, self._centroids)  # L2^2 apostasi query kathe centroid
        clusters_idx, _ = topk_smallest(dist_c, min(p.nprobe, p.kclusters))  # pare ta nprobe kontina clusters
        cand: list[int] = []  # candidates indices pou tha elegxoume akrivws
        for c in clusters_idx.tolist():  # gia kathe epilegeimeno cluster
            cand.extend(self._lists[int(c)].tolist())  # prosthetoume ola ta point indices tou cluster sta candidates
        if not cand:  # an den vrethike kanenas candidate
            return AnnQueryResult(  # epistrefei adeio apotelesma
                indices=np.empty((0,), dtype=np.int32),  # adeio indices
                distances=np.empty((0,), dtype=np.float32),  # adeio distances
            )
        cand_idx = np.asarray(cand, dtype=np.int32)  # metatrepei candidates se numpy array
        dist_sq = l2_sq_to_many(qv, self._X[cand_idx])  # ypologizei L2^2 query me kathe candidate point
        local_idx, dists = topk_smallest(dist_sq, k)  # pairnei top-k mikrotteres distances mesa stous candidates
        return AnnQueryResult(indices=cand_idx[local_idx], distances=dists)  # epistrefei global indices kai distances
