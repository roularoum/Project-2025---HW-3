"""
kmeans.py
=========
Ylopoiisi k-means clustering  gia na spame ta vectors se k clusters.
To xrisimopoioume sto IVF (kclusters) gia na ftiaxoume "coarse centroids" kai na anaθέσουμε kathe vector se ena cluster.
Periexei:
- kmeans_fit: ekpaideuei ta centroids me iterative assignment + update
- predict_labels: vriskei se poio centroid (cluster) antistoixei kathe vector
- kmeans++ init: pio kalh arxikopoiisi gia na min kollaei se kakous centroids
"""

from __future__ import annotations 

from dataclasses import dataclass  # dataclasses gia params kai result
from typing import Tuple  

import numpy as np  # NumPy gia arrays/math

from .utils import ensure_float32_2d  # sigourevei 2D float32 array


@dataclass
class KMeansParams:
    k: int  # posa clusters (centroids) theloume
    max_iter: int = 30  # max iterations tou kmeans
    tol: float = 1e-4  # poso mikri prepei na ginei h allagh sta centroids gia na poume oti teleiose (0.0001)
    seed: int = 1  # seed gia random init/reinit


@dataclass(frozen=True)
class KMeansResult:
    centroids: np.ndarray  # (k, d), float32 (oi telikoi centroids)
    labels: np.ndarray  # (n,), int32 (se poio cluster pige kathe point)


def kmeans_fit(X: np.ndarray, params: KMeansParams) -> KMeansResult:
    X = ensure_float32_2d(X)  # kanonikopoiisi: 2D kai float32
    n, d = X.shape  # n points, d dimensions

    # validations: min trexoume se adia h paraλογα inputs
    if n == 0:
        raise ValueError("Empty dataset")  # den exei points
    if params.k <= 0:
        raise ValueError("k must be positive")  # clusters prepei na einai >0
    if params.k > n:
        raise ValueError("k cannot exceed number of points")  # den ginetai perissotera clusters apo points

    rng = np.random.default_rng(params.seed)  # random generator me seed gia reproducibility
    C = _init_kmeans_pp(X, params.k, rng)  # arxikopoiisi centroids me kmeans++ (pio kalh apo random)

    last_shift = np.inf  # krataei tin teleutaia metakinisi centroids 
    labels = np.zeros((n,), dtype=np.int32)  # arxika labels (tha upologistoun sto loop)

    for _ in range(params.max_iter):  # epanalipseis mexri max_iter
        oldC = C.copy()  # kratame ta palia centroids gia na metrisoume shift

        labels = predict_labels(X, C)  # (step 1) kathe point -> pio kontino centroid
        C = _recompute_centroids(X, labels, params.k, rng, oldC)  # (step 2) nea centroids = mesos oros twn points tou cluster

        shift = float(np.max(np.linalg.norm(C - oldC, axis=1)))  # max metakinisi centroid (L2) se olo to k
        last_shift = shift  # plhroforia

        if shift <= params.tol:  # an i metakinisi einai poli mikri -> exoume sygklisi
            break  # stamatame to kmeans

    return KMeansResult(centroids=C, labels=labels)  # epistrefoume telika centroids + labels


def predict_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
     # vres pio kontino centroid gia kathe point
    X = ensure_float32_2d(X)  # sigourevei swsto dtype/shape
    C = ensure_float32_2d(centroids)  # kai ta centroids 2D float32

    # Ypologizoume L2^2 apostaseis apo ola ta points X (n,d) pros ola ta centroids C (k,d) xwris loops.
    # Xrhsimopoioume: ||x-c||^2 = ||x||^2 + ||c||^2 - 2*(x·c) gia grigoro matrix compute.

    x_norm = np.sum(X * X, axis=1, keepdims=True)      # (n,1) ta ||x||^2 gia kathe point
    c_norm = np.sum(C * C, axis=1)[None, :]            # (1,k) ta ||c||^2 gia kathe centroid
    dot = X @ C.T                                      # (n,k) dot products x·c
    dist_sq = x_norm + c_norm - 2.0 * dot              # (n,k) squared L2 apostaseis


    return np.argmin(dist_sq, axis=1).astype(np.int32, copy=False)  # gia kathe point pare to centroid me mikroteri dist^2


def _init_kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    # kmeans++ initialization: dialegoume arxika centroids me pithnothta analogi tis apostasis
    n, d = X.shape # to NumPy array X exei idi metadata gia tis diastaseis tou.
               # me to X.shape pairnoume ena tuple (n, d) pou leei:
               # - posa stoixeia exei stin 1i diastasi (grammes) -> n
               # - posa stoixeia exei stin 2i diastasi (stiles)  -> d
               # diladi: X einai matrix me n vectors kai kathe vector exei d diastaseis.
    C = np.empty((k, d), dtype=np.float32)  # pinakas pou tha krataei ta k centroids

    idx0 = int(rng.integers(0, n))  # dialegoume tuxaia to prwto centroid
    C[0] = X[idx0]  # to prwto centroid einai ena tuxaio point

    # apostash apo nearest centroid for each point.
    dist_sq = np.sum((X - C[0][None, :]) ** 2, axis=1)  # L2^2 apo kathe point sto prwto centroid

    for i in range(1, k):  # gia kathe epomeno centroid
        total = float(dist_sq.sum())  # athroisma apostasewn 
        if total <= 0.0:
            idx = int(rng.integers(0, n))  # an ola einai idia, den exei noima kmeans++ -> pare tuxaio
            C[i] = X[idx]
            continue

        probs = dist_sq / total  #  pio makria -> megaluteri pithanotita na dialexthei
        idx = int(rng.choice(n, p=probs))  # dialegoume neo centroid me vasi tis pithanotites
        C[i] = X[idx]  # apothikeuoume to centroid

        new_dist = np.sum((X - C[i][None, :]) ** 2, axis=1)  # apostasi kathe point apo to neo centroid
        dist_sq = np.minimum(dist_sq, new_dist)  # kratame tin apostasi sto kontiniotero centroid (nearest-so-far)

    return C  # epistrefoume arxika centroids (kmeans++)


def _recompute_centroids(
    X: np.ndarray,  # dataset (n,d)
    labels: np.ndarray,  # labels (n,) se poio cluster antistoixei kathe point
    k: int,  # plithos clusters
    rng: np.random.Generator,  #random generator pou to xrhsimopoioume an xreiastei na xanabaloume centroid kapou tuxaia an ena cluster meinei adeio
    fallback: np.ndarray,  # palia centroids (fallback base)
) -> np.ndarray:
    n, d = X.shape

    sums = np.zeros((k, d), dtype=np.float32)  # athroisma vectors ana cluster
    counts = np.bincount(labels, minlength=k).astype(np.int32)  # posa points exei kathe cluster
    np.add.at(sums, labels, X)  # prosthetei kathe point X[i] sto sums[label[i]] 

    C = fallback.copy()  # ksekinaμε apo palia centroids (se periptwsi adeiou cluster)
    for j in range(k):  # gia kathe cluster
        if counts[j] > 0:  # an exei points
            C[j] = sums[j] / float(counts[j])  # centroid = mesos oros twn points (mean)
        else:
            idx = int(rng.integers(0, n))  # an cluster adeiasei, to xanavazoume se random point
            C[j] = X[idx]

    return C  # epistrefoume ta nea centroids
