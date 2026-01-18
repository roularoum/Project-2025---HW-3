"""
neural_lsh.py
=============
Ylopoiisi "Neural LSH" gia ANN search.
1) Xwrizoume to dataset se m_bins "bins" me KMeans (dhladi labels 0..m_bins-1).
2) Ekpaideuoume ena MLP pou, dinoume ena embedding vector, provlepei se poio bin anikei.
3) Sto query, to MLP dinei probabilities gia ola ta bins .
   Emeis epilegoume ta top-T pio pithana bins (multi-probe) kai psaxnoume mono mesa se auta.
4) Mesa stous candidates kanoume exact L2 distances kai pairnoume top-k.
 den psaxnoume se olo to dataset, mono se ligoystous "swstous" bins.
"""

from __future__ import annotations  

from dataclasses import dataclass  

import numpy as np  

from .base import AnnIndex, AnnQueryResult  # interface + result type
from .kmeans import KMeansParams, kmeans_fit  
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest  # helpers + distances + topk


@dataclass
class NeuralLshParams:
    # params gia to "hashing"
    m_bins: int = 100           # posa bins/klaseis tha exoume (kmeans clusters)
    top_T: int = 5              # posa bins tha kanoume probe sto query (multi-probe)
    kmeans_max_iter: int = 20   # iterations gia kmeans
    kmeans_train_size: int | None = 20000  # train subset gia kmeans (gia taxytita se megalo n)
    seed: int = 1               # seed gia reproducibility

    # MLP hyperparameters 
    layers: int = 3             # posa hidden layers
    hidden_units: int = 256     # neurons ana hidden layer
    epochs: int = 10            # posa passes sto dataset
    batch_size: int = 256       # batch size gia SGD/Adam
    lr: float = 1e-3            # learning rate
    device: str = "auto"        # auto|cpu|cuda (pou tha trexei to torch)


class NeuralLshIndex(AnnIndex):
    """
    Neural LSH: "learned" partition predictor + inverted bins + multi-probe.

    - Vgazoume labels/partitions me KMeans (m bins) gia na einai scalable se polla points.
    - Meta ekpaideuoume ena MLP classifier pou provlepei se poio bin anikei ena embedding.
    - Sto query, ypologizoume softmax probabilities gia ola ta bins kai kanoume probe ta top-T pio pithana bins.
    Mesa stin enwsi (union) aftwn twn bins kanoume exact L2 distances.
    """ # 3 vimatwn (kmeans bins -> MLP -> top-T bins search)

    def __init__(self, params: NeuralLshParams) -> None:
        self.params = params  # apothikeusi params
        self._rng = np.random.default_rng(params.seed)  # RNG gia sampling

        # fields pou gemizoun sto build()
        self._X: np.ndarray | None = None      # original vectors
        self._labels: np.ndarray | None = None # bin label gia kathe point
        self._bins: list[np.ndarray] | None = None  # inverted bins: gia kathe bin lista indices
        self._model = None   # torch model (MLP classifier)
        self._torch = None   # torch module reference (gia na min einai global import)
        self._device = None  # 'cpu' h 'cuda'

    def build(self, vectors: np.ndarray) -> None:
        X = ensure_float32_2d(vectors)  # ensure 2D float32
        n, d = X.shape                  # n points, d dims
        p = self.params  # apothikeuoume ta params se p gia na ta xrhsimopoioume pio eukola 
              

        # validations gia logikes times
        if p.m_bins <= 1:
            raise ValueError("m_bins must be > 1")  # theloume panw apo 1 bin
        if p.m_bins > n:
            raise ValueError("m_bins cannot exceed number of points")  # den ginetai perissotera bins apo points
        if p.top_T <= 0:
            raise ValueError("top_T must be positive")  # probe bins >= 1

        # -------------------------
        # 1) (labels)
        # -------------------------
        # KMeans: ftiaxnoume m_bins clusters kai dinoume label se kathe point
        train = X
        if p.kmeans_train_size is not None and int(p.kmeans_train_size) < n:
            # gia taxytita, ekpaideuoume kmeans se tyxaio subset
            idx = self._rng.choice(n, size=int(p.kmeans_train_size), replace=False)
            train = X[idx]

        km = kmeans_fit(
            train,
            KMeansParams(k=p.m_bins, max_iter=p.kmeans_max_iter, seed=p.seed),
        )
        centroids = km.centroids  # (m_bins, d)

        # Assign all points to nearest centroid -> labels 0..m_bins-1
        labels = _predict_labels(X, centroids).astype(np.int32, copy=False)

        # Inverted structure: gia kathe bin, poioi indices anisoun ekei
        bins: list[list[int]] = [[] for _ in range(p.m_bins)]
        for i, lab in enumerate(labels.tolist()):
            bins[int(lab)].append(i)
        bins_np = [np.asarray(b, dtype=np.int32) for b in bins]

        # -------------------------
        # -------------------------
        # 2) Train MLP classifier
        # -------------------------
        # Ekpaideuoume ena MLP pou pairnei embeddings (d) kai provlepei bin label (m_bins classes).
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(p.seed)   # gia reproducibility sto torch
        np.random.seed(p.seed)      # gia reproducibility sto numpy

        device = _resolve_device(torch, p.device)
        # device = "cpu" (processor) h "cuda" (NVIDIA GPU) an yparxei. To "auto" dialegει mono tou.

        model = _make_mlp(
            input_dim=d,
            output_dim=p.m_bins,
            layers=p.layers,
            hidden_units=p.hidden_units,
        )
        model.to(device)  # stelnei to model sto cpu/gpu

        # Ftiaxnoume dataset (X -> input, labels -> target class)
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(labels).long())
        loader = DataLoader(ds, batch_size=p.batch_size, shuffle=True)

        optim = torch.optim.Adam(model.parameters(), lr=float(p.lr))
        loss_fn = nn.CrossEntropyLoss()  # classification loss

        model.train()
        for _epoch in range(p.epochs):
            for xb, yb in loader:
                xb = xb.to(device)  # metaferoume ta input data sto cpu h gpu (opou trexei to model)->to pairnoume apo suarthsh resolve device
                yb = yb.to(device)  # metaferoume kai ta labels sto idio device gia na min exoume mismatch
                optim.zero_grad()  # mhdenizoume ta gradients apo to proigoumeno batch (alliws tha xalasei to training)

                logits = model(xb)         # raw scores (oxi probabilities)
                loss = loss_fn(logits, yb) # poso lathos einai
                loss.backward()            # gradients
                optim.step()               # update weights
        model.eval()

        # apothikeuoume state gia na to xrhsimopoihsoume sto query()
        self._X = X
        self._labels = labels
        self._bins = bins_np
        self._model = model
        self._torch = torch
        self._device = device



    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:
        # Prepei na exei xtistei to index me build(), alliws den yparxoun bins/model.
        if self._X is None or self._bins is None or self._model is None or self._torch is None or self._device is None:
            raise RuntimeError("Index is not built")

        qv = ensure_float32_1d(q)
        # Ensures oti to query einai 1D float32 vector (idios typos me ta embeddings)

        p = self.params
        # Shortcut gia na grafoume p.top_T anti self.params.top_T

        # 1) MLP inference: vriskei pithanotites gia kathe bin
        torch = self._torch
        with torch.no_grad():
            qt = torch.from_numpy(qv).float().to(self._device).unsqueeze(0)
            # metatrepoume query se torch tensor
            # unsqueeze(0) -> apo shape (d,) ginetai (1,d) giati to model thelei batch dimension

            logits = self._model(qt)  # raw scores gia kathe bin (1, m_bins)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            # softmax -> metatrepei scores se probabilities pou athroizoun se 1
            # to [0] -> pernoume to mono query apo to batch

        # 2) Epilogi top-T bins: dialegoume ta T bins me megalyteri pithanotita
        T = min(int(p.top_T), probs.shape[0])
        top_bins = np.argpartition(-probs, T - 1)[:T]
        #  grigoros tropos na pareis top-T xwris na kaneis full sort se ola ta bins

        top_bins = top_bins[np.argsort(probs[top_bins])[::-1]]
        # meta kanoume sort MONO sta top_bins, gia na einai se seira (pio pithano -> ligotero)

        # 3) Candidate set: mazevoume olous tous indices apo auta ta bins
        cand: list[int] = []
        for b in top_bins.tolist():
            cand.extend(self._bins[int(b)].tolist())
        # edw ginete i meiwsi search space: psaxnoume mono se points pou peftoun sta top-T bins

        if not cand:
            # an den vrethike kanenas candidate (px ta bins itan adeia),
            # epistrefoume adeio result gia na min skasei o kwdikas
            return AnnQueryResult(
                indices=np.empty((0,), dtype=np.int32),
                distances=np.empty((0,), dtype=np.float32),
            )

        # 4) Exact L2 search mesa sta candidates: ypologizoume distances kai pairnoume top-k
        cand_idx = np.asarray(cand, dtype=np.int32)
        dist_sq = l2_sq_to_many(qv, self._X[cand_idx])   # squared L2 distances mono gia candidates
        local_idx, dists = topk_smallest(dist_sq, k)     # pare top-k mikroteres
        return AnnQueryResult(indices=cand_idx[local_idx], distances=dists)
