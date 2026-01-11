from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AnnIndex, AnnQueryResult
from .kmeans import KMeansParams, kmeans_fit
from .utils import ensure_float32_1d, ensure_float32_2d, l2_sq_to_many, topk_smallest


@dataclass
class NeuralLshParams:
    m_bins: int = 100
    top_T: int = 5
    kmeans_max_iter: int = 20
    kmeans_train_size: int | None = 20000
    seed: int = 1

    # MLP hyperparameters
    layers: int = 3
    hidden_units: int = 256
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "auto"  # auto|cpu|cuda


class NeuralLshIndex(AnnIndex):
    """Neural LSH: learned partition predictor + inverted bins + multi-probe.

    Implementation notes:
    - We derive partition labels via KMeans (m bins) for scalability.
    - We then train an MLP classifier to predict the bin label from embeddings.
    - At query time, we probe the top-T bins by softmax probability and search
      exact L2 distances within their union.
    """

    def __init__(self, params: NeuralLshParams) -> None:
        self.params = params
        self._rng = np.random.default_rng(params.seed)

        self._X: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._bins: list[np.ndarray] | None = None
        self._model = None
        self._torch = None
        self._device = None

    def build(self, vectors: np.ndarray) -> None:
        X = ensure_float32_2d(vectors)
        n, d = X.shape
        p = self.params
        if p.m_bins <= 1:
            raise ValueError("m_bins must be > 1")
        if p.m_bins > n:
            raise ValueError("m_bins cannot exceed number of points")
        if p.top_T <= 0:
            raise ValueError("top_T must be positive")

        # -------------------------
        # 1) Partitioning (labels)
        # -------------------------
        train = X
        if p.kmeans_train_size is not None and int(p.kmeans_train_size) < n:
            idx = self._rng.choice(n, size=int(p.kmeans_train_size), replace=False)
            train = X[idx]

        km = kmeans_fit(
            train,
            KMeansParams(k=p.m_bins, max_iter=p.kmeans_max_iter, seed=p.seed),
        )
        centroids = km.centroids
        # Assign all points to nearest centroid
        labels = _predict_labels(X, centroids).astype(np.int32, copy=False)

        bins: list[list[int]] = [[] for _ in range(p.m_bins)]
        for i, lab in enumerate(labels.tolist()):
            bins[int(lab)].append(i)
        bins_np = [np.asarray(b, dtype=np.int32) for b in bins]

        # -------------------------
        # 2) Train MLP classifier
        # -------------------------
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(p.seed)
        np.random.seed(p.seed)

        device = _resolve_device(torch, p.device)
        model = _make_mlp(input_dim=d, output_dim=p.m_bins, layers=p.layers, hidden_units=p.hidden_units)
        model.to(device)

        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(labels).long())
        loader = DataLoader(ds, batch_size=p.batch_size, shuffle=True)

        optim = torch.optim.Adam(model.parameters(), lr=float(p.lr))
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _epoch in range(p.epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optim.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()

        model.eval()

        self._X = X
        self._labels = labels
        self._bins = bins_np
        self._model = model
        self._torch = torch
        self._device = device

    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:
        if self._X is None or self._bins is None or self._model is None or self._torch is None or self._device is None:
            raise RuntimeError("Index is not built")
        qv = ensure_float32_1d(q)
        p = self.params

        torch = self._torch
        with torch.no_grad():
            qt = torch.from_numpy(qv).float().to(self._device).unsqueeze(0)
            logits = self._model(qt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        T = min(int(p.top_T), probs.shape[0])
        top_bins = np.argpartition(-probs, T - 1)[:T]
        # Sort by probability descending
        top_bins = top_bins[np.argsort(probs[top_bins])[::-1]]

        cand: list[int] = []
        for b in top_bins.tolist():
            cand.extend(self._bins[int(b)].tolist())
        if not cand:
            return AnnQueryResult(
                indices=np.empty((0,), dtype=np.int32),
                distances=np.empty((0,), dtype=np.float32),
            )

        cand_idx = np.asarray(cand, dtype=np.int32)
        dist_sq = l2_sq_to_many(qv, self._X[cand_idx])
        local_idx, dists = topk_smallest(dist_sq, k)
        return AnnQueryResult(indices=cand_idx[local_idx], distances=dists)


def _resolve_device(torch, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _make_mlp(input_dim: int, output_dim: int, layers: int, hidden_units: int):
    import torch.nn as nn

    mods: list[nn.Module] = []
    dim = input_dim
    for _ in range(max(0, int(layers))):
        mods.append(nn.Linear(dim, hidden_units))
        mods.append(nn.ReLU())
        dim = hidden_units
    mods.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*mods)


def _predict_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
    x_norm = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    c_norm = np.sum(centroids * centroids, axis=1)[None, :]  # (1,k)
    dot = X @ centroids.T  # (n,k)
    dist_sq = x_norm + c_norm - 2.0 * dot
    return np.argmin(dist_sq, axis=1)

