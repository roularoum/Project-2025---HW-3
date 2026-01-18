from __future__ import annotations  # epitrepei type hints me forward refs

from typing import Tuple  # type hint gia tuple return types

import numpy as np  # numpy gia arrays kai arithmitikes prakseis


def ensure_float32_2d(x: np.ndarray) -> np.ndarray:  # x einai 2D float32 array
    arr = np.asarray(x, dtype=np.float32)  # metatrepei se numpy array float32
    if arr.ndim != 2:  # elegxei diastaseis
        raise ValueError("Expected 2D array (n, d)")  # error an den einai 2D
    return arr  # epistrefei to array


def ensure_float32_1d(x: np.ndarray) -> np.ndarray:  # x einai 1D float32 array
    arr = np.asarray(x, dtype=np.float32)  # metatrepei se numpy array float32
    if arr.ndim != 1:  # elegxei diastaseis
        raise ValueError("Expected 1D array (d,)")  # error an den einai 1D
    return arr  # epistrefei to array


def l2_sq_to_many(q: np.ndarray, X: np.ndarray) -> np.ndarray:  # ypologizei L2^2 apostaseis apo q pros ola ta rows tou X
    """Squared L2 distances from q to each row of X."""  # docstring: epistrefei squared distances
    q = ensure_float32_1d(q)  # q einai 1D float32
    X = ensure_float32_2d(X)  # X einai 2D float32
    diff = X - q[None, :]  # afairei to q apo kathe row tou X 
    return np.einsum("ij,ij->i", diff, diff).astype(np.float32, copy=False)  


def topk_smallest(dist_sq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:  # vriskei ta k mikrottera distances
    """Return (indices, distances) for the k smallest distances."""  # docstring: gyrnaei indices kai distances
    if k <= 0:  # k prepei > 0
        raise ValueError("k must be positive")  # error an k <= 0
    n = int(dist_sq.shape[0])  # posa stoixeia exei to dist_sq
    if n == 0:  # an den exei kanena
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)  # epistrefei adeia arrays
    k_eff = min(k, n)  # an k>n krataei k_eff=n
    part = np.argpartition(dist_sq, k_eff - 1)[:k_eff]  # vriskei grhgora ta k_eff pio mikra 
    order = np.argsort(dist_sq[part])  # taxinomei ta k_eff pio mikra me swsti seira
    idx = part[order].astype(np.int32, copy=False)  # telika indices me thn seira
    d = np.sqrt(dist_sq[idx]).astype(np.float32, copy=False)  # metatrepei L2^2 -> L2 
    return idx, d  # epistrefei (indices, distances)
