from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class AnnQueryResult:
    indices: np.ndarray  # int32, shape (k,)
    distances: np.ndarray  # float32, shape (k,)


class AnnIndex(ABC):
    """Base interface for ANN indexes (Euclidean / L2)."""

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Build the index from database vectors."""

    @abstractmethod
    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:
        """Return top-k nearest neighbors for a single query vector."""

