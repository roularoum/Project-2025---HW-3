from __future__ import annotations  # epitrepei type hints me forward refs

from abc import ABC, abstractmethod  # ABC gia abstract base class, abstractmethod gia ypoxreotiki methodos gia subclasses
from dataclasses import dataclass  # gia dataclass 

import numpy as np  # numpy arrays gia vectors


@dataclass  # dataclass gia apotelesma query
class AnnQueryResult:  # container me ta apotelesmata tou ANN query
    indices: np.ndarray  # indices twn top-k geitonwn 
    distances: np.ndarray  # distances gia tous top-k geitones


class AnnIndex(ABC):  # abstract base class gia ola ta ANN indexes
    """Base interface for ANN indexes (Euclidean / L2)."""  # koino interface build kai query se L2

    @abstractmethod  # prepei na ylopoieitai apo kathe subclass
    def build(self, vectors: np.ndarray) -> None:  # xtizei to index panw sta database vectors
        """Build the index from database vectors."""  # docstring: build stage

    @abstractmethod  # prepei na ylopoieitai apo kathe subclass
    def query(self, q: np.ndarray, k: int) -> AnnQueryResult:  # kanei query kai epistrefei top-k
        """Return top-k nearest neighbors for a single query vector."""  # docstring: top-k gia ena mono query vector
