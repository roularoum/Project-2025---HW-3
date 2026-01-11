from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VectorStore:
    """In-memory vector store with ID mapping."""

    vectors: np.ndarray  # shape: (n, d), float32
    ids: list[str]  # len n

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be 2D (n, d)")
        if self.vectors.shape[0] != len(self.ids):
            raise ValueError("vectors rows must match ids length")


def _default_ids_path(vectors_path: Path) -> Path:
    # Prefer a stable sibling name that does not depend on extension.
    return vectors_path.with_name("ids.txt")


def save_vectors(vectors_path: str | Path, vectors: np.ndarray, ids: Sequence[str]) -> Path:
    """Save vectors to a binary NumPy file at *exactly* vectors_path.

    The assignment uses a `.dat` extension, but the content is a standard `.npy`.
    We avoid NumPy auto-appending `.npy` by writing to a file handle.
    """
    p = Path(vectors_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    vec = np.asarray(vectors, dtype=np.float32)
    if vec.ndim != 2:
        raise ValueError("vectors must be 2D (n, d)")

    ids_list = list(map(str, ids))
    if vec.shape[0] != len(ids_list):
        raise ValueError("vectors rows must match ids length")

    with p.open("wb") as f:
        np.save(f, vec, allow_pickle=False)

    ids_path = _default_ids_path(p)
    with ids_path.open("w", encoding="utf-8", newline="\n") as f:
        for pid in ids_list:
            f.write(pid + "\n")

    return p


def load_vectors(vectors_path: str | Path, ids_path: str | Path | None = None) -> VectorStore:
    """Load vectors and IDs.

    Accepts:
    - vectors file containing `.npy` content (any extension)
    - optionally, an `.npz` file containing arrays `vectors` and `ids`
    """
    p = Path(vectors_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    loaded = np.load(p, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "vectors" not in loaded.files:
            raise ValueError(f"NPZ missing 'vectors': {p}")
        vec = np.asarray(loaded["vectors"], dtype=np.float32)
        if "ids" in loaded.files:
            raw_ids = loaded["ids"]
            ids_list = [str(x) for x in raw_ids.tolist()]
        else:
            if ids_path is None:
                ids_path = _default_ids_path(p)
            ids_list = _load_ids(ids_path)
        return VectorStore(vectors=vec, ids=ids_list)

    vec = np.asarray(loaded, dtype=np.float32)
    if ids_path is None:
        ids_path = _default_ids_path(p)
        # Backwards-compatible: allow <vectors>.ids.txt
        if not Path(ids_path).exists():
            alt = Path(str(p) + ".ids.txt")
            if alt.exists():
                ids_path = alt
    ids_list = _load_ids(ids_path)
    return VectorStore(vectors=vec, ids=ids_list)


def _load_ids(ids_path: str | Path) -> list[str]:
    p = Path(ids_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing ids mapping file: {p}")
    with p.open("r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if not ids:
        raise ValueError(f"Empty ids file: {p}")
    return ids

