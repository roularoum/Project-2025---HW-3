from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ESM2Embedder:
    """ESM-2 embedder (last layer, mean pooling over residues).

    Spec notes (reference.pdf):
    - Use ESM-2 t6 8M (hidden size 320)
    - Truncate sequences to 1022 AAs to fit special tokens (<cls>, <eos>)
    """

    model_name: str = "esm2_t6_8M_UR50D"
    device: str = "auto"  # auto|cpu|cuda
    seed: int = 1

    def __post_init__(self) -> None:
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._torch = torch
        self._device = self._resolve_device(self.device)

        self._model, self._alphabet = self._load_model(self.model_name)
        self._model.eval()
        self._model.to(self._device)
        self._batch_converter = self._alphabet.get_batch_converter()

        # For esm2_t6_* models, last layer index is 6.
        self._last_layer = int(getattr(self._model, "num_layers", 6))

    def embed_sequences(self, sequences: Sequence[str], batch_size: int = 8) -> np.ndarray:
        """Embed a list of amino-acid sequences into an (N, d) float32 array."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        embs: list[np.ndarray] = []
        for batch in _batched(sequences, batch_size):
            embs.append(self._embed_batch(batch))
        return np.concatenate(embs, axis=0)

    # -----------------
    # internal helpers
    # -----------------

    def _embed_batch(self, sequences: Sequence[str]) -> np.ndarray:
        torch = self._torch

        data = [(f"seq{i}", _truncate_aa(seq)) for i, seq in enumerate(sequences)]
        _labels, _strs, tokens = self._batch_converter(data)
        tokens = tokens.to(self._device)

        with torch.no_grad():
            out = self._model(tokens, repr_layers=[self._last_layer], return_contacts=False)
            reps = out["representations"][self._last_layer]  # (B, T, d)

        # Mean pooling over residue tokens only (exclude <cls>, <eos>, padding).
        pad_idx = self._alphabet.padding_idx
        cls_idx = self._alphabet.cls_idx
        eos_idx = self._alphabet.eos_idx

        mask = (tokens != pad_idx) & (tokens != cls_idx) & (tokens != eos_idx)
        # mask: (B, T) -> (B, T, 1)
        mask_f = mask.unsqueeze(-1).to(reps.dtype)
        summed = (reps * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts  # (B, d)

        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)

    def _resolve_device(self, device: str) -> str:
        torch = self._torch
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, model_name: str):
        import esm

        name = model_name.lower().strip()
        if name in {"esm2_t6_8m_ur50d", "t6"}:
            return esm.pretrained.esm2_t6_8M_UR50D()
        raise ValueError(f"Unsupported model_name: {model_name}")


def _truncate_aa(seq: str, max_residues: int = 1022) -> str:
    s = seq.strip().replace(" ", "").upper()
    if len(s) > max_residues:
        return s[:max_residues]
    return s


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

