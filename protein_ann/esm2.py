from __future__ import annotations  # epitrepei type hints me forward refs

from dataclasses import dataclass  # gia na orisoume dataclass (auto init, klt)
from typing import Iterable, Sequence  # type hints: Sequence gia lista/tuple, Iterable gia generator output

import numpy as np  # numpy gia arrays kai concatenate


@dataclass  # dataclass gia na kratame parametrous kai state tou embedder
class ESM2Embedder:  # klasi pou vgazei embeddings me ESM-2
    """ESM-2 embedder (last layer, mean pooling over residues).  # perigrafi: pairnei last layer reps kai kanei mean pooling

    Spec notes (reference.pdf):  # shmeiwseis apo to assignment spec
    - Use ESM-2 t6 8M (hidden size 320)  # xrisimopoiei to t6 8M model (d=320)
    - Truncate sequences to 1022 AAs to fit special tokens (<cls>, <eos>)  # kovei seq mexri 1022 gia na xwrane ta special tokens
    """

    model_name: str = "esm2_t6_8M_UR50D"  # default model variant
    device: str = "auto"  # device pou tha trexei i ektelesi tou model: auto|cpu|cuda
    seed: int = 1  # seed gia reproducibility

    def __post_init__(self) -> None:  # trexei meta to dataclass init (gia setup model)
        import torch  # fortwnei pytorch

        torch.manual_seed(self.seed)  # set seed gia torch
        np.random.seed(self.seed)  # set seed gia numpy

        self._torch = torch  # krataei reference sto torch module
        self._device = self._resolve_device(self.device)  # apofasizei cpu h cuda

        self._model, self._alphabet = self._load_model(self.model_name)  # fortwnei model kai alphabet
        self._model.eval()  # eval mode oxi training
        self._model.to(self._device)  # metaferei to model sto device
        self._batch_converter = self._alphabet.get_batch_converter()  # converter sequences se tokens

        # For esm2_t6_* models, last layer index is 6.  # gia t6 model teleutaio layer einai 6
        self._last_layer = int(getattr(self._model, "num_layers", 6))  # pairnei num_layers fallback 6

    def embed_sequences(self, sequences: Sequence[str], batch_size: int = 8) -> np.ndarray:  # embedding gia lista sequences
        """Embed a list of amino-acid sequences into an (N, d) float32 array."""  # vgazei array float32
        if batch_size <= 0:  # elegxos batch size
            raise ValueError("batch_size must be positive")  # prepei > 0

        embs: list[np.ndarray] = []  # lista gia ta batch embeddings
        for batch in _batched(sequences, batch_size):  # spaei se batches
            embs.append(self._embed_batch(batch))  # vgazei embeddings gia to batch kai ta krataei
        return np.concatenate(embs, axis=0)  # enwnei ola ta batches se ena array 

   
    # eswterikes voithitikes synarthseis

    def _embed_batch(self, sequences: Sequence[str]) -> np.ndarray:  # embedding gia ena batch sequences
        torch = self._torch  # local alias gia torch

        data = [(f"seq{i}", _truncate_aa(seq)) for i, seq in enumerate(sequences)]  # ftiaxnei (label, seq) kai kanei truncate
        _labels, _strs, tokens = self._batch_converter(data)  # metatrepei se tokens
        tokens = tokens.to(self._device)  # stelnei ta tokens sto device

        with torch.no_grad():  # xwris gradients pio grigoro kai ligoteri mnimi
            out = self._model(tokens, repr_layers=[self._last_layer], return_contacts=False)  # trexei to model kai zitaei representations
            reps = out["representations"][self._last_layer]  # pairnei reps tou teleutaiou layer

        # mean mono sta residue tokens 
        pad_idx = self._alphabet.padding_idx  # token id gia padding
        cls_idx = self._alphabet.cls_idx  # token id gia <cls>
        eos_idx = self._alphabet.eos_idx  # token id gia <eos>

        mask = (tokens != pad_idx) & (tokens != cls_idx) & (tokens != eos_idx)  # mask True mono sta kanonika residues
        # to kanoume 3D gia broadcast me reps
        mask_f = mask.unsqueeze(-1).to(reps.dtype)  # kanei (B,T,1) kai to kanei float idio dtype me reps
        summed = (reps * mask_f).sum(dim=1)  # athroizei reps sta valid tokens 
        counts = mask_f.sum(dim=1).clamp(min=1.0)  # posa valid tokens exei kathe seq
        pooled = summed / counts  # mesos oros 

        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)  # fernei se CPU, numpy, kai to kanei float32

    def _resolve_device(self, device: str) -> str:  # dialegei device me vasi to arg
        torch = self._torch  # alias torch
        if device == "auto":  # an auto
            return "cuda" if torch.cuda.is_available() else "cpu"  # an yparxei GPU -> cuda alliws cpu
        return device  # alliws gyrnaei oti edose o xristis 

    def _load_model(self, model_name: str):  # fortwnei to ESM2 model apo to esm library
        import esm 

        name = model_name.lower().strip()  # normalize to onoma
        if name in {"esm2_t6_8m_ur50d", "t6"}:  # epitreptes epiloges gia to t6 model
            return esm.pretrained.esm2_t6_8M_UR50D()  # fortwnei pretrained weights + alphabet
        raise ValueError(f"Unsupported model_name: {model_name}")  # error an dosoume allo model name


def _truncate_aa(seq: str, max_residues: int = 1022) -> str:  # kovei sequence sto max_residues
    s = seq.strip().replace(" ", "").upper()  # katharizei: trim, afairei spaces, kanei kefalaia
    if len(s) > max_residues:  # an einai poly makry
        return s[:max_residues]  # krataei mono ta prwta max_residues
    return s  # alliws gyrnaei olo


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:  # spaei mia lista se batches
    for i in range(0, len(items), batch_size):  # vhma batch_size
        yield items[i : i + batch_size]  # epistrefei kathe batch ws slice
