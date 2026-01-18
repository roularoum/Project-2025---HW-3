"""
vectors.py
==========
To arxeio auto xeirizetai apothikefsi kai fortwsi twn embeddings vectors (NumPy arrays)
mazi me to mapping twn protein IDs.
- save_vectors: grafei ta vectors se arxeio (p.x. protein_vectors.dat) me periexomeno .npy
  kai grafei kai ids.txt dipla (1 id ana grammi).
- load_vectors: fortwnei ta vectors kai ta ids kai ta epistrefei se VectorStore.
open_memmap: grafei to array se disk ws .npy "siga-siga" (chunks), 
wste na min kollaei se file systems/low space

"""

from __future__ import annotations  # type hints me forward refs / kaluteri symperifora

from dataclasses import dataclass  # dataclass 
from pathlib import Path  # Path = asfales paths (Linux/Windows)
from typing import Sequence  # Sequence = list/tuple me indexing (ids)

import numpy as np  # NumPy gia arrays kai save/load
from numpy.lib.format import open_memmap  # open_memmap grafei .npy arxeio via memmap (chunked/robust)


@dataclass(frozen=True)  # frozen=True = den allazei meta 
class VectorStore:
    """In-memory vector store with ID mapping."""  # store vectors + antistoixa ids

    vectors: np.ndarray  # shape: (n, d), float32 (n proteins, d dimensions embedding)
    ids: list[str]  # lista me ids, mikos n (1 id ana grammi/vector)

    def __post_init__(self) -> None:
        # trexei meta to init, gia na kanei validations/elenxous
        if self.vectors.ndim != 2:  # theloume 2D pinaka: grammes=proteins, stiles=features
            raise ValueError("vectors must be 2D (n, d)")
        if self.vectors.shape[0] != len(self.ids):  # prepei na tairiazei #vectors me #ids
            raise ValueError("vectors rows must match ids length")


def _default_ids_path(vectors_path: Path) -> Path:
    # ftiaxnei to default path gia to ids file (dipla sto vectors file)
     # dhladi pantote "ids.txt", oxi "protein_vectors.dat.ids"
    return vectors_path.with_name("ids.txt")  # allazei mono to onoma arxeiou, krataei idio folder


def save_vectors(vectors_path: str | Path, vectors: np.ndarray, ids: Sequence[str]) -> Path:
    """"
    Apothikeuei ta vectors se δυαδικο arxeio NumPy sto *akribws* path pou dinetai (vectors_path).

    Stin ergasia xrisimopoieitai katali3i `.dat`, alla to periexomeno einai kanoniko format `.npy`.
    Theloume na apofygoume to NumPy na prosthesei mono tou `.npy` sto filename,
    opote grafoume to arxeio me file handle / memmap.

   se kapoia filesystems (eidika WSL DrvFS kai otan exei ligonxwro),
    to `np.save` mporei na apotyxei me `OSError` pou erxetai apo to `ndarray.tofile`.
    H xrhsh `.npy` memmap writer apofeugei auto to provlima giati grafei ta dedomena
    tmhmatika (se mikra chunks/selides), kai oxi ola mazi.
    """
    # eksigei oti grafei .npy periexomeno se .dat kai giati xrhsimopoiei memmap

    p = Path(vectors_path)  # metatrepei to path se Path object
    p.parent.mkdir(parents=True, exist_ok=True)  # dimiourgei folder an den yparxei (me parents)

    vec = np.asarray(vectors, dtype=np.float32)  # metatrepei se NumPy array float32 (statheros typos)
    if vec.ndim != 2:  # prepei na einai 2D (n,d)
        raise ValueError("vectors must be 2D (n, d)")

    ids_list = list(map(str, ids))  # kanoume ola ta ids strings kai ta krataμε se lista
    if vec.shape[0] != len(ids_list):  # elegxos oti exoume 1 id ana vector
        raise ValueError("vectors rows must match ids length")

    # grafei proper .npy format se opoiadipote extension
    # dhladi meta kanoume np.load kai douleuei
    if p.exists():  # an yparxei idi arxeio me idio onoma
        p.unlink()  # to svinoume gia na min mpei conflict me memmap write

    mm = open_memmap(p, mode="w+", dtype=np.float32, shape=vec.shape)  # dimiourgei memmap .npy file me to idio shape
    # grafei se kommatia gia na min "skasei" se disk/network
    chunk_rows = 8192  # posa vectors (grammes) grafoume ana batch ~10MB write per chunk - 2^13 - sunhthismeno gia mnhmh
    for i0 in range(0, vec.shape[0], chunk_rows):  # loop apo 0 mexri n me vima 8192
        i1 = min(vec.shape[0], i0 + chunk_rows)  # teliko index gia to chunk (min gia na min perasei to n)
        mm[i0:i1] = vec[i0:i1]  # antigrafi chunk apo RAM array -> sto memmap file

    mm.flush()  # sigourevei oti grafthikan sto disk
    del mm  # kleinoume to memmap object (apodesmeusi resources)

    ids_path = _default_ids_path(p)  # path gia ids.txt sto idio folder
    with ids_path.open("w", encoding="utf-8", newline="\n") as f:  # grafei ids me utf-8 kai LF newlines
        for pid in ids_list:  # gia kathe protein id
            f.write(pid + "\n")  # grafei 1 id ana grammi

    return p  # epistrefei to Path tou vectors arxeiou pou swthike


def load_vectors(vectors_path: str | Path, ids_path: str | Path | None = None) -> VectorStore:
    """Load vectors and IDs.
    """  # fortwnei h .dat/.npy h .npz kai pairnei vectors + ids

    p = Path(vectors_path)  # path -> Path object
    if not p.exists():  # an den yparxei to vectors file
        raise FileNotFoundError(str(p))  # petame error

    loaded = np.load(p, allow_pickle=False)  # fortwnei to arxeio 
    if isinstance(loaded, np.lib.npyio.NpzFile):  # an einai .npz (zip me polla arrays)
        if "vectors" not in loaded.files:  # prepei na yparxei key "vectors"
            raise ValueError(f"NPZ missing 'vectors': {p}")

        vec = np.asarray(loaded["vectors"], dtype=np.float32)  # fortwnei vectors kai ta kanei float32 (mikroteros xoros/embeddings den thelun tosh akribeia/etsi bazoume sta ANN indexes->grhgoro)

        if "ids" in loaded.files:  # an yparxoun kai ids mesa sto npz
            raw_ids = loaded["ids"]  # pairnei to array ids
            ids_list = [str(x) for x in raw_ids.tolist()]  # to metatrepei se python list apo strings
        else:
            if ids_path is None:  # an den dothike ids_path apo caller
                ids_path = _default_ids_path(p)  # tote psaxnoume to default ids.txt dipla
            ids_list = _load_ids(ids_path)  # fortwnoume ids apo to file

        return VectorStore(vectors=vec, ids=ids_list)  # epistrefoume store (vectors + ids)

    vec = np.asarray(loaded, dtype=np.float32)  # an den einai npz, einai απλό .npy array
    if ids_path is None:  # an den mas dothike ids file
        ids_path = _default_ids_path(p)  # default: ids.txt sto idio folder
        # palia version mporei na eixe allo onoma <vectors>.ids.txt
        if not Path(ids_path).exists():  # an den yparxei to ids.txt
            alt = Path(str(p) + ".ids.txt")  # dokimazoume protein_vectors.dat.ids.txt
            if alt.exists():  # an yparxei
                ids_path = alt  # to xrhsimopoioume

    ids_list = _load_ids(ids_path)  # fortwnoume ids apo arxeio
    return VectorStore(vectors=vec, ids=ids_list)  # ftiaxnoume kai epistrefoume store


def _load_ids(ids_path: str | Path) -> list[str]:
    p = Path(ids_path)  # metatrepei ids_path se Path
    if not p.exists():  # an den yparxei to arxeio ids
        raise FileNotFoundError(f"Missing ids mapping file: {p}")  # petame error gia na kserei o user ti leipei

    with p.open("r", encoding="utf-8") as f:  # anoigei gia diavasma
        ids = [ln.strip() for ln in f if ln.strip()]  # diavazei grammes, kanei strip, kai petaei adeies grammes

    if not ids:  # an to file einai adeio
        raise ValueError(f"Empty ids file: {p}")  # petame error giati den exoume mapping

    return ids  # epistrefoume lista me ids
