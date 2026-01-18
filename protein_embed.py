#!/usr/bin/env python3  # trexei to script me ton default python3 interpreter tou systimatos
"""
protein_embed.py
================
Auto to script ylopoiei to Step 1 tis ergasias.

Diavazei ena FASTA arxeio me proteins (database sequences) kai gia kathe protein
paragei ena embedding vector xrhsimopoiwntas to ESM-2 model.

- Fortwnei sequences apo to FASTA se batches (gia taxytita kai ligoteri mnhmh).
- Kalei ton ESM2Embedder gia na vgalei ena vector (embedding) gia kathe sequence.
- Enwnei ola ta batch embeddings se ena pinaka vectors shape (N, d).
- Apothikeuei ta vectors se binary NumPy arxeio (px .dat me periexomeno .npy)
  kai ftiaxnei dipla ena ids.txt pou exei ta protein IDs me tin idia seira.
"""
from __future__ import annotations  # epitrepei pio “xalara” type hints (forward refs, | None, etc.)

import argparse  # gia na paroume arguments apo to terminal (CLI)
import sys       # gia na peiraksoume to sys.path kai na kanoume local imports
from pathlib import Path  # safer paths (Linux/Windows) anti gia plain strings

_THIS_DIR = Path(__file__).resolve().parent  # to folder pou vrisketai auto to arxeio script
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
    # vazoume to project folder sto sys.path wste na douleuoun ta imports (protein_ann.*)
    # px an trexoume apo allo working directory, na min “xanetai” to package

from protein_ann.fasta import count_fasta_records, read_fasta  # FASTA parsing: metrame & stream records
from protein_ann.vectors import save_vectors  # apothikeusi vectors se .dat (npy content) + ids.txt


def parse_args() -> argparse.Namespace:
    # ftiaxnoume CLI parser: ti flags tha dexetai to script (input/output/model/etc.)
    parser = argparse.ArgumentParser(
        description="Assignment 3 - Step 1: Generate protein embeddings (ESM-2)."
    )

    parser.add_argument(
        "-i",                 # short flag
        "--input",            # long flag
        dest="input_fasta",   # to onoma pou tha exei mesa sto args (args.input_fasta)
        required=True,        # prepei na to dwsoume αλλιώς error
        help="Input FASTA file with database proteins (e.g. swissprot_50k.fasta)",
    )

    parser.add_argument(
        "-o",                 # short flag (output)
        "-0",                 # kapoies fores sto PDF fainetai -0 anti -o, opote to dexomaste kai etsi
        "--output",           # long flag
        dest="output_vectors",
        required=True,
        help="Output vectors file (binary NumPy, e.g. protein_vectors.dat)",
    )

    parser.add_argument(
        "-model",             # flag gia to ESM2 model name
        dest="model_name",
        default="esm2_t6_8M_UR50D",  # default variant (mikro/efkolo)
        help="ESM-2 model variant (default: esm2_t6_8M_UR50D)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,             # prepei na einai integer
        default=8,            # default batch size gia inference
        help="Batch size for embedding inference (default: 8)",
    )
    # batch-size = posa sequences per batch. megalytero -> pio grigoro, alla thelei pio poli RAM/VRAM

    parser.add_argument(
        "--device",
        default="auto",                   # an den dwsoume tipota, dialegει mono
        choices=["auto", "cpu", "cuda"],  # epitreptes times
        help="Device to use (default: auto)",
    )
    # device:
    # - cpu: trexei ston epeksergasti
    # - cuda: trexei se NVIDIA GPU (an yparxei)
    # - auto: an yparxei GPU -> cuda, alliws cpu

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    # seed = na vgazei idia apotelesmata stis tyxaies diadikasies

    parser.add_argument(
        "--overwrite",
        action="store_true",  # an yparxei flag, tote args.overwrite=True
        help="Overwrite output vectors file if it already exists",
    )
    # overwrite = an to output file yparxei, na to patas (alliws stamataei gia asfaleia)

    return parser.parse_args()  # epistrefei Namespace me ola ta args


def main() -> None:
    args = parse_args()  # diabazoume ta arguments pou edwse o xrhsths

    # Den kanoume import to ESM2Embedder apo tin arxi giati fortwnei  libraries (torch, transformers)
    # pou argooun kai katainalwnoun resources.
    # To kanoume import mono otan trexei ontws to script gia embeddings, gia pio grigoro start
    # kai gia na mporoun alla modules na to kanoun import xwris na fortwnoun GPU/torch.
    from protein_ann.esm2 import ESM2Embedder

    embedder = ESM2Embedder(
        model_name=args.model_name,  # poio ESM2 variant tha xrhsimopoiisoume
        device=args.device,          # cpu/cuda/auto
        seed=args.seed,              # random seed
    )

    out_path = Path(args.output_vectors)  # metatrepoume to output path se Path object
    if out_path.exists() and not args.overwrite:
        # an yparxei hdh output file kai den dwsame --overwrite -> stamataei gia na min xathoun palia apotelesmata
        raise FileExistsError(
            f"Output exists: {out_path}. Use --overwrite to replace it."
        )

    input_fasta = Path(args.input_fasta)  # input FASTA path
    if not input_fasta.exists():
        # an o user edwse lathos path, dokimazoume fallback sto datasets/ mesa sto project
        fallback = _THIS_DIR / "datasets" / "swissprot_50k.fasta"
        if fallback.exists():
            input_fasta = fallback  # xrhsimopoioume to default dataset
        else:
            # an oute to fallback yparxei, tote den yparxei input -> error
            raise FileNotFoundError(f"Input FASTA not found: {args.input_fasta}")

    total = count_fasta_records(input_fasta)  # posa proteins exei to FASTA (gia progress bar)

    try:
        from tqdm import tqdm  # progress bar (an yparxei installed)
    except Exception:  
        tqdm = None     # an den yparxei, den tha deiknoume progress

    ids: list[str] = []  # lista me IDs protein se seira (tha grapsei ids.txt)
    vec_chunks: list["__import__('numpy').ndarray"] = []  # embeddings chunks (NumPy arrays) apo kathe batch
    batch_ids: list[str] = []   # IDs gia to trexon batch
    batch_seqs: list[str] = []  # sequences gia to trexon batch

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total, desc="Embedding proteins", unit="seq")
        # desc = ti grafei, unit="seq" = metraei sequences

    # pername to FASTA record-record (den fortwnoume olo to file sti mnhmh)
    for rec in read_fasta(input_fasta):
        batch_ids.append(rec.id)         # prosthetoume ID tou record sto batch
        batch_seqs.append(rec.sequence)  # prosthetoume sequence tou record sto batch

        if len(batch_seqs) >= args.batch_size:
            # otan to batch gemisei, trexoume inference gia embeddings
            vec = embedder.embed_sequences(batch_seqs, batch_size=len(batch_seqs))
            # vec exei shape (batch, d) = ena vector ana sequence

            vec_chunks.append(vec)   # kratame to chunk gia meta concatenate
            ids.extend(batch_ids)    # kratame ta IDs me idia seira me ta vectors

            batch_ids, batch_seqs = [], []  # katharizoume to batch gia to epomeno

            if pbar is not None:
                pbar.update(vec.shape[0])  # ananeonoume progress kata posa sequences epeksergastikame

    # an emeine kapoio "misogemato" batch sto telos, to trexoume kai auto
    if batch_seqs:
        vec = embedder.embed_sequences(batch_seqs, batch_size=len(batch_seqs))
        vec_chunks.append(vec)
        ids.extend(batch_ids)
        if pbar is not None:
            pbar.update(vec.shape[0])

    if pbar is not None:
        pbar.close()  # kleinoume to progress bar gia na min meinei open

    import numpy as np  # local import: to fortwnoume mono edw giati to xreiazomaste mono sto telos
    # (gia concatenate + np.empty).
    # To vazoume mesa sti main gia na min ginetai import NumPy an kapoios kanei import to arxeio ws module
    # h an to script stamathsei nwris (px error sta args).
    # Epishs voithaei na meinei pio "elafry" to start tou script.
    vectors = (
        np.concatenate(vec_chunks, axis=0)  # enwnei ola ta batches se ena megalo array (N, d)
        if vec_chunks
        else np.empty((0, 0), dtype=np.float32)  # an den yparxei tipota , ftiaxnei adeio array
    )

    try:
        # apothikeuei:
        # - vectors sto out_path (binary .npy content me opoiadipote extension, px .dat)
        # - ids se ids.txt dipla sto output file
        save_vectors(out_path, vectors, ids)

        print(f"[protein_embed] Saved vectors: {out_path}")  # minima epityxias
        print(f"[protein_embed] Saved ids: {out_path.with_name('ids.txt')}")

    except OSError as e:
        # an apotyxei i eggrafi (px disk quota, disk full), den theloume na xathoun ta embeddings
        try:
            out_path.unlink(missing_ok=True)  # svinei to miso-graphmeno output file an yparxei
        except Exception:
            pass  # an den ginei unlink, synexizoume

        # fallback folder mesa sto HOME 
        fallback_dir = Path.home() / "assignment3_artifacts"
        fallback_dir.mkdir(parents=True, exist_ok=True)  # ftiaxnei folder an den yparxei
        fallback_path = fallback_dir / out_path.name     # idio onoma file, allo folder

        print(f"[protein_embed] ERROR: failed to write vectors to: {out_path}")
        print(f"[protein_embed] Underlying error: {e}")
        print(f"[protein_embed] Retrying save to fallback: {fallback_path}")

        # ksana-apothikeuei se fallback location
        save_vectors(fallback_path, vectors, ids)
        print(f"[protein_embed] Saved vectors (fallback): {fallback_path}")
        print(f"[protein_embed] Saved ids (fallback): {fallback_path.with_name('ids.txt')}")

        raise  # ksana petame to exception gia na fainetai oti to arxiko write apetyxe


if __name__ == "__main__":  
    # Auto einai to "entry point" tou arxeiou.
    # Otan trexoume to script direct apo terminal:
    #    python protein_embed.py ...
    # tote h Python vazei __name__ = "__main__" kai kaleitai h main().
    #
    # An omws to arxeio ginei import apo allo arxeio (px import protein_embed),
    # tote __name__ den einai "__main__", opote DEN trexei h main() automata.
    # Etσι apofevgoume na trexei o kwdikas kata lathos otan kanoume import.
    main()

