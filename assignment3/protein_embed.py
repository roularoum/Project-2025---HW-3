#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from protein_ann.fasta import count_fasta_records, read_fasta
from protein_ann.vectors import save_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 3 - Step 1: Generate protein embeddings (ESM-2)."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_fasta",
        required=True,
        help="Input FASTA file with database proteins (e.g. swissprot_50k.fasta)",
    )
    parser.add_argument(
        "-o",
        "-0",  # some PDF renderings show -0
        "--output",
        dest="output_vectors",
        required=True,
        help="Output vectors file (binary NumPy, e.g. protein_vectors.dat)",
    )
    parser.add_argument(
        "-model",
        dest="model_name",
        default="esm2_t6_8M_UR50D",
        help="ESM-2 model variant (default: esm2_t6_8M_UR50D)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding inference (default: 8)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output vectors file if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Deferred import: heavy deps are only needed when running embedding.
    from protein_ann.esm2 import ESM2Embedder

    embedder = ESM2Embedder(
        model_name=args.model_name,
        device=args.device,
        seed=args.seed,
    )

    out_path = Path(args.output_vectors)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {out_path}. Use --overwrite to replace it."
        )

    input_fasta = Path(args.input_fasta)
    if not input_fasta.exists():
        fallback = _THIS_DIR / "datasets" / "swissprot_50k.fasta"
        if fallback.exists():
            input_fasta = fallback
        else:
            raise FileNotFoundError(f"Input FASTA not found: {args.input_fasta}")

    total = count_fasta_records(input_fasta)
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        tqdm = None

    ids: list[str] = []
    vec_chunks: list["__import__('numpy').ndarray"] = []
    batch_ids: list[str] = []
    batch_seqs: list[str] = []

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total, desc="Embedding proteins", unit="seq")

    for rec in read_fasta(input_fasta):
        batch_ids.append(rec.id)
        batch_seqs.append(rec.sequence)
        if len(batch_seqs) >= args.batch_size:
            vec = embedder.embed_sequences(batch_seqs, batch_size=len(batch_seqs))
            vec_chunks.append(vec)
            ids.extend(batch_ids)
            batch_ids, batch_seqs = [], []
            if pbar is not None:
                pbar.update(vec.shape[0])

    if batch_seqs:
        vec = embedder.embed_sequences(batch_seqs, batch_size=len(batch_seqs))
        vec_chunks.append(vec)
        ids.extend(batch_ids)
        if pbar is not None:
            pbar.update(vec.shape[0])

    if pbar is not None:
        pbar.close()

    import numpy as np

    vectors = (
        np.concatenate(vec_chunks, axis=0)
        if vec_chunks
        else np.empty((0, 0), dtype=np.float32)
    )

    save_vectors(out_path, vectors, ids)
    print(f"[protein_embed] Saved vectors: {out_path}")
    print(f"[protein_embed] Saved ids: {out_path.with_name('ids.txt')}")


if __name__ == "__main__":
    main()

