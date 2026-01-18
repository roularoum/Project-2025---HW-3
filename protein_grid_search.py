#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from protein_ann.blast import ensure_blast_db, parse_blast_tabular, run_blastp
from protein_ann.fasta import read_fasta
from protein_ann.metrics import mean, recall_at_n
from protein_ann.vectors import load_vectors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assignment 3 - Grid search for ANN hyperparameters (Recall@N vs QPS)."
    )
    p.add_argument("-d", "--data", dest="vectors_path", required=True)
    p.add_argument("-q", "--queries", dest="queries_fasta", required=True)
    p.add_argument("-o", "--output", dest="output_csv", required=True)
    p.add_argument(
        "--method",
        choices=["lsh", "hypercube", "ivfflat", "ivfpq", "neural"],
        required=True,
        help="Method to grid-search",
    )
    p.add_argument("--recall-n", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--blast-fasta",
        default="swissprot.fasta",
        help="FASTA used to build the BLAST DB (default: swissprot.fasta)",
    )
    p.add_argument(
        "--blast-db-dir",
        default=".blast_db_cache",
        help="Directory for BLAST DB cache (default: .blast_db_cache)",
    )

    # Query embeddings
    p.add_argument("--embed-model", default="esm2_t6_8M_UR50D")
    p.add_argument("--embed-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--embed-batch-size", type=int, default=8)

    # Grids (comma-separated)
    p.add_argument("--lsh-k-grid", default="4")
    p.add_argument("--lsh-L-grid", default="5")
    p.add_argument("--lsh-w-grid", default="4.0")

    p.add_argument("--hc-kproj-grid", default="14")
    p.add_argument("--hc-w-grid", default="4.0")
    p.add_argument("--hc-M-grid", default="1000")
    p.add_argument("--hc-probes-grid", default="10")

    p.add_argument("--ivf-kclusters-grid", default="50")
    p.add_argument("--ivf-nprobe-grid", default="5")

    p.add_argument("--pq-M-grid", default="16")
    p.add_argument("--pq-nbits-grid", default="8")

    p.add_argument("--nlsh-m-grid", default="100")
    p.add_argument("--nlsh-T-grid", default="5")
    p.add_argument("--nlsh-epochs", type=int, default=10)
    p.add_argument("--nlsh-layers", type=int, default=3)
    p.add_argument("--nlsh-nodes", type=int, default=256)
    p.add_argument("--nlsh-lr", type=float, default=1e-3)
    p.add_argument("--nlsh-batch-size", type=int, default=256)
    return p.parse_args()


def _parse_int_grid(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    from protein_ann.esm2 import ESM2Embedder
    from protein_ann.ann.hypercube import HypercubeIndex, HypercubeParams
    from protein_ann.ann.ivf_flat import IvfFlatIndex, IvfFlatParams
    from protein_ann.ann.ivf_pq import IvfPqIndex, IvfPqParams
    from protein_ann.ann.lsh import LshIndex, LshParams
    from protein_ann.ann.neural_lsh import NeuralLshIndex, NeuralLshParams

    store = load_vectors(args.vectors_path)
    X = store.vectors
    ids = store.ids

    queries_fasta = Path(args.queries_fasta)
    if not queries_fasta.exists():
        fallback_q = _THIS_DIR / "datasets" / "targets.fasta"
        if fallback_q.exists():
            queries_fasta = fallback_q
        else:
            raise FileNotFoundError(f"Query FASTA not found: {args.queries_fasta}")

    queries = list(read_fasta(queries_fasta))
    if not queries:
        raise ValueError("No queries found")
    q_ids = [q.id for q in queries]
    q_seqs = [q.sequence for q in queries]

    embedder = ESM2Embedder(model_name=args.embed_model, device=args.embed_device, seed=args.seed)
    Q = embedder.embed_sequences(q_seqs, batch_size=args.embed_batch_size)

    blast_fasta = Path(args.blast_fasta)
    if not blast_fasta.exists():
        fallback = _THIS_DIR / "datasets" / "swissprot_50k.fasta"
        if fallback.exists():
            blast_fasta = fallback
        else:
            raise FileNotFoundError(f"BLAST FASTA not found: {args.blast_fasta}")
    db_prefix = ensure_blast_db(blast_fasta, args.blast_db_dir)
    cache_dir = Path(args.blast_db_dir)
    blast_out = cache_dir / f"blast_all_top{int(args.recall_n)}.tsv"
    if not blast_out.exists():
        run_blastp(db_prefix=db_prefix, query_fasta=queries_fasta, out_path=blast_out, max_target_seqs=int(args.recall_n))
    hits_by_q = parse_blast_tabular(blast_out)

    blast_top_by_q: dict[str, list[str]] = {}
    for qid in q_ids:
        hits = hits_by_q.get(qid, [])
        blast_top_by_q[qid] = [h.sseqid for h in hits[: int(args.recall_n)]]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare grid.
    rows: list[dict[str, object]] = []

    if args.method == "lsh":
        grid = itertools.product(
            _parse_int_grid(args.lsh_k_grid),
            _parse_int_grid(args.lsh_L_grid),
            _parse_float_grid(args.lsh_w_grid),
        )
        for k, L, w in grid:
            idx = LshIndex(LshParams(k=k, L=L, w=w, seed=args.seed))
            rows.extend(_eval_index(idx, "Euclidean LSH", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n), {"k": k, "L": L, "w": w}))

    elif args.method == "hypercube":
        grid = itertools.product(
            _parse_int_grid(args.hc_kproj_grid),
            _parse_float_grid(args.hc_w_grid),
            _parse_int_grid(args.hc_M_grid),
            _parse_int_grid(args.hc_probes_grid),
        )
        for kproj, w, M, probes in grid:
            idx = HypercubeIndex(HypercubeParams(kproj=kproj, w=w, M=M, probes=probes, seed=args.seed))
            rows.extend(_eval_index(idx, "Hypercube", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n), {"kproj": kproj, "w": w, "M": M, "probes": probes}))

    elif args.method == "ivfflat":
        grid = itertools.product(
            _parse_int_grid(args.ivf_kclusters_grid),
            _parse_int_grid(args.ivf_nprobe_grid),
        )
        for kclusters, nprobe in grid:
            idx = IvfFlatIndex(IvfFlatParams(kclusters=kclusters, nprobe=nprobe, seed=args.seed))
            rows.extend(_eval_index(idx, "IVF-Flat", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n), {"kclusters": kclusters, "nprobe": nprobe}))

    elif args.method == "ivfpq":
        grid = itertools.product(
            _parse_int_grid(args.ivf_kclusters_grid),
            _parse_int_grid(args.ivf_nprobe_grid),
            _parse_int_grid(args.pq_M_grid),
            _parse_int_grid(args.pq_nbits_grid),
        )
        for kclusters, nprobe, M, nbits in grid:
            idx = IvfPqIndex(IvfPqParams(kclusters=kclusters, nprobe=nprobe, pq_M=M, pq_nbits=nbits, seed=args.seed))
            rows.extend(_eval_index(idx, "IVF-PQ", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n), {"kclusters": kclusters, "nprobe": nprobe, "M": M, "nbits": nbits}))

    elif args.method == "neural":
        grid = itertools.product(
            _parse_int_grid(args.nlsh_m_grid),
            _parse_int_grid(args.nlsh_T_grid),
        )
        for m_bins, T in grid:
            idx = NeuralLshIndex(
                NeuralLshParams(
                    m_bins=m_bins,
                    top_T=T,
                    epochs=int(args.nlsh_epochs),
                    layers=int(args.nlsh_layers),
                    hidden_units=int(args.nlsh_nodes),
                    lr=float(args.nlsh_lr),
                    batch_size=int(args.nlsh_batch_size),
                    seed=int(args.seed),
                    device="auto",
                )
            )
            rows.extend(_eval_index(idx, "Neural LSH", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n), {"m": m_bins, "T": T}))

    # Write CSV
    if not rows:
        raise RuntimeError("No grid rows produced")

    fieldnames = sorted(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[grid] Wrote {len(rows)} rows to: {out_path}")


def _eval_index(
    index,
    method_name: str,
    X: "np.ndarray",
    ids: list[str],
    Q: "np.ndarray",
    q_ids: list[str],
    blast_top_by_q: dict[str, list[str]],
    recall_n: int,
    params_dict: dict[str, object],
) -> list[dict[str, object]]:
    # Build
    t0 = time.perf_counter()
    index.build(X)
    t1 = time.perf_counter()
    build_s = t1 - t0

    times: list[float] = []
    recalls: list[float] = []
    for qi, qid in enumerate(q_ids):
        qvec = Q[qi]
        s0 = time.perf_counter()
        res = index.query(qvec, k=recall_n)
        s1 = time.perf_counter()
        times.append(s1 - s0)

        ann_ids = [ids[int(i)] for i in res.indices.tolist()]
        recalls.append(recall_at_n(ann_ids, blast_top_by_q.get(qid, []), recall_n))

    total = float(sum(times))
    qps = float(len(q_ids) / total) if total > 0 else 0.0
    avg_time = float(total / len(q_ids)) if q_ids else 0.0
    avg_recall = float(mean(recalls))

    row: dict[str, object] = {
        "method": method_name,
        "recall_n": recall_n,
        "avg_recall": avg_recall,
        "avg_time_per_query_s": avg_time,
        "qps": qps,
        "build_time_s": float(build_s),
    }
    for k, v in params_dict.items():
        row[f"param_{k}"] = v
    return [row]


if __name__ == "__main__":
    main()

