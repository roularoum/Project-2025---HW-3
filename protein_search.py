#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from protein_ann.fasta import read_fasta
from protein_ann.fasta import write_fasta
from protein_ann.metrics import recall_at_n
from protein_ann.vectors import load_vectors
from protein_ann.output_format import NeighborRow, SummaryRow, write_method_neighbors, write_neighbors_section_header, write_query_header, write_summary_table
from protein_ann.blast import ensure_blast_db, parse_blast_tabular, run_blastp, write_seqidlist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 3 - Step 2/3: ANN search benchmark + BLAST comparison."
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="vectors_path",
        required=True,
        help="Vectors file from protein_embed.py (e.g. protein_vectors.dat)",
    )
    parser.add_argument(
        "-q",
        "--queries",
        dest="queries_fasta",
        required=True,
        help="Query proteins FASTA file (e.g. targets.fasta)",
    )
    parser.add_argument(
        "-o",
        "-0",
        "--output",
        dest="output_path",
        required=True,
        help="Output results file (e.g. results.txt)",
    )
    parser.add_argument(
        "-method",
        dest="method",
        required=True,
        choices=[
            "all",
            "lsh",
            "hypercube",
            "neural",
            "ivf",
            "ivfflat",
            "ivfpq",
        ],
        help="ANN method to run (or 'all')",
    )
    parser.add_argument(
        "--recall-n",
        type=int,
        default=50,
        help="N for Recall@N vs BLAST Top-N (default: 50)",
    )
    parser.add_argument(
        "--print-topk",
        type=int,
        default=10,
        help="How many neighbors to print per method (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    parser.add_argument(
        "--blast-fasta",
        default="swissprot.fasta",
        help="FASTA used to build the BLAST DB (default: swissprot.fasta)",
    )
    parser.add_argument(
        "--blast-db-dir",
        default=".blast_db_cache",
        help="Directory for BLAST DB cache (default: .blast_db_cache)",
    )
    parser.add_argument(
        "--embed-model",
        default="esm2_t6_8M_UR50D",
        help="ESM-2 model variant for query embeddings (default: esm2_t6_8M_UR50D)",
    )
    parser.add_argument(
        "--embed-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for ESM-2 inference (default: auto)",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=8,
        help="Batch size for query embeddings (default: 8)",
    )

    # -----------------
    # Method parameters
    # -----------------
    # Defaults tuned on the provided SwissProt(50k)+targets set via a small grid search
    # (see protein_grid_search.py outputs) to balance Recall@N vs QPS.
    parser.add_argument("--lsh-k", type=int, default=6)
    parser.add_argument("--lsh-L", type=int, default=10)
    parser.add_argument("--lsh-w", type=float, default=4.0)

    parser.add_argument("--hc-kproj", type=int, default=14)
    parser.add_argument("--hc-w", type=float, default=4.0)
    parser.add_argument("--hc-M", type=int, default=1000)
    parser.add_argument("--hc-probes", type=int, default=20)

    parser.add_argument("--ivf-kclusters", type=int, default=100)
    parser.add_argument("--ivf-nprobe", type=int, default=5)

    parser.add_argument("--pq-M", type=int, default=16)
    parser.add_argument("--pq-nbits", type=int, default=8)

    parser.add_argument("--nlsh-m", type=int, default=200)
    parser.add_argument("--nlsh-T", type=int, default=5)
    parser.add_argument("--nlsh-epochs", type=int, default=10)
    parser.add_argument("--nlsh-layers", type=int, default=3)
    parser.add_argument("--nlsh-nodes", type=int, default=256)
    parser.add_argument("--nlsh-lr", type=float, default=1e-3)
    parser.add_argument("--nlsh-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np

    from protein_ann.esm2 import ESM2Embedder
    from protein_ann.ann import HypercubeIndex, IvfFlatIndex, IvfPqIndex, LshIndex, NeuralLshIndex
    from protein_ann.ann.hypercube import HypercubeParams
    from protein_ann.ann.ivf_flat import IvfFlatParams
    from protein_ann.ann.ivf_pq import IvfPqParams
    from protein_ann.ann.lsh import LshParams
    from protein_ann.ann.neural_lsh import NeuralLshParams
    from protein_ann.ann.utils import topk_smallest

    store = load_vectors(args.vectors_path)

    queries_fasta = Path(args.queries_fasta)
    if not queries_fasta.exists():
        fallback_q = _THIS_DIR / "datasets" / "targets.fasta"
        if fallback_q.exists():
            queries_fasta = fallback_q
        else:
            raise FileNotFoundError(f"Query FASTA not found: {args.queries_fasta}")

    queries = list(read_fasta(queries_fasta))

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not queries:
        raise ValueError("No queries found in FASTA")

    # -----------------------
    # Build ANN indexes once
    # -----------------------
    methods: list[tuple[str, object]] = []
    if args.method in {"all", "lsh"}:
        methods.append(
            (
                "Euclidean LSH",
                LshIndex(LshParams(k=args.lsh_k, L=args.lsh_L, w=args.lsh_w, seed=args.seed)),
            )
        )
    if args.method in {"all", "hypercube"}:
        methods.append(
            (
                "Hypercube",
                HypercubeIndex(
                    HypercubeParams(
                        kproj=args.hc_kproj,
                        w=args.hc_w,
                        M=args.hc_M,
                        probes=args.hc_probes,
                        seed=args.seed,
                    )
                ),
            )
        )
    if args.method in {"all", "neural"}:
        methods.append(
            (
                "Neural LSH",
                NeuralLshIndex(
                    NeuralLshParams(
                        m_bins=args.nlsh_m,
                        top_T=args.nlsh_T,
                        epochs=args.nlsh_epochs,
                        layers=args.nlsh_layers,
                        hidden_units=args.nlsh_nodes,
                        lr=args.nlsh_lr,
                        batch_size=args.nlsh_batch_size,
                        seed=args.seed,
                        device="auto",
                    )
                ),
            )
        )
    if args.method in {"all", "ivf", "ivfflat"}:
        methods.append(
            (
                "IVF-Flat",
                IvfFlatIndex(
                    IvfFlatParams(
                        kclusters=args.ivf_kclusters,
                        nprobe=args.ivf_nprobe,
                        seed=args.seed,
                    )
                ),
            )
        )
    if args.method in {"all", "ivf", "ivfpq"}:
        methods.append(
            (
                "IVF-PQ",
                IvfPqIndex(
                    IvfPqParams(
                        kclusters=args.ivf_kclusters,
                        nprobe=args.ivf_nprobe,
                        seed=args.seed,
                        pq_M=args.pq_M,
                        pq_nbits=args.pq_nbits,
                    )
                ),
            )
        )

    if not methods:
        raise ValueError(f"No methods selected for -method {args.method}")

    X = store.vectors
    ids = store.ids

    for name, index in methods:
        # Build can be heavy; keep output minimal but informative.
        print(f"[protein_search] Building index: {name} ...")
        index.build(X)  # type: ignore[attr-defined]

    # -----------------------
    # Embed queries (on-the-fly)
    # -----------------------
    embedder = ESM2Embedder(
        model_name=args.embed_model,
        device=args.embed_device,
        seed=args.seed,
    )
    q_ids = [q.id for q in queries]
    q_seqs = [q.sequence for q in queries]
    Q = embedder.embed_sequences(q_seqs, batch_size=args.embed_batch_size)

    # -----------------------
    # BLAST baseline (Top-N)
    # -----------------------
    blast_fasta = Path(args.blast_fasta)
    if not blast_fasta.exists():
        fallback = _THIS_DIR / "datasets" / "swissprot_50k.fasta"
        if fallback.exists():
            blast_fasta = fallback
        else:
            raise FileNotFoundError(
                f"BLAST FASTA not found: {args.blast_fasta}. "
                f"Provide --blast-fasta or place swissprot.fasta next to the script."
            )

    db_prefix = ensure_blast_db(blast_fasta, args.blast_db_dir)
    cache_dir = Path(args.blast_db_dir)
    blast_all_out = cache_dir / f"blast_all_top{int(args.recall_n)}.tsv"
    print("[protein_search] Running BLAST for all queries (Top-N ground truth) ...")
    t_blast_total = run_blastp(
        db_prefix=db_prefix,
        query_fasta=queries_fasta,
        out_path=blast_all_out,
        max_target_seqs=int(args.recall_n),
        seqidlist=None,
        evalue=1e6,
    )
    hits_by_q = parse_blast_tabular(blast_all_out)
    blast_time_per_query = t_blast_total / float(len(queries)) if (queries and t_blast_total > 0.0) else 0.0
    blast_qps = float(len(queries) / t_blast_total) if t_blast_total > 0.0 else 0.0

    # -----------------------
    # Write results
    # -----------------------
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for qi, qid in enumerate(q_ids):
            blast_hits = hits_by_q.get(qid, [])
            blast_top = [h.sseqid for h in blast_hits[: int(args.recall_n)]]
            blast_top_set = set(blast_top)

            # Run ANN searches + collect printed neighbors for identity lookup.
            summary: list[SummaryRow] = []
            per_method_printed: dict[str, list[tuple[str, float]]] = {}
            union_printed_ids: set[str] = set()

            qvec = Q[qi]
            for method_name, index in methods:
                t0 = time.perf_counter()
                res = index.query(qvec, k=int(args.recall_n))  # type: ignore[attr-defined]
                # Strict output: ensure we can always report exactly N neighbors if needed.
                if int(res.indices.size) < int(args.recall_n):
                    dist_sq_all = np.einsum("ij,ij->i", (X - qvec[None, :]), (X - qvec[None, :])).astype(np.float32, copy=False)
                    full_idx, full_dist = topk_smallest(dist_sq_all, int(args.recall_n))
                    res.indices = full_idx
                    res.distances = full_dist
                t1 = time.perf_counter()
                dt = t1 - t0
                qps = (1.0 / dt) if dt > 0.0 else 0.0

                ann_ids = [ids[int(i)] for i in res.indices.tolist()]
                rec = recall_at_n(ann_ids, blast_top, int(args.recall_n))
                summary.append(
                    SummaryRow(
                        method=method_name,
                        time_per_query_s=float(dt),
                        qps=float(qps),
                        recall_at_n=float(rec),
                    )
                )

                topk = min(int(args.print_topk), len(ann_ids))
                printed = [(ann_ids[r], float(res.distances[r])) for r in range(topk)]
                per_method_printed[method_name] = printed
                for pid, _ in printed:
                    union_printed_ids.add(pid)

            # Add BLAST reference row (average across all queries).
            summary.append(
                SummaryRow(
                    method="BLAST (Ref)",
                    time_per_query_s=float(blast_time_per_query),
                    qps=float(blast_qps),
                    recall_at_n=1.0,
                )
            )

            # BLAST identities for printed neighbors (restricted search).
            pident_by_id: dict[str, float] = {}
            if union_printed_ids:
                q_tmp = cache_dir / f"query_{qid}.fasta"
                write_fasta(q_tmp, [queries[qi]])
                ids_tmp = cache_dir / f"seqidlist_{qid}.txt"
                write_seqidlist(ids_tmp, sorted(union_printed_ids))
                blast_cand_out = cache_dir / f"blast_candidates_{qid}.tsv"
                run_blastp(
                    db_prefix=db_prefix,
                    query_fasta=q_tmp,
                    out_path=blast_cand_out,
                    max_target_seqs=max(1, len(union_printed_ids)),
                    seqidlist=ids_tmp,
                )
                cand_hits = parse_blast_tabular(blast_cand_out).get(qid, [])
                # Keep best (highest bitscore) per subject.
                for h in cand_hits:
                    if h.sseqid not in pident_by_id:
                        pident_by_id[h.sseqid] = h.pident

            # Write per-query section.
            write_query_header(out, qid, int(args.recall_n))
            write_summary_table(out, summary)
            write_neighbors_section_header(out, int(args.print_topk))

            # Detailed neighbor tables per method.
            for method_name, _index in methods:
                printed = per_method_printed.get(method_name, [])
                rows: list[NeighborRow] = []
                for rank, (nid, l2) in enumerate(printed, start=1):
                    rows.append(
                        NeighborRow(
                            rank=rank,
                            neighbor_id=nid,
                            l2_dist=float(l2),
                            blast_identity_pct=pident_by_id.get(nid),
                            in_blast_top_n=(nid in blast_top_set),
                            bio_comment="--",
                        )
                    )
                write_method_neighbors(out, method_name, rows)
            out.write("\n")

    print(f"[protein_search] Wrote results to: {out_path}")


if __name__ == "__main__":
    main()

