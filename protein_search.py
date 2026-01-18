#!/usr/bin/env python3  
from __future__ import annotations  # epitrepei pio eukola type hints me forward refs

import argparse  # gia command line arguments CLI
import sys  # gia sys.path
import time  # gia xronometra 
from pathlib import Path  

_THIS_DIR = Path(__file__).resolve().parent  # to folder pou vrisketai to trexon arxeio
if str(_THIS_DIR) not in sys.path:  # an den einai sto python path
    sys.path.insert(0, str(_THIS_DIR))  # to vazei prwto gia na vrei ta local modules

from protein_ann.fasta import read_fasta  # diavazei FASTA records id + sequence
from protein_ann.fasta import write_fasta  # grafei FASTA records se arxeio
from protein_ann.metrics import recall_at_n  # ypologizei recall@N
from protein_ann.vectors import load_vectors  # fortwnei embeddings store vectors + ids
from protein_ann.output_format import NeighborRow, SummaryRow, write_method_neighbors, write_neighbors_section_header, write_query_header, write_summary_table  # gia output txt
from protein_ann.blast import ensure_blast_db, parse_blast_tabular, run_blastp, write_seqidlist  # BLAST: DB, run, parse, kai seqidlist file

 # orizei ola ta CLI arguments
def parse_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(  # ftiaxnei argument parser
        description="Assignment 3 - Step 2/3: ANN search benchmark + BLAST comparison."  # benchmark ANN kai sugkrisi me BLAST
    )
    parser.add_argument(  # input vectors dataset
        "-d",
        "--data",
        dest="vectors_path",
        required=True,
        help="Vectors file from protein_embed.py (e.g. protein_vectors.dat)",
    )
    parser.add_argument(  # input queries fasta
        "-q",
        "--queries",
        dest="queries_fasta",
        required=True,
        help="Query proteins FASTA file (e.g. targets.fasta)",
    )
    parser.add_argument(  # output results arxeio
        "-o",
        "-0",
        "--output",
        dest="output_path",
        required=True,
        help="Output results file (e.g. results.txt)",
    )
    parser.add_argument(  # epilogi methodou ANN
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
    parser.add_argument(  # N gia Recall@N 
        "--recall-n",
        type=int,
        default=50,
        help="N for Recall@N vs BLAST Top-N (default: 50)",
    )
    parser.add_argument(  # posa neighbors na typwsei ana methodo
        "--print-topk",
        type=int,
        default=10,
        help="How many neighbors to print per method (default: 10)",
    )
    parser.add_argument(  # seed gia reproducibility
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    parser.add_argument(  # fasta pou xrisimopoiei o BLAST gia database
        "--blast-fasta",
        default="swissprot.fasta",
        help="FASTA used to build the BLAST DB (default: swissprot.fasta)",
    )
    parser.add_argument(  # folder pou kratame BLAST cache db + outputs
        "--blast-db-dir",
        default=".blast_db_cache",
        help="Directory for BLAST DB cache (default: .blast_db_cache)",
    )
    parser.add_argument(  # ESM2 model gia query embeddings
        "--embed-model",
        default="esm2_t6_8M_UR50D",
        help="ESM-2 model variant for query embeddings (default: esm2_t6_8M_UR50D)",
    )
    parser.add_argument(  # device pou tha trexei i ektelesi tou model (cpu/cuda/auto)
        "--embed-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for ESM-2 inference (default: auto)",
    )
    parser.add_argument(  # batch size gia query embeddings
        "--embed-batch-size",
        type=int,
        default=8,
        help="Batch size for query embeddings (default: 8)",
    )

    
    parser.add_argument("--lsh-k", type=int, default=6)  # LSH: k hashes ana pinaka
    parser.add_argument("--lsh-L", type=int, default=10)  # LSH: L pinakes hash tables
    parser.add_argument("--lsh-w", type=float, default=4.0)  # LSH: bucket width 

    parser.add_argument("--hc-kproj", type=int, default=14)  # Hypercube: posa projections/bits
    parser.add_argument("--hc-w", type=float, default=4.0)  # Hypercube: quantization width
    parser.add_argument("--hc-M", type=int, default=1000)  # Hypercube: max candidates pou elegxoume
    parser.add_argument("--hc-probes", type=int, default=20)  # Hypercube: posa geitonika vertices psaxnoume

    parser.add_argument("--ivf-kclusters", type=int, default=100)  # IVF: posa clusters 
    parser.add_argument("--ivf-nprobe", type=int, default=5)  # IVF: posa clusters psaxnoume sto query

    parser.add_argument("--pq-M", type=int, default=16)  # PQ: posa sub-vectors
    parser.add_argument("--pq-nbits", type=int, default=8)  # PQ: bits ana subvector

    parser.add_argument("--nlsh-m", type=int, default=200)  # Neural LSH: poses katigories
    parser.add_argument("--nlsh-T", type=int, default=5)  # Neural LSH: top T candidates
    parser.add_argument("--nlsh-epochs", type=int, default=10)  # epochs training
    parser.add_argument("--nlsh-layers", type=int, default=3)  # posa layers sto network
    parser.add_argument("--nlsh-nodes", type=int, default=256)  # neurons ana layer
    parser.add_argument("--nlsh-lr", type=float, default=1e-3)  # learning rate
    parser.add_argument("--nlsh-batch-size", type=int, default=256)  # batch size training
    return parser.parse_args()  # epistrefei ta args

# load data, build indexes, run queries, compare me BLAST, write report
def main() -> None:  
    args = parse_args()  # parse CLI args

    import numpy as np  # numpy gia ypologismous 

    from protein_ann.esm2 import ESM2Embedder  # embedder gia proteins 
    from protein_ann.ann import HypercubeIndex, IvfFlatIndex, IvfPqIndex, LshIndex, NeuralLshIndex  # classes indices
    from protein_ann.ann.hypercube import HypercubeParams  # params hypercube
    from protein_ann.ann.ivf_flat import IvfFlatParams  # params IVF-Flat
    from protein_ann.ann.ivf_pq import IvfPqParams  # params IVF-PQ
    from protein_ann.ann.lsh import LshParams  # params LSH
    from protein_ann.ann.neural_lsh import NeuralLshParams  # params Neural LSH
    from protein_ann.ann.utils import topk_smallest  # pare top-k mikrotera

    store = load_vectors(args.vectors_path)  # fortwnei to vector store dataset embeddings

    queries_fasta = Path(args.queries_fasta)  # path tou query fasta
    if not queries_fasta.exists():  # an den yparxei sto path pou edoses
        fallback_q = _THIS_DIR / "datasets" / "targets.fasta"  # fallback queries
        if fallback_q.exists():  # an yparxei fallback
            queries_fasta = fallback_q  # xrisimopoiei fallback
        else:
            raise FileNotFoundError(f"Query FASTA not found: {args.queries_fasta}")  # error an den vrethei

    queries = list(read_fasta(queries_fasta))  # diavazei ola ta query records

    out_path = Path(args.output_path)  # path gia output results
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ftiaxnei folder output an leipei

    if not queries:  # an den exei queries
        raise ValueError("No queries found in FASTA")  # error

   
    # xtizoyme indices mia fora oxi ana query
    methods: list[tuple[str, object]] = []  # lista: onoma methodou, index object
    if args.method in {"all", "lsh"}:  # an trexoume LSH
        methods.append(  # prosthetei LSH stis methods
            (
                "Euclidean LSH",  # label pou tha fanei sto report
                LshIndex(LshParams(k=args.lsh_k, L=args.lsh_L, w=args.lsh_w, seed=args.seed)),  # index me params
            )
        )
    if args.method in {"all", "hypercube"}:  # an trexoume Hypercube
        methods.append(
            (
                "Hypercube",  # label
                HypercubeIndex(
                    HypercubeParams(
                        kproj=args.hc_kproj,  # projections/bits
                        w=args.hc_w,  # width
                        M=args.hc_M,  # max candidates
                        probes=args.hc_probes,  # geitonika vertices
                        seed=args.seed,  # seed
                    )
                ),
            )
        )
    if args.method in {"all", "neural"}:  # an trexoume Neural LSH
        methods.append(
            (
                "Neural LSH",  # label
                NeuralLshIndex(
                    NeuralLshParams(
                        m_bins=args.nlsh_m,  # posa bins
                        top_T=args.nlsh_T,  # top T
                        epochs=args.nlsh_epochs,  # epochs
                        layers=args.nlsh_layers,  # layers
                        hidden_units=args.nlsh_nodes,  # neurons
                        lr=args.nlsh_lr,  # learning rate
                        batch_size=args.nlsh_batch_size,  # batch size
                        seed=args.seed,  # seed
                        device="auto",  # device (auto)
                    )
                ),
            )
        )
    if args.method in {"all", "ivf", "ivfflat"}:  # an trexoume IVF-Flat
        methods.append(
            (
                "IVF-Flat",  # label
                IvfFlatIndex(
                    IvfFlatParams(
                        kclusters=args.ivf_kclusters,  # posa clusters
                        nprobe=args.ivf_nprobe,  # posa clusters psaxnei
                        seed=args.seed,  # seed
                    )
                ),
            )
        )
    if args.method in {"all", "ivf", "ivfpq"}:  # an trexoume IVF-PQ 
        methods.append(
            (
                "IVF-PQ",  # label
                IvfPqIndex(
                    IvfPqParams(
                        kclusters=args.ivf_kclusters,  # clusters
                        nprobe=args.ivf_nprobe,  # probes
                        seed=args.seed,  # seed
                        pq_M=args.pq_M,  # PQ M
                        pq_nbits=args.pq_nbits,  # PQ nbits
                    )
                ),
            )
        )

    if not methods:  # an den epilextike kamia methodos
        raise ValueError(f"No methods selected for -method {args.method}")  # error

    X = store.vectors  # dataset vectors
    ids = store.ids  # dataset ids

    for name, index in methods:  # build kathe index mia fora
        # to build mporei na einai bary typwnoume mono ta aparaithta
        print(f"[protein_search] Building index: {name} ...")  # print progress
        index.build(X) # xtizei to index panw sto dataset

    # ypologizoume embeddings twn queries
    embedder = ESM2Embedder(  # ftiaxnei embedder
        model_name=args.embed_model,  # poio modelo
        device=args.embed_device,  # poio device
        seed=args.seed,  # seed
    )
    q_ids = [q.id for q in queries]  # ids twn queries
    q_seqs = [q.sequence for q in queries]  # sequences twn queries
    Q = embedder.embed_sequences(q_seqs, batch_size=args.embed_batch_size)  # query embeddings

    
    # trexoume BLAST gia ground truth
    blast_fasta = Path(args.blast_fasta)  # fasta pou ginetai BLAST DB
    if not blast_fasta.exists():  # an den yparxei
        fallback = _THIS_DIR / "datasets" / "swissprot_50k.fasta"  # fallback corpus
        if fallback.exists():  # an yparxei fallback
            blast_fasta = fallback  # xrisimopoiei fallback
        else:
            raise FileNotFoundError(  # error me odigies
                f"BLAST FASTA not found: {args.blast_fasta}. "
                f"Provide --blast-fasta or place swissprot.fasta next to the script."
            )

    db_prefix = ensure_blast_db(blast_fasta, args.blast_db_dir)  # ftiaxnei BLAST DB kai gyrnaei prefix
    cache_dir = Path(args.blast_db_dir)  # folder cache BLAST
    blast_all_out = cache_dir / f"blast_all_top{int(args.recall_n)}.tsv"  # output file gia BLAST top-N
    print("[protein_search] Running BLAST for all queries (Top-N ground truth) ...")  # print progress
    t_blast_total = run_blastp(  # trexei BLAST gia ola ta queries
        db_prefix=db_prefix,  # poio DB
        query_fasta=queries_fasta,  # query fasta
        out_path=blast_all_out,  # pou na grapsei TSV
        max_target_seqs=int(args.recall_n),  # top-N hits
        seqidlist=None,  # oxi restricted list psaxnei olo to DB
        evalue=1e6,  # evalue gia na parei arketa hits
    )
    hits_by_q = parse_blast_tabular(blast_all_out)  # kanei parse: qid -> hits
    blast_time_per_query = t_blast_total / float(len(queries)) if (queries and t_blast_total > 0.0) else 0.0  # mesos xronos BLAST ana query
    blast_qps = float(len(queries) / t_blast_total) if t_blast_total > 0.0 else 0.0  # BLAST queries per second

    # grafoume to report se ena arxeio
    with out_path.open("w", encoding="utf-8", newline="\n") as out:  # anoigei output file
        for qi, qid in enumerate(q_ids):  # gia kathe query
            blast_hits = hits_by_q.get(qid, [])  # pare BLAST hits gia auto to query
            blast_top = [h.sseqid for h in blast_hits[: int(args.recall_n)]]  # krataei ta prwta N ids
            blast_top_set = set(blast_top)  # set gia grhgoro membership check

            # trexoume ANN kai mazeyoume ids gia extra BLAST identity
            summary: list[SummaryRow] = []  # pinakas summary rows ana method
            per_method_printed: dict[str, list[tuple[str, float]]] = {}  # lista (id, distance) pou tha typwthei
            union_printed_ids: set[str] = set()  # ola ta printed ids mazi gia na kanoume ena restricted BLAST

            qvec = Q[qi]  # embedding vector tou query
            for method_name, index in methods:  # trexei kathe ANN method
                t0 = time.perf_counter()  # start timer gia query
                res = index.query(qvec, k=int(args.recall_n))  # ANN query top-N
                # theloyme na exoume panta N apotelesmata
                if int(res.indices.size) < int(args.recall_n):  # an epestrepse ligotera apo N
                    dist_sq_all = np.einsum("ij,ij->i", (X - qvec[None, :]), (X - qvec[None, :])).astype(np.float32, copy=False)  # full L2^2 se olo to dataset
                    full_idx, full_dist = topk_smallest(dist_sq_all, int(args.recall_n))  # pare akrivws ta top-N me brute force
                    res.indices = full_idx  # antikatesthse indices me full search
                    res.distances = full_dist  # antikatesthse distances me full search
                t1 = time.perf_counter()  # stop timer
                dt = t1 - t0  # xronos gia auto to query
                qps = (1.0 / dt) if dt > 0.0 else 0.0  # queries per second gia auto to query

                ann_ids = [ids[int(i)] for i in res.indices.tolist()]  # metatrepei indices se dataset ids
                rec = recall_at_n(ann_ids, blast_top, int(args.recall_n))  # recall@N se sxesi me BLAST top-N
                summary.append(  # prosthetei mia grammi sto summary
                    SummaryRow(
                        method=method_name,  # onoma methodou
                        time_per_query_s=float(dt),  # xronos ana query
                        qps=float(qps),  # qps
                        recall_at_n=float(rec),  # recall
                    )
                )

                topk = min(int(args.print_topk), len(ann_ids))  # posa na typwsei 
                printed = [(ann_ids[r], float(res.distances[r])) for r in range(topk)]  # lista me id, dist
                per_method_printed[method_name] = printed  # apothikeuei ta printed ana method
                for pid, _ in printed:  # mazeyei ola ta printed ids
                    union_printed_ids.add(pid)  # union printed ids 

            # prosthetei BLAST ws reference sto summary
            summary.append(
                SummaryRow(
                    method="BLAST (Ref)",  # label BLAST
                    time_per_query_s=float(blast_time_per_query),  # mesos xronos BLAST
                    qps=float(blast_qps),  # BLAST qps
                    recall_at_n=1.0,  # BLAST einai to ground truth me recall 1.0
                )
            )

            # ypologizei % identity mono gia printed neighbors
            pident_by_id: dict[str, float] = {}  # neighbor_id einai to blast percent identity
            if union_printed_ids:  # an exoume printed ids
                q_tmp = cache_dir / f"query_{qid}.fasta"  # proswrino fasta me 1 query
                write_fasta(q_tmp, [queries[qi]])  # grafei mono to trexon query se fasta
                ids_tmp = cache_dir / f"seqidlist_{qid}.txt"  # proswrino seqidlist file
                write_seqidlist(ids_tmp, sorted(union_printed_ids))  # grafei lista ids pou epitrepetai na giroun hits
                blast_cand_out = cache_dir / f"blast_candidates_{qid}.tsv"  # output tsv gia restricted BLAST
                run_blastp(  # trexei BLAST mono se ayta ta candidates
                    db_prefix=db_prefix,  # BLAST DB
                    query_fasta=q_tmp,  # 1 query fasta
                    out_path=blast_cand_out,  # output
                    max_target_seqs=max(1, len(union_printed_ids)),  # posa hits na zitisei
                    seqidlist=ids_tmp,  # restrict sto union printed ids
                )
                cand_hits = parse_blast_tabular(blast_cand_out).get(qid, [])  # pare hits gia to query
                # kratame to kalutero hit ana subject
                for h in cand_hits:  # loop sta hits
                    if h.sseqid not in pident_by_id:  # an den to exoume idi valei
                        pident_by_id[h.sseqid] = h.pident  # apothikeyoyme percent identity

            # grafei to section gia auto to query sto results arxeio
            write_query_header(out, qid, int(args.recall_n))  # header query 
            write_summary_table(out, summary)  # summary table me times/qps/recall
            write_neighbors_section_header(out, int(args.print_topk))  # header gia neighbors

            # analutika neighbors ana method
            for method_name, _index in methods:  # gia kathe method
                printed = per_method_printed.get(method_name, [])  # pare printed lista id, dist
                rows: list[NeighborRow] = []  # grammes gia neighbor table
                for rank, (nid, l2) in enumerate(printed, start=1):  # rank starting apo 1
                    rows.append(  # ftiaxnei mia grammi neighbor
                        NeighborRow(
                            rank=rank,  # thesi sto top-k
                            neighbor_id=nid,  # id tou neighbor
                            l2_dist=float(l2),  # L2 distance 
                            blast_identity_pct=pident_by_id.get(nid),  # % identity apo restricted BLAST
                            in_blast_top_n=(nid in blast_top_set),  # an einai mesa sta BLAST top-N
                            bio_comment="--",  # placeholder bio sxolio
                        )
                    )
                write_method_neighbors(out, method_name, rows)  # grafei ton pinaka neighbors gia ayti ti method
            out.write("\n")  # keno meta apo kathe query gia kaluterh anagnwsi

    print(f"[protein_search] Wrote results to: {out_path}") 


if __name__ == "__main__": 
    main() 
