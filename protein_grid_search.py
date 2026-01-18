#!/usr/bin/env python3
from __future__ import annotations

import argparse  # gia na diavazoume arguments apo command line
import csv  # gia na grapsoume apotelesmata se CSV
import itertools  # gia cartesian product grid search combinations
import sys  # gia na peiraksoume to sys.path kai na kanoume local imports
import time  # gia xronometra build/query times
from dataclasses import asdict  # metatrepei dataclass se dict
from pathlib import Path  # swstos xeirismos paths 
from typing import Iterable  # type hint gia iterables

_THIS_DIR = Path(__file__).resolve().parent # to folder pou vrisketai to arxeio
if str(_THIS_DIR) not in sys.path:  # an to local folder den einai sto python path
    sys.path.insert(0, str(_THIS_DIR))  # to vazοume prwto gia na vrei ta project modules

from protein_ann.blast import ensure_blast_db, parse_blast_tabular, run_blastp  # BLAST: build/run/parse
from protein_ann.fasta import read_fasta  # diavazei FASTA queries
from protein_ann.metrics import mean, recall_at_n  # metrics: mesos oros kai recall@N
from protein_ann.vectors import load_vectors  # fortwnei embeddings kai ids apo disk


# synarthsh poy orizei ta CLI arguments
def parse_args() -> argparse.Namespace:  
    p = argparse.ArgumentParser(  # ftiaxnei parser gia command line
        description="Assignment 3 - Grid search for ANN hyperparameters (Recall@N vs QPS)."  
    )
    p.add_argument("-d", "--data", dest="vectors_path", required=True)  # path sto vector dataset 
    p.add_argument("-q", "--queries", dest="queries_fasta", required=True)  # FASTA me query sequences
    p.add_argument("-o", "--output", dest="output_csv", required=True)  # pou tha grapsei to CSV

    # epilegi method ANN pou tha trexei grid search
    p.add_argument(  
        "--method",
        choices=["lsh", "hypercube", "ivfflat", "ivfpq", "neural"],
        required=True,
        help="Method to grid-search",
    )
    p.add_argument("--recall-n", type=int, default=50)  # N gia Recall@N 
    p.add_argument("--seed", type=int, default=1)  # seed gia reproducibility

    p.add_argument(  # fasta pou xrisimopoiei o BLAST gia na ftiaxei database
        "--blast-fasta",
        default="swissprot.fasta",
        help="FASTA used to build the BLAST DB (default: swissprot.fasta)",
    )
    p.add_argument(  # pou tha apothikeuei BLAST db kai cached results
        "--blast-db-dir",
        default=".blast_db_cache",
        help="Directory for BLAST DB cache (default: .blast_db_cache)",
    )

    # embeddings twn queries
    p.add_argument("--embed-model", default="esm2_t6_8M_UR50D")  # onoma ESM2 model
    p.add_argument("--embed-device", default="auto", choices=["auto", "cpu", "cuda"])  # pou tha trexei
    p.add_argument("--embed-batch-size", type=int, default=8)  # batch size gia embedding 

    # ta grids dinontai ws strings "1,2,3"
    p.add_argument("--lsh-k-grid", default="4")  # LSH: k hashes ana pinaka
    p.add_argument("--lsh-L-grid", default="5")  # LSH: L pinakes katakermatismou 
    p.add_argument("--lsh-w-grid", default="4.0")  # LSH: platos kadoy 

    p.add_argument("--hc-kproj-grid", default="14")  # Hypercube: poses provoles/bits 
    p.add_argument("--hc-w-grid", default="4.0")  # Hypercube: platos quantization 
    p.add_argument("--hc-M-grid", default="1000")  # Hypercube: megistos arithmos candidates pou elegxoume 
    p.add_argument("--hc-probes-grid", default="10")  # Hypercube: posa geitonika vertices psaxnoume 

    p.add_argument("--ivf-kclusters-grid", default="50")  # IVF: arithmos clusters 
    p.add_argument("--ivf-nprobe-grid", default="5")  # IVF: posa clusters psaxnei sto query 

    p.add_argument("--pq-M-grid", default="16")  # PQ: posa sub-vectors / omades 
    p.add_argument("--pq-nbits-grid", default="8")  # PQ: posa bits ana subvector 

    p.add_argument("--nlsh-m-grid", default="100")  # Neural LSH: posa bins/katigories 
    p.add_argument("--nlsh-T-grid", default="5")  # Neural LSH: posa top candidates kratame 
    p.add_argument("--nlsh-epochs", type=int, default=10)  # posa epochs gia training
    p.add_argument("--nlsh-layers", type=int, default=3)  # posa layers sto diktyo
    p.add_argument("--nlsh-nodes", type=int, default=256)  # posa neurons/monades ana layer
    p.add_argument("--nlsh-lr", type=float, default=1e-3)  # vima ekpaideusis
    p.add_argument("--nlsh-batch-size", type=int, default=256)  # batch size gia training


    return p.parse_args()  # epistrefei ta arguments


def _parse_int_grid(s: str) -> list[int]:  # metatrepei akeraious se lista
    return [int(x.strip()) for x in s.split(",") if x.strip()] 


def _parse_float_grid(s: str) -> list[float]:  # metatrepei dekadikoys se lista
    return [float(x.strip()) for x in s.split(",") if x.strip()]  


def main() -> None:  
    args = parse_args()  # diavazei arguments apo CLI

    from protein_ann.esm2 import ESM2Embedder  # embedder gia protein sequences 
    from protein_ann.ann.hypercube import HypercubeIndex, HypercubeParams  # hypercube index kai params
    from protein_ann.ann.ivf_flat import IvfFlatIndex, IvfFlatParams  # IVF-Flat index kai params
    from protein_ann.ann.ivf_pq import IvfPqIndex, IvfPqParams  # IVF-PQ index kai params
    from protein_ann.ann.lsh import LshIndex, LshParams  # LSH index kai params
    from protein_ann.ann.neural_lsh import NeuralLshIndex, NeuralLshParams  # Neural LSH index kai params

    store = load_vectors(args.vectors_path)  # fortwnei embeddings store
    X = store.vectors  # dataset vectors 
    ids = store.ids  # ids gia kathe vector 

    queries_fasta = Path(args.queries_fasta)  # path tou query fasta
    if not queries_fasta.exists():  # an den yparxei
        fallback_q = _THIS_DIR / "datasets" / "targets.fasta"  # fallback file mesa sto repo
        if fallback_q.exists():  # an yparxei fallback
            queries_fasta = fallback_q  # xrisimopoiei fallback
        else:
            raise FileNotFoundError(f"Query FASTA not found: {args.queries_fasta}")  # skasei me error

    queries = list(read_fasta(queries_fasta))  # diavazei queries apo FASTA
    if not queries:  # an den yparxei kanena query
        raise ValueError("No queries found")  # error

    q_ids = [q.id for q in queries]  # lista me query IDs
    q_seqs = [q.sequence for q in queries]  # lista me query sequences

    embedder = ESM2Embedder(model_name=args.embed_model, device=args.embed_device, seed=args.seed)  # setup embedder
    Q = embedder.embed_sequences(q_seqs, batch_size=args.embed_batch_size)  # embeddings gia queries 

    blast_fasta = Path(args.blast_fasta)  # FASTA pou tha ginei BLAST database
    if not blast_fasta.exists():  # an den yparxei
        fallback = _THIS_DIR / "datasets" / "swissprot_50k.fasta"  # fallback dataset
        if fallback.exists():  # an yparxei fallback
            blast_fasta = fallback  # to xrisimopoiei
        else:
            raise FileNotFoundError(f"BLAST FASTA not found: {args.blast_fasta}")  # error

    db_prefix = ensure_blast_db(blast_fasta, args.blast_db_dir)  # ftiaxnei to BLAST database kai gyrnaei to prefix path
    cache_dir = Path(args.blast_db_dir)  # folder pou kratame cache gia BLAST 
    blast_out = cache_dir / f"blast_all_top{int(args.recall_n)}.tsv"   # arxeio cache me BLAST top-N hits ana query

    if not blast_out.exists():  # an den yparxei cached BLAST output
        run_blastp(  # trexei BLASTP
            db_prefix=db_prefix,  # poio DB xrisimopoiei
            query_fasta=queries_fasta,  # queries fasta
            out_path=blast_out,  # pou na grapsei output
            max_target_seqs=int(args.recall_n),  # top-N hits
        )

    hits_by_q = parse_blast_tabular(blast_out)  # kanei parse to BLAST TSV kai ftiaxnei dict: qid -> lista apo hits

    blast_top_by_q: dict[str, list[str]] = {}  # teliko ground-truth: qid -> lista ids
    for qid in q_ids:  # gia kathe query
        hits = hits_by_q.get(qid, [])  # pare hits alliws adeia lista
        blast_top_by_q[qid] = [h.sseqid for h in hits[: int(args.recall_n)]]  # krataei prwta N ids

    out_path = Path(args.output_csv)  # path CSV output
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ftiaxnei folder an leipei

    rows: list[dict[str, object]] = []  # lista rows gia to CSV ena row ana config

    if args.method == "lsh":  # an epileksame LSH
        grid = itertools.product(  # oloi oi syndyasmoi twn hyperparams
            _parse_int_grid(args.lsh_k_grid),
            _parse_int_grid(args.lsh_L_grid),
            _parse_float_grid(args.lsh_w_grid),
        )
        for k, L, w in grid:  # loop se kathe syndyasmo
            idx = LshIndex(LshParams(k=k, L=L, w=w, seed=args.seed))  # ftiaxnei index me ta params
            rows.extend(_eval_index(  # evaluate kai prosthetei to apotelesma
                idx, "Euclidean LSH", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n),
                {"k": k, "L": L, "w": w}
            ))

    elif args.method == "hypercube":  # an epileksame Hypercube
        grid = itertools.product(  # syndyasmoi
            _parse_int_grid(args.hc_kproj_grid),
            _parse_float_grid(args.hc_w_grid),
            _parse_int_grid(args.hc_M_grid),
            _parse_int_grid(args.hc_probes_grid),
        )
        for kproj, w, M, probes in grid:  # loop
            idx = HypercubeIndex(HypercubeParams(kproj=kproj, w=w, M=M, probes=probes, seed=args.seed))  # index
            rows.extend(_eval_index(
                idx, "Hypercube", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n),
                {"kproj": kproj, "w": w, "M": M, "probes": probes}
            ))

    elif args.method == "ivfflat":  # an epileksame IVF-Flat
        grid = itertools.product(
            _parse_int_grid(args.ivf_kclusters_grid),
            _parse_int_grid(args.ivf_nprobe_grid),
        )
        for kclusters, nprobe in grid:
            idx = IvfFlatIndex(IvfFlatParams(kclusters=kclusters, nprobe=nprobe, seed=args.seed))  # index
            rows.extend(_eval_index(
                idx, "IVF-Flat", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n),
                {"kclusters": kclusters, "nprobe": nprobe}
            ))

    elif args.method == "ivfpq":  # an epileksame IVF-PQ
        grid = itertools.product(
            _parse_int_grid(args.ivf_kclusters_grid),
            _parse_int_grid(args.ivf_nprobe_grid),
            _parse_int_grid(args.pq_M_grid),
            _parse_int_grid(args.pq_nbits_grid),
        )
        for kclusters, nprobe, M, nbits in grid:
            idx = IvfPqIndex(IvfPqParams(kclusters=kclusters, nprobe=nprobe, pq_M=M, pq_nbits=nbits, seed=args.seed))  # index
            rows.extend(_eval_index(
                idx, "IVF-PQ", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n),
                {"kclusters": kclusters, "nprobe": nprobe, "M": M, "nbits": nbits}
            ))

    elif args.method == "neural":  # an epileksame Neural LSH
        grid = itertools.product(
            _parse_int_grid(args.nlsh_m_grid),
            _parse_int_grid(args.nlsh_T_grid),
        )
        for m_bins, T in grid:
            idx = NeuralLshIndex(  # neural index mporei na kanei training mesa sto build
                NeuralLshParams(
                    m_bins=m_bins,  # posa bins
                    top_T=T,  # top candidates
                    epochs=int(args.nlsh_epochs),  # training epochs
                    layers=int(args.nlsh_layers),  # layers
                    hidden_units=int(args.nlsh_nodes),  # neurons
                    lr=float(args.nlsh_lr),  # learning rate
                    batch_size=int(args.nlsh_batch_size),  # batch size
                    seed=int(args.seed),  # seed
                    device="auto",  # device
                )
            )
            rows.extend(_eval_index(
                idx, "Neural LSH", X, ids, Q, q_ids, blast_top_by_q, int(args.recall_n),
                {"m": m_bins, "T": T}
            ))

    if not rows:  # an den egine eval tipota
        raise RuntimeError("No grid rows produced")  # error

    fieldnames = sorted(rows[0].keys())  # statheri seira stilwn sto CSV
    with out_path.open("w", encoding="utf-8", newline="") as f:  # anoigei file gia grapsimo
        w = csv.DictWriter(f, fieldnames=fieldnames)  # writer pou pairnei dict per row
        w.writeheader()  # grafei headers
        for row in rows:  # gia kathe result row
            w.writerow(row)  # grafei mia grammi sto CSV

    print(f"[grid] Wrote {len(rows)} rows to: {out_path}")  # posa rows kai pou

 #kanei build kai evaluation gia mia config
def _eval_index( 
    index,  # to ANN index instance
    method_name: str,  # onoma methodou gia to CSV
    X: "np.ndarray",  # dataset vectors
    ids: list[str],  # dataset ids
    Q: "np.ndarray",  # query vectors
    q_ids: list[str],  # query ids
    blast_top_by_q: dict[str, list[str]],  # ground-truth list apo BLAST
    recall_n: int,  # top-N gia recall
    params_dict: dict[str, object],  # params pou tha graphtoun sto CSV
) -> list[dict[str, object]]:

    t0 = time.perf_counter()  # start timer gia build
    index.build(X)  # xtizei to index panw sto X (preprocessing/structures)
    t1 = time.perf_counter()  # stop timer
    build_s = t1 - t0  # build time se seconds

    times: list[float] = []  # xronoi ana query
    recalls: list[float] = []  # recall ana query

    for qi, qid in enumerate(q_ids):  # loop se ola ta queries
        qvec = Q[qi]  # embedding tou trexontos query

        s0 = time.perf_counter()  # start timer gia query
        res = index.query(qvec, k=recall_n)  # pairnei top-k nearest 
        s1 = time.perf_counter()  # stop timer
        times.append(s1 - s0)  # kratame ton xrono

        ann_ids = [ids[int(i)] for i in res.indices.tolist()]  # metatrepei indices se protein ids
        recalls.append(recall_at_n(ann_ids, blast_top_by_q.get(qid, []), recall_n))  # recall@N vs BLAST

    total = float(sum(times))  # synolikos xronos gia ola ta queries
    qps = float(len(q_ids) / total) if total > 0 else 0.0  # queries per second
    avg_time = float(total / len(q_ids)) if q_ids else 0.0  # mesos xronos ana query
    avg_recall = float(mean(recalls))  # meso recall se ola ta queries

    row: dict[str, object] = {  # ena row apotelesmatwn gia auto to config
        "method": method_name,  # onoma methodou
        "recall_n": recall_n,  # N
        "avg_recall": avg_recall,  # meso recall
        "avg_time_per_query_s": avg_time,  # mesos xronos query
        "qps": qps,  # taxythta
        "build_time_s": float(build_s),  # xronos build
    }
    for k, v in params_dict.items():  # gia kathe hyperparameter
        row[f"param_{k}"] = v  # apothikeuei sto row me prefix param_

    return [row]  # epistrefei lista me ena row 

if __name__ == "__main__":  
    main()  