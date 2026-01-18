"""
output_format.py
================
To arxeio auto orizei to format tis eksodou (results.txt) pou zitaei i ergasia.
Dimiourgei:
- ena header gia kathe query (Query Protein + N)
- ena summary table me Time/query, QPS, Recall@N gia kathe methodo
- ena section me tous top-K geitones ana methodo (Rank, ID, L2 distance, BLAST identity, ktl)
"""

from __future__ import annotations  # type hints meionoun provlimata me forward refs / py<3.11

from dataclasses import dataclass  # dataclass = aplo "container" gia dedomena
from typing import Iterable, TextIO  # Iterable = opoiadipote loop-able, TextIO = file-like object gia text


@dataclass(frozen=True)  # frozen = den theloume na allazei meta 
class SummaryRow:
    method: str  # onoma methodou (LSH, Hypercube, IVF, Neural, BLAST)
    time_per_query_s: float  # mesos xronos ana query se seconds
    qps: float  # queries per second 
    recall_at_n: float  # recall@N se sxesi me BLAST top-N


@dataclass(frozen=True)  # frozen = immutable row gia top-k neighbor print
class NeighborRow:
    rank: int  # thesi sti lista (1 = pio konta)
    neighbor_id: str  # ID tis protein-pou-vrethike ws geitonas
    l2_dist: float  # euclidean distance sto embedding space (mikrotero = pio similar)
    blast_identity_pct: float | None  # pososto % identity apo BLAST (None an den exoume/ypologisoume)
    in_blast_top_n: bool | None  # True/False an einai mesa sto BLAST top-N (None an den exoume BLAST list)
    bio_comment: str = "--"  # proairetiko sxolio biologias, default "--" an den yparxei


def write_query_header(out: TextIO, query_id: str, recall_n: int) -> None:
    # grafoume header gia kathe query me to style tis ekfwnisis
    out.write(f"Query Protein: <{query_id}>\n")  # grafoume to ID tou query se <> opws zitaei to format
    out.write(f"N = {recall_n} (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)\n\n")  # grafoume to N pou xrisimopoieitai gia recall + 2 newlines


def write_summary_table(out: TextIO, rows: Iterable[SummaryRow]) -> None:
    # grafoume tin epikefalida tou summary section (1)
    out.write("[1] Συνοπτική σύγκριση μεθόδων\n")  # titlos section
    out.write("Method\t|\tTime/query (s)\t|\tQPS\t|\tRecall@N vs BLAST Top-N\n")  # header grammi pinaka me tabs kai separators

    for r in rows:  # gia kathe methodo 
        time_s = f"{r.time_per_query_s:.3f}"  # format time me 3 dekadika (px 0.123)
        qps = f"{r.qps:.3g}"  # format QPS me 3 significant digits (px 50, 0.7) opws sto paradeigma
        rec = f"{r.recall_at_n:.2f}"  # format recall me 2 dekadika (px 0.00, 1.00)

        if r.method.lower().startswith("blast"):  # an einai i BLAST grammi 
            out.write(f"{r.method}\t|\t{time_s}\t|\t{qps}\t|\t{rec}\t(ορίζει το Top-N)\n")  # extra sxolio oti i BLAST orizei to ground truth
        else:
            out.write(f"{r.method}\t|\t{time_s}\t|\t{qps}\t|\t{rec}\n")  # kanoniki grammi gia ANN methodous

    out.write("\n\n")  # keno meta ton pinaka gia na xwristoun ta sections


def write_neighbors_section_header(out: TextIO, print_topk: int) -> None:
    # grafoume tin epikefalida tou section (2) pou periexei tous top geitones
    out.write(f"[2] Top-N γείτονες ανά μέθοδο (εδώ π.χ. N = {print_topk} για εκτύπωση)\n\n")  # deixnoume oti tiponoume topK gia parousiasi


def write_method_neighbors(out: TextIO, method: str, rows: Iterable[NeighborRow]) -> None:
    # grafoume to block apotelesmaton gia mia methodo (p.x. LSH)
    out.write(f"Method: {method}\n\n")  # titlos methodou
    out.write("Rank\t|\tNeighbor ID\t|\tL2 Dist\t|\tBLAST Identity\t|\tIn BLAST Top-N?\t|\tBio comment\n")  # header twn stilon

    for r in rows:  # gia kathe geitona sto top-k
        ident = "0%" if r.blast_identity_pct is None else f"{int(round(r.blast_identity_pct))}%"  # an den exoume identity(pososto theseon pou einai id) -> 0%, alliws stroggylopoiisi se akeraio %
        in_top = "-" if r.in_blast_top_n is None else ("Yes" if r.in_blast_top_n else "No")  # an den exoume BLAST info -> "-", alliws Yes/No
        comment = (r.bio_comment or "--").strip()  # an einai adeio/None sxolio -> "--", kai trim spaces

        out.write(  # grafoume mia grammi me ola ta pedia se format
            f"{r.rank}\t|\t<{r.neighbor_id}>\t|\t{r.l2_dist:.2f}\t|\t{ident}\t|\t{in_top}\t|\t{comment}\n"
            # r.l2_dist:.2f = 2 dekadika, neighbor_id mesa se <> opws zitaei to format
        )

    out.write("\n\n")  # keno meta to block tis methodou
