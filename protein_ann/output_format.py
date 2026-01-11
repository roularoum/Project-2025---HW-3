from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TextIO


@dataclass(frozen=True)
class SummaryRow:
    method: str
    time_per_query_s: float
    qps: float
    recall_at_n: float


@dataclass(frozen=True)
class NeighborRow:
    rank: int
    neighbor_id: str
    l2_dist: float
    blast_identity_pct: float | None  # None if BLAST identity unavailable
    in_blast_top_n: bool | None  # None if BLAST top-N unavailable
    bio_comment: str = "--"


def write_query_header(out: TextIO, query_id: str, recall_n: int) -> None:
    """Write the per-query header exactly in the style of the assignment spec."""
    out.write(f"Query Protein: <{query_id}>\n")
    out.write(f"N = {recall_n} (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)\n\n")


def write_summary_table(out: TextIO, rows: Iterable[SummaryRow]) -> None:
    out.write("[1] Συνοπτική σύγκριση μεθόδων\n")
    out.write("Method\t|\tTime/query (s)\t|\tQPS\t|\tRecall@N vs BLAST Top-N\n")
    for r in rows:
        time_s = f"{r.time_per_query_s:.3f}"
        qps = f"{r.qps:.3g}"  # matches example-style (e.g. 50, 0.7)
        rec = f"{r.recall_at_n:.2f}"
        if r.method.lower().startswith("blast"):
            out.write(f"{r.method}\t|\t{time_s}\t|\t{qps}\t|\t{rec}\t(ορίζει το Top-N)\n")
        else:
            out.write(f"{r.method}\t|\t{time_s}\t|\t{qps}\t|\t{rec}\n")
    out.write("\n\n")


def write_neighbors_section_header(out: TextIO, print_topk: int) -> None:
    out.write(f"[2] Top-N γείτονες ανά μέθοδο (εδώ π.χ. N = {print_topk} για εκτύπωση)\n\n")


def write_method_neighbors(out: TextIO, method: str, rows: Iterable[NeighborRow]) -> None:
    out.write(f"Method: {method}\n\n")
    out.write("Rank\t|\tNeighbor ID\t|\tL2 Dist\t|\tBLAST Identity\t|\tIn BLAST Top-N?\t|\tBio comment\n")
    for r in rows:
        ident = "0%" if r.blast_identity_pct is None else f"{int(round(r.blast_identity_pct))}%"
        in_top = "-" if r.in_blast_top_n is None else ("Yes" if r.in_blast_top_n else "No")
        comment = (r.bio_comment or "--").strip()
        out.write(
            f"{r.rank}\t|\t<{r.neighbor_id}>\t|\t{r.l2_dist:.2f}\t|\t{ident}\t|\t{in_top}\t|\t{comment}\n"
        )
    out.write("\n\n")

