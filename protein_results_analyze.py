#!/usr/bin/env python3  # trexoume to script me ton default python3 interpreter
"""
protein_results_analyze.py
==================
Auto to script diabazei to results.txt pou paragei to protein_search.py kai to metatrepei se xrhsima CSV.
Ti vgainei:
1) summary_per_query.csv
   - ana query kai ana method: time/query, QPS, Recall@N
2) summary_by_method.csv
   - mesoi oroi (mean) ana method se ola ta queries
3) candidates_remote_homologs.csv
   - "ypoptoi" remote homologs: zeugaria query-neighbor me xamilo BLAST identity
     (px < 30%) kai (proairetika) mono auta pou DEN einai sto BLAST Top-N.
- To results.txt einai readable gia anthrwpo, alla gia graphs/report theloume CSV.
"""
from __future__ import annotations  

import argparse  # gia CLI arguments
import csv       # gia na grapsoume CSV files
import re        # regex parsing apo to results.txt
from dataclasses import dataclass  # gia "struct-like" classes
from pathlib import Path           # paths
from typing import Iterable, Iterator  # types


# -------------------------
# Data structures 
# -------------------------

@dataclass(frozen=True)
class SummaryEntry:
    # mia grammi apo to summary table gia ena query + method
    query_id: str              # ID tou query protein
    method: str                # onoma methodou (LSH/Hypercube/IVF/Neural/BLAST)
    time_per_query_s: float    # xronos ana query (δευτερόλεπτα)
    qps: float                 # queries per second (1 / time_per_query)
    recall_at_n: float         # Recall@N se sxesi me BLAST Top-N


@dataclass(frozen=True)
class NeighborEntry:
    # mia grammi apo to section me tous "geitones" (top-k results) ana query + method
    query_id: str              # poio query
    method: str                # apo poia methodo proerkhetai auto to neighbor
    rank: int                  # seira/top position (1,2,3,...)
    neighbor_id: str           # ID tou neighbor protein
    l2_dist: float             # L2 apostasi sto embedding space
    blast_identity_pct: float  # BLAST % identity
    in_blast_top_n: bool       # an o neighbor einai mesa sto BLAST top-N ("Yes/No")
    bio_comment: str           # bio sxolio (an yparxei), alliws "--"


# -------------------------
# CLI arguments
# -------------------------

def parse_args() -> argparse.Namespace:
    # ftiaxnoume parser gia flags apo terminal
    p = argparse.ArgumentParser(
        description="Parse results.txt and export CSV summaries + candidate remote-homolog list."
    )

    p.add_argument(
        "-i", "--input",
        dest="results_path",
        required=True,
        help="results.txt",
    )
    # to input arxeio pou tha kanoume parse (results.txt)

    p.add_argument(
        "--out-dir",
        default=".",
        help="Output directory (default: current)",
    )
    # pou tha grapsoume ta CSV (default: current directory)

    p.add_argument(
        "--identity-threshold",
        type=float,
        default=30.0,
        help="BLAST identity threshold (default: 30)",
    )
    # identity-threshold = orio gia to poso "idia" einai i query me ton geitona.
    # An BLAST identity >= threshold (px 30%), tote einai arketa similar -> den to kratame ws remote-homolog candidate.
    # Kratame MONO auta me identity < threshold, dhladi pio "makrina" / remote omoiotita.


    p.add_argument(
        "--require-not-in-blast",
        action="store_true",
        help="Only keep candidates with In BLAST Top-N? = No",
    )
    # an to valeis, krataei mono auta pou DEN einai sto BLAST top-N

    return p.parse_args()  # epistrofi Namespace


# -------------------------
# Main flow
# -------------------------

def main() -> None:
    args = parse_args()  # diabazoume flags
    results = Path(args.results_path)  # path sto results.txt
    out_dir = Path(args.out_dir)       # output folder
    out_dir.mkdir(parents=True, exist_ok=True)  # dimiourgei folder an den yparxei

    per_query = list(parse_results(results))
    # parse_results() epistrefei iterator ana query.
    # to kanoume list gia na to xrhsimopoiisoume 2 fores (flatten + candidates)

    # Flatten: mazepse OLA ta rows se eniaies listes
    summary_rows: list[SummaryEntry] = []   # olo to summary apo ola ta queries
    neighbor_rows: list[NeighborEntry] = [] # oloi oi geitones apo ola ta queries

    for s, ns in per_query:
        summary_rows.extend(s)   # prosthetoume summary entries tou query
        neighbor_rows.extend(ns) # prosthetoume neighbor entries tou query

    # -------------------------
    # 1) Write summary per query
    # -------------------------

    _write_csv(
        out_dir / "summary_per_query.csv",  # output file path
        ["query_id", "method", "time_per_query_s", "qps", "recall_at_n"],  # stiles CSV
        (
            {
                "query_id": r.query_id,
                "method": r.method,
                "time_per_query_s": f"{r.time_per_query_s:.6f}",  # format se string
                "qps": f"{r.qps:.6f}",
                "recall_at_n": f"{r.recall_at_n:.6f}",
            }
            for r in summary_rows
        ),
    )
    # edw pername generator (oxi lista) pou ftiaxnei dict ana row

    # -------------------------
    # 2) Aggregate per method
    # -------------------------

    by_method: dict[str, list[SummaryEntry]] = {}
    # dict: key=method, value=list me rows apo ola ta queries

    for r in summary_rows:
        by_method.setdefault(r.method, []).append(r)
        # an den yparxei to method sto dict, dimiourgei adeia lista
        # kai meta prosthetei to row

    agg_rows = []  # rows gia to summary_by_method.csv

    for method, rows in by_method.items():
        if not rows:
            continue  # an einai adeio

        mean_time = sum(r.time_per_query_s for r in rows) / len(rows)
        # mesos xronos ana query

        # QPS = Queries Per Second = 1 / (time_per_query).
        # Otan exoume polla queries, to swsto gia report einai na paroume ton meso oro
        # twn QPS pou metrhthikan se kathe query.
        mean_qps = sum(r.qps for r in rows) / len(rows)

        mean_recall = sum(r.recall_at_n for r in rows) / len(rows)
        # mesos recall@N

        agg_rows.append(
            {
                "method": method,
                "mean_time_per_query_s": f"{mean_time:.6f}",
                "mean_qps": f"{mean_qps:.6f}",
                "mean_recall_at_n": f"{mean_recall:.6f}",
                "n_queries": str(len(rows)),  # posa queries xrhsimopoihthikan
            }
        )

    _write_csv(
        out_dir / "summary_by_method.csv",
        ["method", "mean_time_per_query_s", "mean_qps", "mean_recall_at_n", "n_queries"],
        agg_rows,
    )

    # -------------------------
    # 3) Candidate remote-homolog list
    # -------------------------

    cand = []  # lista me candidate neighbors 

    for r in neighbor_rows:
        # Filtro 1 (identity):
        # To BLAST identity % mas leei poso idia einai i amino-acid sequence tou query me tou neighbor.
        # An to identity einai megalO (>= threshold, px 30%), tote oi proteins einai arketa similar
        # kai den mas endiaferei ws "remote homolog" (einai pio kontini omoiotita).
        # Emeis theloume candidates pou mοιάζουν "ligo" (identity < threshold), diladi pio remote.
        if r.blast_identity_pct >= float(args.identity_threshold):
            continue  # an einai polu idia me BLAST, den to theloume ws "remote"

        # filtro 2 (optional): na min einai mesa sto BLAST top-N
        if args.require_not_in_blast and r.in_blast_top_n:
            continue

        cand.append(
            {
                "query_id": r.query_id,
                "method": r.method,
                "rank": str(r.rank),
                "neighbor_id": r.neighbor_id,
                "l2_dist": f"{r.l2_dist:.6f}",
                "blast_identity_pct": f"{r.blast_identity_pct:.2f}",
                "in_blast_top_n": "Yes" if r.in_blast_top_n else "No",
                "bio_comment": r.bio_comment,
            }
        )

    _write_csv(
        out_dir / "candidates_remote_homologs.csv",
        [
            "query_id",
            "method",
            "rank",
            "neighbor_id",
            "l2_dist",
            "blast_identity_pct",
            "in_blast_top_n",
            "bio_comment",
        ],
        cand,
    )

    # console prints gia na kseroume pou grapsame ta outputs
    print(f"[analyze] Wrote: {out_dir / 'summary_per_query.csv'}")
    print(f"[analyze] Wrote: {out_dir / 'summary_by_method.csv'}")
    print(f"[analyze] Wrote: {out_dir / 'candidates_remote_homologs.csv'}")


# -------------------------
# Parser for results.txt
# -------------------------

def parse_results(path: Path) -> Iterator[tuple[list[SummaryEntry], list[NeighborEntry]]]:
    """Yield (summary_entries, neighbor_entries) per query."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # diabazei olo to results.txt se lista grammwn
    # errors="replace": an yparxoun periergoi xaraktires, den skaei

    qid_re = re.compile(r"^Query Protein:\s*<([^>]+)>")
    # regex: vriskei grammi pou leei "Query Protein: <ID>" kai pianei to ID mesa sto <...>

    method_line_re = re.compile(r"^Method:\s*(.+)$")
    # regex: vriskei grammi "Method: LSH" kai piάνει to onoma methodou

    # Sto results.txt, oi stiles xwrizontai me whitespace + '|' + whitespace.
    # Alla ta UniProt IDs exoun '|' mesa (px "sp|P12345|NAME"),
    # opote den prepei na spame me σκeto '|', giati tha xalasoume to ID.
    col_sep_re = re.compile(r"\s+\|\s+")
    # diaxoristis stilis: "   |   " (me kena girw)

    i = 0  # pointer se poia grammi eimaste
    while i < len(text):
        line = text[i].strip()       # trexousa grammi xwris kena
        m = qid_re.match(line)       # koita an einai query header
        if not m:
            i += 1
            continue  

        query_id = m.group(1).strip()  # to ID tou query
        i += 1

        # -------------------------
        # Parse summary table
        # -------------------------

        # Skip mexri na vroume header line ("Method | Time/query...")
        while i < len(text) and "Method" not in text[i]:
            i += 1

        summary_entries: list[SummaryEntry] = []

        # an eimaste se summary header, arxizoume na diavazoume grammές mexri keno line
        if i < len(text) and text[i].lstrip().startswith("Method"):
            i += 1  # pername tin header grammi
            while i < len(text) and text[i].strip():
                cols = [c.strip() for c in col_sep_re.split(text[i].strip())]
                # spame ti grammi se stiles me ton "safe" separator

                if len(cols) >= 4:
                    method = cols[0]        # method name
                    time_s = _to_float(cols[1])  # time/query
                    qps = _to_float(cols[2])     # QPS
                    recall = _to_float(cols[3])  # Recall@N

                    summary_entries.append(
                        SummaryEntry(
                            query_id=query_id,
                            method=method,
                            time_per_query_s=time_s,
                            qps=qps,
                            recall_at_n=recall,
                        )
                    )
                i += 1  # epomeni grammi

        # -------------------------
        # Parse neighbor sections
        # -------------------------

        neighbor_entries: list[NeighborEntry] = []

        while i < len(text):
            line = text[i].strip()

            # an ksekinaei kainourio query, stamataei to neighbor parsing gia to proigoumeno
            if qid_re.match(line):
                break

            mm = method_line_re.match(line)
            if not mm:
                i += 1
                continue  # den einai "Method:" line, proxora

            method = mm.group(1).strip()  # krata to method name (px "LSH")
            i += 1

            # Skip mexri na vroume header "Rank | Neighbor ID | ..."
            while i < len(text) and not text[i].lstrip().startswith("Rank"):
                if qid_re.match(text[i].strip()):
                    break  # an petuxame neo query, stamataei
                i += 1

            if i < len(text) and text[i].lstrip().startswith("Rank"):
                i += 1  # pername tin header grammi tou neighbors table

            # Parse neighbor rows mexri na vroume keni grammi
            while i < len(text) and text[i].strip():
                cols = [c.strip() for c in col_sep_re.split(text[i].strip())]
                # Expected cols: rank, <id>, l2, ident, yes/no, comment

                if len(cols) >= 6:
                    rank = int(cols[0])  # seira
                    neighbor_id = cols[1].strip().strip("<>").strip()  # afairei <...>
                    l2 = _to_float(cols[2])     # apostasi
                    ident = _to_float(cols[3])  # identity (%)
                    in_top = cols[4].strip().lower().startswith("y")  # Yes/No -> bool
                    comment = cols[5].strip()   # bio comment

                    neighbor_entries.append(
                        NeighborEntry(
                            query_id=query_id,
                            method=method,
                            rank=rank,
                            neighbor_id=neighbor_id,
                            l2_dist=l2,
                            blast_identity_pct=ident,
                            in_blast_top_n=in_top,
                            bio_comment=comment,
                        )
                    )
                i += 1

            i += 1  # pername to blank line meta apo to section

        # epistrefoume ta parsed rows gia auto to query
        yield summary_entries, neighbor_entries


# -------------------------
# Helper: parse float safely
# -------------------------

def _to_float(s: str) -> float:
    # Regex gia na kanoume extract ena float apo string pou mporei na exei extra text.
    # Piaνει arithmous se morfes:
    #  - 0.112
    #  - -12.5
    #  - 31%        (tha piasei to 31)
    #  - 1.00 (text) (tha piasei to 1.00)
    #  - 1.9e+03    (scientific notation = 1900)
    #
    # To pattern:
    #  [-+]?         optional sign (+/-)
    #  \d*\.?\d+     digits me optional decimal point
    #  (?:[eE]...)?  optional scientific exponent (e.g. e+03)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)


    if not m:
        return 0.0  # an den vrethike arithmos, dose 0

    try:
        return float(m.group(0))  # metatrepoume to match se float
    except Exception:
        return 0.0  # an kati paei lathos, fallback 0


# -------------------------
# Helper: write CSV
# -------------------------

def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]] | list[dict[str, str]]) -> None:
    # anoigei file kai grafei CSV me header + rows
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)  # DictWriter = dineis dict ana row
        w.writeheader()  # grafei tin 1i grammi (column names)
        for r in rows:
            w.writerow(r)  # grafei kathe row


if __name__ == "__main__":
    # entry point: trexei mono an to script klithei direct (oxi an ginei import)
    main()
