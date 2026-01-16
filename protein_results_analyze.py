#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class SummaryEntry:
    query_id: str
    method: str
    time_per_query_s: float
    qps: float
    recall_at_n: float


@dataclass(frozen=True)
class NeighborEntry:
    query_id: str
    method: str
    rank: int
    neighbor_id: str
    l2_dist: float
    blast_identity_pct: float
    in_blast_top_n: bool
    bio_comment: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse results.txt and export CSV summaries + candidate remote-homolog list."
    )
    p.add_argument("-i", "--input", dest="results_path", required=True, help="results.txt")
    p.add_argument("--out-dir", default=".", help="Output directory (default: current)")
    p.add_argument("--identity-threshold", type=float, default=30.0, help="BLAST identity threshold (default: 30)")
    p.add_argument("--require-not-in-blast", action="store_true", help="Only keep candidates with In BLAST Top-N? = No")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = Path(args.results_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query = list(parse_results(results))

    # Flatten
    summary_rows: list[SummaryEntry] = []
    neighbor_rows: list[NeighborEntry] = []
    for s, ns in per_query:
        summary_rows.extend(s)
        neighbor_rows.extend(ns)

    # Write per-query summary
    _write_csv(
        out_dir / "summary_per_query.csv",
        ["query_id", "method", "time_per_query_s", "qps", "recall_at_n"],
        (
            {
                "query_id": r.query_id,
                "method": r.method,
                "time_per_query_s": f"{r.time_per_query_s:.6f}",
                "qps": f"{r.qps:.6f}",
                "recall_at_n": f"{r.recall_at_n:.6f}",
            }
            for r in summary_rows
        ),
    )

    # Aggregate per method (mean over queries)
    by_method: dict[str, list[SummaryEntry]] = {}
    for r in summary_rows:
        by_method.setdefault(r.method, []).append(r)

    agg_rows = []
    for method, rows in by_method.items():
        if not rows:
            continue
        mean_time = sum(r.time_per_query_s for r in rows) / len(rows)
        # QPS is not additive; use mean of per-query QPS for reporting.
        mean_qps = sum(r.qps for r in rows) / len(rows)
        mean_recall = sum(r.recall_at_n for r in rows) / len(rows)
        agg_rows.append(
            {
                "method": method,
                "mean_time_per_query_s": f"{mean_time:.6f}",
                "mean_qps": f"{mean_qps:.6f}",
                "mean_recall_at_n": f"{mean_recall:.6f}",
                "n_queries": str(len(rows)),
            }
        )

    _write_csv(
        out_dir / "summary_by_method.csv",
        ["method", "mean_time_per_query_s", "mean_qps", "mean_recall_at_n", "n_queries"],
        agg_rows,
    )

    # Candidate remote-homolog list
    cand = []
    for r in neighbor_rows:
        if r.blast_identity_pct >= float(args.identity_threshold):
            continue
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

    print(f"[analyze] Wrote: {out_dir / 'summary_per_query.csv'}")
    print(f"[analyze] Wrote: {out_dir / 'summary_by_method.csv'}")
    print(f"[analyze] Wrote: {out_dir / 'candidates_remote_homologs.csv'}")


def parse_results(path: Path) -> Iterator[tuple[list[SummaryEntry], list[NeighborEntry]]]:
    """Yield (summary_entries, neighbor_entries) per query."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    qid_re = re.compile(r"^Query Protein:\s*<([^>]+)>")
    method_line_re = re.compile(r"^Method:\s*(.+)$")

    i = 0
    while i < len(text):
        line = text[i].strip()
        m = qid_re.match(line)
        if not m:
            i += 1
            continue
        query_id = m.group(1).strip()
        i += 1

        # Skip until summary header line ("Method | Time/query...")
        while i < len(text) and "Method" not in text[i]:
            i += 1
        # Now parse summary rows until blank line.
        summary_entries: list[SummaryEntry] = []
        if i < len(text) and text[i].lstrip().startswith("Method"):
            i += 1
            while i < len(text) and text[i].strip():
                cols = [c.strip() for c in text[i].split("|")]
                if len(cols) >= 4:
                    method = cols[0]
                    time_s = _to_float(cols[1])
                    qps = _to_float(cols[2])
                    recall = _to_float(cols[3])
                    summary_entries.append(
                        SummaryEntry(
                            query_id=query_id,
                            method=method,
                            time_per_query_s=time_s,
                            qps=qps,
                            recall_at_n=recall,
                        )
                    )
                i += 1

        # Skip to neighbor section
        neighbor_entries: list[NeighborEntry] = []
        while i < len(text):
            line = text[i].strip()
            if qid_re.match(line):
                break
            mm = method_line_re.match(line)
            if not mm:
                i += 1
                continue
            method = mm.group(1).strip()
            i += 1
            # Skip header line
            while i < len(text) and not text[i].lstrip().startswith("Rank"):
                if qid_re.match(text[i].strip()):
                    break
                i += 1
            if i < len(text) and text[i].lstrip().startswith("Rank"):
                i += 1
            # Parse rows until blank line
            while i < len(text) and text[i].strip():
                cols = [c.strip() for c in text[i].split("|")]
                # Expected: rank, <id>, l2, ident, yes/no, comment
                if len(cols) >= 6:
                    rank = int(cols[0])
                    neighbor_id = cols[1].strip().strip("<>").strip()
                    l2 = _to_float(cols[2])
                    ident_str = cols[3].strip().rstrip("%")
                    ident = _to_float(ident_str)
                    in_top = cols[4].strip().lower().startswith("y")
                    comment = cols[5].strip()
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
            i += 1

        yield summary_entries, neighbor_entries


def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]] | list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()

