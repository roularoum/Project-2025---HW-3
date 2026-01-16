#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from protein_ann.fasta import FastaRecord, read_fasta, write_fasta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Utility: write a FASTA subset (first N records). Useful for smoke-tests."
    )
    p.add_argument("-i", "--input", required=True, help="Input FASTA")
    p.add_argument("-o", "--output", required=True, help="Output FASTA")
    p.add_argument("--n", type=int, required=True, help="Number of records to keep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = int(args.n)
    if n <= 0:
        raise ValueError("--n must be positive")

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    records: list[FastaRecord] = []
    for rec in read_fasta(inp):
        records.append(rec)
        if len(records) >= n:
            break

    if not records:
        raise ValueError("No FASTA records found")

    write_fasta(out, records)
    print(f"[subset] Wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()

