from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class BlastHit:
    qseqid: str
    sseqid: str
    pident: float
    bitscore: float
    evalue: float


def require_blast_tools() -> tuple[str, str]:
    """Return (makeblastdb, blastp) executables or raise with actionable error."""
    makeblastdb = shutil.which("makeblastdb")
    blastp = shutil.which("blastp")
    if not makeblastdb or not blastp:
        raise RuntimeError(
            "BLAST+ tools not found in PATH. Required: makeblastdb, blastp.\n"
            "Install NCBI BLAST+ and ensure executables are available in PATH."
        )
    return makeblastdb, blastp


def ensure_blast_db(fasta_path: str | Path, db_dir: str | Path, db_name: str = "swissprot_db") -> Path:
    """Ensure a protein BLAST database exists, building it if missing.

    Returns the DB prefix path (without file extension).
    """
    makeblastdb, _blastp = require_blast_tools()
    fasta = Path(fasta_path)
    if not fasta.exists():
        raise FileNotFoundError(str(fasta))

    out_dir = Path(db_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / db_name

    # BLAST DB marker files for protein DB.
    pin = prefix.with_suffix(".pin")
    psq = prefix.with_suffix(".psq")
    phr = prefix.with_suffix(".phr")
    # When building with -parse_seqids, BLAST also emits extra index files (.pog/.pos)
    # that allow accession-based filtering (needed for -seqidlist in our pipeline).
    pog = prefix.with_suffix(".pog")
    pos = prefix.with_suffix(".pos")
    if pin.exists() and psq.exists() and phr.exists() and pog.exists() and pos.exists():
        return prefix

    cmd = [
        makeblastdb,
        "-in",
        str(fasta),
        "-dbtype",
        "prot",
        "-parse_seqids",
        "-out",
        str(prefix),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return prefix


def run_blastp(
    db_prefix: str | Path,
    query_fasta: str | Path,
    out_path: str | Path,
    max_target_seqs: int,
    seqidlist: str | Path | None = None,
    evalue: float = 1e6,
) -> float:
    """Run blastp and return elapsed seconds."""
    _makeblastdb, blastp = require_blast_tools()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        blastp,
        "-db",
        str(db_prefix),
        "-query",
        str(query_fasta),
        "-outfmt",
        "6 qseqid sseqid pident bitscore evalue",
        "-max_target_seqs",
        str(int(max_target_seqs)),
        "-max_hsps",
        "1",
        "-evalue",
        str(float(evalue)),
        "-out",
        str(out),
    ]
    if seqidlist is not None:
        cmd.extend(["-seqidlist", str(seqidlist)])

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()
    return t1 - t0


def parse_blast_tabular(path: str | Path) -> dict[str, list[BlastHit]]:
    """Parse BLAST outfmt 6 with fields: qseqid sseqid pident bitscore evalue."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    hits_by_q: dict[str, list[BlastHit]] = {}
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 5:
                continue
            qseqid, sseqid, pident, bitscore, evalue = parts[:5]
            hit = BlastHit(
                qseqid=qseqid,
                sseqid=sseqid,
                pident=float(pident),
                bitscore=float(bitscore),
                evalue=float(evalue),
            )
            hits_by_q.setdefault(qseqid, []).append(hit)

    # Ensure sorted by bitscore descending (defensive).
    for q, hits in hits_by_q.items():
        hits.sort(key=lambda h: h.bitscore, reverse=True)
        hits_by_q[q] = hits
    return hits_by_q


def write_seqidlist(path: str | Path, ids: Iterable[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for pid in ids:
            pid = str(pid).strip()
            if pid:
                f.write(pid + "\n")
    return p

