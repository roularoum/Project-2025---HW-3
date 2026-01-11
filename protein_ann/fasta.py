from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO


@dataclass(frozen=True)
class FastaRecord:
    """A FASTA record."""

    id: str
    sequence: str


def _parse_fasta_stream(handle: TextIO) -> Iterator[FastaRecord]:
    header: str | None = None
    seq_chunks: list[str] = []

    for raw in handle:
        line = raw.strip()
        if not line:
            continue

        if line.startswith(">"):
            if header is not None:
                seq = "".join(seq_chunks).replace(" ", "").upper()
                yield FastaRecord(id=header, sequence=seq)
            header = line[1:].split()[0]
            seq_chunks = []
        else:
            seq_chunks.append(line)

    if header is not None:
        seq = "".join(seq_chunks).replace(" ", "").upper()
        yield FastaRecord(id=header, sequence=seq)


def read_fasta(path: str | Path) -> Iterator[FastaRecord]:
    """Stream FASTA records from a file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        yield from _parse_fasta_stream(f)


def count_fasta_records(path: str | Path) -> int:
    """Count FASTA records quickly by counting '>' header lines."""
    p = Path(path)
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith(">"):
                n += 1
    return n


def write_fasta(path: str | Path, records: Iterable[FastaRecord]) -> None:
    """Write FASTA records to a file."""
    p = Path(path)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(f">{rec.id}\n")
            # Wrap at 60 chars for readability.
            seq = rec.sequence.strip().replace(" ", "")
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")

