"""
fasta.py
========
To arxeio auto periexei voithitikes synartiseis gia na diavazoume, na grafoume
kai na metrame FASTA arxeia (protein sequences). To FASTA einai format opou:
- grammes pou arxizoun me '>' einai header (id tis protein)
- oi epomenes grammes einai to sequence (amino acid letters)
To parsing ginetai stream (me yield), wste na mporoume na doulepoume kai me megala arxeia
xwris na fortonoume ola ta dedomena sti mnimi.
"""

from __future__ import annotations  # epitrepei type hints san strings kai "forward references" xwris provlimata

from dataclasses import dataclass  # gia na ftiaxnoume apla classes pou krataνε dedomena (records)
from pathlib import Path  # gia safe xrhsh paths (Windows/Linux)
from typing import Iterable, Iterator, TextIO  # typoi gia iterable/iterator kai file handles


@dataclass(frozen=True)  # dataclass = auto-generate init/eq/repr, frozen=True = immutable (den allazei meta)
class FastaRecord:
 # ena record FASTA = (id, sequence)

    id: str  # to id/protein name apo to header
    sequence: str  # to amino-acid sequence tis protein


def _parse_fasta_stream(handle: TextIO) -> Iterator[FastaRecord]:
    # diavazei line-by-line apo file handle kai kanei yield FastaRecord objects
    header: str | None = None  # kratame to trexon header, None an den exoume diavasei akoma '>'
    seq_chunks: list[str] = []  # lista me kommata sequence, ta kollame meta se 1 string

    for raw in handle:  # loop se kathe grammh tou arxeiou
        line = raw.strip()  # afairei kena apo arxh/telos
        if not line:  # an i grammh einai adeia
            continue  

        if line.startswith(">"):  # an ksekinaei me '>' einai header
            if header is not None:  # an exoume proigoumeno header, prepei na "kleisoume" to record
                seq = "".join(seq_chunks).replace(" ", "").upper()  # enwnoume ta chunks, afairoume spaces, kanoume kefalaia
                yield FastaRecord(id=header, sequence=seq)  # epistrefoume record gia tin proigoumeni protein
            header = line[1:].split()[0]  # pairnoume to header xwris to '>' kai kratame mono to prwto token (mexri to prwto space)
            seq_chunks = []  # reset to sequence gia to kainourgio record
        else:
            seq_chunks.append(line)  # an den einai header, einai kommati sequence -> to prosthetoume

    if header is not None:  # telos arxeiou: an exoume record pou den exei ginei yield akoma
        seq = "".join(seq_chunks).replace(" ", "").upper()  # ftiaxnoume to teliko sequence
        yield FastaRecord(id=header, sequence=seq)  # yield to teleutaio record


def read_fasta(path: str | Path) -> Iterator[FastaRecord]:
    # diavazei ena FASTA arxeio kai gurnaei iterator apo records
    p = Path(path)  # metatrepei string se Path (an einai idi Path, einai ok)
    with p.open("r", encoding="utf-8") as f:  # anoigei arxeio se read mode me utf-8 encoding
        yield from _parse_fasta_stream(f)  # dinei to file handle sto parser kai "proothei" ola ta records


def count_fasta_records(path: str | Path) -> int:
     # metraei posa records exei ena FASTA arxeio
    p = Path(path)  # metatrepei path se Path object
    n = 0  # counter gia poses proteins/records
    with p.open("r", encoding="utf-8") as f:  # anoigei arxeio gia diavasma
        for raw in f:  # pername grammi-grammi
            if raw.startswith(">"):  # kathe header grammh antistoixei se ena record
                n += 1  # auksanoume metriti
    return n  # epistrefoume ton arithmo records


def write_fasta(path: str | Path, records: Iterable[FastaRecord]) -> None:
     # grafei lista/iterable apo FastaRecord se FASTA format
    p = Path(path)  # path se Path
    with p.open("w", encoding="utf-8", newline="\n") as f:  # anoigei arxeio se write mode, me newline \n gia consistent output tupou Linux
        for rec in records:  # gia kathe record pou theloume na grapsoume
            f.write(f">{rec.id}\n")  # grapse header line me '>' kai to id
            # spame to sequence se grammes 60 xarakthron gia na einai euanagnwsto
            seq = rec.sequence.strip().replace(" ", "")  # katharizoume sequence (trim + afairesh spaces)
            for i in range(0, len(seq), 60):  # kanoume loop ana 60 xarakthres
                f.write(seq[i : i + 60] + "\n")  # grafoume 60-char chunk kai newline
