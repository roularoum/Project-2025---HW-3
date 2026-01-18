from __future__ import annotations  # epitrepei type hints me forward refs

import shutil  # gia shutil.which gia na vrei executables sto PATH
import subprocess  # gia na trexei external entoles px makeblastdb, blastp
import time  # gia xronometra 
from dataclasses import dataclass  # gia dataclass BlastHit
from pathlib import Path  # gia xeirismo paths arxeiwn
from typing import Dict, Iterable, List, Sequence  # type hints 


@dataclass(frozen=True)  # immutable record gia ena BLAST hit
class BlastHit:  # apothikeuei ta pedia pou mas endiaferoun apo BLAST
    qseqid: str  # id tou query sequence
    sseqid: str  # id tou subject database sequence hit
    pident: float  # percent identity poso moiazoun
    bitscore: float  # bitscore pio megalo = pio kalo hit
    evalue: float  # e-value pio mikro = pio shmantiko


def require_blast_tools() -> tuple[str, str]:  # elegxei oti yparxoun ta BLAST kai executables
    """Return (makeblastdb, blastp) executables or raise with actionable error."""  # docstring poy gyrnaei paths h kanei error
    makeblastdb = shutil.which("makeblastdb")  # psaxnei to makeblastdb sto PATH
    blastp = shutil.which("blastp")  # psaxnei to blastp sto PATH
    if not makeblastdb or not blastp:  # an leipei kapoio
        raise RuntimeError(  # petaei error me odigies
            "BLAST+ tools not found in PATH. Required: makeblastdb, blastp.\n"
            "Install NCBI BLAST+ and ensure executables are available in PATH."
        )
    return makeblastdb, blastp  # epistrefei ta paths twn executables


def ensure_blast_db(fasta_path: str | Path, db_dir: str | Path, db_name: str = "swissprot_db") -> Path:  # ftiaxnei BLAST protein DB
    """Ensure a protein BLAST database exists, building it if missing.  # docstring: an den yparxei DB tin ftiaxnei

    Returns the DB prefix path (without file extension).  # epistrefei to prefix xwris .pin/.psq/.phr
    """
    makeblastdb, _blastp = require_blast_tools()  # pairnei to makeblastdb kai blastp 
    fasta = Path(fasta_path)  # metatrepei to fasta path se Path
    if not fasta.exists():  # an den yparxei to FASTA
        raise FileNotFoundError(str(fasta))  # error

    out_dir = Path(db_dir)  # folder pou tha mpei i DB
    out_dir.mkdir(parents=True, exist_ok=True)  # ftiaxnei folder an leipei
    prefix = out_dir / db_name  # prefix onoma DB mesa sto folder

    # arxeia oti yparxei h DB
    pin = prefix.with_suffix(".pin")  # index file .pin
    psq = prefix.with_suffix(".psq")  # sequences file .psq
    phr = prefix.with_suffix(".phr")  # header file .phr
    # extra arxeia gia filtering me accession ids
    # xreiazontai gia to -seqidlist
    pog = prefix.with_suffix(".pog")  # file gia seqid index
    pos = prefix.with_suffix(".pos")  # file gia seqid index
    if pin.exists() and psq.exists() and phr.exists() and pog.exists() and pos.exists():  # an yparxoun ola
        return prefix  # tote i DB einai etoimi gyrnaei to prefix

    cmd = [  # entoli makeblastdb gia na xtisei protein database
        makeblastdb,  # executable
        "-in",  # input fasta flag
        str(fasta),  # to fasta arxeio
        "-dbtype",  # typos database
        "prot",  # protein DB
        "-parse_seqids",  # krataei ta sequence ids gia na doulevei -seqidlist
        "-out",  # output prefix flag
        str(prefix),  # output prefix
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # trexei tin entoli kai kanei error an apotyxei
    return prefix  # epistrefei to prefix pou xrisimopoieitai meta


def run_blastp(  # trexei blastp kai metraei poso krataei
    db_prefix: str | Path,  # prefix tis BLAST database
    query_fasta: str | Path,  # FASTA me queries
    out_path: str | Path,  # pou na grapsei to output tsv
    max_target_seqs: int,  # posa top hits na zitaei ana query
    seqidlist: str | Path | None = None,  # periorizei ta hits se sygkekrimena ids
    evalue: float = 1e6,  # evalue threshold 
) -> float:
    """Run blastp and return elapsed seconds."""  # docstring: epistrefei xrono ektelesis
    _makeblastdb, blastp = require_blast_tools()  # pairnei to blastp executable

    out = Path(out_path)  # Path object gia output
    out.parent.mkdir(parents=True, exist_ok=True)  # ftiaxnei output folder an leipei

    cmd = [  # blastp command me outfmt 6
        blastp,  # executable
        "-db",  # database flag
        str(db_prefix),  # database prefix
        "-query",  # query flag
        str(query_fasta),  # query fasta
        "-outfmt",  # output format flag
        "6 qseqid sseqid pident bitscore evalue",  # fields pou theloume sto TSV
        "-max_target_seqs",  # top hits flag
        str(int(max_target_seqs)),  # posa hits
        "-max_hsps",  # periorizei HSPs ana pair
        "1",  # 1 HSP ana hit 
        "-evalue",  # evalue flag
        str(float(evalue)),  # evalue threshold
        "-out",  # output flag
        str(out),  # output file
    ]
    if seqidlist is not None:  # an exoume restrict list
        cmd.extend(["-seqidlist", str(seqidlist)])  # prosthese -seqidlist gia filtering

    t0 = time.perf_counter()  # start timer
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # trexei blastp kai kanei error an apotyxei
    t1 = time.perf_counter()  # stop timer
    return t1 - t0  # epistrefei elapsed seconds


def parse_blast_tabular(path: str | Path) -> dict[str, list[BlastHit]]:  # kanei parse to BLAST TSV se dict
    """Parse BLAST outfmt 6 with fields: qseqid sseqid pident bitscore evalue."""  # docstring: poia pedia perimenei
    p = Path(path)  # Path object
    if not p.exists():  # an den yparxei to file
        raise FileNotFoundError(str(p))  # error

    hits_by_q: dict[str, list[BlastHit]] = {}  # qseqid -> lista hits
    with p.open("r", encoding="utf-8") as f:  # anoigei TSV gia diavasma
        for ln in f:  # diavazei grammi grammi
            ln = ln.strip()  # trim
            if not ln:  # agnoei adeies grammes
                continue
            parts = ln.split("\t")  # spaei se stiles me tab
            if len(parts) < 5:  # an den exei ola ta pedia
                continue  # agnoei grammi
            qseqid, sseqid, pident, bitscore, evalue = parts[:5]  # pairnei ta prwta 5 pedia
            hit = BlastHit(  # ftiaxnei BlastHit object
                qseqid=qseqid,  # query id
                sseqid=sseqid,  # subject id
                pident=float(pident),  # percent identity se float
                bitscore=float(bitscore),  # bitscore se float
                evalue=float(evalue),  # evalue se float
            )
            hits_by_q.setdefault(qseqid, []).append(hit)  # prosthetei to hit sti lista tou qseqid

    # ta taxinomei me bitscore pio kalo prwto
    for q, hits in hits_by_q.items():  # loop ana query
        hits.sort(key=lambda h: h.bitscore, reverse=True)  # sort me bitscore fthinousa
        hits_by_q[q] = hits  # xana assign 
    return hits_by_q  # epistrefei to dict


def write_seqidlist(path: str | Path, ids: Iterable[str]) -> Path:  # grafei lista ids se arxeio 1 id ana grammi
    p = Path(path)  # Path object
    p.parent.mkdir(parents=True, exist_ok=True)  # ftiaxnei folder an leipei
    with p.open("w", encoding="utf-8", newline="\n") as f:  # anoigei gia grapsimo me unix newlines
        for pid in ids:  # gia kathe id
            pid = str(pid).strip()  # kanei string kai trim
            if pid:  # an den einai adeio
                f.write(pid + "\n")  # grafei to id kai newline
    return p  # epistrefei to path tou arxeiou pou egine save
