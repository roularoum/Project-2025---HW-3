#!/usr/bin/env python3
from __future__ import annotations  # kanei ta type hints na doulevoun kai me forward references

import argparse  # gia command line arguments CLI
import sys  # gia sys.path 
from pathlib import Path  

_THIS_DIR = Path(__file__).resolve().parent  # to folder pou vrisketai to trexon arxeio

if str(_THIS_DIR) not in sys.path:  # an to folder den einai sto python path
    sys.path.insert(0, str(_THIS_DIR))  # to vazei prwto
    
from protein_ann.fasta import FastaRecord, read_fasta, write_fasta  # typos record kai read/write FASTA 

 # synarthsh pou orizei ta CLI args kai ta epistrefei
def parse_args() -> argparse.Namespace: 
    p = argparse.ArgumentParser(  # ftiaxnei parser gia command line
        description="Utility: write a FASTA subset (first N records). Useful for smoke-tests."  #krataei ta prwta N records
    )
    p.add_argument("-i", "--input", required=True, help="Input FASTA")  # input fasta arxeio
    p.add_argument("-o", "--output", required=True, help="Output FASTA")  # output fasta arxeio
    p.add_argument("--n", type=int, required=True, help="Number of records to keep")  # posa records na kratisei
    return p.parse_args()  # diavazei kai gyrnaei ta args

# diavazei fasta kai grafei subset
def main() -> None:  
    args = parse_args()  # pairnei ta CLI args
    n = int(args.n)  # metatrepei to n se int 
    if n <= 0: 
        raise ValueError("--n must be positive")  # to n prepei na einai thetiko

    inp = Path(args.input)  # path object gia input fasta
    out = Path(args.output)  # path object gia output fasta
    out.parent.mkdir(parents=True, exist_ok=True)  # ftiaxnei to output folder an den yparxei

    records: list[FastaRecord] = []  # lista pou tha kratisei ta prwta N records
    for rec in read_fasta(inp):  # diavazei ena-ena ta records apo to input FASTA
        records.append(rec)  # prosthetei to record sti lista
        if len(records) >= n:  # an ftasame ta N records
            break  # stamataei to diavasma den xreiazetai allo

    if not records:  # an den vrethike kanena record sto FASTA
        raise ValueError("No FASTA records found")  # skasei me error

    write_fasta(out, records)  # grafei ta records sto output FASTA
    print(f"[subset] Wrote {len(records)} records to {out}")  # posa records egrafse kai pou


if __name__ == "__main__": 
    main()  