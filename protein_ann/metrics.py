"""
metrics.py
==========
To arxeio auto exei apla metrics/voithitika gia tin aksiologisi twn methodwn ANN
se sxesi me to BLAST. Ypologizei:
- Recall@N: posoi apo tous top-N geitones pou vrike to ANN einai kai mesa sto BLAST top-N
- mean: mesos oros gia times (px xronos ana query)
Episis exei ena dataclass (MethodSummary) pou krataei summary stats gia kathe methodo.
"""

from __future__ import annotations  # kanei type hints pio eukola/secure 

from dataclasses import dataclass  # gia na ftiaxnoume apla "data holder" classes
from typing import Iterable, Sequence  # Iterable = opoiadipote lista/generator, Sequence = list/tuple me indexing


@dataclass(frozen=True)  # frozen=True = immutable (den theloume na allazei meta)
class MethodSummary:
    method: str  # onoma methodou (lsh, hypercube, ivf, neural, ktl)
    time_per_query_s: float  # mesos xronos ana query se seconds
    qps: float  # queries per second (posa queries trexoun to deyterolepto)
    recall_at_n: float  # recall@N se sxesi me BLAST (poso kala vriskei "swstous" geitones)


def recall_at_n(
    ann_ids: Sequence[str],  # lista me IDs pou evgale i ANN methodos (kata seira rank)
    blast_top_ids: Sequence[str],  # lista me IDs pou evgale i BLAST (kata seira rank)
    n: int,  # to N tou Recall@N (px 10, 50)
) -> float:
    
    # eksigei oti metrame posoi koina stoixeia exoun ANN kai BLAST sta top-N, dia N

    if n <= 0:  # an kapoios dwsei n=0 h arnhtiko, den exei noima
        raise ValueError("n must be positive")  # petame error gia na min vgoun lathos metrics

    blast_set = set(blast_top_ids[:n])  # pairnoume ta prwta N BLAST IDs kai ta kanoume set gia grigoro membership
    ann_set = set(ann_ids[:n])  # pairnoume ta prwta N ANN IDs kai ta kanoume set

    return len(ann_set & blast_set) / float(n)  # h tomh tous dinei ta koinά IDs -> dia N = recall@N

#pragmatikoi arithmoi pou mporoume na tous kanoume loop
def mean(values: Iterable[float]) -> float:
    vals = list(values)  # metatrepoume to iterable se lista gia na mporoume na to ksanaxrhsimopoiisoume
    if not vals:  # an einai adeio (px den yparxoun times)
        return 0.0  
    return sum(vals) / float(len(vals))  # mesos oros = athroisma / plithos
