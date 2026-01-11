from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MethodSummary:
    method: str
    time_per_query_s: float
    qps: float
    recall_at_n: float


def recall_at_n(
    ann_ids: Sequence[str],
    blast_top_ids: Sequence[str],
    n: int,
) -> float:
    """Recall@N against BLAST Top-N IDs.

    Defined as |ANN ∩ BLAST| / N.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    blast_set = set(blast_top_ids[:n])
    ann_set = set(ann_ids[:n])
    return len(ann_set & blast_set) / float(n)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))

