# -------------------------------
# file: src/utils/onr.py
# -------------------------------
"""ONR computation utilities.
ONR = (# of gold offensive spans that are neutralized) / (# of gold offensive spans)
We consider a span neutralized if post-neutralization string no longer contains a lexeme matching the original offensive lexicon OR
if similarity to an allowed synonym exceeds a threshold (hook for future semantic check).
"""
from typing import List, Tuple
import re

def span_overlap(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])

def compute_onr(gold_spans: List[Tuple[int,int]], replaced_spans: List[Tuple[int,int]]) -> float:
    if not gold_spans:
        return 100.0
    covered = 0
    for g in gold_spans:
        if any(span_overlap(g, r) for r in replaced_spans):
            covered += 1
    return covered / len(gold_spans) * 100.0
