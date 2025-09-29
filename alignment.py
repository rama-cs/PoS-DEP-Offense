# -------------------------------
# file: src/utils/alignment.py
# -------------------------------
"""Alignment helpers using simalign (if available)."""
from typing import List, Tuple
try:
    from simalign import SentenceAligner
    _HAS_SIMALIGN = True
except Exception:
    _HAS_SIMALIGN = False

class Aligner:
    def __init__(self, method: str = 'itermax'):
        if _HAS_SIMALIGN:
            self.aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods=[method])
        else:
            self.aligner = None
    def align_counts(self, src_tokens: List[str], tgt_tokens: List[str]) -> Tuple[int,int]:
        """Return (unaligned_count, total_target_tokens)"""
        if self.aligner is None:
            # Fallback heuristic: consider identical (lowercased) tokens as aligned
            src_set = set([s.lower() for s in src_tokens])
            unaligned = sum(1 for t in tgt_tokens if t.lower() not in src_set)
            return unaligned, len(tgt_tokens)
        out = self.aligner.get_word_aligns(src_tokens, tgt_tokens)
        aligned_tgt = set([j for (_, j) in out['itermax']])
        unaligned = sum(1 for j in range(len(tgt_tokens)) if j not in aligned_tgt)
        return unaligned, len(tgt_tokens)
