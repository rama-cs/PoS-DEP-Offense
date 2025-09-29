 -------------------------------
# file: src/utils/metrics.py
# -------------------------------
from typing import List, Tuple
import numpy as np
import sacrebleu
from sklearn.metrics import f1_score

def compute_bleu(refs: List[str], hyps: List[str]) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs]).score

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

# Hallucination Rate (HR)
# Given per-sentence counts of unaligned target tokens and total target tokens

def hallucination_rate(unaligned_counts: List[int], total_counts: List[int]) -> float:
    u = np.sum(unaligned_counts); t = np.sum(total_counts)
    return float(u)/float(t+1e-8) * 100.0

# Hallucination-Free Outputs percentage

def hallucination_free_pct(unaligned_counts: List[int]) -> float:
    arr = np.array(unaligned_counts)
    return float(np.sum(arr==0))/float(len(arr)+1e-8) * 100.0

# Average Hallucinations per Sentence

def avg_hallucinations(unaligned_counts: List[int]) -> float:
    return float(np.mean(unaligned_counts))
