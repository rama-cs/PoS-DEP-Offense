import torch
import torch.nn as nn

def seq_ce_loss(logits, labels, ignore_index=-100):
    return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)

def sentence_ce_loss(logits, labels):
    return nn.functional.cross_entropy(logits, labels)

def info_nce(z_q, z_k, temperature: float = 0.1):
    """Simple InfoNCE across sequence positions (masked avg). z_*: (B,L,P)"""
    B, L, P = z_q.size()
    q = z_q.view(B*L, P)              # (N,P)
    k = z_k.view(B*L, P)              # (N,P)
    sim = torch.matmul(q, k.t())      # (N,N)
    labels = torch.arange(q.size(0), device=q.device)
    return nn.functional.cross_entropy(sim/temperature, labels)
