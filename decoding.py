# -------------------------------
# file: src/decoding.py
# -------------------------------
"""Greedy & Beam search decoding wrappers for generation."""
from typing import List
import torch
from transformers import AutoTokenizer

@torch.no_grad()
def greedy_decode(model, tokenizer: AutoTokenizer, src_ids, src_mask, pos_ids, dep_ids, max_len=160):
    B = src_ids.size(0)
    ys = torch.full((B,1), tokenizer.bos_token_id or tokenizer.eos_token_id, device=src_ids.device, dtype=torch.long)
    for _ in range(max_len):
        out = model(src_ids, src_mask, pos_ids, dep_ids, ys)
        next_token = out['lm_logits'][:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == tokenizer.eos_token_id).all():
            break
    return ys

@torch.no_grad()
def beam_search_decode(model, tokenizer: AutoTokenizer, src_ids, src_mask, pos_ids, dep_ids, beam_size=4, max_len=160):
    # Lightweight beam search (batch=1 for simplicity)
    hyps = []
    for i in range(src_ids.size(0)):
        beams = [(torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=src_ids.device), 0.0)]
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                out = model(src_ids[i:i+1], src_mask[i:i+1], pos_ids[i:i+1], dep_ids[i:i+1], seq)
                logp = out['lm_logits'][:,-1].log_softmax(-1)
                topk = torch.topk(logp, beam_size, dim=-1)
                for k in range(beam_size):
                    tok = topk.indices[0, k].view(1,1)
                    sc = score + float(topk.values[0, k])
                    new_beams.append((torch.cat([seq, tok], dim=1), sc))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(b[0][0,-1].item()==tokenizer.eos_token_id for b in beams):
                break
        hyps.append(beams[0][0])
    return torch.nn.utils.rnn.pad_sequence([h.squeeze(0) for h in hyps], batch_first=True, padding_value=tokenizer.pad_token_id)
