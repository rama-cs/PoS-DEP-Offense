from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .layers import SyntaxBiasedAttention, ContrastiveProjection

class POSDEP_Offense_Trans(nn.Module):
    def __init__(self, base_model: str, n_pos: int, n_dep: int, n_offense: int, n_heads: int = 8):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)
        d_model = self.cfg.hidden_size
        # Syntax embeddings
        self.pos_embed = nn.Embedding(n_pos, d_model, padding_idx=0)
        self.dep_embed = nn.Embedding(n_dep, d_model, padding_idx=0)
        # Attention layer to inject syntax
        self.syntax_attn = SyntaxBiasedAttention(d_model, n_heads)
        # Decoder head (tie weights with LM head approach)
        self.lm_head = nn.Linear(d_model, self.cfg.vocab_size)
        # Offense heads
        self.offense_sent = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_model, n_offense))
        self.offense_tok = nn.Linear(d_model, n_offense)
        # POS/DEP auxiliary heads (predict on encoder outputs)
        self.pos_head = nn.Linear(d_model, n_pos)
        self.dep_head = nn.Linear(d_model, n_dep)
        # Contrastive projection
        self.proj_enc = ContrastiveProjection(d_model)
        self.proj_pos = ContrastiveProjection(d_model)
        self.proj_dep = ContrastiveProjection(d_model)

    def forward(self, src_ids, src_mask, pos_ids, dep_ids, tgt_ids=None):
        enc = self.encoder(input_ids=src_ids, attention_mask=src_mask)
        h = enc.last_hidden_state  # (B, L, D)
        pos_e = self.pos_embed(pos_ids)
        dep_e = self.dep_embed(dep_ids)
        h2, attn_w = self.syntax_attn(h, pos_e, dep_e, key_padding_mask=src_mask==0)
        # LM logits for translation (teacher forcing uses tgt_ids outside)
        lm_logits = self.lm_head(h2)
        # Sentence offense (pool)
        pooled = (h2 * src_mask.unsqueeze(-1)).sum(dim=1) / (src_mask.sum(dim=1, keepdim=True) + 1e-8)
        offense_sent_logits = self.offense_sent(pooled)
        # Token offense
        offense_tok_logits = self.offense_tok(h2)
        # POS/DEP predictions (aux)
        pos_logits = self.pos_head(h2)
        dep_logits = self.dep_head(h2)
        # Contrastive projections
        z_enc = self.proj_enc(h2)              # (B,L,P)
        z_pos = self.proj_pos(pos_e).detach()  # stop-grad targets
        z_dep = self.proj_dep(dep_e).detach()
        return {
            'lm_logits': lm_logits,
            'offense_sent_logits': offense_sent_logits,
            'offense_tok_logits': offense_tok_logits,
            'pos_logits': pos_logits,
            'dep_logits': dep_logits,
            'z_enc': z_enc,
            'z_pos': z_pos,
            'z_dep': z_dep,
            'attn_w': attn_w
        }
