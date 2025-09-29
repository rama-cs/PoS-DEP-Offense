# -------------------------------
# file: src/evaluate_domains.py
# -------------------------------
"""Evaluate per domain: Formal / Noisy / Offensive-Enriched.
Config expects parallel/formal TSVs and optional code-mixed/offensive TSVs.
"""
import yaml, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.datasets import MTMultiTaskDataset
from data.tagging import POS_VOCAB, DEP_VOCAB
from models.posdep_offense_trans import POSDEP_Offense_Trans
from utils.metrics import compute_bleu, macro_f1, hallucination_rate, hallucination_free_pct, avg_hallucinations
from utils.alignment import Aligner
from utils.logger import log_kv


def collate(batch, pad_id: int = 1):
    keys = batch[0].keys(); out = {}
    for k in keys:
        if k == 'offense_label': out[k] = torch.stack([b[k] for b in batch])
        else: out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=pad_id)
    return out

@torch.no_grad()
def eval_one(ds, model, tokenizer):
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    aligner = Aligner()
    refs, hyps = [], []
    y_true, y_pred = [], []
    unaligned, totals = [], []
    for batch in loader:
        out = model(batch['src_ids'], batch['src_mask'], batch['pos_ids'], batch['dep_ids'], batch['tgt_ids'])
        pred_ids = out['lm_logits'].argmax(-1)
        for i in range(pred_ids.size(0)):
            refs.append(tokenizer.decode(batch['tgt_ids'][i], skip_special_tokens=True))
            hyps.append(tokenizer.decode(pred_ids[i], skip_special_tokens=True))
            y_true.append(int(batch['offense_label'][i]))
            y_pred.append(int(out['offense_sent_logits'][i].argmax(-1)))
            s_toks = tokenizer.convert_ids_to_tokens(batch['src_ids'][i])
            t_toks = tokenizer.convert_ids_to_tokens(pred_ids[i])
            u, t = aligner.align_counts(s_toks, t_toks)
            unaligned.append(u); totals.append(t)
    return {
        'BLEU': compute_bleu(refs, hyps),
        'F1': macro_f1(y_true, y_pred),
        'HR': hallucination_rate(unaligned, totals),
        'HFO': hallucination_free_pct(unaligned),
        'AHS': avg_hallucinations(unaligned)
    }


def main(cfg_path: str, ckpt_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    model = POSDEP_Offense_Trans(cfg['model_name'], len(POS_VOCAB)+1, len(DEP_VOCAB)+1, len(cfg['offense_labels']))
    sd = torch.load(ckpt_path, map_location='cpu'); model.load_state_dict(sd['model']); model.eval()

    results = {}
    if 'parallel_tsv' in cfg:
        ds_formal = MTMultiTaskDataset(cfg['parallel_tsv'], cfg.get('offense_tsv'), tokenizer, cfg['max_len'], cfg['spacy_en'], cfg['stanza_ta'], {n:i for i,n in enumerate(cfg['offense_labels'])})
        results['Formal'] = eval_one(ds_formal, model, tokenizer)
    if cfg.get('code_mixed_tsv'):
        ds_noisy = MTMultiTaskDataset(cfg['code_mixed_tsv'], cfg.get('offense_tsv'), tokenizer, cfg['max_len'], cfg['spacy_en'], cfg['stanza_ta'], {n:i for i,n in enumerate(cfg['offense_labels'])})
        results['Noisy'] = eval_one(ds_noisy, model, tokenizer)
    if cfg.get('offensive_enriched_tsv'):
        ds_off = MTMultiTaskDataset(cfg['offensive_enriched_tsv'], cfg.get('offense_tsv'), tokenizer, cfg['max_len'], cfg['spacy_en'], cfg['stanza_ta'], {n:i for i,n in enumerate(cfg['offense_labels'])})
        results['Offensive'] = eval_one(ds_off, model, tokenizer)

    for k, v in results.items():
        log_kv(f"Domain: {k}", {kk: round(vv, 3) for kk, vv in v.items()})

if __name__ == '__main__':
    main('config.yaml', 'outputs/checkpoint_epoch5.pt')

