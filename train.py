# -------------------------------
# file: src/train.py
# -------------------------------
import os, math, yaml, pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from utils.seed import set_seed
from utils.logger import log_kv, console
from data.datasets import MTMultiTaskDataset
from data.tagging import POS_VOCAB, DEP_VOCAB
from models.posdep_offense_trans import POSDEP_Offense_Trans
from losses import seq_ce_loss, sentence_ce_loss, info_nce


def collate(batch, pad_id: int = 1):
    # pad_id=1 for MBART (</s>)
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == 'offense_label':
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=pad_id)
    return out


def main(cfg_path: str = 'config.yaml'):
    set_seed()
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(cfg['output_dir'], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    offense_map = {name: i for i, name in enumerate(cfg['offense_labels'])}

    dataset = MTMultiTaskDataset(
        parallel_tsv=cfg['parallel_tsv'],
        offense_tsv=cfg['offense_tsv'],
        tokenizer=tokenizer,
        max_len=cfg['max_len'],
        spacy_en=cfg['spacy_en'],
        stanza_ta=cfg['stanza_ta'],
        offense_label_map=offense_map
    )

    N = len(dataset)
    train_n = int(0.8*N); val_n = N - train_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], collate_fn=collate)

    model = POSDEP_Offense_Trans(
        base_model=cfg['model_name'],
        n_pos=len(POS_VOCAB)+1,
        n_dep=len(DEP_VOCAB)+1,
        n_offense=len(cfg['offense_labels']),
        n_heads=8
    ).cuda() if torch.cuda.is_available() else POSDEP_Offense_Trans(
        base_model=cfg['model_name'],
        n_pos=len(POS_VOCAB)+1,
        n_dep=len(DEP_VOCAB)+1,
        n_offense=len(cfg['offense_labels']),
        n_heads=8
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total_steps = cfg['num_epochs'] * math.ceil(len(train_loader))
    sched = get_linear_schedule_with_warmup(optim, int(cfg['warmup_ratio']*total_steps), total_steps)

    for epoch in range(1, cfg['num_epochs']+1):
        model.train()
        losses = []
        for batch in train_loader:
            for k in batch: batch[k] = batch[k].cuda() if torch.cuda.is_available() else batch[k]
            out = model(batch['src_ids'], batch['src_mask'], batch['pos_ids'], batch['dep_ids'], batch['tgt_ids'])
            # Losses
            L_mt = seq_ce_loss(out['lm_logits'], batch['tgt_ids'], ignore_index=tokenizer.pad_token_id)
            L_off_s = sentence_ce_loss(out['offense_sent_logits'], batch['offense_label'])
            # token-level labels not provided -> use sent label broadcast as weak supervision
            tok_labels = batch['offense_label'].unsqueeze(1).repeat(1, out['offense_tok_logits'].size(1))
            L_off_t = seq_ce_loss(out['offense_tok_logits'], tok_labels, ignore_index=-100)
            # POS/DEP self-supervision (predict tags)
            L_pos = seq_ce_loss(out['pos_logits'], batch['pos_ids'])
            L_dep = seq_ce_loss(out['dep_logits'], batch['dep_ids'])
            # Contrastive consistency: encoder vs POS/DEP projections
            L_con = info_nce(out['z_enc'], out['z_pos']) + info_nce(out['z_enc'], out['z_dep'])
            loss = (
                L_mt
                + cfg['lambda_offense_sent']*L_off_s
                + cfg['lambda_offense_tok']*L_off_t
                + cfg['lambda_pos']*L_pos
                + cfg['lambda_dep']*L_dep
                + cfg['lambda_contrastive']*L_con
            )
            optim.zero_grad(); loss.backward(); optim.step(); sched.step()
            losses.append(float(loss.detach().cpu()))
        log_kv(f"Epoch {epoch}", {"train_loss": np.mean(losses)})
        # (Optional) save
        if epoch % cfg['save_every'] == 0:
            ckpt = os.path.join(cfg['output_dir'], f"checkpoint_epoch{epoch}.pt")
            torch.save({'model': model.state_dict(), 'cfg': cfg}, ckpt)
            console.print(f"Saved {ckpt}")

if __name__ == '__main__':
    main()

