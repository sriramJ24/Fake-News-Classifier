"""Fine-tune DistilBERT on ISOT only (optionally + B part) and score it on held-out sets.

python bert_exp.py --cleaner v2 --train A --out bert_A
Saves model + P(fake) on A-test and B to <out>/. Never reads McIntire.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import HERE, load_sources, scores

LABELS = ["FAKE", "REAL"]


def loader(tok, texts, labels, bs, shuffle):
    def collate(rows):
        enc = tok([t for t, _ in rows], truncation=True, max_length=256, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([LABELS.index(y) for _, y in rows])
        return enc
    return DataLoader(list(zip(texts, labels)), batch_size=bs, shuffle=shuffle, collate_fn=collate)


def p_fake(model, tok, texts, device):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader(tok, texts, ["FAKE"] * len(texts), 64, False):
            batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            out += torch.softmax(logits, -1)[:, 0].cpu().tolist()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaner", default="v2")
    ap.add_argument("--train", default="A")
    ap.add_argument("--sample", type=int, default=12000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = HERE / args.out
    out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    a, b = load_sources(args.cleaner)
    a_tr, a_te = train_test_split(a, test_size=0.2, random_state=42, stratify=a["label"])
    train = a_tr.sample(min(args.sample, len(a_tr)), random_state=42)
    a_te = a_te.sample(4000, random_state=42)

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()
    dl = loader(tok, train["cleaned"].tolist(), train["label"].tolist(), 16, True)
    for step, batch in enumerate(dl):
        loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 100 == 0:
            print(f"step {step}/{len(dl)} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    res = {}
    for name, df in [("A_test", a_te), ("B", b)]:
        p = p_fake(model, tok, df["cleaned"].tolist(), device)
        np.save(out / f"p_{name}.npy", p)
        df[["label"]].to_pickle(out / f"y_{name}.pkl")
        res[name] = scores(df["label"], np.where(p >= 0.5, "FAKE", "REAL"))
        print(name, res[name], flush=True)
    (out / "val.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
