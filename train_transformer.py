"""Optional: fine-tune DistilBERT, which reads meaning rather than single words.

Needs:  pip install -r requirements-transformer.txt
Run:    python train_transformer.py [--sample 12000] [--epochs 1]
Saves the model to models/distilbert/ — the app uses it automatically if present,
averaging its prediction with the word model's.
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import load_data

BASE = "distilbert-base-uncased"
OUT = Path("models/distilbert")
LABELS = ["FAKE", "REAL"]


def batches(tokenizer, texts, labels, batch_size, shuffle):
    data = list(zip(texts, labels))
    def collate(rows):
        enc = tokenizer([t for t, _ in rows], truncation=True, max_length=256, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([LABELS.index(y) for _, y in rows])
        return enc
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=12000, help="training articles (CPU/laptop friendly)")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    df = load_data()
    # Same split as train.py, so the test articles are unseen by both models
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train = train.sample(min(args.sample, len(train)), random_state=42)
    test = test.sample(min(4000, len(test)), random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    model.train()
    for epoch in range(args.epochs):
        loader = batches(tokenizer, train["cleaned"].tolist(), train["label"].tolist(), 16, True)
        for step, batch in enumerate(loader):
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} loss {loss.item():.4f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in batches(tokenizer, test["cleaned"].tolist(), test["label"].tolist(), 32, False):
            batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            preds += [LABELS[i] for i in logits.argmax(-1).cpu().numpy()]
    accuracy = accuracy_score(test["label"], preds)
    print(f"DistilBERT test accuracy: {accuracy:.2%} ({len(test):,} articles)")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    (OUT / "metrics.json").write_text(json.dumps({"accuracy": accuracy, "train_size": len(train), "test_size": len(test)}, indent=2))
    print(f"Saved to {OUT}/")


if __name__ == "__main__":
    main()
