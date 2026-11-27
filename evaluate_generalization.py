"""Check how well the model holds up on articles from a dataset it never saw.

Same-dataset test scores are high partly because each dataset has its own writing style
(one outlet's wire copy vs. a handful of blogs). This trains on ISOT (plus the parts of
WELFake that are not ISOT or McIntire, if WELFake is present) and tests on the separate
McIntire "fake_or_real_news" dataset. WELFake was partly built from McIntire, so every
training article that matches a McIntire article by text, headline or opening is removed first.

Needs:  data/Fake.csv, data/True.csv (ISOT), data/fake_or_real_news.csv
        (https://github.com/lutzhamel/fake-news/tree/master/data), optional data/WELFake_Dataset.csv
Run:    python evaluate_generalization.py [--isot-only]
        (after train.py; adds the result to metrics.json unless --isot-only)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from text_utils import clean_text, join_title
from train import fit_model, load_isot, load_welfake


def prepare(df):
    df = df.copy()
    df["text"] = [join_title(t, x) for t, x in zip(df["title"].fillna(""), df["text"].fillna(""))]
    df["cleaned"] = df["text"].map(clean_text)
    df = df[df["cleaned"].str.split().str.len() >= 20].drop_duplicates("cleaned")
    return df[~df.duplicated("cleaned", keep=False)].reset_index(drop=True)


def overlap_keys(df):
    """Loose fingerprints, so copies with different formatting still match."""
    letters = df["cleaned"].str.replace(r"[^a-z]", "", regex=True)
    title = df["title"].fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
    return {"text": set(df["cleaned"]),
            "opening": set(letters.str[50:250][letters.str.len() > 300]),
            "title": set(title[title.str.len() > 20])}


def drop_overlap(df, keys):
    letters = df["cleaned"].str.replace(r"[^a-z]", "", regex=True)
    title = df["title"].fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
    hit = df["cleaned"].isin(keys["text"]) | letters.str[50:250].isin(keys["opening"]) | title.isin(keys["title"])
    return df[~hit].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isot-only", action="store_true", help="train on ISOT alone")
    args = parser.parse_args()

    test = prepare(pd.read_csv(Path("data") / "fake_or_real_news.csv"))
    isot = prepare(load_isot())
    test = drop_overlap(test, overlap_keys(isot))  # a few McIntire articles are also in ISOT
    test_keys = overlap_keys(test)
    isot = drop_overlap(isot, test_keys)
    train = isot
    welfake = None if args.isot_only else load_welfake()
    if welfake is not None:
        extra = drop_overlap(drop_overlap(prepare(welfake), test_keys), overlap_keys(isot))
        train = pd.concat([isot, extra], ignore_index=True)
    sources = ", ".join(f"{s} {n:,}" for s, n in train["source"].value_counts().items())

    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(train, train["label"])
    same = [accuracy_score(train["label"].iloc[b], fit_model(train.iloc[a]).predict(train["text"].iloc[b]))
            for a, b in folds]

    pred = fit_model(train).predict(test["text"])
    is_fake = test["label"].values == "FAKE"
    result = {
        "dataset": "McIntire fake_or_real_news",
        "test_size": len(test),
        "trained_on": {s: int(n) for s, n in train["source"].value_counts().items()},
        "same_dataset_accuracy": float(np.mean(same)),
        "accuracy": accuracy_score(test["label"], pred),
        "fake_recall": float((pred[is_fake] == "FAKE").mean()),
        "real_recall": float((pred[~is_fake] == "REAL").mean()),
    }
    print(f"Trained on: {sources}")
    print(f"Same datasets (5-fold):          {np.mean(same):.1%} +/- {np.std(same):.1%}")
    print(f"New dataset (McIntire, {len(test):,}):  {result['accuracy']:.1%}")
    print(f"  fake caught {result['fake_recall']:.0%}, real kept {result['real_recall']:.0%}")

    metrics_path = Path("metrics.json")
    if not args.isot_only and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        metrics["unseen_dataset"] = result
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print("Added the result to metrics.json")


if __name__ == "__main__":
    main()
