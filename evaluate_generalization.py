"""Check how well the word model holds up on articles from a dataset it never saw.

Same-dataset test scores are high partly because each dataset has its own
writing style (one outlet's wire copy vs. a handful of blogs). This trains on
ISOT only and tests on the separate McIntire "fake_or_real_news" dataset,
after removing any article that also appears in ISOT.

Needs:  data/Fake.csv, data/True.csv (ISOT) and data/fake_or_real_news.csv
        (https://github.com/lutzhamel/fake-news/tree/master/data)
Run:    python evaluate_generalization.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from text_utils import clean_text
from train import build_model, load_isot


def prepare(df):
    df = df.copy()
    df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    df["cleaned"] = df["text"].map(clean_text)
    df = df[df["cleaned"].str.split().str.len() >= 20].drop_duplicates("cleaned")
    return df[~df.duplicated("cleaned", keep=False)].reset_index(drop=True)


def main():
    isot = prepare(load_isot())
    other = prepare(pd.read_csv(Path("data") / "fake_or_real_news.csv"))
    other = other[~other["cleaned"].isin(set(isot["cleaned"]))]

    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(isot, isot["label"])
    same = [accuracy_score(isot["label"].iloc[b], build_model().fit(isot["text"].iloc[a], isot["label"].iloc[a])
                           .predict(isot["text"].iloc[b])) for a, b in folds]

    pred = build_model().fit(isot["text"], isot["label"]).predict(other["text"])
    is_fake = other["label"].values == "FAKE"
    print(f"Same dataset (ISOT, 5-fold):      {np.mean(same):.1%} +/- {np.std(same):.1%}")
    print(f"New dataset (McIntire, {len(other):,}):  {accuracy_score(other['label'], pred):.1%}")
    print(f"  fake caught {(pred[is_fake] == 'FAKE').mean():.0%}, real kept {(pred[~is_fake] == 'REAL').mean():.0%}")


if __name__ == "__main__":
    main()
