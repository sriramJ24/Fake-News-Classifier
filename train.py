"""Train the fake news classifier and save it for the app.

Run:  python train.py
Saves model.pkl (vectorizer + classifier in one pipeline) and metrics.json.
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from text_utils import clean_text

DATA = Path("data")


def load_isot():
    fake = pd.read_csv(DATA / "Fake.csv").assign(label="FAKE")
    real = pd.read_csv(DATA / "True.csv").assign(label="REAL")
    return pd.concat([fake, real]).assign(source="ISOT")


def load_welfake():
    """Optional extra dataset (72k articles from Kaggle, McIntire, Reuters, BuzzFeed)."""
    path = DATA / "WELFake_Dataset.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # The dataset's docs disagree on which label means fake, so check the data:
    # whichever label holds the Reuters wire stories is the real one.
    reuters_share = df["text"].fillna("").str.contains(r"\(Reuters\)").groupby(df["label"]).mean()
    real_label = reuters_share.idxmax()
    df["label"] = (df["label"] == real_label).map({True: "REAL", False: "FAKE"})
    return df.assign(source="WELFake")


def load_data():
    frames = [load_isot(), load_welfake()]
    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    df["text"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    df["cleaned"] = df["text"].map(clean_text)

    before = len(df)
    df = df[df["cleaned"].str.split().str.len() >= 20]  # empty / near-empty articles
    df = df.drop_duplicates("cleaned")
    # The same article must not appear under both labels
    df = df[~df.duplicated("cleaned", keep=False)]
    print(f"Loaded {before:,} articles, kept {len(df):,} after removing empty and duplicate ones")
    return df.reset_index(drop=True)


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),   # single words and two-word phrases
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            max_features=100_000,
        )),
        ("clf", LogisticRegression(C=10, max_iter=2000)),
    ])


def main():
    df = load_data()
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    # Split first, then fit, so the vectorizer never sees test articles
    model = build_model().fit(train["text"], train["label"])

    pred = model.predict(test["text"])
    accuracy = accuracy_score(test["label"], pred)
    report = classification_report(test["label"], pred, output_dict=True)
    cm = confusion_matrix(test["label"], pred, labels=["FAKE", "REAL"])

    print("=" * 50)
    print(f"Test accuracy: {accuracy:.2%}  ({len(test):,} articles)")
    print(classification_report(test["label"], pred, digits=4))
    print("Confusion matrix (rows = actual, cols = predicted)")
    print(f"            FAKE   REAL\nFAKE      {cm[0][0]:6d} {cm[0][1]:6d}\nREAL      {cm[1][0]:6d} {cm[1][1]:6d}")

    by_source = {
        src: accuracy_score(part["label"], model.predict(part["text"]))
        for src, part in test.groupby("source")
    }
    headline_accuracy = accuracy_score(test["label"], model.predict(test["title"].fillna("")))
    for src, acc in by_source.items():
        print(f"{src} test accuracy: {acc:.2%}")
    print(f"Headline-only accuracy: {headline_accuracy:.2%}")
    print("=" * 50)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "accuracy": accuracy,
        "headline_accuracy": headline_accuracy,
        "accuracy_by_dataset": by_source,
        "fake_recall": report["FAKE"]["recall"],
        "real_recall": report["REAL"]["recall"],
        "train_size": len(train),
        "test_size": len(test),
        "datasets": sorted(df["source"].unique()),
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model.pkl and metrics.json")


if __name__ == "__main__":
    main()
