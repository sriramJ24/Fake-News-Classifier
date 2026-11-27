"""Validation experiments. Uses only A (ISOT) and B (WELFake residual). No McIntire.

Usage: python run_experiments.py name [name ...]   (appends to results.jsonl)
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import HERE, _BOILER, _BYLINE_CS, _DATELINE2, _DATELINE_V1, _QUOTES, load_sources, scores

OUT = HERE / "results.jsonl"


# ---------------- stylometric features ----------------
STYLO = ["log_words", "sent_len", "sent_len_sd", "word_len", "ttr", "excl", "quest", "quotes", "allcaps",
         "capitalized", "digits", "first_sg", "first_pl", "second", "said", "parens", "colons"]


def keep_case(text):
    text = str(text).translate(_QUOTES)
    text = _DATELINE_V1.sub(r"\1", text, count=1)
    text = _DATELINE2.sub(r"\1", text, count=1)
    text = _BYLINE_CS.sub(" ", text)
    return _BOILER.sub(" ", text)


def stylometry(raw):
    s = pd.Series(raw).map(keep_case)
    words = s.str.findall(r"[A-Za-z']+")
    n = words.str.len().clip(lower=1)
    sents = s.str.split(r"(?<=[.!?])\s+").map(lambda xs: [len(x.split()) for x in xs if x.strip()] or [0])
    low = s.str.lower()
    f = pd.DataFrame({
        "log_words": np.log1p(n),
        "sent_len": sents.map(np.mean),
        "sent_len_sd": sents.map(np.std),
        "word_len": words.map(lambda w: np.mean([len(x) for x in w]) if w else 0),
        "ttr": words.map(lambda w: len(set(x.lower() for x in w[:300])) / max(len(w[:300]), 1)),
        "excl": s.str.count("!") / n,
        "quest": s.str.count(r"\?") / n,
        "quotes": s.str.count('"') / n,
        "allcaps": s.str.count(r"\b[A-Z]{2,}\b") / n,
        "capitalized": s.str.count(r"\b[A-Z][a-z]+") / n,
        "digits": s.str.count(r"\d+") / n,
        "first_sg": low.str.count(r"\b(i|me|my|mine)\b") / n,
        "first_pl": low.str.count(r"\b(we|us|our|ours)\b") / n,
        "second": low.str.count(r"\b(you|your|yours)\b") / n,
        "said": low.str.count(r"\b(said|says|told)\b") / n,
        "parens": s.str.count(r"\(") / n,
        "colons": s.str.count(":") / n,
    })
    return f.fillna(0).values


class Stylo(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(stylometry(X))
        return self

    def transform(self, X):
        return np.clip(self.scaler_.transform(stylometry(X)), -5, 5)


def tfidf(**kw):
    base = dict(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=100_000)
    base.update(kw)
    return TfidfVectorizer(**base)


def text_model(C=10, **kw):
    # input: DataFrame with 'cleaned' and 'raw'
    return Pipeline([("f", ColumnTransformer([("w", tfidf(**kw), "cleaned")])),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


def stylo_model(C=1):
    return Pipeline([("f", ColumnTransformer([("s", Stylo(), "raw")])), ("clf", LogisticRegression(C=C, max_iter=3000))])


def combo_model(C=1, weight=1.0, **kw):
    return Pipeline([("f", ColumnTransformer([("w", tfidf(**kw), "cleaned"), ("s", Stylo(), "raw")],
                                             transformer_weights={"w": 1.0, "s": weight})),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


def char_model(C=1):
    return Pipeline([("f", ColumnTransformer([("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                                                    max_df=0.9, sublinear_tf=True, max_features=200_000), "cleaned")])),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


CONFIGS = {
    "base_v1_C10": ("v1", lambda: text_model(10)),
    "v2_C10": ("v2", lambda: text_model(10)),
    "v2days_C10": ("v2days", lambda: text_model(10)),
    "v2_C3": ("v2", lambda: text_model(3)),
    "v2_C1": ("v2", lambda: text_model(1)),
    "v2_C0.3": ("v2", lambda: text_model(0.3)),
    "v2_C0.1": ("v2", lambda: text_model(0.1)),
    "v1_C1": ("v1", lambda: text_model(1)),
    "v2_uni_C1": ("v2", lambda: text_model(1, ngram_range=(1, 1))),
    "v2_mindf20_C1": ("v2", lambda: text_model(1, min_df=20, max_features=30_000)),
    "v2_binary_uni_C1": ("v2", lambda: text_model(1, ngram_range=(1, 1), binary=True, sublinear_tf=False, use_idf=False, norm="l2")),
    "v2_stylo_only": ("v2", lambda: stylo_model(1)),
    "v2_tfidf+stylo_C1": ("v2", lambda: combo_model(1, 1.0)),
    "v2_tfidf+stylo_w0.3_C1": ("v2", lambda: combo_model(1, 0.3)),
    "v2_char_C1": ("v2", lambda: char_model(1)),
}


def fit_eval(cfg_fn, train, test):
    m = cfg_fn().fit(train[["cleaned", "raw"]], train["label"])
    return m, scores(test["label"], m.predict(test[["cleaned", "raw"]]))


def run(name):
    cleaner, fn = CONFIGS[name]
    a, b = load_sources(cleaner)
    t0 = time.time()
    tr, te = train_test_split(a, test_size=0.2, random_state=42, stratify=a["label"])
    _, same = fit_eval(fn, tr, te)
    _, ab = fit_eval(fn, a, b)
    _, ba = fit_eval(fn, b, a)
    row = {"name": name, "same_A": same["acc"], "A->B": ab, "B->A": ba, "loso_bal": (ab["bal"] + ba["bal"]) / 2,
           "secs": round(time.time() - t0)}
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{name:26s} same {same['acc']:.3f} | A->B acc {ab['acc']:.3f} bal {ab['bal']:.3f} (fake {ab['fake']:.2f} real {ab['real']:.2f})"
          f" | B->A bal {ba['bal']:.3f} | LOSO {row['loso_bal']:.3f}  [{row['secs']}s]", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
