"""Does adding a second training source help on outlets neither source contains?

B is split by outlet: B1 = NYT real + half of B's fake, B2 = Breitbart/other real + the other half of fake.
Compare  A -> B2  vs  A+B1 -> B2  (and the mirror). Optionally weight sources equally.
python multi_exp.py CONFIG_NAME
"""
import json
import sys

import numpy as np
import pandas as pd

from common import load_sources, scores
from run_experiments import CONFIGS, OUT


def split_b(b):
    fake = b[b["label"] == "FAKE"].sample(frac=1, random_state=0)
    half = len(fake) // 2
    real = b[b["label"] == "REAL"]
    b1 = pd.concat([real[real["outlet"] == "nyt"], fake.iloc[:half]])
    b2 = pd.concat([real[real["outlet"] != "nyt"], fake.iloc[half:]])
    return b1, b2


def fit(fn, train, weight_sources):
    X, y = train[["cleaned", "raw"]], train["label"]
    m = fn()
    if weight_sources:
        w = train["src"].map(1 / train["src"].value_counts(normalize=True)).values
        m.fit(X, y, clf__sample_weight=w / w.mean())
    else:
        m.fit(X, y)
    return m


def main(name):
    cleaner, fn = CONFIGS[name]
    a, b = load_sources(cleaner)
    a = a.assign(src="A")
    b1, b2 = split_b(b.assign(src="B"))
    rows = {}
    for held, other in [("B2", (b1, b2)), ("B1", (b2, b1))]:
        add, test = other
        for label, train, ws in [("A only", a, False), ("A+B", pd.concat([a, add]), False), ("A+B weighted", pd.concat([a, add]), True)]:
            s = scores(test["label"], fit(fn, train, ws).predict(test[["cleaned", "raw"]]))
            rows.setdefault(label, []).append(s)
            print(f"{name} {label:13s} -> {held}: acc {s['acc']:.3f} bal {s['bal']:.3f} fake {s['fake']:.2f} real {s['real']:.2f}", flush=True)
    summary = {k: float(np.mean([r["bal"] for r in v])) for k, v in rows.items()}
    print(name, "mean balanced acc on held-out outlets:", summary)
    with OUT.open("a") as f:
        f.write(json.dumps({"name": f"multi:{name}", "detail": rows, "summary": summary}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
