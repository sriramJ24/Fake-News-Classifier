"""One-time final test on McIntire. Settings were fixed on validation before this ran.

Every approach is scored on the same McIntire test set that evaluate_generalization.py builds.
Same-dataset = 80/20 holdout of that approach's own training data (DistilBERT: its 4,000-article
ISOT holdout from bert_exp.py).
python experiments/final_test.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from common import clean_v1, scores  # noqa: E402
from evaluate_generalization import drop_overlap, overlap_keys, prepare  # noqa: E402
from train import build_model, fit_model, load_isot, load_welfake  # noqa: E402

LABELS = ["FAKE", "REAL"]


def baseline_model():  # the model before this work: old cleaner, words only, C=10
    return Pipeline([("tfidf", TfidfVectorizer(preprocessor=clean_v1, ngram_range=(1, 2), min_df=3, max_df=0.9,
                                               sublinear_tf=True, max_features=100_000)),
                     ("clf", LogisticRegression(C=10, max_iter=2000))])


def with_old_text(df):
    """Keep the old headline + body join (no outlet-suffix stripping) for the baseline."""
    return df.assign(old_text=df["title"].fillna("") + ". " + df["text"].fillna(""))


def p_fake(model, texts):
    return model.predict_proba(texts)[:, list(model.classes_).index("FAKE")]


def bert_p_fake(path, texts):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to("mps").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=256, padding=True, return_tensors="pt").to("mps")
            out += torch.softmax(model(**enc).logits, -1)[:, 0].cpu().tolist()
    return np.array(out)


def row(name, same, y, p):
    s = scores(y, np.where(p >= 0.5, "FAKE", "REAL"))
    r = {"approach": name, "same": same, "unseen": s["acc"], "fake_recall": s["fake"], "real_recall": s["real"]}
    print(f"{name:42s} same {same:.1%}  McIntire {s['acc']:.1%}  fake {s['fake']:.0%}  real {s['real']:.0%}", flush=True)
    return r


def holdout_same(make, df, fit=None):
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    m = fit(tr) if fit else make().fit(tr["text"], tr["label"])
    return float((m.predict(te["text"]) == te["label"]).mean())


def main():
    test = prepare(with_old_text(pd.read_csv("data/fake_or_real_news.csv")))
    isot = prepare(with_old_text(load_isot()))
    test = drop_overlap(test, overlap_keys(isot))
    keys = overlap_keys(test)
    isot = drop_overlap(isot, keys)
    extra = drop_overlap(drop_overlap(prepare(load_welfake()), keys), overlap_keys(isot))
    both = pd.concat([isot, extra], ignore_index=True)
    y = test["label"].values
    print(f"McIntire test articles: {len(test):,}; ISOT {len(isot):,}; WELFake extra {len(extra):,}", flush=True)

    rows = []
    # 1. Baseline: old cleaning + words only, trained on ISOT (raw title. text, as before)
    base_tr = isot.assign(text=isot["old_text"])
    m = baseline_model().fit(base_tr["text"], base_tr["label"])
    rows.append(row("Baseline: words only, ISOT", holdout_same(baseline_model, base_tr), y, p_fake(m, test["old_text"])))

    # 2. New model (cleaning + words + style), ISOT only
    m_isot = build_model().fit(isot["text"], isot["label"])
    p_isot = p_fake(m_isot, test["text"])
    rows.append(row("Words + style, ISOT", holdout_same(build_model, isot), y, p_isot))

    # 3. New model, ISOT + WELFake extra
    m_both = build_model().fit(both["text"], both["label"])
    p_both = p_fake(m_both, test["text"])
    rows.append(row("Words + style, ISOT + WELFake extra", holdout_same(build_model, both), y, p_both))

    # 3b. Same, each source weighted equally  <- the setup chosen on validation
    m_w = fit_model(both)
    p_w = p_fake(m_w, test["text"])
    rows.append(row("CHOSEN: words + style, ISOT + WELFake extra, weighted", holdout_same(None, both, fit_model), y, p_w))

    # 4. DistilBERT trained on ISOT only, and averaged with the word+style models
    bert_dir = HERE / "bert_A"
    if (bert_dir / "config.json").exists():
        val = json.loads((bert_dir / "val.json").read_text())
        p_bert = bert_p_fake(str(bert_dir), test["cleaned"].tolist())
        rows.append(row("DistilBERT, ISOT", val["A_test"]["acc"], y, p_bert))
        rows.append(row("Ensemble: DistilBERT + words/style (ISOT)", float("nan"), y, (p_bert + p_isot) / 2))
        rows.append(row("Ensemble: DistilBERT + chosen model", float("nan"), y, (p_bert + p_w) / 2))
    (HERE / "final_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
