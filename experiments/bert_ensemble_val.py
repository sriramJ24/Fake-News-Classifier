"""Validation (B only) for DistilBERT alone and averaged with the words+style model, both trained on ISOT."""
import sys; from pathlib import Path; HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from common import load_sources, scores
from run_experiments import CONFIGS
a, b = load_sources("v2")
bert = HERE / sys.argv[1]
p_bert = np.load(bert / "p_B.npy"); y = pd.read_pickle(bert / "y_B.pkl")["label"].values
assert (y == b["label"].values).all()
m = CONFIGS["v2_tfidf+stylo_C1"][1]().fit(a[["cleaned", "raw"]], a["label"])
p_ws = m.predict_proba(b[["cleaned", "raw"]])[:, list(m.classes_).index("FAKE")]
for name, p in [("DistilBERT", p_bert), ("words+style", p_ws), ("average", (p_bert + p_ws) / 2)]:
    s = scores(y, np.where(p >= 0.5, "FAKE", "REAL")); print(f"{name:12s} A->B acc {s['acc']:.3f} bal {s['bal']:.3f} fake {s['fake']:.2f} real {s['real']:.2f}")
