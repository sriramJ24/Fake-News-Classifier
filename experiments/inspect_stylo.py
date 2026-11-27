import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import load_sources
from run_experiments import stylometry, STYLO
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
a, b = load_sources("v2")
fa, fb = stylometry(a.raw), stylometry(b.raw)
da = pd.DataFrame(fa, columns=STYLO).groupby(a.label.values).mean().T
db = pd.DataFrame(fb, columns=STYLO).groupby(b.label.values).mean().T
out = pd.concat([da.add_prefix("ISOT_"), db.add_prefix("B_")], axis=1)
for name, f, y in [("ISOT", fa, a.label), ("B", fb, b.label)]:
    sc = StandardScaler().fit(f); out[f"coef(+REAL)_{name}"] = LogisticRegression(C=1, max_iter=3000).fit(sc.transform(f), y).coef_[0]
print(out.round(4).sort_values("coef(+REAL)_ISOT").to_string())
