"""Build B: WELFake minus anything matching ISOT or McIntire (by text opening or headline)."""
import re, sys, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.chdir(HERE.parent); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import pandas as pd
from common import clean_v1 as clean_text  # the original cleaner, as used when B was first built
from train import load_isot, load_welfake

def body_key(t):
    t = re.sub(r"[^a-z]", "", clean_text(t))
    return t[50:250] if len(t) > 300 else t
def title_key(t):
    return re.sub(r"[^a-z]", "", str(t).lower())

isot = load_isot(); wel = load_welfake(); mc = pd.read_csv("data/fake_or_real_news.csv")
for d in (isot, wel, mc):
    d["bk"] = d["text"].fillna("").map(body_key); d["tk"] = d["title"].fillna("").map(title_key)
print(len(isot), len(wel), len(mc))
print(wel.label.value_counts())
for name, ref in [("ISOT", isot), ("McIntire", mc)]:
    bk, tk = set(ref.bk[ref.bk.str.len() > 50]), set(ref.tk[ref.tk.str.len() > 20])
    hit_b, hit_t = wel.bk.isin(bk), wel.tk.isin(tk)
    print(name, "body-match", hit_b.sum(), "title-match", hit_t.sum(), "either", (hit_b | hit_t).sum())
rest = wel[~(wel.bk.isin(set(isot.bk[isot.bk.str.len()>50])) | wel.tk.isin(set(isot.tk[isot.tk.str.len()>20]))
            | wel.bk.isin(set(mc.bk[mc.bk.str.len()>50])) | wel.tk.isin(set(mc.tk[mc.tk.str.len()>20])))]
print("WELFake residual", len(rest)); print(rest.label.value_counts())
print("reuters in residual real:", rest[rest.label=="REAL"].text.fillna("").str.contains("Reuters").mean())
print(rest.sample(8, random_state=1)[["label","title"]].to_string())
rest = rest.copy()
rest["suffix"] = rest.title.fillna("").str.extract(r"\s[-|–]\s([^-|–]{3,40})$")[0]
print(rest.groupby("label").suffix.apply(lambda s: s.value_counts().head(12)))
print("suffix share by label", rest.suffix.notna().groupby(rest.label).mean())
for lab in ["REAL","FAKE"]:
    for t in rest[rest.label==lab].text.fillna("").sample(4, random_state=3):
        print(lab, "|", t[:220].replace("\n"," "), "...", t[-150:].replace("\n"," "))
print("curly quote rate", rest.text.fillna("").str.contains("[’“”]").groupby(rest.label).mean())
print("ISOT curly", isot.text.fillna("").str.contains("[’“”]").groupby(isot.label).mean())
rest.drop(columns=["bk","tk"]).to_pickle(HERE / "welfake_residual.pkl")
