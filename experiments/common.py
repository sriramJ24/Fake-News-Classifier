"""Shared data + cleaning variants for the generalization experiments.

Never touches McIntire except for removing overlap (in final_test.py).
Sources:  A = ISOT,  B = WELFake minus anything matching ISOT or McIntire (mostly Kaggle 2016).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from train import load_isot  # noqa: E402

import os  # noqa: E402
HERE = Path(__file__).resolve().parent
os.chdir(REPO)  # train.py reads data/ relative to the repo
# Frozen copy of the original cleaner (text_utils.py at commit 59a7cbd), for the baseline
_V1_DATELINE = re.compile(r"^.{0,150}?\((reuters|ap|afp)\)\s*[-–—]\s*", re.I | re.S)
_V1_TELLS = re.compile(r"\breuters\b|21st century wire|featured image|getty images?|image via\b"
                       r"|photo by\b|pic\.twitter\.com\S*|https?://\S+|www\.\S+|@\w+", re.I)


def clean_v1(text):
    text = _V1_DATELINE.sub("", str(text))
    text = _V1_TELLS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

# ---------- v2 cleaning: stronger source/style stripping ----------
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " "})
_TITLE_SUFFIX = re.compile(r"\s+[-|]\s+[^-|]{2,40}$")
# WASHINGTON (Reuters) - ; keeps the headline in front of it (v1 deleted it, but only for real articles)
_DATELINE_V1 = re.compile(r"^(.{0,250}?)(?:\b[A-Z]{2,}(?:[ ./,]{1,2}[A-Z]{2,}){0,3}\s*)?\((?i:reuters|ap|afp)\)\s*-\s*", re.S)
# NEW YORK - / BEIJING/HONG KONG (AP) - , possibly after the title; keep what came before it
_DATELINE2 = re.compile(r"^(.{0,250}?)\b[A-Z]{2,}(?:[ ./,]{1,2}[A-Z]{2,}){0,3}\s*(?:\([A-Za-z .]{1,20}\))?\s*-\s+", re.S)
_BOILER = re.compile(
    r"\breuters\b|21st century wire|featured image|getty images?|image via\b|photo by\b"
    r"|\bvia\s+(twitter|youtube|facebook)\b|pic\.twitter\.com\S*|https?://\S+|www\.\S+|@\w+"
    r"|\bthe new york times\b|\bnew york times\b|\bbreitbart( news)?\b"
    r"|follow [\w .]{2,40} on (twitter|facebook)[^.]*"
    r"|\b(tweet widget|print this|share this|share on facebook|related:|read more:?|subscribe)\b"
    r"|[\(\[](video|images?|photos?|watch|tweets?)[\)\]]"
    r"|\babout the author\b",
    re.I,
)
_BYLINE_CS = re.compile(r"\bBy [A-Z][a-z]+ [A-Z][a-z]+\b")
_DAYS = re.compile(r"\b(on )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_SPACES = re.compile(r"\s+")


def clean_v2(text, drop_days=False):
    text = str(text).translate(_QUOTES)
    text = _DATELINE_V1.sub(r"\1", text, count=1)
    text = _DATELINE2.sub(r"\1", text, count=1)
    text = _BYLINE_CS.sub(" ", text)
    text = _BOILER.sub(" ", text)
    if drop_days:
        text = _DAYS.sub(" ", text)
    return _SPACES.sub(" ", text).strip().lower()


def clean_v2_days(text):
    return clean_v2(text, drop_days=True)


CLEANERS = {"v1": clean_v1, "v2": clean_v2, "v2days": clean_v2_days}


def join(df, strip_suffix):
    title = df["title"].fillna("")
    if strip_suffix:
        title = title.str.translate(_QUOTES).str.replace(_TITLE_SUFFIX, "", regex=True)
    return title + ". " + df["text"].fillna("")


def prepare(df, cleaner, strip_suffix):
    df = df.copy()
    df["raw"] = join(df, strip_suffix)
    df["cleaned"] = df["raw"].map(CLEANERS[cleaner])
    df = df[df["cleaned"].str.split().str.len() >= 20].drop_duplicates("cleaned")
    return df[~df.duplicated("cleaned", keep=False)].reset_index(drop=True)


def load_sources(cleaner="v1"):
    """Return (A, B) prepared with the given cleaner. Cached."""
    path = CACHE / f"sources_{cleaner}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    strip = cleaner != "v1"
    a = prepare(load_isot(), cleaner, strip)
    b = prepare(pd.read_pickle(HERE / "welfake_residual.pkl"), cleaner, strip)
    b = b[~b["cleaned"].isin(set(a["cleaned"]))].reset_index(drop=True)
    # outlet group for B's real articles (from the original title suffix), used for outlet-disjoint splits
    suffix = b["title"].fillna("").str.extract(r"\s[-|–]\s([^-|–]{3,40})$")[0]
    b["outlet"] = np.where(suffix.eq("The New York Times"), "nyt", np.where(suffix.eq("Breitbart"), "breitbart", "other"))
    pd.to_pickle((a, b), path)
    return a, b


def recall(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    fake = (pred[y == "FAKE"] == "FAKE").mean()
    real = (pred[y == "REAL"] == "REAL").mean()
    return fake, real


def scores(y, pred):
    fake, real = recall(y, pred)
    return {"acc": float((np.asarray(y) == np.asarray(pred)).mean()), "bal": (fake + real) / 2, "fake": fake, "real": real}
