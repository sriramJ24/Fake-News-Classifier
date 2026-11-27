import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# ---------- Cleaning: remove clues about the *source*, keep the content ----------
# Left in, the model just learns "Reuters = real, blog with Getty photos = fake".

# Curly vs straight quotes is a formatting habit of each site, not a sign of truth.
_PLAIN = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " "})

# "WASHINGTON (Reuters) - " or "(AP) - ". Keeps whatever came before it (the headline).
_WIRE_DATELINE = re.compile(
    r"^(.{0,250}?)(?:\b[A-Z]{2,}(?:[ ./,]{1,2}[A-Z]{2,}){0,3}\s*)?\((?i:reuters|ap|afp)\)\s*-\s*", re.S)
# "NEW YORK - " style datelines without a wire service.
_CITY_DATELINE = re.compile(r"^(.{0,250}?)\b[A-Z]{2,}(?:[ ./,]{1,2}[A-Z]{2,}){0,3}\s*-\s+", re.S)
_BYLINE = re.compile(r"\bBy [A-Z][a-z]+ [A-Z][a-z]+\b")  # "By Jane Smith" (capital B only)
_SOURCE_TELLS = re.compile(
    r"\breuters\b|21st century wire|featured image|getty images?|image via\b|photo by\b"
    r"|\bvia\s+(twitter|youtube|facebook)\b|pic\.twitter\.com\S*|https?://\S+|www\.\S+|@\w+"
    r"|\b(the )?new york times\b|\bbreitbart( news)?\b"
    r"|follow [\w .]{2,40} on (twitter|facebook)[^.]*"
    r"|\b(tweet widget|print this|share this|share on facebook|related:|read more:?|subscribe)\b"
    r"|[\(\[](video|images?|photos?|watch|tweets?)[\)\]]"
    r"|\babout the author\b",
    re.I,
)
# Outlet name tacked onto a headline: "Comey Confirms Inquiry - The New York Times"
_TITLE_SUFFIX = re.compile(r"\s+[-|]\s+[^-|]{2,40}$")
_SPACES = re.compile(r"\s+")


def strip_sources(text):
    """Remove source giveaways but keep capitals and punctuation (used for style features)."""
    text = str(text).translate(_PLAIN)
    text = _WIRE_DATELINE.sub(r"\1", text, count=1)
    text = _CITY_DATELINE.sub(r"\1", text, count=1)
    text = _BYLINE.sub(" ", text)
    return _SOURCE_TELLS.sub(" ", text)


def clean_text(text):
    """Strip source giveaways and lowercase, so the word model learns from content."""
    return _SPACES.sub(" ", strip_sources(text)).strip().lower()


def join_title(title, text):
    """Headline + body as one string, with any trailing outlet name removed from the headline."""
    title = _TITLE_SUFFIX.sub("", str(title or "").translate(_PLAIN))
    return f"{title}. {text or ''}"


# ---------- Writing-style features ----------
# How an article is written rather than which words it uses. These point the same way across
# outlets: fake articles shout (!, ALL CAPS), ask questions and talk to "you"; real reporting
# attributes claims ("said", "told") and quotes people.

STYLE_FEATURES = ["log_words", "sentence_length", "sentence_length_spread", "word_length", "vocabulary_variety",
                  "exclamations", "questions", "quote_marks", "all_caps_words", "capitalized_words", "numbers",
                  "i_me_my", "we_us_our", "you_your", "said_told", "parentheses", "colons"]


def style_features(texts):
    rows = []
    for text in texts:
        t = strip_sources(text)
        words = re.findall(r"[A-Za-z']+", t)
        n = max(len(words), 1)
        sentences = [len(s.split()) for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()] or [0]
        low = t.lower()
        first300 = [w.lower() for w in words[:300]]
        rows.append([
            np.log1p(n),
            np.mean(sentences),
            np.std(sentences),
            np.mean([len(w) for w in words]) if words else 0,
            len(set(first300)) / max(len(first300), 1),
            t.count("!") / n,
            t.count("?") / n,
            t.count('"') / n,
            len(re.findall(r"\b[A-Z]{2,}\b", t)) / n,
            len(re.findall(r"\b[A-Z][a-z]+", t)) / n,
            len(re.findall(r"\d+", t)) / n,
            len(re.findall(r"\b(i|me|my|mine)\b", low)) / n,
            len(re.findall(r"\b(we|us|our|ours)\b", low)) / n,
            len(re.findall(r"\b(you|your|yours)\b", low)) / n,
            len(re.findall(r"\b(said|says|told)\b", low)) / n,
            t.count("(") / n,
            t.count(":") / n,
        ])
    return np.array(rows, dtype=float)


class StyleFeatures(BaseEstimator, TransformerMixin):
    """sklearn step: text -> scaled style features (clipped so one odd article can't dominate)."""

    def fit(self, texts, y=None):
        self.scaler_ = StandardScaler().fit(style_features(texts))
        return self

    def transform(self, texts):
        return np.clip(self.scaler_.transform(style_features(texts)), -5, 5)

    def get_feature_names_out(self, input_features=None):
        return np.array(STYLE_FEATURES)
