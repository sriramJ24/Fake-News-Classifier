import re

# Phrases that reveal the *source* of an article rather than whether it's true.
# Left in, the model just learns "Reuters = real, blog with Getty photos = fake".
_DATELINE = re.compile(r"^.{0,150}?\((reuters|ap|afp)\)\s*[-–—]\s*", re.I | re.S)
_SOURCE_TELLS = re.compile(
    r"\breuters\b|21st century wire|featured image|getty images?|image via\b"
    r"|photo by\b|pic\.twitter\.com\S*|https?://\S+|www\.\S+|@\w+",
    re.I,
)
_SPACES = re.compile(r"\s+")


def clean_text(text):
    """Lowercase and strip source giveaways so the model learns from content."""
    text = _DATELINE.sub("", str(text))
    text = _SOURCE_TELLS.sub(" ", text)
    return _SPACES.sub(" ", text).strip().lower()
