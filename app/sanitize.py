"""Strip invisible/deceptive Unicode (NFKC alone is insufficient — ranges are stripped explicitly)."""
import re
import unicodedata

_INVISIBLE = re.compile(
    "["
    "\\u200b-\\u200f"            # zero-width space/joiners, bidi marks
    "\\u202a-\\u202e"            # bidi embeddings + overrides (RLO/LRO)
    "\\u2060-\\u2064"            # word-joiner, invisible math operators
    "\\u2066-\\u2069"            # bidi isolates
    "\\ufe00-\\ufe0f"            # variation selectors
    "\\ufeff"                   # BOM
    "\\U000e0000-\\U000e007f"    # Unicode Tags block — ASCII smuggling
    "\\U000e0100-\\U000e01ef"    # variation selectors supplement
    "]"
)


def sanitize_text(text):
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
