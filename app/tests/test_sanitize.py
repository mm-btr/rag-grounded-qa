"""Unit tests for sanitize.sanitize_text — strips invisible/deceptive Unicode (smuggling).
Invisible chars built via chr() so they don't depend on surviving file encoding.
Run: PYTHONPATH=/app python tests/test_sanitize.py
"""
from sanitize import sanitize_text


def test_strips_zero_width_space():
    assert sanitize_text("при" + chr(0x200B) + "вет") == "привет"


def test_strips_zero_width_nonjoiner():          # zero-width binary uses 200B/200C
    assert sanitize_text("a" + chr(0x200C) + "b") == "ab"


def test_strips_bidi_override():
    assert chr(0x202E) not in sanitize_text("invoice" + chr(0x202E) + "gpj.exe")


def test_strips_unicode_tags():                  # ASCII smuggling block U+E0000..E007F
    assert sanitize_text("hi" + chr(0xE0048) + chr(0xE0049)) == "hi"


def test_strips_variation_selector():
    assert sanitize_text("a" + chr(0xFE0F) + "b") == "ab"


def test_strips_bom():
    assert sanitize_text(chr(0xFEFF) + "text") == "text"


def test_keeps_normal_text():
    s = "Wildberries: возврат 14 дней — §25, стр. 12."
    assert sanitize_text(s) == s


def test_nfkc_normalizes_ligature():             # ﬁ U+FB01 -> fi
    assert sanitize_text(chr(0xFB01) + "le") == "file"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all sanitize tests passed")
