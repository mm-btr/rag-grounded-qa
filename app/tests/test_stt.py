"""Unit tests for stt._parse_transcript — extracts the transcript from Scribe's JSON.
Pure function, no network. Run: PYTHONPATH=/app python tests/test_stt.py
"""
from stt import _parse_transcript, STTError


def _raises(data):
    try:
        _parse_transcript(data)
    except STTError:
        return True
    return False


def test_extracts_text():
    assert _parse_transcript({"text": "какая комиссия на Ozon", "language_code": "rus"}) \
        == "какая комиссия на Ozon"


def test_strips_whitespace():
    assert _parse_transcript({"text": "  привет  "}) == "привет"


def test_empty_text_raises():
    assert _raises({"text": ""})


def test_missing_text_raises():
    assert _raises({"language_code": "eng"})


def test_whitespace_only_raises():
    assert _raises({"text": "   \n  "})


def test_null_text_raises():
    assert _raises({"text": None})


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all stt tests passed")
