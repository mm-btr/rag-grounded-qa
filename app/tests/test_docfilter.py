"""Unit tests for the doc-filter fuzzy resolve (retrieve._match_docs): pure logic, no Qdrant.

Run: PYTHONPATH=/app python tests/test_docfilter.py
"""
from retrieve import _match_docs

DOCS = {
    "acme_policy_2025.pdf": {"label": "Project Alpha", "descr": "rules and governance"},
    "acme_report_2025.pdf": {"label": "Project Alpha", "descr": None},
    "market_review.pdf": {"label": "Project Alpha", "descr": None},
}


def test_filename_substring_resolves_single_doc():
    assert _match_docs(DOCS, "policy") == ["acme_policy_2025.pdf"]
    assert _match_docs(DOCS, "market") == ["market_review.pdf"]


def test_filename_match_is_case_insensitive():
    assert _match_docs(DOCS, "REPORT") == ["acme_report_2025.pdf"]
    assert _match_docs(DOCS, "POLICY") == ["acme_policy_2025.pdf"]


def test_label_matches_all_group_docs():
    assert sorted(_match_docs(DOCS, "project")) == sorted(DOCS)


def test_unresolved_and_empty_return_none():
    assert _match_docs(DOCS, "unknown") is None               # no such doc -> widen upstream
    assert _match_docs(DOCS, "") is None
    assert _match_docs(DOCS, "   ") is None
    assert _match_docs({}, "policy") is None                  # empty corpus


def test_ambiguous_filename_match_is_unresolved():
    # "acme" is a substring of two file names -> not a document pointer -> widen
    assert _match_docs(DOCS, "ACME") is None
    assert _match_docs(DOCS, "acme") is None
    assert _match_docs(DOCS, "2025") is None                  # year hits several names too


def test_label_axis_wins_over_file_axis():
    docs = {"a.pdf": {"label": "reports 2025", "descr": None},
            "b.pdf": {"label": "reports 2025", "descr": None},
            "reports_archive.pdf": {"label": "archive", "descr": None}}
    assert sorted(_match_docs(docs, "reports")) == ["a.pdf", "b.pdf"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all docfilter tests passed")
