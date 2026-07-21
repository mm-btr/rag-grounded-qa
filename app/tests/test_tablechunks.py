"""Unit tests for the prose re-merge with table isolation (parse._merge_prose):
pure logic, no Docling models.

Run: PYTHONPATH=/app python tests/test_tablechunks.py
"""
from parse import _merge_prose

WORDS = lambda s: len(s.split())          # fake tokenizer: 1 word = 1 token


def _row(text, section="S", table=False, page=1):
    return {"text": text, "embed_text": f"{section}\n{text}" if section else text,
            "page": page, "section": section, "is_table": table}


def test_adjacent_prose_same_section_merges():
    rows = [_row("один два"), _row("три четыре")]
    out = _merge_prose(rows, WORDS, 100)
    assert len(out) == 1
    assert out[0]["text"] == "один два\nтри четыре"
    assert out[0]["embed_text"] == "S\nодин два\nтри четыре"   # headings prefix not duplicated


def test_table_never_merges_with_prose():
    rows = [_row("проза до"), _row("| a | b |", table=True), _row("проза после")]
    out = _merge_prose(rows, WORDS, 100)
    assert len(out) == 3                                        # table stays alone
    assert out[1]["is_table"] is True


def test_two_tables_do_not_merge():
    rows = [_row("| t1 |", table=True), _row("| t2 |", table=True)]
    assert len(_merge_prose(rows, WORDS, 100)) == 2


def test_section_boundary_blocks_merge():
    rows = [_row("один", section="A"), _row("два", section="B")]
    assert len(_merge_prose(rows, WORDS, 100)) == 2


def test_token_limit_blocks_merge():
    rows = [_row("один два три"), _row("четыре пять")]
    out = _merge_prose(rows, WORDS, 5)                          # 3+1(S)+2 > 5 after join
    assert len(out) == 2


def test_page_taken_from_first_filled():
    rows = [_row("один", page=None), _row("два", page=7)]
    out = _merge_prose(rows, WORDS, 100)
    assert len(out) == 1 and out[0]["page"] == 7


def test_input_rows_not_mutated():
    rows = [_row("один"), _row("два")]
    _merge_prose(rows, WORDS, 100)
    assert rows[0]["text"] == "один"                            # merge works on copies


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all tablechunks tests passed")
