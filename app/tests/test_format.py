"""Unit tests for the source-card formatter and citation tidy (display tags are \\[N\\] —
canonical literal-bracket markdown emitted by _render_answer itself).
Run: PYTHONPATH=/app PYTHONIOENCODING=utf-8 python tests/test_format.py
"""
from bot import _clean_source, _locator_lines, _tidy_citations


def test_clean_source_drops_extension_and_parens():
    assert _clean_source("Оферта Wildberries (продавцы).pdf") == "Оферта Wildberries"


def test_clean_source_plain():
    assert _clean_source("rules.docx") == "rules"


def test_clean_source_empty():
    assert _clean_source("") == "документ"


def test_locator_lines_full_card():
    p = {"label": "Ozon", "source": "правила.pdf",
         "page": 43, "section": "11.3. Компенсация:"}
    assert _locator_lines(p) == ["Ozon", "правила", "§ 11.3. Компенсация:", "стр. 43"]


def test_locator_lines_section_no_page():                # txt/docx: no fake "стр. 1"
    p = {"source": "договор.docx", "section": "Условия возврата"}
    assert _locator_lines(p) == ["договор", "§ Условия возврата"]


def test_locator_lines_no_label():
    p = {"source": "правила.pdf", "page": 5}
    assert _locator_lines(p) == ["правила", "стр. 5"]


def test_tidy_pulls_stranded_tag_onto_prose():
    assert _tidy_citations("текст про НДС.\n\\[1\\]") == "текст про НДС. \\[1\\]"


def test_tidy_collapses_duplicate_run():
    assert _tidy_citations("факт \\[1\\]\\[1\\]") == "факт \\[1\\]"
    assert _tidy_citations("факт \\[1\\] \\[1\\]") == "факт \\[1\\]"


def test_tidy_keeps_distinct_tags():
    assert _tidy_citations("факт \\[1\\]\\[2\\]") == "факт \\[1\\]\\[2\\]"


def test_tidy_keeps_paragraph_breaks():
    assert _tidy_citations("а.\n\\[1\\]\n\nб.") == "а. \\[1\\]\n\nб."


def test_tidy_does_not_eat_blank_line_before_tag():
    # a tag already standing as its own paragraph stays one (the blank line is preserved)
    assert _tidy_citations("пар.\n\n\\[1\\]") == "пар.\n\n\\[1\\]"


def test_tidy_bullets_each_inline():
    assert _tidy_citations("• а.\n\\[1\\]\n• б.\n\\[1\\]") == "• а. \\[1\\]\n• б. \\[1\\]"


def test_tidy_detaches_tag_from_table():
    # gluing onto a table row makes the parser DROP the tag as an excess cell —
    # the tag must be detached into its own paragraph instead
    src = "| A | B |\n|---|---|\n| C | D |\n\\[1\\]"
    assert _tidy_citations(src) == "| A | B |\n|---|---|\n| C | D |\n\n\\[1\\]"


def test_tidy_detaches_tag_from_fence():
    src = "```\nx = 1\n```\n\\[1\\]"
    assert _tidy_citations(src) == "```\nx = 1\n```\n\n\\[1\\]"


def test_tidy_leaves_fence_interior_alone():
    src = "```\n\\[1\\]\nx\n```"
    assert _tidy_citations(src) == src


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all format tests passed")
