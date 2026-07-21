"""Render-path invariants: model markdown (machine tags 【id#n】) + our own pre-escaped
display tags \\[N\\] -> plain text + entity spans. What must hold:
  - literal '<'/'>' survive to the user (the text-loss class the HTML mode had);
  - stray asterisks (financial footnotes) don't turn into phantom bold;
  - the display tag \\[N\\] renders as literal [N] in prose, tables and detached-after-table;
  - '\\[N\\](…)' can NEVER parse as a link — the trap is dead by construction;
  - model's own bracketed indexes in code (`a[0]`) are untouched — no escape apparatus;
  - the sources block is ONE expandable_blockquote entity with exact UTF-16 offsets;
  - lib entities map onto aiogram MessageEntity.
Run: PYTHONPATH=/app PYTHONIOENCODING=utf-8 python tests/test_render.py
"""
import telegramify_markdown as tgmd
from keys import CITE_RE
from bot import _sources_block, _tg_entities, _tidy_citations


def _convert(md):
    return tgmd.convert(md, latex_escape=False)


def test_machine_grammar_is_cjk_brackets():
    assert CITE_RE.findall("факт 【9b0679d790#105】.") == [("9b0679d790", "105")]
    assert CITE_RE.findall("факт [9b0679d790#105].") == []      # old grammar is dead
    assert CITE_RE.findall("факт \\[9b0679d790#105\\].") == []  # and so is its escaped form


def test_covenant_angles_survive_with_bold():
    text, ents = _convert("Ковенант: **Чистый долг/EBITDA** < 3,0, покрытие процентов > 4,0 \\[1\\].")
    assert "< 3,0" in text and "> 4,0" in text and "[1]" in text
    assert any(e.type == "bold" for e in ents)


def test_user_angle_fragment_survives():
    text, _ = _convert("Ответ: < если в тексте вот такая штука есть она исчезнет ? > и точка.")
    assert "< если в тексте вот такая штука есть она исчезнет ? >" in text


def test_footnote_stars_stay_literal():
    text, ents = _convert("Показатель OIBDA* вырос, а EBITDA* снизилась.")
    assert "OIBDA*" in text and "EBITDA*" in text
    assert not ents                                   # no phantom bold/italic from stray stars


def test_display_tag_never_parses_as_link():
    # our \[N\] is canonical literal-bracket markdown: even right before a parenthetical
    # it stays text — a '[1](…)' sequence can never parse as a link
    text, ents = _convert("Выручка выросла \\[1\\](по итогам 2025 года).")
    assert "[1]" in text and "(по итогам 2025 года)" in text
    assert not any("link" in e.type for e in ents)


def test_model_code_brackets_untouched():
    # no blanket escaping anymore: the model's own bracketed index in code renders clean
    text, _ = _convert("формула `a[0] + b[1]` конец")
    assert "a[0] + b[1]" in text and "\\" not in text


def test_broken_markdown_degrades_to_text():
    text, _ = _convert("Битый маркдаун: **жирный без закрытия и _хвост")
    assert "жирный без закрытия" in text and "хвост" in text


def test_markdown_table_renders_as_mono_block():
    md = ("Сравнение по годам:\n\n"
          "| Показатель | 2024 | 2025 |\n|---|---|---|\n"
          "| Выручка | 703 741 | 807 186 |\n\n"
          "Рост составил 14,7% \\[1\\].")
    text, ents = _convert(md)
    assert "Выручка" in text and "807 186" in text and "[1]" in text
    assert any(e.type in ("pre", "code") for e in ents)   # the table -> monospace block


def test_citation_after_table_survives_via_tidy_detach():
    body = _tidy_citations("| A | B |\n|---|---|\n| C | D |\n\\[1\\]")
    text, ents = _convert(body)
    assert "[1]" in text                                  # not swallowed as a bogus table cell
    assert any(e.type in ("pre", "code") for e in ents)   # table still a mono block


def test_citation_inside_table_cell_clean():
    text, _ = _convert("| M | 2025 |\n|---|---|\n| Rev | 807 186 \\[1\\] |")
    assert "807 186 [1]" in text and "\\" not in text


def test_sources_block_single_expandable_entity_utf16():
    body = "тело 📈 ответа"                            # emoji: 1 codepoint, 2 UTF-16 units
    lines = ["[1] Метка", "Док", "стр. 5", "", "[2] Метка2", "Док2"]
    text, ents = _sources_block(body, [], lines)
    bq = ents[-1]
    assert bq.type == "expandable_blockquote"
    u16 = text.encode("utf-16-le")
    span = u16[bq.offset * 2:(bq.offset + bq.length) * 2].decode("utf-16-le")
    assert span == "\n".join(lines)                    # offsets exact despite the emoji


def test_tg_entities_maps_fields_and_drops_lib_extras():
    ents = [tgmd.MessageEntity(type="bold", offset=3, length=7),
            tgmd.MessageEntity(type="pre", offset=0, length=2, language="python")]
    out = _tg_entities(ents)
    assert (out[0].type, out[0].offset, out[0].length) == ("bold", 3, 7)
    assert out[1].language == "python"
    assert _tg_entities([]) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all render tests passed")
