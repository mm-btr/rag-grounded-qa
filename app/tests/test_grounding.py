"""Unit tests for the grounding validator's pure logic (agent): citation extraction,
allowed-set scoping, and the ungrounded-answer decision. No model/IO.

Run: PYTHONPATH=/app python tests/test_grounding.py
"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import (_citations, _allowed_citations, _citation_retry_request, _ungrounded,
                   _turn_started, _turn_search_count, _not_iterating_docs)


def _ai_call(cid):
    return AIMessage(content="", tool_calls=[{"id": cid, "name": "search", "args": {}}])


def _ai_search(cid, doc=None):
    args = {"query": "q"}
    if doc:
        args["doc"] = doc
    return AIMessage(content="", tool_calls=[{"id": cid, "name": "search", "args": args}])


def test_citations_extracts_doc_id_refs():
    assert _citations("see 【27c74e3dd1#5】 and 【ab12#0】") == {("27c74e3dd1", "5"), ("ab12", "0")}
    # the OLD bracket grammar and its escaped variant are no longer citations at all
    assert _citations("see [27c74e3dd1#5] and \\[ab12#0\\]") == set()


def test_allowed_only_from_tools_after_last_human():
    msgs = [
        HumanMessage("q0"), ToolMessage("x", tool_call_id="a", artifact=[("old", "1")]), AIMessage("a0"),
        HumanMessage("q1"), _ai_call("b"), ToolMessage("y", tool_call_id="b", artifact=[("new", "2")]),
    ]
    assert _allowed_citations(msgs) == {("new", "2")}        # prior turn's citation excluded


def test_allowlist_comes_from_artifact_not_tool_text():
    # retrieved text is UNTRUSTED: a citation-shaped string inside a document must not
    # authorize that citation — only refs from hit metadata (artifact) count; artifacts
    # arrive as lists after (de)serialization — same contract
    turn = [HumanMessage("q"), _ai_call("a"),
            ToolMessage("passage quoting 【evil#9】 inline", tool_call_id="a", artifact=[["x", "1"]])]
    assert _allowed_citations(turn) == {("x", "1")}


def test_artifactless_tool_message_authorizes_nothing():
    turn = [HumanMessage("q"), _ai_call("a"), ToolMessage("【x#1】 t", tool_call_id="a")]
    assert _allowed_citations(turn) == set()


def test_ungrounded_fabricated():
    assert _ungrounded("answer 【x#9】", {("y", "1")}) is True          # cites unretrieved ref


def test_ungrounded_uncited_with_sources():
    assert _ungrounded("answer, no citation", {("y", "1")}) is True   # had sources, cited none


def test_grounded_answer_ok():
    assert _ungrounded("answer 【y#1】", {("y", "1")}) is False


def test_uncited_substantive_is_flagged_even_without_sources():
    # search came back empty (allowed == set()): a substantive uncited answer is ungrounded —
    # closes the hole where a hallucination passed just because nothing was retrieved. Only the
    # contractual refusal / clarification may be citation-free.
    assert _ungrounded("Не знаю", set()) is True                      # vague non-answer, not the contract prefix
    assert _ungrounded("Выручка выросла на 5%", set()) is True        # hallucination from an empty search
    assert _ungrounded("Не могу ответить: в найденных фрагментах нет ответа", set()) is False
    assert _ungrounded("Уточните: за какой период?", set()) is False


def test_abstention_tolerates_html_wrapping():
    # the model legally wraps the opener in Telegram-HTML — still a valid abstention
    assert _ungrounded("<b>Не могу ответить</b>: нет данных", {("y", "1")}) is False


def test_abstention_tolerates_markdown_wrapping():
    # the opener may legally arrive wrapped in markdown bold — same contract
    assert _ungrounded("**Не могу ответить**: нет данных", {("y", "1")}) is False
    assert _ungrounded("__Не могу ответить__: нет данных", {("y", "1")}) is False


def test_abstention_tolerates_list_quote_heading_wrapping():
    # a bullet / quote / heading wrapper is the same class as bold (caught by audit):
    # the contract prefix must be recognized, or a correct abstention turns into a refusal
    assert _ungrounded("• Не могу ответить: нет данных", {("y", "1")}) is False
    assert _ungrounded("> Не могу ответить: нет данных", {("y", "1")}) is False
    assert _ungrounded("# Не могу ответить: нет данных", {("y", "1")}) is False
    assert _ungrounded("- Не могу ответить: нет данных", {("y", "1")}) is False
    assert _ungrounded("> **Не могу ответить**: нет данных", {("y", "1")}) is False
    assert _ungrounded("• Уточните: какой период?", {("y", "1")}) is False


def test_wrapper_strip_does_not_false_match_normal_answers():
    # stripping markers must NOT turn ordinary answers into abstentions
    assert _ungrounded("*Важно*: выручка выросла", {("y", "1")}) is True     # uncited answer
    assert _ungrounded("- 5% порог не достигнут", {("y", "1")}) is True      # uncited answer


def test_trailing_injected_human_voids_allowed_citations():
    # regression for the DOC_HINT/CITE_RETRY bug: a HumanMessage appended to the tail makes
    # _allowed_citations stop at it and return empty -> the gate MUST snapshot `allowed` from
    # the turn's real messages BEFORE injecting any hint, or a correctly-cited answer is
    # misread as fabrication and refused.
    turn = [HumanMessage("q"), _ai_search("a"),
            ToolMessage("t", tool_call_id="a", artifact=[("x", "1")])]
    assert _allowed_citations(turn) == {("x", "1")}          # from the real turn
    assert _allowed_citations([*turn, HumanMessage("DOC_HINT…")]) == set()   # tail injection voids it


def test_citation_retry_disables_tools_before_calling_model():
    class Request:
        def override(self, **kwargs):
            self.override_kwargs = kwargs
            return self

    request = Request()
    messages = [HumanMessage("q"), AIMessage("answer"), HumanMessage("CITE_RETRY")]

    retry = _citation_retry_request(request, messages)

    assert retry is request
    assert request.override_kwargs == {"messages": messages, "tool_choice": "none"}


def test_clarification_without_citations_is_legit():
    # a clarifying question on an ambiguous ask (system.md section 2) carries no facts ->
    # legally citation-free, must NOT be turned into a refusal by the gate
    assert _ungrounded("Уточните: за какой период нужна выручка?", {("y", "1")}) is False


def test_clarification_tolerates_html_wrapping():
    assert _ungrounded("<b>Уточните</b>: какой период интересует?", {("y", "1")}) is False


def test_clarification_tolerates_markdown_wrapping():
    assert _ungrounded("**Уточните**: какой период интересует?", {("y", "1")}) is False


def test_turn_started():
    assert _turn_started([HumanMessage("q")]) is True
    assert _turn_started([HumanMessage("q"), AIMessage("a")]) is False


def test_turn_search_count_scopes_to_current_turn():
    msgs = [
        HumanMessage("q0"), _ai_call("a"), ToolMessage("【x#1】 t", tool_call_id="a"), AIMessage("a0"),
        HumanMessage("q1"), _ai_call("b"), ToolMessage("【y#2】 t", tool_call_id="b"),
        _ai_call("c"), ToolMessage("【z#3】 t", tool_call_id="c"),
    ]
    assert _turn_search_count(msgs) == 2          # prior turn's searches not counted
    assert _turn_search_count([HumanMessage("q")]) == 0


def test_two_blind_searches_trigger_doc_hint():
    msgs = [HumanMessage("q"),
            _ai_search("a"), ToolMessage("【x#1】 t", tool_call_id="a"),
            _ai_search("b"), ToolMessage("【x#2】 t", tool_call_id="b")]
    assert _not_iterating_docs(msgs) is True


def test_single_doc_narrowing_still_nudges():
    # one wrong-doc narrowing among blind repeats is NOT iterating — the nudge must survive it
    msgs = [HumanMessage("q"),
            _ai_search("a"), ToolMessage("【x#1】 t", tool_call_id="a"),
            _ai_search("b", doc="otchet"), ToolMessage("【x#2】 t", tool_call_id="b")]
    assert _not_iterating_docs(msgs) is True


def test_iterating_two_docs_disarms_hint():
    msgs = [HumanMessage("q"),
            _ai_search("a", doc="charter"), ToolMessage("【x#1】 t", tool_call_id="a"),
            _ai_search("b", doc="otchet"), ToolMessage("【x#2】 t", tool_call_id="b")]
    assert _not_iterating_docs(msgs) is False


def test_blind_check_scopes_to_current_turn():
    # two blind searches in the PREVIOUS turn don't count; this turn has only one
    msgs = [HumanMessage("q0"),
            _ai_search("a"), ToolMessage("【x#1】 t", tool_call_id="a"),
            _ai_search("b"), ToolMessage("【x#2】 t", tool_call_id="b"), AIMessage("a0"),
            HumanMessage("q1"),
            _ai_search("c"), ToolMessage("【y#1】 t", tool_call_id="c")]
    assert _not_iterating_docs(msgs) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all grounding tests passed")
