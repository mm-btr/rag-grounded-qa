"""Unit tests for agent._normalize_messages — the history-normalization invariant.

The model must NEVER receive an unanswered tool call (it invalidates the whole thread,
OpenAI 400), the window must start on a human turn, and PAST turns collapse to
question + answer with citation tags stripped. Pure logic, no models/IO.

Run: PYTHONPATH=/app python -m pytest tests/test_normalize.py   (or run this file directly).
"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import _normalize_messages


def _ai_call(cid):
    return AIMessage(content="", tool_calls=[{"id": cid, "name": "search", "args": {"query": "x"}}])


def _complete_turn(i):
    return [HumanMessage(f"q{i}"), _ai_call(str(i)), ToolMessage("r", tool_call_id=str(i)), AIMessage(f"a{i}")]


def test_drops_interrupted_trailing_turn():
    # complete turn, then an interrupted one (tool call 'b' never answered);
    # turn 0 is PAST (a later human exists) -> compacted to question + answer
    msgs = _complete_turn(0) + [HumanMessage("q2"), _ai_call("b")]
    out = _normalize_messages(msgs, 50)
    assert [m.type for m in out] == ["human", "ai"]
    assert out[0].content == "q0" and out[-1].content == "a0"


def test_keeps_complete_log_untouched():
    msgs = _complete_turn(0)
    out = _normalize_messages(msgs, 50)
    assert len(out) == 4


def test_window_starts_on_human():
    msgs = _complete_turn(0) + _complete_turn(1) + _complete_turn(2)
    out = _normalize_messages(msgs, 5)            # forces a cut mid-history
    assert out[0].type == "human"
    assert len(out) <= 5


def test_only_interrupted_turn_clears_all():
    msgs = [HumanMessage("q1"), _ai_call("a")]    # nothing but a broken turn
    out = _normalize_messages(msgs, 50)
    assert out == []


def test_answered_tool_call_is_kept():
    # interrupted-looking but the call IS answered -> not dropped
    msgs = [HumanMessage("q"), _ai_call("a"), ToolMessage("r", tool_call_id="a")]
    out = _normalize_messages(msgs, 50)
    assert len(out) == 3


def test_past_turn_compacts_to_question_and_answer():
    past = [HumanMessage("q0"), _ai_call("0"), ToolMessage("r", tool_call_id="0"),
            AIMessage("Выручка 703 741 млн руб.【1a2fff7161#89】")]
    msgs = past + _complete_turn(1)
    out = _normalize_messages(msgs, 50)
    assert [m.type for m in out] == ["human", "ai", "human", "ai", "tool", "ai"]
    assert out[1].content == "Выручка 703 741 млн руб."   # tag stripped, fact kept


def test_current_turn_untouched_including_tags():
    msgs = [HumanMessage("q"), _ai_call("a"), ToolMessage("r", tool_call_id="a"),
            AIMessage("a 【d#1】")]
    out = _normalize_messages(msgs, 50)
    assert len(out) == 4 and out[-1].content == "a 【d#1】"  # single turn = current


def test_compaction_is_idempotent():
    msgs = _complete_turn(0) + _complete_turn(1)
    once = _normalize_messages(msgs, 50)
    twice = _normalize_messages(once, 50)
    assert len(once) == len(twice) and all(a is b for a, b in zip(once, twice))


def test_past_block_content_flattens_to_text():
    blocks = [{"type": "reasoning", "reasoning": "…"}, {"type": "text", "text": "x 【d#7】"}]
    msgs = [HumanMessage("q0"), _ai_call("0"), ToolMessage("r", tool_call_id="0"),
            AIMessage(blocks)] + _complete_turn(1)
    out = _normalize_messages(msgs, 50)
    assert out[1].content == "x"                           # blocks flattened, tag stripped


def test_answerless_past_turn_keeps_only_question():
    msgs = [HumanMessage("q0"), _ai_call("0"), ToolMessage("r", tool_call_id="0")] \
           + _complete_turn(1)
    out = _normalize_messages(msgs, 50)
    assert out[0].content == "q0" and out[1].type == "human"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all normalize tests passed")
