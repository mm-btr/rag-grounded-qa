"""Integrity tests for the golden datasets (golden.json, smoke.json) — the mechanical
half of dataset review, frozen as a test so any future edit is checked automatically.
Pure stdlib: python eval/tests/test_golden.py
"""
import json
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from scoring import CORRECTNESS_JUDGED_NEGATIVES

GOLDEN = json.load(open(os.path.join(_HERE, "..", "golden.json"), encoding="utf-8"))
SMOKE = json.load(open(os.path.join(_HERE, "..", "smoke.json"), encoding="utf-8"))

CID = re.compile(r"^[0-9a-f]{10}#\d+$")


def _check_gold_groups(q):
    groups = q.get("gold_chunks")
    assert isinstance(groups, list) and groups, f"№{q['id']}: gold_chunks пуст"
    for g in groups:
        assert isinstance(g, list) and g, f"№{q['id']}: пустая gold-группа"
        for cid in g:
            assert isinstance(cid, str) and CID.match(cid), f"№{q['id']}: битый id {cid!r}"


def test_golden_shape_frozen():
    # 158 total = 128 CORE (tuned) + 30 HELD-OUT (fresh, never shown to the bot). The core is
    # frozen by type/subtype balance; held-out is a separate control slice (held_out:true).
    # Compute (derived) is an operation over a type, not a type — those core questions are
    # multi-hop/comparison as in the original 100.
    assert len(GOLDEN) == 158
    assert len(set(q["id"] for q in GOLDEN)) == 158
    held = [q for q in GOLDEN if q.get("held_out")]
    core = [q for q in GOLDEN if not q.get("held_out")]
    assert len(held) == 30 and sorted(q["id"] for q in held) == list(range(182, 212))
    hans = [q for q in held if q["answerable"]]
    assert len(hans) == 24
    assert Counter(q["type"] for q in hans) == {
        "factoid": 7, "multi-hop": 4, "aggregation": 3, "comparison": 2,
        "conditional": 2, "explanation": 2, "temporal": 2, "absence": 1, "superlative": 1}
    assert Counter(q["negative_type"] for q in held if not q["answerable"]) == {
        "missing-info": 1, "temporal-mismatch": 1, "out-of-scope-relevant": 1,
        "false-premise": 1, "underspecified": 1, "epistemic": 1}
    # CORE frozen (128 = 101 answerable + 27 negatives)
    assert sorted(q["id"] for q in core) == list(range(1, 101)) + list(range(121, 142)) + list(range(145, 152))
    ans = [q for q in core if q["answerable"]]
    neg = [q for q in core if not q["answerable"]]
    assert len(ans) == 101 and len(neg) == 27
    assert Counter(q["type"] for q in ans) == {
        "factoid": 20, "multi-hop": 14, "comparison": 11, "aggregation": 10,
        "conditional": 10, "explanation": 10, "temporal": 8,
        "absence": 3, "cross-doc": 3, "trap": 3,
        "negation": 3, "superlative": 3, "verification": 3}
    assert Counter(q["negative_type"] for q in neg) == {
        "missing-info": 4, "false-premise": 4, "temporal-mismatch": 4,
        "out-of-scope-relevant": 3, "underspecified": 3, "contradiction": 3,
        "epistemic": 3, "injection": 3}


def test_answerables_have_reference_and_gold_groups():
    for q in GOLDEN:
        if q["answerable"]:
            assert isinstance(q["reference_answer"], str) and q["reference_answer"].strip()
            _check_gold_groups(q)


def test_negatives_have_no_gold_and_correct_references():
    for q in GOLDEN:
        if q["answerable"]:
            continue
        assert not q.get("gold_chunks"), f"№{q['id']}: негатив с голдом"
        if q["negative_type"] in CORRECTNESS_JUDGED_NEGATIVES:
            # judge must reward the substantive behaviour (premise correction / epistemic
            # sandwich), not a refusal -> custom behavioural reference required
            assert isinstance(q["reference_answer"], str) and q["reference_answer"].strip(), \
                f"№{q['id']}: негатив с содержательным поведением без reference (судья потребует отказ)"
        else:
            assert q["reference_answer"] is None, \
                f"№{q['id']}: отказный негатив с reference (перекроет NEGATIVE_REFERENCE)"


def test_twins_point_to_answerables():
    by_id = {q["id"]: q for q in GOLDEN}
    twins = [(q["id"], q["twin_of"]) for q in GOLDEN if q.get("twin_of")]
    assert len(twins) >= 6
    for nid, aid in twins:
        assert not by_id[nid]["answerable"], f"№{nid}: близнец должен быть негативом"
        assert by_id.get(aid, {}).get("answerable"), f"№{nid}: twin_of {aid} не отвечаемый"


def test_smoke_integrity():
    ids = [q["id"] for q in SMOKE]
    assert len(set(ids)) == len(ids)
    for q in SMOKE:
        if q["answerable"]:
            assert isinstance(q["reference_answer"], str) and q["reference_answer"].strip()
            _check_gold_groups(q)
        else:
            assert not q.get("gold_chunks")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all golden tests passed")
