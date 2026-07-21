"""Unit tests for the pure eval math (eval/scoring.py). No app/langfuse imports —
run anywhere: python eval/tests/test_scoring.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import (citation_precision, citation_recall, gold_groups, mcnemar_p,
                     percentile, retrieval_scores)


def test_single_gold_hit_equals_recall():
    s = retrieval_scores([["a#1"]], ["x#9", "a#1", "y#2"], ["x#9", "a#1", "y#2"])
    assert s["hit@5"] == 1.0 and s["recall@5"] == 1.0
    assert s["mrr"] == 0.5                          # rank 2
    assert s["recall_any_search"] == 1.0 and s["all_gold_found"] == 1.0


def test_multi_gold_hit_flatters_recall_tells_truth():
    # comparison question: two required gold groups, first search covers only one
    s = retrieval_scores([["a#1"], ["b#2"]], ["a#1", "x#9"], ["a#1", "x#9"])
    assert s["hit@5"] == 1.0                        # hit flatters: some group = full credit
    assert s["recall@5"] == 0.5                     # the honest one: half the groups
    assert s["all_gold_found"] == 0.0               # complete answer impossible


def test_second_search_completes_the_gold():
    s = retrieval_scores([["a#1"], ["b#2"]], ["a#1", "x#9"], ["a#1", "x#9", "b#2"])
    assert s["recall@5"] == 0.5                     # first search unchanged
    assert s["all_gold_found"] == 1.0               # but the turn found everything


def test_gold_found_only_by_repeat_search():
    s = retrieval_scores([["a#1"]], ["x#9", "y#2"], ["x#9", "y#2", "a#1"])
    assert s["hit@5"] == 0.0 and s["mrr"] == 0.0
    assert s["recall_any_search"] == 1.0 and s["all_gold_found"] == 1.0


def test_top5_window_is_hard():
    s = retrieval_scores([["a#1"]], ["1#1", "2#2", "3#3", "4#4", "5#5", "a#1"], ["a#1"])
    assert s["hit@5"] == 0.0 and s["recall@5"] == 0.0   # rank 6 is outside the window


def test_zero_hit_turn():
    s = retrieval_scores([["a#1"]], [], [])
    assert all(v == 0.0 for v in s.values())


def test_gold_groups_single_shape():
    assert gold_groups([["a#1", "b#2"], ["c#3"]]) == [{"a#1", "b#2"}, {"c#3"}]
    assert gold_groups(None) == [] and gold_groups([]) == []
    for bad in (["a#1", "b#2"], [["a#1"], "b#2"], [[]]):   # bare ids / empty group
        try:
            gold_groups(bad)
            assert False, f"{bad!r} должен быть ошибкой формата"
        except ValueError:
            pass


def test_duplicate_carriers_any_of_covers_the_group():
    # one gold group has two interchangeable carriers: finding either covers it fully
    s = retrieval_scores([["a#1", "b#2"]], ["b#2", "x#9"], ["b#2"])
    assert s["recall@5"] == 1.0 and s["all_gold_found"] == 1.0 and s["mrr"] == 1.0


def test_groups_mix_alternatives_and_required_groups():
    # group1 carried by a|b, group2 only by c: top-5 has b -> half covered;
    # union adds c -> complete answer possible
    s = retrieval_scores([["a#1", "b#2"], ["c#3"]], ["b#2"], ["b#2", "c#3"])
    assert s["hit@5"] == 1.0 and s["recall@5"] == 0.5 and s["all_gold_found"] == 1.0


def test_citation_valid_against_any_group_member():
    assert citation_precision({"b#2"}, [["a#1", "b#2"], ["c#3"]]) == 1.0
    assert citation_precision({"b#2", "z#7"}, [["a#1", "b#2"]]) == 0.5


def test_citation_precision_cases():
    assert citation_precision(set(), [["a#1"]]) is None            # nothing cited
    assert citation_precision({"a#1"}, [["a#1"]]) == 1.0
    assert citation_precision({"a#1", "x#9"}, [["a#1"]]) == 0.5    # one stray citation
    assert citation_precision({"x#9"}, [["a#1"]]) == 0.0           # full miss
    assert citation_precision({"a#1", "b#2"}, [["a#1"], ["b#2"], ["c#3"]]) == 1.0


def test_citation_recall_measures_completeness():
    gold = [["a#1", "b#2"], ["c#3"]]
    assert citation_recall({"b#2", "c#3"}, gold) == 1.0    # each group has a carrier
    assert citation_recall({"b#2"}, gold) == 0.5           # precision would still be 1.0
    assert citation_recall(set(), gold) == 0.0             # uncited groups, not None
    assert citation_recall({"x#9"}, None) is None          # negatives: nothing to attribute


def test_mcnemar_exact_values():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(1, 1) == 1.0                   # perfectly balanced flips
    assert mcnemar_p(6, 0) == 0.03125               # smallest significant split
    assert mcnemar_p(5, 0) == 0.0625                # one flip short of significance
    assert abs(mcnemar_p(7, 2) - 0.1796875) < 1e-9  # 2*(1+9+36)/512
    assert mcnemar_p(0, 6) == mcnemar_p(6, 0)       # direction-symmetric


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile(list(range(1, 101)), 50) == 50


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all scoring tests passed")
