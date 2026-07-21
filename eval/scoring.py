"""Pure eval math shared by the runner (run_langfuse) and the exporter (export_run).
No langfuse/app imports — unit-testable anywhere:
    python eval/tests/test_scoring.py
"""
import math

# Negative subtypes whose required behaviour is a substantive reply, not a refusal —
# judged by Correctness against their behavioural reference.
# ONE definition — imported by the runner, the exporter and the dataset validator.
CORRECTNESS_JUDGED_NEGATIVES = ("contradiction", "false-premise", "epistemic", "injection")


def gold_groups(gold):
    """Normalize gold_chunks into groups of interchangeable carrier chunks.

    Only a list of non-empty groups is accepted; anything else is a ValueError.
    """
    groups = []
    for g in gold or []:
        if not isinstance(g, (list, tuple, set)) or not g:
            raise ValueError(f"gold_chunks must be a list of non-empty gold groups, got {g!r}")
        groups.append(set(g))
    return groups


def retrieval_scores(gold, first, union):
    """Retrieval of one turn against gold carrier groups.

    A group is covered when any interchangeable carrier is retrieved. ``first`` contains
    ids from the first search in rerank order; ``union`` contains ids from all searches.
    """
    groups = gold_groups(gold)
    flat = set().union(*groups) if groups else set()
    union = set(union)
    top5 = set(list(first)[:5])
    rank = next((i for i, cid in enumerate(list(first)[:5], 1) if cid in flat), 0)
    covered5 = sum(1 for g in groups if g & top5)
    covered_union = sum(1 for g in groups if g & union)
    return {
        "hit@5": 1.0 if rank else 0.0,
        "recall@5": (covered5 / len(groups)) if groups else 0.0,
        "mrr": (1.0 / rank) if rank else 0.0,
        "recall_any_search": 1.0 if flat & union else 0.0,
        "all_gold_found": 1.0 if groups and covered_union == len(groups) else 0.0,
    }


def citation_precision(cited, gold):
    """Share of the answer's citations pointing at any gold carrier. None when the answer
    cites nothing — nothing to measure."""
    cited = set(cited)
    if not cited:
        return None
    flat = set().union(*gold_groups(gold)) if gold else set()
    return len(cited & flat) / len(cited)


def citation_recall(cited, gold):
    """Share of gold groups covered by at least one citation.

    None when the question has no gold groups; zero citations scores 0.0, not None.
    """
    groups = gold_groups(gold)
    if not groups:
        return None
    cited = set(cited)
    return sum(1 for g in groups if g & cited) / len(groups)


def mcnemar_p(b, c):
    """Exact two-sided McNemar over the discordant pairs of two paired runs (b fail->pass,
    c pass->fail): p = min(1, 2*P(X <= min(b,c))), X ~ Binomial(b+c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def percentile(values, q):
    """Nearest-rank percentile (q in 0..100); None on empty input."""
    if not values:
        return None
    vs = sorted(values)
    idx = max(0, math.ceil(q / 100.0 * len(vs)) - 1)
    return vs[min(idx, len(vs) - 1)]
