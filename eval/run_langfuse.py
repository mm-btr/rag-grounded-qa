"""Langfuse eval: upsert the golden dataset, run the agent over it, attach code metrics.
Run inside the app image: python eval/run_langfuse.py --name vN-<set size> [--smoke].

LLM judges (Correctness, Faithfulness) run in the Langfuse UI on every new experiment run.
Faithfulness context contract: retrieved passages are written to the item's ROOT SPAN
metadata under `retrieved_context` — renaming the key breaks the UI evaluator.
"""
import argparse
import asyncio
import glob
import hashlib
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (eval/ beside app/)
sys.path.insert(0, os.path.join(_ROOT, "app"))                         # import agent/config/... from app/

from langfuse import Evaluation, get_client

from agent import Ctx, _is_abstention, build_agent, content_text
from config import (AGENT_RECURSION_LIMIT, AGENT_TIMEOUT, COLLECTION, DEFAULT_TENANT,
                    EMBED_MODEL, LLM_MODEL, LLM_REASONING_EFFORT, RERANK_MODEL,
                    TOP_K_RERANK, TOP_K_RETRIEVE)
from keys import CITE_RE

import scoring   # eval-local pure math (this file's dir is on sys.path when run as a script)

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(_HERE, "golden.json")
SMOKE = os.path.join(_HERE, "smoke.json")         # same schema, fresh unseen questions

# Reference for negatives (golden reference_answer is null): tells the Correctness judge
# that the RIGHT behaviour is an abstention, so a refusal scores high, a made-up answer low.
NEGATIVE_REFERENCE = (
    "Ответа в корпусе нет. Правильное поведение — отказ, начинающийся с «Не могу ответить» "
    "(а для неоднозначного вопроса — короткий уточняющий вопрос), без выдуманных фактов."
)

# Retrieval of each item, filled by the task, read by the retrieval_quality evaluator.
# golden id -> {"first": ["doc_id#chunk"...] in rerank order (the forced first search),
#               "union": same ids for ALL searches of the turn}. Keyed by golden id.
RETRIEVED = {}


def load_dataset(path):
    """Load one immutable byte snapshot and return its rows plus content revision."""
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()[:12]


# Dataset-item metadata: presence-filtered (absent keys are dropped, not written as null).
_META_KEYS = (
    "id", "type", "negative_type", "near", "answerable", "gold_chunks",
    "content", "lexical", "twin_of", "held_out", "sources",
)


def _item_metadata(q):
    return {k: q[k] for k in _META_KEYS if k in q}


def upsert_dataset(lf, name, rows):
    """Idempotent upsert; the dataset MIRRORS the local file: items gone from the file are
    archived (excluded from runs), and a count gate aborts before the experiment if the
    active set diverges from the file."""
    lf.create_dataset(name=name)
    for q in rows:
        lf.create_dataset_item(
            dataset_name=name,
            id=f"{name}-{q['id']}",
            input=q["question"],                       # string -> clean {{query}}
            expected_output=q["reference_answer"] or NEGATIVE_REFERENCE,
            metadata=_item_metadata(q),
        )
    keep = {f"{name}-{q['id']}" for q in rows}
    stale, active, page = 0, 0, 1
    while True:
        res = lf.api.dataset_items.list(dataset_name=name, page=page, limit=100)
        for it in res.data:
            if "ARCHIVED" in str(getattr(it, "status", "")).upper():
                continue
            if it.id in keep:
                active += 1
            else:
                lf.create_dataset_item(dataset_name=name, id=it.id, status="ARCHIVED")
                stale += 1
        if page >= res.meta.total_pages:
            break
        page += 1
    if active != len(rows):
        raise SystemExit(f"датасет '{name}' рассинхронизирован: {active} активных ≠ "
                         f"{len(rows)} вопросов в файле — прогон остановлен")
    return stale


# --- item-level code evaluators (signature fixed by the Experiment SDK) ---------------------
# Refusal/clarification detection is imported from the agent — one normalizer, no drift.

def retrieval_quality(*, output, metadata, **kwargs):
    """Retrieval scores of one answerable item by gold-group membership: hit@5 /
    recall@5 / mrr over the first search, recall_any_search / all_gold_found over the turn."""
    if not metadata or not metadata.get("answerable"):
        return []
    groups = scoring.gold_groups(metadata.get("gold_chunks"))
    if not groups:
        return []
    data = RETRIEVED.get(metadata["id"]) or {}
    first, union = data.get("first") or [], data.get("union") or []
    s = scoring.retrieval_scores(groups, first, union)
    n = len(groups)
    in_top5 = round(s["recall@5"] * n)
    in_union = sum(1 for g in groups if g & set(union))
    rank = round(1.0 / s["mrr"]) if s["mrr"] else 0
    return [
        Evaluation(name="hit@5", value=s["hit@5"],
                   comment=(f"носитель на ранге {rank}" if rank
                            else "носителя нет в топ-5 первого поиска")),
        Evaluation(name="recall@5", value=s["recall@5"],
                   comment=f"{in_top5} из {n} gold-групп покрыты топ-5 первого поиска"),
        Evaluation(name="mrr", value=s["mrr"]),
        Evaluation(name="recall_any_search", value=s["recall_any_search"],
                   comment="носитель был в выдаче хотя бы одного поиска" if s["recall_any_search"]
                           else "ни один поиск хода не принёс носителя"),
        Evaluation(name="all_gold_found", value=s["all_gold_found"],
                   comment="все gold-группы покрыты найденными носителями"
                           if s["all_gold_found"]
                           else f"покрыто {in_union} из {n} gold-групп за весь ход"),
    ]


def citation_quality(*, output, metadata, **kwargs):
    """Attribution of the final answer: citation_precision (are the footnotes valid) and
    citation_recall (is every fact signed). Code-only, by gold ids."""
    if not metadata or not metadata.get("answerable"):
        return []
    groups = scoring.gold_groups(metadata.get("gold_chunks"))
    if not groups:
        return []
    cited = {f"{m.group(1)}#{m.group(2)}" for m in CITE_RE.finditer(str(output or ""))}
    out = []
    p = scoring.citation_precision(cited, groups)
    if p is not None:                              # citation-free answer: no precision to measure
        stray = sorted(cited - set().union(*groups))
        out.append(Evaluation(name="citation_precision", value=p,
                              comment="все сноски ведут в носители ответа" if p == 1.0
                              else f"сноски мимо носителей: {', '.join(stray)}"))
    r = scoring.citation_recall(cited, groups)
    out.append(Evaluation(name="citation_recall", value=r,
                          comment=f"процитированы носители {round(r * len(groups))} "
                                  f"из {len(groups)} gold-групп"))
    return out


def over_refusal(*, output, metadata, **kwargs):
    """Answerables only: a refusal on an answerable question is a miss."""
    if not metadata.get("answerable"):
        return []
    refused = _is_abstention(output)
    return [Evaluation(name="over_refusal", value=1.0 if refused else 0.0,
                       comment="отказ на отвечаемом вопросе" if refused else "ответил")]


def run_summary(*, item_results, **kwargs):
    """Run-level averages of every code score (the LLM judges — Correctness, Faithfulness —
    run in the Langfuse UI and aggregate themselves there)."""
    def avg(name):
        vals = [e.value for r in item_results for e in r.evaluations
                if e.name == name and e.value is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    out = [(n, avg(n)) for n in ("hit@5", "recall@5", "mrr", "recall_any_search",
                                 "all_gold_found", "citation_precision", "citation_recall",
                                 "over_refusal")]
    return [Evaluation(name=f"avg_{n}", value=v) for n, v in out if v is not None]


# --- the task: one grounded turn through the real agent --------------------------------------

def make_task(agent):
    from langfuse.langchain import CallbackHandler
    handler = CallbackHandler()   # nest agent spans (model calls, each `search`) under the item's root span

    async def task(*, item, **kwargs):
        messages = [{"role": "user", "content": item.input}]
        out = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": AGENT_RECURSION_LIMIT, "callbacks": [handler]},
                context=Ctx(tenant_id=DEFAULT_TENANT),
            ),
            timeout=AGENT_TIMEOUT,
        )
        msgs = out["messages"]
        answer = content_text(msgs[-1].content)   # Responses API: block list -> text

        # Retrieval provenance: the tool messages of this turn hold the formatted passages,
        # one 【id#n】 header per chunk. Split each search into (ref, chunk_text) blocks.
        tools = [m.content for m in msgs if getattr(m, "type", None) == "tool"]

        def blocks(tool_text):
            ms = list(CITE_RE.finditer(str(tool_text or "")))
            return [((m.group(1), int(m.group(2))),
                     str(tool_text)[m.start(): ms[i + 1].start() if i + 1 < len(ms) else None])
                    for i, m in enumerate(ms)]

        first = blocks(tools[0]) if tools else []
        union = [b for t in tools for b in blocks(t)]

        def cid(ref):
            return f"{ref[0]}#{ref[1]}"           # (doc_id, chunk) -> "doc_id#chunk", gold_chunks form

        gid = (item.metadata or {}).get("id")
        # Count tool messages: a zero-hit search still ran and must not be
        # confused with a turn that never searched.
        RETRIEVED[gid] = {"first": [cid(r) for r, _ in first],
                          "union": [cid(r) for r, _ in union],
                          "n_searches": len(tools)}
        # Span metadata feeds the UI Faithfulness judge; chunks deduped by id, zero-hit
        # turns fall back to the raw tool texts — the judge never gets an empty context.
        seen_ids, ctx = set(), []
        for ref, block in union:
            if ref not in seen_ids:
                seen_ids.add(ref)
                ctx.append(block.strip())
        get_client().update_current_span(
            metadata={"retrieved_context": "\n\n".join(ctx) if ctx else "\n\n".join(tools)})
        return answer
    return task


def _tree_sha(root, subdir):
    """sha256 over one source tree (recursive, tests excluded); each file framed by its
    relative path so a rename or move changes the digest. None if the tree has no .py."""
    h = hashlib.sha256()
    files = sorted(f for f in glob.glob(os.path.join(root, subdir, "**", "*.py"), recursive=True)
                   if os.sep + "tests" + os.sep not in f)
    for fp in files:
        rel = os.path.relpath(fp, root).replace(os.sep, "/")
        h.update(rel.encode("utf-8") + b"\0")
        with open(fp, "rb") as f:
            h.update(f.read() + b"\0")
    return h.hexdigest()[:12] if files else None


def run_passport(dataset_sha):
    """Provenance for the run's metadata (code/prompt shas, retriever knobs, corpus size).
    Dataset revision is mandatory; every environment-derived field is best-effort."""
    p = {"model": LLM_MODEL, "reasoning_effort": LLM_REASONING_EFFORT,
         "top_k_retrieve": TOP_K_RETRIEVE, "top_k_rerank": TOP_K_RERANK,
         "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL,
         "dataset_sha": dataset_sha}
    try:
        with open(os.path.join(_ROOT, "app", "prompts", "system.md"), "rb") as f:
            p["system_prompt_sha"] = hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        pass
    # Judge prompts: per-judge hashes of the eval/judges/ copies (the UI executes its own copy).
    try:
        jp = {}
        for fp in sorted(glob.glob(os.path.join(_ROOT, "eval", "judges", "*.md"))):
            with open(fp, "rb") as f:
                jp[os.path.splitext(os.path.basename(fp))[0]] = hashlib.sha256(f.read()).hexdigest()[:12]
        if jp:
            p["judge_prompts"] = jp
    except OSError:
        pass
    # git is absent on the scp-deployed server, so hash the source directly.
    for key, subdir in (("app_sha", "app"), ("eval_sha", "eval")):
        try:
            sha = _tree_sha(_ROOT, subdir)
            if sha:
                p[key] = sha
        except OSError:
            pass
    try:
        from ingest import _read_filter, _tenant_match, get_client as qdrant_client
        # Published tenant count — same read population as the dump the sha guard compares against.
        flt = _read_filter([_tenant_match(DEFAULT_TENANT)])
        p["corpus_points"] = qdrant_client().count(
            collection_name=COLLECTION, count_filter=flt, exact=True).count
    except Exception:
        pass
    # Corpus content sha via the local dump; hashed only while the dump matches the live
    # index by point count — a stale dump degrades to an absent key.
    try:
        with open(os.path.join(_ROOT, "eval", "corpus", "corpus_flat.json"), "rb") as f:
            raw = f.read()
        if p.get("corpus_points") == len(json.loads(raw)):
            p["corpus_sha"] = hashlib.sha256(raw).hexdigest()[:12]
    except Exception:
        pass
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment run name, vN-<set size>, e.g. v1-100")
    ap.add_argument("--smoke", action="store_true",
                    help="run eval/smoke.json against golden-qa-smoke")
    args = ap.parse_args()

    lf = get_client()
    if not lf.auth_check():
        raise SystemExit("Langfuse auth failed — проверь LANGFUSE_* в .env")

    dataset_name = "golden-qa-smoke" if args.smoke else "golden-qa"
    rows, dataset_sha = load_dataset(SMOKE if args.smoke else GOLDEN)
    stale = upsert_dataset(lf, dataset_name, rows)
    print(f"dataset '{dataset_name}': {len(rows)} items upserted, {stale} stale archived")

    from models import warmup
    warmup()                                    # pay the cold start once, not inside item #1
    agent = build_agent()                       # no checkpointer -> every question starts clean

    passport = run_passport(dataset_sha)
    result = lf.get_dataset(dataset_name).run_experiment(
        name=args.name,
        description=f"model={LLM_MODEL} effort={LLM_REASONING_EFFORT}",
        task=make_task(agent),
        evaluators=[retrieval_quality, citation_quality, over_refusal],
        run_evaluators=[run_summary],
        max_concurrency=1,                      # CPU reranker: parallel turns would just contend
        metadata=passport,
    )
    print(result.format(include_item_results=True))
    lf.flush()
    lf.shutdown()


if __name__ == "__main__":
    main()
