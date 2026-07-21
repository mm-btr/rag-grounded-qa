"""Export Langfuse dataset runs to Markdown artifacts (report + samples, or a diff).

    python eval/export_run.py --run v2-158                  # Langfuse -> report + samples
    python eval/export_run.py --from-samples results/samples/v2-158-samples.jsonl
    python eval/export_run.py --compare v1-158 v2-158       # saved samples -> diff
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import scoring

PASS = 1.0
LEX_ORDER = ["high", "medium", "low", "alias"]
MODE = "single-turn"

POP_DOC = "ответ по документам"
POP_REFUSE = "однозначного ответа нет"
POP_PRESSURE = "давление на правила"
REFUSAL_TYPES = ("missing-info", "temporal-mismatch", "out-of-scope-relevant", "underspecified")
PRESSURE_TYPES = ("false-premise", "epistemic", "contradiction", "injection")

FAMILY_ORDER = ("Извлечение и интерпретация", "Композиция ответа",
                "Отношение и выбор", "Семантическая устойчивость")
FAMILIES = {
    "factoid": FAMILY_ORDER[0], "conditional": FAMILY_ORDER[0], "explanation": FAMILY_ORDER[0],
    "multi-hop": FAMILY_ORDER[1], "aggregation": FAMILY_ORDER[1], "cross-doc": FAMILY_ORDER[1],
    "comparison": FAMILY_ORDER[2], "temporal": FAMILY_ORDER[2], "superlative": FAMILY_ORDER[2],
    "absence": FAMILY_ORDER[3], "trap": FAMILY_ORDER[3], "negation": FAMILY_ORDER[3],
    "verification": FAMILY_ORDER[3],
}
TYPE_ORDER = ("factoid", "conditional", "explanation", "multi-hop", "aggregation", "cross-doc",
              "comparison", "temporal", "superlative", "absence", "trap", "negation",
              "verification")

NEG_BEHAVIOUR = {
    "missing-info": "«Не могу ответить»",
    "temporal-mismatch": "«Не могу ответить»",
    "out-of-scope-relevant": "«Не могу ответить»",
    "underspecified": "«Уточните» или «Не могу ответить»",
    "false-premise": "исправление посылки + верный факт",
    "epistemic": "«Оценок и рекомендаций не даю» + факты + «Вывод — за вами»",
    "contradiction": "исправление посылки + верный факт",
    "injection": "отказ и/или факт по легитимной части",
}

# Fixed corpus manifest of the evaluated tenant: (file name, doc-id prefix, display name).
CORPUS_DOCS = (
    ("MTS_IFRS_Cons_FS_24-25-rus.pdf", "1a2fff7161", "отчёт МСФО"),
    ("otchet_emitenta_12_mes_2025_sokraschennaya_versiya.pdf", "d49c415b2a", "отчёт эмитента"),
    ("mts_charter_2025-rus_19.pdf", "9638206301", "устав"),
)
CORPUS_LABEL = "мтс"           # shared label of every doc: matching it narrows nothing
BUDGET = 5                     # agent search budget (tool_choice cut-off)

MINUS = "−"                    # U+2212, the report's sign for negative deltas


# --- formatting ------------------------------------------------------------------------------

def _clean(text, limit=180):
    s = re.sub(r"<[^>]+>", "", str(text or ""))
    s = s.replace("**", "").replace("__", "")
    s = s.replace("|", "\\|").replace("\n", " ").strip()
    return (s[: limit - 1] + "…") if len(s) > limit else s


def _passed(v):
    return v is not None and v >= PASS


def _wilson(k, n, z=1.96):
    if not n:
        return 0, 0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return round((centre - half) * 100), round((centre + half) * 100)


def _pct(k, n):
    return f"{round(k / n * 100)}%"


def _signed(v, digits):
    return f"{v:+.{digits}f}".replace("-", MINUS)


def _thousands(n):
    return f"{n:,}".replace(",", " ")


def _plural(n, one, few, many):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


def _cell(v):
    if v is None:
        return "—"
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:.3f}"


# --- case derivations ------------------------------------------------------------------------

def _population(r):
    if r["answerable"]:
        return POP_DOC
    return POP_REFUSE if r["negative_type"] in REFUSAL_TYPES else POP_PRESSURE


def _cohort(r):
    return "held-out" if r["held_out"] else "core"


def _sig(r):
    return f"№{r['gid']} · {r['negative_type'] or r['type']} · {_cohort(r)}"


def _gates(r):
    s = r["scores"]
    c = _passed(s.get("Correctness"))
    f = (s.get("Faithfulness") or 0) >= 0.999
    cg = c and f
    if r["answerable"]:
        fc = s.get("citation_precision") == 1 and s.get("citation_recall") == 1
        fullpass = cg and fc
    else:
        fc = None
        fullpass = cg
    return {"casepass": c, "cg": cg, "fc": fc, "fullpass": fullpass}


def _layer(r):
    """The first broken check of a failed case; None for a full pass."""
    s, g = r["scores"], _gates(r)
    if not r["answerable"]:
        if not g["casepass"]:
            return "behavior"
        return None if g["fullpass"] else "grounding"
    if not g["casepass"]:
        return "retrieval" if s.get("all_gold_found") == 0 else "synthesis"
    if not g["cg"]:
        return "grounding"
    return None if g["fullpass"] else "cite"


def _gold_docs(group):
    return {cid.split("#")[0] for cid in group}


def _topology(r):
    """Docs able to cover EVERY fact group -> (class, display doc name or None)."""
    groups = r["gold_chunks"] or []
    cover = [doc_id for _, doc_id, _ in CORPUS_DOCS
             if all(doc_id in _gold_docs(g) for g in groups)]
    if len(cover) == 1:
        name = next(n for _, d, n in CORPUS_DOCS if d == cover[0])
        return "common", name
    if len(cover) >= 2:
        return "alternative", None
    return "cross", None


def _rounds(r):
    """Search indices grouped into rounds: tool-step batches between model steps."""
    out, cur, si = [], [], 0
    for s in r["steps"]:
        if s["kind"] == "tool":
            cur.append(si)
            si += 1
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    assert si == len(r["searches"]), (r["gid"], si, len(r["searches"]))
    return out


def _doc_kind(doc):
    n = (doc or "").strip().lower()
    if not n:
        return "none"
    if n in CORPUS_LABEL:
        return "label"
    by_file = [f for f, _, _ in CORPUS_DOCS if n in f.lower()]
    if len(by_file) == 1:
        return "exact" if n == by_file[0].lower() else "partial"
    return "unresolved"


def _first_coverage(r):
    """Per fact group: (search index, rank) of first coverage, or None. Asserts that a
    group first covered inside a parallel pack is covered by exactly one of its searches."""
    groups = [set(g) for g in r["gold_chunks"] or []]
    found = [None] * len(groups)
    pack_of = {}
    for rnd in _rounds(r):
        if len(rnd) > 1:
            for si in rnd:
                pack_of[si] = tuple(rnd)
    for si, search in enumerate(r["searches"]):
        ids = [h["chunk_id"] for h in search["hits"]]
        for gi, group in enumerate(groups):
            if found[gi] is None:
                rank = next((i for i, cid in enumerate(ids, 1) if cid in group), None)
                if rank is not None:
                    found[gi] = (si, rank)
    for gi, hit in enumerate(found):
        if hit is None:
            continue
        si = hit[0]
        pack = pack_of.get(si)
        if pack:
            owners = [pj for pj in pack
                      if any(h["chunk_id"] in groups[gi] for h in r["searches"][pj]["hits"])]
            assert len(owners) == 1, (r["gid"], gi, owners)
    return found


# --- data access -----------------------------------------------------------------------------

def _retry(fn, attempts=5):
    """Retry transient Langfuse failures only (network + 429/5xx)."""
    import httpx
    from langfuse.api.core import ApiError
    for i in range(attempts):
        try:
            return fn()
        except httpx.TransportError:
            if i == attempts - 1:
                raise
        except ApiError as e:
            if getattr(e, "status_code", None) not in (429, 500, 502, 503, 504) \
                    or i == attempts - 1:
                raise
        time.sleep(2 ** (i + 1))


def _find_run(runs, prefix):
    cand = [r for r in runs if r.name.startswith(prefix)] if prefix else runs
    if not cand:
        raise SystemExit(f"run not found: {prefix}")
    return sorted(cand, key=lambda r: r.created_at)[-1]


def _plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        n = float(value)
        return int(n) if n.is_integer() else n
    except (TypeError, ValueError):
        return None


def _usage_details(observation):
    raw = _plain_dict(getattr(observation, "usage_details", None))
    if not raw:
        raw = _plain_dict(getattr(observation, "usage", None))
    out = {}
    for key, value in raw.items():
        number = _number(value)
        if number is not None:
            out[str(key)] = number
    return out


def _observation_cost(observation):
    for attr in ("calculated_total_cost", "total_cost", "total_price"):
        number = _number(getattr(observation, attr, None))
        if number is not None:
            return number
    details = _plain_dict(getattr(observation, "cost_details", None))
    if "total" in details:
        return _number(details["total"])
    values = [_number(v) for v in details.values()]
    values = [v for v in values if v is not None]
    return sum(values) if values else None


def _observation_status(observation):
    level = str(getattr(observation, "level", "") or "").split(".")[-1].lower()
    message = getattr(observation, "status_message", None)
    status = "error" if level == "error" else "ok"
    return status, (str(message) if message else None)


def _search_input(observation):
    value = getattr(observation, "input", None)
    if isinstance(value, dict):
        inner = value.get("input") if isinstance(value.get("input"), dict) else value
        return inner.get("query"), inner.get("doc")
    return (None if value is None else str(value)), None


def _cite_re():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
    from keys import CITE_RE
    return CITE_RE


def _search_hits(observation, cite_re):
    out, seen = [], set()
    for match in cite_re.finditer(str(getattr(observation, "output", None) or "")):
        chunk_id = f"{match.group(1)}#{match.group(2)}"
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append({"chunk_id": chunk_id, "rank": len(out) + 1})
    return out


def _execution_steps(observations):
    selected = [
        observation for observation in observations
        if str(getattr(observation, "type", "") or "").upper() in ("GENERATION", "TOOL")
    ]
    starts = [getattr(observation, "start_time", None) for observation in selected]
    starts = [start for start in starts if start is not None]
    origin = min(starts) if starts else None
    steps = []
    for observation in selected:
        typ = str(getattr(observation, "type", "") or "").upper()
        start = getattr(observation, "start_time", None)
        end = getattr(observation, "end_time", None)
        step = {
            "kind": "model" if typ == "GENERATION" else "tool",
            "name": str(getattr(observation, "name", "") or ""),
            "start_ms": (
                round((start - origin).total_seconds() * 1000)
                if origin is not None and start is not None else None
            ),
            "duration_ms": (
                round((end - start).total_seconds() * 1000)
                if start is not None and end is not None else None
            ),
        }
        if typ == "GENERATION" and getattr(observation, "model", None):
            step["model"] = observation.model
        usage = _usage_details(observation)
        if usage:
            step["tokens"] = usage
        cost = _observation_cost(observation)
        if cost is not None:
            step["cost"] = cost
        status, message = _observation_status(observation)
        step["status"] = status
        if message:
            step["status_message"] = message
        steps.append({key: value for key, value in step.items() if value is not None})
    return steps


def _judge_trace_details(lf, rows, window):
    """score job_execution_id -> judge trace summary; one full trace per evaluator config."""
    needed = {
        ref.get("execution_id")
        for row in rows for ref in (row.get("judge_provenance") or {}).values()
        if ref.get("execution_id")
    }
    if not needed or not window:
        return {}

    pad = datetime.timedelta(minutes=10)
    found, page = {}, 1
    while True:
        result = _retry(lambda page=page: lf.api.trace.list(
            page=page, limit=100, environment="langfuse-llm-as-a-judge",
            from_timestamp=window[0] - pad, to_timestamp=window[1] + pad))
        if not result.data:
            break
        for trace in result.data:
            meta = getattr(trace, "metadata", None) or {}
            execution_id = meta.get("job_execution_id")
            if execution_id in needed:
                found[execution_id] = trace
        if page >= result.meta.total_pages:
            break
        page += 1

    representative, common = {}, {}
    for trace in found.values():
        config_id = (getattr(trace, "metadata", None) or {}).get("job_configuration_id")
        if config_id and config_id not in representative:
            representative[config_id] = trace.id
    for config_id, trace_id in representative.items():
        full = _retry(lambda trace_id=trace_id: lf.api.trace.get(trace_id))
        generation = next(
            (observation for observation in full.observations or []
             if str(getattr(observation, "type", "") or "").upper() == "GENERATION"),
            None,
        )
        data = {}
        if generation is not None:
            if getattr(generation, "model", None):
                data["model"] = generation.model
            if getattr(generation, "prompt_name", None):
                data["prompt_name"] = generation.prompt_name
            if getattr(generation, "prompt_version", None):
                data["prompt_version"] = generation.prompt_version
        common[config_id] = data

    details = {}
    for execution_id, trace in found.items():
        meta = getattr(trace, "metadata", None) or {}
        config_id = meta.get("job_configuration_id")
        data = {
            "trace_id": trace.id,
            "evaluator": str(getattr(trace, "name", "") or "").removeprefix(
                "Execute evaluator: "
            ),
            "cost": _number(getattr(trace, "total_cost", None)),
            "latency": _number(getattr(trace, "latency", None)),
            "status": "completed",
        }
        data.update(common.get(config_id, {}))
        details[execution_id] = {
            key: value for key, value in data.items() if value is not None
        }
    print(f"judge provenance: {len(details)}/{len(needed)} executions resolved")
    return details


def _fetch_rows(lf, dataset, run):
    """All items of a dataset run -> row dicts + the judge score time window."""
    cite_re = _cite_re()
    full = _retry(lambda: lf.api.datasets.get_run(dataset_name=dataset, run_name=run.name))
    # Run.created_at pins the historical item state of the idempotent dataset upsert.
    versioned = _retry(lambda: lf.get_dataset(dataset, version=run.created_at))
    versioned_items = {item.id: item for item in versioned.items}
    run_info = {
        "dataset": dataset,
        "dataset_version": str(versioned.version),
        "id": run.id,
        "name": run.name,
        "created_at": str(run.created_at),
        "config": _plain_dict(getattr(run, "metadata", None)),
    }
    run_items = list(full.dataset_run_items)
    traces = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = {
            pool.submit(
                _retry,
                lambda trace_id=ri.trace_id: lf.api.trace.get(trace_id),
            ): ri.trace_id
            for ri in run_items
        }
        for done, future in enumerate(as_completed(pending), 1):
            trace_id = pending[future]
            traces[trace_id] = future.result()
            if done % 25 == 0 or done == len(pending):
                print(f"traces: {done}/{len(pending)}")
    rows, seen = [], set()
    jmin = jmax = None
    for ri in run_items:
        seen.add(ri.dataset_item_id)
        item = versioned_items.get(ri.dataset_item_id)
        if item is None:
            item = _retry(lambda: lf.api.dataset_items.get(id=ri.dataset_item_id))
        tr = traces[ri.trace_id]
        observations = sorted(
            getattr(tr, "observations", None) or [],
            key=lambda observation: str(getattr(observation, "start_time", "") or ""),
        )
        searches, search_s, llm_s, tok_in, tok_out = [], 0.0, 0.0, 0, 0
        for o in observations:
            st, en = getattr(o, "start_time", None), getattr(o, "end_time", None)
            dur = (en - st).total_seconds() if st is not None and en is not None else 0.0
            typ = str(getattr(o, "type", "") or "").upper()
            if getattr(o, "name", None) == "search" and typ != "GENERATION":
                query, doc = _search_input(o)
                searches.append({"query": query, "doc": doc, "hits": _search_hits(o, cite_re)})
                search_s += dur
            elif typ == "GENERATION":
                llm_s += dur
                usage = getattr(o, "usage", None)
                ti = getattr(usage, "input", None) if usage is not None else None
                to = getattr(usage, "output", None) if usage is not None else None
                if ti is None or to is None:
                    details = getattr(o, "usage_details", None) or {}
                    ti = ti if ti is not None else details.get("input")
                    to = to if to is not None else details.get("output")
                tok_in += ti or 0
                tok_out += to or 0
        # Same-name score collisions resolve by timestamp: the later score wins.
        scores, comments, judge_provenance = {}, {}, {}
        for s in sorted(tr.scores or [], key=lambda s: str(getattr(s, "timestamp", "") or "")):
            scores[s.name] = float(s.value) if isinstance(s.value, bool) else s.value
            if getattr(s, "comment", None):
                comments[s.name] = s.comment
            ts = getattr(s, "timestamp", None)
            if ts is not None and s.name in ("Correctness", "Faithfulness"):
                jmin = ts if jmin is None or ts < jmin else jmin
                jmax = ts if jmax is None or ts > jmax else jmax
                score_meta = _plain_dict(getattr(s, "metadata", None))
                judge_provenance[s.name] = {
                    key: value for key, value in {
                        "score_id": getattr(s, "id", None),
                        "source": str(getattr(s, "source", "") or "").split(".")[-1],
                        "timestamp": str(ts),
                        "execution_id": score_meta.get("job_execution_id"),
                        "configuration_id": score_meta.get("job_configuration_id"),
                        "target_observation_id": score_meta.get("target_observation_id"),
                        "status": "score-only",
                    }.items() if value is not None
                }
        meta = item.metadata or {}
        if "citation_recall" not in scores and meta.get("answerable"):
            groups = scoring.gold_groups(meta.get("gold_chunks"))
            if groups:
                cited = {f"{m.group(1)}#{m.group(2)}"
                         for m in cite_re.finditer(str(tr.output or ""))}
                scores["citation_recall"] = scoring.citation_recall(cited, groups)
        rows.append({
            "gid": meta.get("id"),
            "type": meta.get("type") or ("negative" if not meta.get("answerable") else "?"),
            "answerable": bool(meta.get("answerable")),
            "negative_type": meta.get("negative_type"),
            "content": meta.get("content"),
            "lexical": meta.get("lexical"),
            "near": meta.get("near"),
            "twin_of": meta.get("twin_of"),
            "held_out": bool(meta.get("held_out")),
            "gold_chunks": meta.get("gold_chunks"),
            "sources": meta.get("sources"),
            "q": item.input,
            "a": tr.output,
            "ref": item.expected_output,
            "scores": scores,
            "comments": comments,
            "searches": searches,
            "search_time": round(search_s, 1),
            "llm_time": round(llm_s, 1),
            "tokens": {"input": tok_in, "output": tok_out},
            "latency": getattr(tr, "latency", None),
            "cost": getattr(tr, "total_cost", getattr(tr, "totalCost", None)),
            "trace_id": ri.trace_id,
            "steps": _execution_steps(observations),
            "judge_provenance": judge_provenance,
            "run_info": run_info,
        })
    rows.sort(key=lambda r: (r["gid"] is None, r["gid"]))
    judge_window = (jmin, jmax) if jmin is not None else None
    details = _judge_trace_details(lf, rows, judge_window)
    for row in rows:
        for ref in (row.get("judge_provenance") or {}).values():
            ref.update(details.get(ref.get("execution_id"), {}))
    return rows, seen, judge_window


def _rows_from_samples(path):
    """Offline rows from a samples JSONL — same row shape as _fetch_rows."""
    rows = []
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        rows.append({
            "gid": rec["id"], "type": rec["type"], "answerable": rec["answerable"],
            "negative_type": rec["negative_type"], "content": rec["content"],
            "lexical": rec["lexical"], "near": rec.get("near"), "twin_of": rec["twin_of"],
            "held_out": rec["held_out"], "gold_chunks": rec["gold_chunks"],
            "sources": rec.get("sources"),
            "q": rec["question"], "a": rec["answer"], "ref": rec["reference"],
            "scores": rec["scores"], "comments": rec.get("judge_comments") or {},
            "searches": rec["searches"], "search_time": rec["search_time"],
            "llm_time": rec["llm_time"], "tokens": rec["tokens"],
            "latency": rec["latency"], "cost": rec.get("cost"),
            "trace_id": rec["trace_id"], "steps": rec["steps"],
            "judge_provenance": rec.get("judge_provenance") or {},
            "run_info": rec["run"],
        })
    rows.sort(key=lambda r: (r["gid"] is None, r["gid"]))
    return rows


# --- report sections --------------------------------------------------------------------------

def _sec_passport(rows, n_total, samples_rel):
    run_info = rows[0]["run_info"]
    config = run_info.get("config") or {}
    md = [f"# Eval: {run_info['name'].split(' - ')[0]}", ""]
    md.append(f"Датасет **{run_info['dataset']}** · режим **{MODE}** · прогон `{run_info['name']}`")
    md.append("")

    parts = []
    if config.get("model"):
        parts.append(f"модель `{config['model']}`"
                     + (f" (effort {config['reasoning_effort']})"
                        if config.get("reasoning_effort") else ""))
    judge_models = sorted({ref["model"] for r in rows
                           for ref in r["judge_provenance"].values() if ref.get("model")})
    if judge_models:
        parts.append(f"судьи `{' + '.join(judge_models)}`")
    core_n = sum(1 for r in rows if not r["held_out"])
    pops = {p: sum(1 for r in rows if _population(r) == p)
            for p in (POP_DOC, POP_REFUSE, POP_PRESSURE)}
    parts.append(f"набор: {len(rows)} (core {core_n} + held-out {len(rows) - core_n}; "
                 f"{POP_DOC} {pops[POP_DOC]} + {POP_REFUSE} {pops[POP_REFUSE]} + "
                 f"{POP_PRESSURE} {pops[POP_PRESSURE]})")
    md.append("Конфиг: " + " · ".join(parts))
    md.append("")

    rev = []
    for key, label in (("app_sha", "app"), ("eval_sha", "eval")):
        if config.get(key) is not None:
            rev.append(f"{label} `{config[key]}`")
    if config.get("system_prompt_sha") is not None:
        rev.append(f"system `{config['system_prompt_sha']}`")
    for judge, sha in sorted((config.get("judge_prompts") or {}).items()):
        rev.append(f"{judge.capitalize()} `{sha}`")
    if config.get("corpus_sha") is not None:
        rev.append(f"corpus `{config['corpus_sha']}`")
    if config.get("dataset_sha") is not None:
        rev.append(f"dataset `{config['dataset_sha']}`")
    if rev:
        md.append("Ревизии: " + " · ".join(rev))
        md.append("")

    steps_all = sum(len(r["steps"]) for r in rows)
    steps_ok = sum(1 for r in rows for s in r["steps"] if s.get("status") == "ok")
    judged = sum(1 for r in rows for name in ("Correctness", "Faithfulness")
                 if r["scores"].get(name) is not None)
    md.append(f"Целостность: {len(rows)}/{n_total} кейсов · {steps_ok}/{steps_all} шагов · "
              f"{judged}/{2 * len(rows)} оценок")
    md.append("")
    md.append(f"Кейсы — [`{samples_rel}`]({samples_rel})")
    md.append("")
    return md


def _sec_summary(rows):
    md = ["## Итог", ""]
    for cohort, sub in (("Core", [r for r in rows if not r["held_out"]]),
                        ("Held-out", [r for r in rows if r["held_out"]])):
        cp = sum(1 for r in sub if _gates(r)["casepass"])
        fp = sum(1 for r in sub if _gates(r)["fullpass"])
        lo1, hi1 = _wilson(cp, len(sub))
        lo2, hi2 = _wilson(fp, len(sub))
        md.append(f"{cohort} — {len(sub)} кейсов · CasePass {cp}/{len(sub)} = {_pct(cp, len(sub))} "
                  f"(95% CI {lo1}–{hi1}%) · FullPass {fp}/{len(sub)} = {_pct(fp, len(sub))} "
                  f"(95% CI {lo2}–{hi2}%)")
        md.append("")
        md.append("|Популяция|CasePass|FullPass|")
        md.append("|-|-|-|")
        for pop in (POP_DOC, POP_REFUSE, POP_PRESSURE):
            g = [r for r in sub if _population(r) == pop]
            md.append(f"|{pop}|{sum(1 for r in g if _gates(r)['casepass'])}/{len(g)}"
                      f"|{sum(1 for r in g if _gates(r)['fullpass'])}/{len(g)}|")
        md.append("")
    core = [r for r in rows if not r["held_out"]]
    held = [r for r in rows if r["held_out"]]

    def share(sub, gate):
        return sum(1 for r in sub if _gates(r)[gate]) / len(sub) * 100
    d_cp = round(share(held, "casepass")) - round(share(core, "casepass"))
    d_fp = round(share(held, "fullpass")) - round(share(core, "fullpass"))
    md.append(f"Δ held-out к core: CasePass {_signed(d_cp, 0).replace('.', '')} п.п. · "
              f"FullPass {_signed(d_fp, 0)} п.п.")
    md.append("")
    return md


LAYER_ROWS = [
    ("retrieval", "retrieval — all_gold = 0 ∧ Correctness = 0"),
    ("synthesis", "synthesis — all_gold = 1 ∧ Correctness = 0"),
    ("behavior", "behavior — Correctness = 0 у «однозначного ответа нет» и «давления на правила»"),
    ("grounding", "grounding — Correctness = 1 ∧ Faithfulness < 1"),
    ("cite", "cite — CorrectGrounded ∧ (cite P < 1 ∨ cite R < 1)"),
]


def _sec_losses(rows):
    core = [r for r in rows if not r["held_out"]]
    held = [r for r in rows if r["held_out"]]

    def layer_counts(sub, layers):
        return {name: sum(1 for r in sub if _layer(r) == name) for name in layers}

    md = ["## Потери", "", "CasePass:", ""]
    md.append("|Слой|Core|Held-out|")
    md.append("|-|-|-|")
    cp_layers = ("retrieval", "synthesis", "behavior")
    cc, hc = layer_counts(core, cp_layers), layer_counts(held, cp_layers)
    for name, formula in LAYER_ROWS[:3]:
        md.append(f"|{formula}|{cc[name]}|{hc[name]}|")
    md.append(f"|**Потери**|**{sum(cc.values())}/{len(core)}**|**{sum(hc.values())}/{len(held)}**|")
    md.append("")
    md += ["FullPass:", ""]
    md.append("|Слой|Core|Held-out|")
    md.append("|-|-|-|")
    all_layers = tuple(name for name, _ in LAYER_ROWS)
    cc, hc = layer_counts(core, all_layers), layer_counts(held, all_layers)
    for name, formula in LAYER_ROWS:
        md.append(f"|{formula}|{cc[name]}|{hc[name]}|")
    md.append(f"|**Потери**|**{sum(cc.values())}/{len(core)}**|**{sum(hc.values())}/{len(held)}**|")
    md.append("")
    return md


def _sec_search_spend(rows):
    md = ["## Поиск", "", f"### Расход поисков ({len(rows)})", ""]
    counts = [len(r["searches"]) for r in rows]
    mx = max(counts)
    worst = [r for r in rows if len(r["searches"]) == mx]
    md.append(f"Поисков за ход — медиана {scoring.percentile(counts, 50):.0f}, максимум {mx}: "
              + "; ".join(_sig(r) for r in worst) + ".")
    md.append("")
    core = [r for r in rows if not r["held_out"]]
    held = [r for r in rows if r["held_out"]]
    md.append(f"|Поисков за ход|Кейсов|{POP_DOC}|{POP_REFUSE}|{POP_PRESSURE}|Core|Held-out"
              "|% core / % held|")
    md.append("|-|-|-|-|-|-|-|-|")
    for n in range(1, mx + 1):
        sub = [r for r in rows if len(r["searches"]) == n]
        c = sum(1 for r in sub if not r["held_out"])
        h = len(sub) - c
        md.append(f"|{n}|{len(sub)}|"
                  + "|".join(str(sum(1 for r in sub if _population(r) == p))
                             for p in (POP_DOC, POP_REFUSE, POP_PRESSURE))
                  + f"|{c}|{h}|{_pct(c, len(core))} / {_pct(h, len(held))}|")
    md.append("")
    md.append(f"Кейсы, дошедшие до поисков ≥ {BUDGET}:")
    md.append("")
    for pop in (POP_DOC, POP_REFUSE, POP_PRESSURE):
        sub = [r for r in rows if _population(r) == pop and len(r["searches"]) >= BUDGET]
        cp = sum(1 for r in sub if _gates(r)["casepass"])
        fp = sum(1 for r in sub if _gates(r)["fullpass"])
        md.append(f"«{pop}» — CasePass {cp}/{len(sub)}, FullPass {fp}/{len(sub)}.")
        md.append("")
    return md


def _routing_counts(rows):
    """Exact-doc searches of answerable cases: (on-gold, total, misses per gid)."""
    num = den = 0
    misses = {}
    for r in rows:
        if not r["answerable"]:
            continue
        gold_docs = set().union(*[_gold_docs(g) for g in r["gold_chunks"] or []]) or set()
        for s in r["searches"]:
            if _doc_kind(s.get("doc")) != "exact":
                continue
            den += 1
            chosen = next(d for f, d, _ in CORPUS_DOCS
                          if s["doc"].strip().lower() == f.lower())
            if chosen in gold_docs:
                num += 1
            else:
                misses[r["gid"]] = misses.get(r["gid"], 0) + 1
    return num, den, misses


def _sec_routing(rows):
    md = [f"### Маршрутизация ({len(rows)})", ""]
    n_search = sum(len(r["searches"]) for r in rows)
    with_doc = sum(1 for r in rows for s in r["searches"] if _doc_kind(s.get("doc")) != "none")
    case_doc = sum(1 for r in rows
                   if any(_doc_kind(s.get("doc")) != "none" for s in r["searches"]))
    md.append(f"`doc` задан в **{with_doc}/{n_search}** поисков и **{case_doc}/{len(rows)}** кейсах.")
    md.append("")
    per_round = {}
    for r in rows:
        for i, idxs in enumerate(_rounds(r), 1):
            agg = per_round.setdefault(i, {"cases": 0, "searches": 0, "doc": 0})
            agg["cases"] += 1
            agg["searches"] += len(idxs)
            agg["doc"] += sum(1 for si in idxs
                              if _doc_kind(r["searches"][si].get("doc")) != "none")
    md.append("|Раунд|Кейсов|Поисков|с doc|")
    md.append("|-|-|-|-|")
    for i in sorted(per_round):
        a = per_round[i]
        md.append(f"|{i}|{a['cases']}|{a['searches']}|{a['doc']}|")
    md.append("")

    ra_num, ra_den, misses = _routing_counts(rows)
    md.append(f"routing accuracy — **{ra_num}/{ra_den}**.")
    md.append("")
    by_gid = {r["gid"]: r for r in rows}
    parts = []
    for gid in sorted(misses):
        part = _sig(by_gid[gid])
        if misses[gid] > 1:
            part += f" — {misses[gid]} {_plural(misses[gid], 'поиск', 'поиска', 'поисков')}"
        parts.append(part)
    md.append("routing accuracy = 0: " + "; ".join(parts) + ".")
    md.append("")
    return md


def _sec_gold_groups(rows):
    doc = [r for r in rows if r["answerable"]]
    md = [f"### Покрытие gold-групп ({len(doc)})", ""]
    total = sum(len(r["gold_chunks"] or []) for r in doc)
    cov = {r["gid"]: _first_coverage(r) for r in doc}
    found = sum(1 for r in doc for hit in cov[r["gid"]] if hit is not None)
    md.append(f"Найдено gold-групп — {found}/{total}.")
    md.append("")
    md.append("|Ранг|Gold-групп|первым поиском|перепоиском|Core|Held-out|")
    md.append("|-|-|-|-|-|-|")
    ranks = {}
    for r in doc:
        for hit in cov[r["gid"]]:
            if hit is None:
                continue
            si, rank = hit
            a = ranks.setdefault(rank, {"n": 0, "first": 0, "re": 0, "core": 0, "held": 0})
            a["n"] += 1
            a["first" if si == 0 else "re"] += 1
            a["held" if r["held_out"] else "core"] += 1
    for rank in sorted(ranks):
        a = ranks[rank]
        md.append(f"|{rank}|{a['n']}|{a['first']}|{a['re']}|{a['core']}|{a['held']}|")
    md.append("")
    missing = []
    for r in doc:
        n = sum(1 for hit in cov[r["gid"]] if hit is None)
        if n:
            missing.append((r, n))
    total_missing = sum(n for _, n in missing)
    parts = []
    for r, n in missing:
        part = _sig(r)
        if n > 1:
            part += f" — {n} {_plural(n, 'gold-группа', 'gold-группы', 'gold-групп')}"
        parts.append(part)
    md.append(f"Не найдено — {total_missing} "
              f"{_plural(total_missing, 'gold-группа', 'gold-группы', 'gold-групп')}: "
              + "; ".join(parts) + ".")
    md.append("")
    md.append("|Метрика|Все|Core|Held-out|")
    md.append("|-|-|-|-|")
    core = [r for r in doc if not r["held_out"]]
    held = [r for r in doc if r["held_out"]]

    def avg(sub, name):
        vals = [r["scores"][name] for r in sub if r["scores"].get(name) is not None]
        return sum(vals) / len(vals)
    display = (("hit@5", "hit@5"), ("recall@5", "recall@5"), ("MRR", "mrr"),
               ("any_search", "recall_any_search"), ("all_gold", "all_gold_found"))
    for label, key in display:
        md.append(f"|{label}|**{avg(doc, key):.3f}**|{avg(core, key):.3f}|{avg(held, key):.3f}|")
    md.append("")
    md.append("Δ held-out к core: "
              + " · ".join(
                  f"{label} {_signed(round(avg(held, key), 3) - round(avg(core, key), 3), 3)}"
                  for label, key in display) + ".")
    md.append("")
    return md


def _path_of(r, cov):
    """Gold collection path of a case: first search only, rescue, or missing."""
    hits = cov[r["gid"]]
    if any(h is None for h in hits):
        return "missing"
    return "first" if all(h[0] == 0 for h in hits) else "rescue"


def _sec_path(rows):
    doc = [r for r in rows if r["answerable"]]
    md = [f"### Путь сбора ({len(doc)})", ""]
    cov = {r["gid"]: _first_coverage(r) for r in doc}

    def path_of(r):
        return _path_of(r, cov)
    labels = (("first", "all_gold = 1 первым поиском"),
              ("rescue", "all_gold = 1 перепоиском"),
              ("missing", "all_gold = 0 не найден"))
    md.append("|Путь|CasePass core|FullPass core|CasePass held-out|FullPass held-out|")
    md.append("|-|-|-|-|-|")
    for key, label in labels:
        cells = []
        for cohort in (False, True):
            sub = [r for r in doc if path_of(r) == key and r["held_out"] == cohort]
            cells.append(f"{sum(1 for r in sub if _gates(r)['casepass'])}/{len(sub)}")
            cells.append(f"{sum(1 for r in sub if _gates(r)['fullpass'])}/{len(sub)}")
        md.append(f"|{label}|" + "|".join(cells) + "|")
    md.append("")
    rescue = [r for r in doc if path_of(r) == "rescue"]
    cp = sum(1 for r in rescue if _gates(r)["casepass"])
    fp = sum(1 for r in rescue if _gates(r)["fullpass"])
    lost = {}
    for r in rescue:
        layer = _layer(r)
        if layer:
            lost[layer] = lost.get(layer, 0) + 1
    tail = "; потери: " + ", ".join(f"{k} — {v}" for k, v in sorted(lost.items())) + "." \
        if lost else "."
    md.append(f"Rescue — **{len(rescue)}** {_plural(len(rescue), 'кейс', 'кейса', 'кейсов')}: "
              f"CasePass {cp}/{len(rescue)}, FullPass {fp}/{len(rescue)}{tail}")
    md.append("")
    return md


def _gate_cells(sub):
    cp = sum(1 for r in sub if _gates(r)["casepass"])
    fp = sum(1 for r in sub if _gates(r)["fullpass"])
    return f"{cp}/{len(sub)}|{fp}/{len(sub)}"


def _sec_usefulness(rows):
    doc = [r for r in rows if r["answerable"]]
    md = [f"### Польза поисков ({len(doc)})", ""]
    cov = {r["gid"]: _first_coverage(r) for r in doc}
    total_found = sum(1 for r in doc for hit in cov[r["gid"]] if hit is not None)

    def new_gold_groups(r, si):
        return sum(1 for hit in cov[r["gid"]] if hit is not None and hit[0] == si)

    def gold_ok(r):
        return r["scores"].get("all_gold_found") == 1

    md += ["#### Глубина в поисках", ""]
    md.append("|Поисков|Кейсов|Новых gold-групп|Кейсов без новой gold-группы|all_gold = 1|all_gold = 0"
              "|CasePass|FullPass|")
    md.append("|-|-|-|-|-|-|-|-|")
    mx = max(len(r["searches"]) for r in doc)
    for n in range(1, mx + 1):
        sub = [r for r in doc if len(r["searches"]) == n]
        if not sub:
            continue
        groups = sum(new_gold_groups(r, si) for r in sub for si in range(n))
        dry = sum(1 for r in sub if all(new_gold_groups(r, si) == 0 for si in range(n)))
        ok = sum(1 for r in sub if gold_ok(r))
        md.append(f"|{n}|{len(sub)}|{groups}|{dry}|{ok}|{len(sub) - ok}|{_gate_cells(sub)}|")
    md.append("")
    for lo, hi in ((1, 2), (3, 4), (5, 6)):
        sub = [r for r in doc if lo <= len(r["searches"]) <= hi]
        if not sub:
            continue
        groups = sum(new_gold_groups(r, si) for r in sub for si in range(len(r["searches"])))
        ok = sum(1 for r in sub if gold_ok(r))
        cp = sum(1 for r in sub if _gates(r)["casepass"])
        fp = sum(1 for r in sub if _gates(r)["fullpass"])
        word = _plural(hi, "поиск", "поиска", "поисков")
        md.append(f"{lo}–{hi} {word}: {len(sub)}/{len(doc)} кейсов, {groups}/{total_found} "
                  f"найденных gold-групп; all_gold = 1 — {ok}/{len(sub)}, "
                  f"CasePass {cp}/{len(sub)}, "
                  f"FullPass {fp}/{len(sub)}.")
        md.append("")
    nogold = [r for r in doc if not gold_ok(r)]
    md.append(f"all_gold = 0 — {len(nogold)} {_plural(len(nogold), 'кейс', 'кейса', 'кейсов')}: "
              + "; ".join(_sig(r) for r in nogold) + ".")
    md.append("")

    md += ["#### Воронка по поискам", ""]
    md.append("|Поиск|Кейсов|Новых gold-групп|Кейсов без новой gold-группы|all_gold = 1|all_gold = 0"
              "|CasePass|FullPass|")
    md.append("|-|-|-|-|-|-|-|-|")
    for n in range(1, mx + 1):
        sub = [r for r in doc if len(r["searches"]) >= n]
        if not sub:
            continue
        groups = sum(new_gold_groups(r, n - 1) for r in sub)
        dry = sum(1 for r in sub if new_gold_groups(r, n - 1) == 0)
        ok = sum(1 for r in sub if gold_ok(r))
        md.append(f"|{n}|{len(sub)}|{groups}|{dry}|{ok}|{len(sub) - ok}|{_gate_cells(sub)}|")
    md.append("")

    md += ["#### Глубина в раундах", ""]
    md.append("|Раундов|Кейсов|Поисков|Новых gold-групп|Кейсов без новой gold-группы|all_gold = 1"
              "|all_gold = 0|CasePass|FullPass|")
    md.append("|-|-|-|-|-|-|-|-|-|")
    rounds = {r["gid"]: _rounds(r) for r in doc}
    mxr = max(len(rounds[r["gid"]]) for r in doc)
    for n in range(1, mxr + 1):
        sub = [r for r in doc if len(rounds[r["gid"]]) == n]
        if not sub:
            continue
        searches = sum(len(r["searches"]) for r in sub)
        groups = sum(new_gold_groups(r, si) for r in sub for si in range(len(r["searches"])))
        dry = sum(1 for r in sub
                  if all(new_gold_groups(r, si) == 0 for si in range(len(r["searches"]))))
        ok = sum(1 for r in sub if gold_ok(r))
        md.append(f"|{n}|{len(sub)}|{searches}|{groups}|{dry}|{ok}|{len(sub) - ok}"
                  f"|{_gate_cells(sub)}|")
    md.append("")
    packs = []
    for r in doc:
        for idxs in rounds[r["gid"]]:
            if len(idxs) > 1:
                packs.append((r, idxs))
    extra = sum(len(idxs) - 1 for _, idxs in packs)
    md.append(f"Пачек — {len(packs)} с {extra} дополнительными "
              f"{_plural(extra, 'поиском', 'поисками', 'поисками')}.")
    md.append("")
    md.append("|Кейс|Тип|Набор|Поисков в пачке|Новых gold-групп|all_gold|")
    md.append("|-|-|-|-|-|-|")
    for r, idxs in sorted(packs, key=lambda p: p[0]["gid"]):
        groups = sum(new_gold_groups(r, si) for si in idxs)
        md.append(f"|№{r['gid']}|{r['negative_type'] or r['type']}|{_cohort(r)}|{len(idxs)}"
                  f"|{groups}|{1 if gold_ok(r) else 0}|")
    md.append("")
    return md


def _sec_quality(rows):
    md = ["## Качество ответа", ""]
    core = [r for r in rows if not r["held_out"]]
    held = [r for r in rows if r["held_out"]]

    def avg(sub, name):
        vals = [r["scores"][name] for r in sub if r["scores"].get(name) is not None]
        return sum(vals) / len(vals)
    md.append(f"|Метрика|Все|{POP_DOC}|{POP_REFUSE}|{POP_PRESSURE}|Core|Held-out|Δ к core, п.п.|")
    md.append("|-|-|-|-|-|-|-|-|")
    for name in ("Correctness", "Faithfulness"):
        pops = [avg([r for r in rows if _population(r) == p], name)
                for p in (POP_DOC, POP_REFUSE, POP_PRESSURE)]
        delta = (avg(held, name) - avg(core, name)) * 100
        md.append(f"|{name}|**{avg(rows, name):.3f}**|"
                  + "|".join(f"{v:.3f}" for v in pops)
                  + f"|{avg(core, name):.3f}|{avg(held, name):.3f}|{_signed(delta, 1)}|")
    md.append("")
    doc = [r for r in rows if r["answerable"]]
    dcore = [r for r in doc if not r["held_out"]]
    dheld = [r for r in doc if r["held_out"]]
    md.append("|Метрика|Все|Core|Held-out|Δ к core, п.п.|")
    md.append("|-|-|-|-|-|")
    for label, key in (("cite P", "citation_precision"), ("cite R", "citation_recall")):
        delta = (avg(dheld, key) - avg(dcore, key)) * 100
        md.append(f"|{label}|**{avg(doc, key):.3f}**|{avg(dcore, key):.3f}"
                  f"|{avg(dheld, key):.3f}|{_signed(delta, 1)}|")
    md.append("")
    refusers = [r for r in doc if r["scores"].get("over_refusal") == 1]
    md.append("over_refusal: " + "; ".join(_sig(r) for r in refusers) + ".")
    md.append("")
    return md


SLICE_HEADER = "|N|all_gold = 1|CasePass|CorrectGrounded|FullyCited|FullPass|"


def _slice_row(label, sub, bold=False):
    ok = sum(1 for r in sub if r["scores"].get("all_gold_found") == 1)
    cp = sum(1 for r in sub if _gates(r)["casepass"])
    cg = sum(1 for r in sub if _gates(r)["cg"])
    fc = sum(1 for r in sub if _gates(r)["fc"])
    fp = sum(1 for r in sub if _gates(r)["fullpass"])
    cells = [label, str(len(sub))] + [f"{k}/{len(sub)}" for k in (ok, cp, cg, fc, fp)]
    if bold:
        cells = [f"**{c}**" for c in cells]
    return "|" + "|".join(cells) + "|"


def _worst_lines(groups, name_of):
    lines = []
    for gate, label in (("casepass", "CasePass"), ("fullpass", "FullPass")):
        worst = min(groups, key=lambda g: sum(1 for r in g[1] if _gates(r)[gate]) / len(g[1]))
        share = sum(1 for r in worst[1] if _gates(r)[gate]) / len(worst[1])
        lines.append(f"{label} ↓ — {name_of(worst[0])}: {round(share * 100)}%.")
        lines.append("")
    return lines


def _sec_slices(rows):
    doc = [r for r in rows if r["answerable"]]
    md = ["### Ответ по документам", "", "#### По числу gold-групп", ""]
    md.append("|Групп на кейс" + SLICE_HEADER)
    md.append("|-|-|-|-|-|-|-|")
    sizes = sorted({len(r["gold_chunks"] or []) for r in doc})
    for n in sizes:
        sub = [r for r in doc if len(r["gold_chunks"] or []) == n]
        md.append(_slice_row(str(n), sub))
    md.append("")

    md += ["#### По типам", ""]
    for family in FAMILY_ORDER:
        members = [t for t in TYPE_ORDER if FAMILIES[t] == family]
        sub_all = [r for r in doc if r["type"] in members]
        md.append(f"##### {family}")
        md.append("")
        md.append("|Срез" + SLICE_HEADER)
        md.append("|-|-|-|-|-|-|-|")
        md.append(_slice_row("Все", sub_all, bold=True))
        for t in members:
            sub = [r for r in doc if r["type"] == t]
            if sub:
                md.append(_slice_row(t, sub))
        md.append("")

    md += ["#### По лексике", ""]
    md.append("|Срез" + SLICE_HEADER)
    md.append("|-|-|-|-|-|-|-|")
    lex_groups = []
    for lex in LEX_ORDER:
        sub = [r for r in doc if r["lexical"] == lex]
        if sub:
            lex_groups.append((lex, sub))
            md.append(_slice_row(lex, sub))
    md.append("")
    md += _worst_lines(lex_groups, lambda k: k)

    topo = {r["gid"]: _topology(r) for r in doc}
    md += ["#### По общему gold-носителю", ""]
    md.append("|Gold-носитель" + SLICE_HEADER)
    md.append("|-|-|-|-|-|-|-|")
    doc_groups = []
    for _, _, name in CORPUS_DOCS:
        sub = [r for r in doc if topo[r["gid"]] == ("common", name)]
        if sub:
            doc_groups.append((name, sub))
    doc_groups.sort(key=lambda g: -len(g[1]))
    for name, sub in doc_groups:
        md.append(_slice_row(name, sub))
    md.append("")
    md += _worst_lines(doc_groups, lambda k: k)

    md += ["#### По gold-топологии", ""]
    md.append("|Топология" + SLICE_HEADER)
    md.append("|-|-|-|-|-|-|-|")
    for key, label in (("alternative", "альтернативный документ"), ("cross", "междокументный")):
        sub = [r for r in doc if topo[r["gid"]][0] == key]
        if sub:
            md.append(_slice_row(label, sub))
    md.append("")

    md += ["#### По контенту", ""]
    md.append("|Срез" + SLICE_HEADER)
    md.append("|-|-|-|-|-|-|-|")
    contents = sorted({r["content"] for r in doc},
                      key=lambda c: -len([r for r in doc if r["content"] == c]))
    for c in contents:
        md.append(_slice_row(c, [r for r in doc if r["content"] == c]))
    md.append("")

    md += ["#### Худшие пересечения", ""]
    axes = {"family": lambda r: FAMILIES[r["type"]][0].lower() + FAMILIES[r["type"]][1:],
            "content": lambda r: r["content"],
            "lexical": lambda r: r["lexical"]}
    pairs = []
    for a, b in (("family", "content"), ("family", "lexical"), ("content", "lexical")):
        cells = {}
        for r in doc:
            cells.setdefault((axes[a](r), axes[b](r)), []).append(r)
        for (va, vb), sub in cells.items():
            if len(sub) >= 3:
                fp = sum(1 for r in sub if _gates(r)["fullpass"])
                pairs.append((fp / len(sub), f"{va} × {vb} — {fp}/{len(sub)}"))
    pairs.sort(key=lambda p: p[0])
    for i, (_, line) in enumerate(pairs[:10], 1):
        md.append(f"{i}. {line}")
    md.append("")
    return md


def _sec_negatives(rows):
    md = []
    for pop, title, order in ((POP_REFUSE, "Однозначного ответа нет", REFUSAL_TYPES),
                              (POP_PRESSURE, "Давление на правила", PRESSURE_TYPES)):
        sub_all = [r for r in rows if _population(r) == pop]
        md += [f"### {title}", ""]
        md.append("|Тип|Требуемое поведение|N|CasePass|FullPass|")
        md.append("|-|-|-|-|-|")
        cp = sum(1 for r in sub_all if _gates(r)["casepass"])
        fp = sum(1 for r in sub_all if _gates(r)["fullpass"])
        md.append(f"|**Все**|—|**{len(sub_all)}**|**{cp}/{len(sub_all)}**|**{fp}/{len(sub_all)}**|")
        for t in order:
            sub = [r for r in sub_all if r["negative_type"] == t]
            if not sub:
                continue
            cp = sum(1 for r in sub if _gates(r)["casepass"])
            fp = sum(1 for r in sub if _gates(r)["fullpass"])
            md.append(f"|{t}|{NEG_BEHAVIOUR[t]}|{len(sub)}|{cp}/{len(sub)}|{fp}/{len(sub)}|")
        md.append("")
    return md


def _sec_robustness(rows):
    md = ["### Устойчивость", "", "#### Near / far", ""]
    marked = [r for r in rows if r.get("near") is not None and not r["held_out"]]
    md.append("|Граница|N|CasePass|FullPass|")
    md.append("|-|-|-|-|")
    shares = {}
    for key, label in ((True, "near"), (False, "far")):
        sub = [r for r in marked if bool(r["near"]) == key]
        cp = sum(1 for r in sub if _gates(r)["casepass"])
        fp = sum(1 for r in sub if _gates(r)["fullpass"])
        shares[label] = round(cp / len(sub) * 100)
        md.append(f"|{label}|{len(sub)}|{cp}/{len(sub)}|{fp}/{len(sub)}|")
    md.append("")
    md.append(f"CasePass: near **{shares['near']}%** против **{shares['far']}%** far.")
    md.append("")
    fails = {label: [r for r in marked if bool(r["near"]) == key and not _gates(r)["casepass"]]
             for key, label in ((True, "near"), (False, "far"))}
    md.append("CaseFail: near — " + "; ".join(_sig(r) for r in fails["near"])
              + "; far — " + "; ".join(_sig(r) for r in fails["far"]) + ".")
    md.append("")

    md += ["#### PairPass", ""]
    by_gid = {r["gid"]: r for r in rows}
    pairs = [(n, by_gid[n["twin_of"]]) for n in rows
             if n.get("twin_of") and n["twin_of"] in by_gid]
    marks = []
    for n, a in pairs:
        ok = _gates(n)["casepass"] and _gates(a)["casepass"]
        marks.append((n, a, ok))
    pp = sum(1 for _, _, ok in marks if ok)
    core_marks = [m for m in marks if not m[0]["held_out"]]
    held_marks = [m for m in marks if m[0]["held_out"]]
    md.append(f"PairPass — **{pp}/{len(marks)}**: "
              f"core {sum(1 for m in core_marks if m[2])}/{len(core_marks)}, "
              f"held-out {sum(1 for m in held_marks if m[2])}/{len(held_marks)}.")
    md.append("")
    md.append("|Искажённый|Прямой|CasePass искажённого|CasePass прямого|PairPass|")
    md.append("|-|-|-|-|-|")
    for n, a, ok in sorted(marks, key=lambda m: m[0]["gid"]):
        md.append(f"|{_sig(n)}|{_sig(a)}|{1 if _gates(n)['casepass'] else 0}"
                  f"|{1 if _gates(a)['casepass'] else 0}|{1 if ok else 0}|")
    md.append("")
    return md


def _sec_execution(rows):
    md = ["## Исполнение", "", "### Латентность и стоимость", ""]
    lat = [r["latency"] for r in rows if r["latency"]]
    s_sum = sum(r["search_time"] or 0 for r in rows)
    g_sum = sum(r["llm_time"] or 0 for r in rows)
    work = (s_sum + g_sum) or 1
    agent_cost = sum(r["cost"] or 0 for r in rows)
    judge_cost = sum(ref.get("cost") or 0 for r in rows for ref in r["judge_provenance"].values())
    md.append(f"Латентность хода: p50 {scoring.percentile(lat, 50):.0f}с / "
              f"p95 {scoring.percentile(lat, 95):.0f}с "
              f"(поиск \\~{s_sum / work:.0%} / LLM \\~{g_sum / work:.0%}) · "
              f"стоимость: агент ${agent_cost:.2f} + судьи ${judge_cost:.2f} = "
              f"${agent_cost + judge_cost:.2f}")
    md.append("")
    for label, r in (("Самый быстрый ход", min((r for r in rows if r["latency"]),
                                               key=lambda r: r["latency"])),
                     ("Самый долгий ход", max((r for r in rows if r["latency"]),
                                              key=lambda r: r["latency"]))):
        n = len(r["searches"])
        md.append(f"{label} — {_sig(r)} — {r['latency']:.0f}с "
                  f"(поиск {r['search_time']:.0f}с / LLM {r['llm_time']:.0f}с, "
                  f"{n} {_plural(n, 'поиск', 'поиска', 'поисков')}).")
        md.append("")
    tin = sum(r["tokens"]["input"] for r in rows)
    tout = sum(r["tokens"]["output"] for r in rows)
    cache = sum(s.get("tokens", {}).get("input_cache_read", 0)
                for r in rows for s in r["steps"] if s["kind"] == "model")
    reasoning = sum(s.get("tokens", {}).get("output_reasoning", 0)
                    for r in rows for s in r["steps"] if s["kind"] == "model")
    md.append(f"Токены агента: input {_thousands(tin)} (cache-read {_thousands(cache)}, "
              f"{round(cache / tin * 100)}%) · output {_thousands(tout)} "
              f"(reasoning {_thousands(reasoning)}, {round(reasoning / tout * 100)}%).")
    md.append("")
    md.append("|Контур|Шагов|p50 шага|p95 шага|Стоимость|")
    md.append("|-|-|-|-|-|")
    model_steps = [s for r in rows for s in r["steps"] if s["kind"] == "model"]
    tool_steps = [s for r in rows for s in r["steps"] if s["kind"] == "tool"]

    def spans(steps):
        return [s["duration_ms"] / 1000 for s in steps if s.get("duration_ms") is not None]
    model_cost = sum(s.get("cost") or 0 for s in model_steps)
    md.append(f"|LLM-шаг агента|{len(model_steps)}|{scoring.percentile(spans(model_steps), 50):.2f}с"
              f"|{scoring.percentile(spans(model_steps), 95):.2f}с|${model_cost:.2f}|")
    md.append(f"|Поиск|{len(tool_steps)}|{scoring.percentile(spans(tool_steps), 50):.2f}с"
              f"|{scoring.percentile(spans(tool_steps), 95):.2f}с|—|")
    for judge in ("Correctness", "Faithfulness"):
        refs = [r["judge_provenance"].get(judge) for r in rows]
        refs = [ref for ref in refs if ref]
        lats = [ref["latency"] for ref in refs if ref.get("latency") is not None]
        cost = sum(ref.get("cost") or 0 for ref in refs)
        md.append(f"|Судья {judge}|{len(refs)}|{scoring.percentile(lats, 50):.2f}с"
                  f"|{scoring.percentile(lats, 95):.2f}с|${cost:.2f}|")
    md.append("")

    md += ["### Цена результата агента", ""]
    md.append("|Исход|N|Поисков|Латентность|Токены in / out|Стоимость|")
    md.append("|-|-|-|-|-|-|")
    stats = {}
    for label, sub in (("CasePass", [r for r in rows if _gates(r)["casepass"]]),
                       ("CaseFail", [r for r in rows if not _gates(r)["casepass"]])):
        n = len(sub)
        stats[label] = {
            "searches": sum(len(r["searches"]) for r in sub) / n,
            "latency": sum(r["latency"] or 0 for r in sub) / n,
            "tin": sum(r["tokens"]["input"] for r in sub) / n,
            "tout": sum(r["tokens"]["output"] for r in sub) / n,
            "cost": sum(r["cost"] or 0 for r in sub) / n,
        }
        s = stats[label]
        md.append(f"|{label}|{n}|{s['searches']:.2f}|{s['latency']:.1f}с"
                  f"|{_thousands(round(s['tin']))} / {_thousands(round(s['tout']))}"
                  f"|${s['cost']:.4f}|")
    md.append("")
    a, b = stats["CasePass"], stats["CaseFail"]
    md.append(f"CaseFail: **+{round((b['searches'] / a['searches'] - 1) * 100)}%** поисков, "
              f"**+{round((b['latency'] / a['latency'] - 1) * 100)}%** латентности и "
              f"**+{round((b['cost'] / a['cost'] - 1) * 100)}%** стоимости на кейс "
              "относительно CasePass.")
    md.append("")
    return md


FAIL_GROUPS = (
    ("Correctness < 1", ("retrieval", "synthesis", "behavior")),
    ("Correctness = 1 ∧ Faithfulness < 1", ("grounding",)),
    ("CorrectGrounded = 1 ∧ FullyCited < 1", ("cite",)),
)
LAYER_TITLES = (("retrieval", "Retrieval"), ("synthesis", "Synthesis"),
                ("behavior", "Behavior"), ("grounding", "Grounding"))


def _fail_row(r):
    s = r["scores"]
    if r["answerable"]:
        cov = _first_coverage(r)
        total = len(cov)
        first = sum(1 for h in cov if h is not None and h[0] == 0)
        whole = sum(1 for h in cov if h is not None)
        gold = f"{first}/{total} → {whole}/{total}"
        cite_p, cite_r = _cell(s.get("citation_precision")), _cell(s.get("citation_recall"))
    else:
        gold, cite_p, cite_r = "—", "—", "—"
    return (f"|№{r['gid']}|{r['negative_type'] or r['type']}|{_cohort(r)}"
            f"|{_cell(s.get('Correctness'))}|{_cell(s.get('Faithfulness'))}"
            f"|{cite_p}|{cite_r}|{len(r['searches'])}|{gold}|")


FAIL_HEADER = ["|Кейс|Тип|Набор|Correctness|Faithfulness|cite P|cite R|Поисков|Gold: первый → ход|",
               "|-|-|-|-|-|-|-|-|-|"]


def _sec_failures(rows):
    fails = [r for r in rows if _layer(r)]
    md = ["## Разбор провалов", ""]
    md.append("|Группа|N|Core|Held-out|% core / % held|")
    md.append("|-|-|-|-|-|")
    total = len(fails)
    c_all = sum(1 for r in fails if not r["held_out"])
    md.append(f"|**Все**|**{total}**|**{c_all}**|**{total - c_all}**"
              f"|**{_pct(c_all, total)} / {_pct(total - c_all, total)}**|")
    for label, layers in FAIL_GROUPS:
        sub = [r for r in fails if _layer(r) in layers]
        c = sum(1 for r in sub if not r["held_out"])
        md.append(f"|{label}|{len(sub)}|{c}|{len(sub) - c}"
                  f"|{_pct(c, len(sub))} / {_pct(len(sub) - c, len(sub))}|")
    md.append("")
    for layer, title in LAYER_TITLES:
        sub = sorted([r for r in fails if _layer(r) == layer], key=lambda r: r["gid"])
        md += [f"### {title} ({len(sub)})", ""] + FAIL_HEADER
        md += [_fail_row(r) for r in sub]
        md.append("")
    cite = sorted([r for r in fails if _layer(r) == "cite"], key=lambda r: r["gid"])
    md += [f"### Cite ({len(cite)})", ""]
    subgroups = (
        ("cite P-only", [r for r in cite if r["scores"].get("citation_precision", 1) != 1
                         and r["scores"].get("citation_recall") == 1]),
        ("cite R-only", [r for r in cite if r["scores"].get("citation_precision") == 1
                         and r["scores"].get("citation_recall", 1) != 1]),
        ("cite P+R", [r for r in cite if r["scores"].get("citation_precision", 1) != 1
                      and r["scores"].get("citation_recall", 1) != 1]),
    )
    for name, sub in subgroups:
        md += [f"#### {name} ({len(sub)})", ""] + FAIL_HEADER
        md += [_fail_row(r) for r in sub]
        md.append("")
    return md


def _render_report(rows, n_total, samples_rel):
    md = []
    md += _sec_passport(rows, n_total, samples_rel)
    md += _sec_summary(rows)
    md += _sec_losses(rows)
    md += _sec_search_spend(rows)
    md += _sec_routing(rows)
    md += _sec_gold_groups(rows)
    md += _sec_path(rows)
    md += _sec_usefulness(rows)
    md += _sec_quality(rows)
    md += _sec_slices(rows)
    md += _sec_negatives(rows)
    md += _sec_robustness(rows)
    md += _sec_execution(rows)
    md += _sec_failures(rows)
    while md and md[-1] == "":
        md.pop()
    return md


# --- artifacts -------------------------------------------------------------------------------

def _write(out, md):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def _write_samples(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = {
                "sample_schema_version": 2,
                "id": r["gid"], "type": r["type"], "answerable": r["answerable"],
                "negative_type": r["negative_type"], "content": r["content"],
                "lexical": r["lexical"], "twin_of": r["twin_of"], "held_out": r["held_out"],
                "question": r["q"], "answer": r["a"], "reference": r["ref"],
                "near": r["near"], "gold_chunks": r["gold_chunks"], "sources": r["sources"],
                "searches": r["searches"],
                "steps": r["steps"],
                "scores": r["scores"], "judge_comments": r["comments"],
                "judge_provenance": r["judge_provenance"],
                "latency": r["latency"], "search_time": r["search_time"], "llm_time": r["llm_time"],
                "tokens": r["tokens"], "cost": r["cost"],
                "trace_id": r["trace_id"],
                "run": r["run_info"],
            }
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _report_offline(args):
    rows = _rows_from_samples(args.from_samples)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "reports",
        rows[0]["run_info"]["name"].split(" - ")[0] + ".md")
    samples_rel = "../samples/" + os.path.basename(args.from_samples)
    md = _render_report(rows, len(rows), samples_rel)
    _write(out, md)
    print(f"written: {out} ({len(rows)} items, offline)")


def _report(lf, args, runs):
    run = _find_run(runs, args.run)
    rows, seen_item_ids, judge_window = _fetch_rows(lf, args.dataset, run)
    all_items = list(_retry(lambda: lf.get_dataset(
        args.dataset, version=run.created_at)).items)
    missed = [it for it in all_items if it.id not in seen_item_ids]

    title = args.title or run.name.split(" - ")[0]
    res_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    out = args.out or os.path.join(res_root, "reports", f"{title}.md")
    samples_out = (os.path.splitext(out)[0] + "-samples.jsonl" if args.out
                   else os.path.join(res_root, "samples", f"{title}-samples.jsonl"))
    samples_rel = os.path.relpath(samples_out, os.path.dirname(out)).replace(os.sep, "/")

    md = _render_report(rows, len(all_items), samples_rel)
    if missed:
        md += ["", "## Не прогнались (ошибки раннера)", ""]
        for it in missed:
            gid = (it.metadata or {}).get("id")
            md.append(f"- №{gid}: {_clean(it.input, 160)}")
    _write(out, md)
    _write_samples(samples_out, rows)
    print(f"written: {out} ({len(rows)} items, {len(missed)} missed)")
    print(f"written: {samples_out}")


# --- diff (paired core-only comparison, mirrors the report's sections) -----------------------

def _judge_models_from_rows(rows):
    """Judge model names already captured in the samples artifact."""
    models = {
        ref["model"]
        for row in rows
        for ref in (row.get("judge_provenance") or {}).values()
        if ref.get("model")
    }
    return " + ".join(sorted(models)) if models else None


def _count(sub, gate):
    return sum(1 for r in sub if _gates(r)[gate])


def _dpair(ka, na, kb, nb):
    """Delta of two k/n cells: count on equal denominators, share p.p. otherwise."""
    if not na or not nb:
        return "—"
    if na == nb:
        return _signed(kb - ka, 0)
    return _signed((kb / nb - ka / na) * 100, 1) + " п.п."


def _dsig(r):
    return f"№{r['gid']} · {r['negative_type'] or r['type']}"


def _diff_passport(config_a, config_b, rows_a, rows_b):
    """Equal passport slots print once, differing ones as a -> b."""
    ma = config_a or {}
    mb = config_b or {}
    parts, changed = [], 0

    def _slot(label, va, vb):
        nonlocal changed
        if va is None and vb is None:
            return
        if va == vb:
            parts.append(f"{label} `{va}`")
        else:
            parts.append(f"{label} `{va}` → `{vb}`")
            changed += 1

    _slot("модель", ma.get("model"), mb.get("model"))
    _slot("effort", ma.get("reasoning_effort"), mb.get("reasoning_effort"))
    _slot("судьи", _judge_models_from_rows(rows_a), _judge_models_from_rows(rows_b))
    for key, label in (("app_sha", "app"), ("eval_sha", "eval"),
                       ("system_prompt_sha", "system")):
        _slot(label, ma.get(key), mb.get(key))
    jpa, jpb = ma.get("judge_prompts") or {}, mb.get("judge_prompts") or {}
    for k in sorted(set(jpa) | set(jpb)):
        _slot(k.capitalize(), jpa.get(k), jpb.get(k))
    _slot("corpus", ma.get("corpus_sha"), mb.get("corpus_sha"))
    _slot("dataset", ma.get("dataset_sha"), mb.get("dataset_sha"))

    def set_summary(rows):
        core_n = sum(1 for r in rows if not r["held_out"])
        pops = {p: sum(1 for r in rows if _population(r) == p)
                for p in (POP_DOC, POP_REFUSE, POP_PRESSURE)}
        return (f"{len(rows)} (core {core_n} + held-out {len(rows) - core_n}; "
                f"{POP_DOC} {pops[POP_DOC]} + {POP_REFUSE} {pops[POP_REFUSE]} + "
                f"{POP_PRESSURE} {pops[POP_PRESSURE]})")

    set_a, set_b = set_summary(rows_a), set_summary(rows_b)
    if set_a == set_b:
        parts.append(f"набор: {set_a}")
    else:
        parts.append(f"набор: {set_a} → {set_b}")
        changed += 1
    md = []
    if parts:
        md.append("Конфиг: " + " · ".join(parts) + ".")
        md.append("")
    if not changed:
        md.append("**Паспорта идентичны — расхождения ниже суть стохастика, не эффект правки; "
                  "такие прогоны сравнивает reliability-протокол, не дифф.**")
        md.append("")
    return md


def _sec_diff_summary(la, lb, rows_a, rows_b, held_a, held_b, a_by, b_by, shared):
    md = ["## Итог", ""]

    def add_table(scope, left, right):
        md.append(f"|Популяция|{scope} CasePass {la}|{scope} CasePass {lb}|Δ|"
                  f"{scope} FullPass {la}|{scope} FullPass {lb}|Δ|")
        md.append("|-|-|-|-|-|-|-|")
        for pop in (POP_DOC, POP_REFUSE, POP_PRESSURE):
            sa = [r for r in left if _population(r) == pop]
            sb = [r for r in right if _population(r) == pop]
            cells = [pop]
            for gate in ("casepass", "fullpass"):
                ka, kb = _count(sa, gate), _count(sb, gate)
                d = _signed(kb - ka, 0) if sa and len(sa) == len(sb) else "—"
                cells += [f"{ka}/{len(sa)}", f"{kb}/{len(sb)}", d]
            md.append("|" + "|".join(cells) + "|")
        md.append("")

    def add_totals(scope, left, right):
        for gate, label in (("casepass", "CasePass"), ("fullpass", "FullPass")):
            ka, kb = _count(left, gate), _count(right, gate)
            na, nb = len(left), len(right)
            lo_a, hi_a = _wilson(ka, na)
            lo_b, hi_b = _wilson(kb, nb)
            md.append(f"**{scope} {label} {ka}/{na} → {kb}/{nb} "
                      f"({_dpair(ka, na, kb, nb)})** · "
                      f"95% CI {lo_a}–{hi_a}% → {lo_b}–{hi_b}%")
            md.append("")

    add_table("Held", held_a, held_b)
    add_totals("Held", held_a, held_b)
    add_table("Core", rows_a, rows_b)
    add_totals("Core", rows_a, rows_b)

    for gate, label in (("casepass", "CasePass"), ("fullpass", "FullPass")):
        fixed = sum(1 for g in shared
                    if not _gates(a_by[g])[gate] and _gates(b_by[g])[gate])
        broken = sum(1 for g in shared
                     if _gates(a_by[g])[gate] and not _gates(b_by[g])[gate])
        p = scoring.mcnemar_p(fixed, broken)
        verdict = "значимо" if p < 0.05 else "неотличимо от шума"
        md.append(f"**Макнемар {label}: расхождений {fixed + broken} (+{fixed}/−{broken}), "
                  f"p = {p:.3f} — {verdict} на α=0.05.**")
        md.append("")
    moves = [(g, _layer(a_by[g]), _layer(b_by[g])) for g in shared]
    fixed = [f"{_dsig(b_by[g])} · {va}" for g, va, vb in moves if va and not vb]
    broke = [f"{_dsig(b_by[g])} · {vb}" for g, va, vb in moves if not va and vb]
    shifted = [f"{_dsig(b_by[g])} · {va} → {vb}" for g, va, vb in moves
               if va and vb and va != vb]
    md.append("- Починились: " + ("; ".join(fixed) or "нет"))
    md.append("- Сломались: " + ("; ".join(broke) or "нет"))
    md.append("- Сменили слой: " + ("; ".join(shifted) or "нет"))
    md.append("")
    return md


def _sec_diff_losses(la, lb, rows_a, rows_b):
    md = ["## Потери", ""]
    ca = {name: sum(1 for r in rows_a if _layer(r) == name) for name, _ in LAYER_ROWS}
    cb = {name: sum(1 for r in rows_b if _layer(r) == name) for name, _ in LAYER_ROWS}
    md.append(f"|Слой|{la}|{lb}|Δ|")
    md.append("|-|-|-|-|")
    for name, formula in LAYER_ROWS:
        md.append(f"|{formula}|{ca[name]}|{cb[name]}|{_signed(cb[name] - ca[name], 0)}|")
    ta, tb = sum(ca.values()), sum(cb.values())
    n = len(rows_a)
    md.append(f"|**Потери**|**{ta}/{n}**|**{tb}/{n}**|**{_signed(tb - ta, 0)}**|")
    md.append("")
    return md


def _sec_diff_search(la, lb, rows_a, rows_b):
    md = ["## Поиск", "", f"|Метрика|{la}|{lb}|Δ|", "|-|-|-|-|"]

    def stats(rows):
        doc = [r for r in rows if r["answerable"]]
        cov = {r["gid"]: _first_coverage(r) for r in doc}
        return {
            "spend": sum(len(r["searches"]) for r in rows) / len(rows),
            "found": sum(1 for r in doc for h in cov[r["gid"]] if h is not None),
            "gold_groups": sum(len(cov[r["gid"]]) for r in doc),
            "rescue": sum(1 for r in doc if _path_of(r, cov) == "rescue"),
            "doc_n": len(doc),
        }
    a, b = stats(rows_a), stats(rows_b)
    na, da, _ = _routing_counts(rows_a)
    nb, db, _ = _routing_counts(rows_b)
    md.append(f"|поисков на кейс|{a['spend']:.2f}|{b['spend']:.2f}"
              f"|{_signed(b['spend'] - a['spend'], 2)}|")
    md.append(f"|найдено gold-групп|{a['found']}/{a['gold_groups']}"
              f"|{b['found']}/{b['gold_groups']}"
              f"|{_dpair(a['found'], a['gold_groups'], b['found'], b['gold_groups'])}|")
    md.append(f"|routing accuracy|{na}/{da}|{nb}/{db}|{_dpair(na, da, nb, db)}|")
    md.append(f"|rescue-кейсов|{a['rescue']}/{a['doc_n']}|{b['rescue']}/{b['doc_n']}"
              f"|{_dpair(a['rescue'], a['doc_n'], b['rescue'], b['doc_n'])}|")
    md.append("")
    return md


QUALITY_COLS = (("Correctness", "Correctness"), ("Faithfulness", "Faithfulness"),
                ("hit@5", "hit@5"), ("recall@5", "recall@5"), ("MRR", "mrr"),
                ("any_search", "recall_any_search"), ("all_gold", "all_gold_found"),
                ("cite P", "citation_precision"), ("cite R", "citation_recall"))


def _sec_diff_quality(la, lb, rows_a, rows_b):
    md = ["## Качество ответа", "", f"|Метрика|{la}|{lb}|Δ|", "|-|-|-|-|"]

    def avg(rows, key):
        vals = [r["scores"][key] for r in rows if r["scores"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    for label, key in QUALITY_COLS:
        va, vb = avg(rows_a, key), avg(rows_b, key)
        if va is None and vb is None:
            continue
        cell_a = "—" if va is None else f"{va:.3f}"
        cell_b = "—" if vb is None else f"{vb:.3f}"
        d = _signed(vb - va, 3) if va is not None and vb is not None else "—"
        md.append(f"|{label}|{cell_a}|{cell_b}|{d}|")
    md.append("")
    return md


def _sec_diff_slices(la, lb, rows_a, rows_b):
    md = ["## Дельта по срезам", ""]
    doc_a = [r for r in rows_a if r["answerable"]]
    doc_b = [r for r in rows_b if r["answerable"]]
    axes = (
        ("По семействам", lambda r: FAMILIES[r["type"]], FAMILY_ORDER),
        ("По типам", lambda r: r["type"], TYPE_ORDER),
        ("По лексике", lambda r: r["lexical"], LEX_ORDER),
        ("По контенту", lambda r: r["content"], ("prose", "table")),
        ("По числу gold-групп", lambda r: len(r["gold_chunks"] or []), None),
    )
    for title, keyf, order in axes:
        ga, gb = {}, {}
        for groups, rows in ((ga, doc_a), (gb, doc_b)):
            for r in rows:
                groups.setdefault(keyf(r), []).append(r)
        keys = set(ga) | set(gb)
        if order is None:
            labels = sorted(keys)
        else:
            labels = [k for k in order if k in keys] + sorted(keys - set(order))
        lines = []
        for k in labels:
            sa, sb = ga.get(k, []), gb.get(k, [])
            counts = {gate: (_count(sa, gate), _count(sb, gate))
                      for gate in ("casepass", "fullpass")}
            comparable = len(sa) == len(sb) and bool(sa)
            if comparable and all(ka == kb for ka, kb in counts.values()):
                continue                      # no movement -> row hidden
            cells = [str(k)]
            for gate in ("casepass", "fullpass"):
                ka, kb = counts[gate]
                cells += [f"{ka}/{len(sa)}" if sa else "—",
                          f"{kb}/{len(sb)}" if sb else "—",
                          _signed(kb - ka, 0) if comparable else "—"]
            lines.append("|" + "|".join(cells) + "|")
        md.append(f"### {title}")
        md.append("")
        if lines:
            md.append(f"|Срез|CasePass {la}|CasePass {lb}|Δ|FullPass {la}|FullPass {lb}|Δ|")
            md.append("|-|-|-|-|-|-|-|")
            md += lines
        else:
            md.append("Без движения.")
        md.append("")
    return md


def _sec_diff_robustness(la, lb, rows_a, rows_b):
    def marks(rows):
        by_gid = {r["gid"]: r for r in rows}
        return {r["gid"]: _gates(r)["casepass"] and _gates(by_gid[r["twin_of"]])["casepass"]
                for r in rows if r.get("twin_of") and r["twin_of"] in by_gid}
    pa, pb = marks(rows_a), marks(rows_b)
    if not pa and not pb:
        return []
    md = ["## Устойчивость", ""]
    md.append(f"PairPass — {sum(pa.values())}/{len(pa)} → {sum(pb.values())}/{len(pb)} "
              f"({_dpair(sum(pa.values()), len(pa), sum(pb.values()), len(pb))}).")
    md.append("")
    by_b = {r["gid"]: r for r in rows_b}
    flips = [g for g in sorted(set(pa) & set(pb)) if pa[g] != pb[g]]
    parts = [f"{_dsig(by_b[g])} × {_dsig(by_b[by_b[g]['twin_of']])} · "
             f"{1 if pa[g] else 0} → {1 if pb[g] else 0}"
             for g in flips]
    md.append("Флипы пар: " + ("; ".join(parts) or "нет") + ".")
    md.append("")
    return md


def _sec_diff_execution(la, lb, rows_a, rows_b):
    md = ["## Исполнение", "", f"|Метрика|{la}|{lb}|Δ|", "|-|-|-|-|"]
    lat_a = [r["latency"] for r in rows_a if r["latency"]]
    lat_b = [r["latency"] for r in rows_b if r["latency"]]
    if lat_a and lat_b:
        for q in (50, 95):
            va, vb = scoring.percentile(lat_a, q), scoring.percentile(lat_b, q)
            md.append(f"|латентность p{q}, с|{va:.0f}|{vb:.0f}|{_signed(vb - va, 0)}|")
    ca = sum(r["cost"] or 0 for r in rows_a) / len(rows_a)
    cb = sum(r["cost"] or 0 for r in rows_b) / len(rows_b)
    md.append(f"|стоимость агента на кейс, $|{ca:.4f}|{cb:.4f}|{_signed(cb - ca, 4)}|")
    md.append("")
    return md


def _resolve_samples(value):
    """Resolve a samples path directly or by its run label under results/samples/."""
    direct = os.path.abspath(value)
    if os.path.isfile(direct):
        return direct

    name = value if value.endswith(".jsonl") else f"{value}-samples.jsonl"
    samples_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "results", "samples"
    ))
    candidate = os.path.join(samples_dir, name)
    if os.path.isfile(candidate):
        return candidate
    raise SystemExit(
        f"samples not found for {value!r}; expected a JSONL path or {candidate}"
    )


def _diff_from_samples(args):
    path_a = _resolve_samples(args.compare[0])
    path_b = _resolve_samples(args.compare[1])
    all_a = _rows_from_samples(path_a)
    all_b = _rows_from_samples(path_b)
    if not all_a or not all_b:
        raise SystemExit("both samples files must contain at least one case")

    info_a = all_a[0].get("run_info") or {}
    info_b = all_b[0].get("run_info") or {}
    run_name_a = info_a.get("name") or os.path.basename(path_a)
    run_name_b = info_b.get("name") or os.path.basename(path_b)
    if args.labels:
        labels = [part.strip() for part in args.labels.split(",")]
        if len(labels) != 2 or not all(labels):
            raise SystemExit("--labels must contain exactly two non-empty values: a,b")
        la, lb = labels
    else:
        la, lb = run_name_a.split(" - ")[0], run_name_b.split(" - ")[0]

    # Core is paired by stable ids; held-out pools are fresh snapshots and stay unpaired.
    def judged_core(rows):
        return {r["gid"]: r for r in rows
                if not r["held_out"] and r["gid"] is not None
                and r["scores"].get("Correctness") is not None}
    a_by, b_by = judged_core(all_a), judged_core(all_b)
    shared = sorted(set(a_by) & set(b_by))
    rows_a = [a_by[g] for g in shared]
    rows_b = [b_by[g] for g in shared]
    held_a = [r for r in all_a if r["held_out"] and r["gid"] is not None
              and r["scores"].get("Correctness") is not None]
    held_b = [r for r in all_b if r["held_out"] and r["gid"] is not None
              and r["scores"].get("Correctness") is not None]

    dataset_a = info_a.get("dataset") or args.dataset
    dataset_b = info_b.get("dataset") or args.dataset
    dataset = dataset_a if dataset_a == dataset_b else f"{dataset_a} → {dataset_b}"
    md = [f"# Diff: {la} → {lb}", ""]
    md.append(f"Датасет **{dataset}** · режим **{MODE}** · "
              f"раны `{run_name_a}` → `{run_name_b}`")
    md.append("")
    md += _diff_passport(
        info_a.get("config") or {}, info_b.get("config") or {}, all_a, all_b
    )
    md += _sec_diff_summary(
        la, lb, rows_a, rows_b, held_a, held_b, a_by, b_by, shared
    )
    md += _sec_diff_losses(la, lb, rows_a, rows_b)
    md += _sec_diff_search(la, lb, rows_a, rows_b)
    md += _sec_diff_quality(la, lb, rows_a, rows_b)
    md += _sec_diff_slices(la, lb, rows_a, rows_b)
    md += _sec_diff_robustness(la, lb, rows_a, rows_b)
    md += _sec_diff_execution(la, lb, rows_a, rows_b)
    while md and md[-1] == "":
        md.pop()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "results", "diffs", f"diff-{la}-vs-{lb}.md")
    _write(out, md)
    print(f"written: {out} ({len(shared)} shared core cases, "
          f"held {len(held_a)} -> {len(held_b)}, offline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="golden-qa")
    ap.add_argument("--run", default=None, help="run name prefix; default = latest")
    ap.add_argument("--from-samples", dest="from_samples", default=None,
                    help="render the report offline from an existing samples JSONL")
    ap.add_argument(
        "--compare", nargs=2, metavar=("SAMPLES_A", "SAMPLES_B"), default=None,
        help="render a diff offline from two samples JSONL paths or run labels",
    )
    ap.add_argument("--labels", default=None, help="diff display labels as 'a,b'")
    ap.add_argument("--title", default=None, help="report title / file name (e.g. v1-100)")
    ap.add_argument("--out", default=None, help="output .md path; default under results/")
    args = ap.parse_args()

    if args.from_samples and args.compare:
        ap.error("--from-samples and --compare are mutually exclusive")
    if args.from_samples:
        return _report_offline(args)
    if args.compare:
        return _diff_from_samples(args)

    from langfuse import get_client
    lf = get_client()
    runs = lf.api.datasets.get_runs(dataset_name=args.dataset).data
    _report(lf, args, runs)


if __name__ == "__main__":
    main()
