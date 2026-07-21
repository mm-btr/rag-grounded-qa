"""LangGraph agent: tenant-scoped `search` tool + cloud LLM, grounded & abstaining.
tenant_id reaches the tool via ToolRuntime.context — runtime-injected, hidden from the LLM."""
import asyncio
import os
import re
from dataclasses import dataclass, replace
from retrieve import search as _search, resolve_doc, _tenant_docs
from config import LLM_MODEL, LLM_REASONING_EFFORT, TOP_K_RERANK, MEMORY_WINDOW, SEARCH_BUDGET
from keys import CITE_RE


@dataclass
class Ctx:
    tenant_id: str


def _format_context(hits):
    # One machine tag 【doc_id#chunk】 per passage (a second bracket tag made the model copy the
    # wrong one). The section heading is load-bearing for table chunks: the measurement unit
    # often lives only there.
    lines = []
    for h in hits:
        tags = []
        if h.get("label"):
            tags.append(f"метка: {h['label']}")
        if h.get("source"):
            tags.append(f"док: {h['source']}")
        if h.get("section"):
            tags.append(f"раздел: {h['section']}")
        head = f"【{h['doc_id']}#{h['chunk']}】"
        if tags:
            head += " (" + " · ".join(tags) + ")"
        lines.append(f"{head}\n{h['text']}")
    return "\n\n".join(lines)


GROUNDING_REFUSAL = (
    "Не могу ответить с опорой на источники: в найденных фрагментах нет ответа."
)
CITE_RETRY = (
    "Твой ответ без ссылок на источник. Перепиши его, поставив в конце каждого утверждения (или "
    "группы предложений из одного пассажа — один тег на группу) точный тег источника 【id#n】 — "
    "ровно как он напечатан перед нужным фрагментом в результатах поиска (например "
    "【9b0679d790#105】). Если ответа в найденных фрагментах нет — начни ответ с «Не могу ответить» "
    "и коротко поясни, чего не хватает."
)
# CONDITIONAL wording: a healthy multi-hop turn legally searches broadly and then answers —
# the hint must not talk it out of a correct answer.
DOC_HINT = (
    "Если в найденных пассажах уже есть ядро ответа — отвечай сейчас. Если нет — не повторяй "
    "поиск вслепую: повтори его с параметром doc по каждому подходящему документу корпуса "
    "(передавай точное имя файла), и только после этого отвечай или отказывайся."
)


def _doc_line(source, info):
    if info.get("descr"):
        return f"{source} — {info['descr']}"
    if info.get("label"):
        return f"{source} (метка: {info['label']})"
    return source
# An UNCITED reply passes the gate ONLY when it opens with this exact prefix — deterministic
# startswith, no fuzzy matching. Accepted residual: a fact smuggled after the prefix passes
# uncited; eval faithfulness surfaces it.
ABSTAIN_PREFIX = "Не могу ответить"
CLARIFY_PREFIX = "Уточните"


def _strip_wrapping(content):
    # The model legally wraps the opener in HTML/markdown; still a deterministic startswith.
    text = re.sub(r"<[^>]+>", "", str(content)).strip()
    return re.sub(r"^[\s>#*_•+-]+", "", text)


def _is_abstention(content):
    return _strip_wrapping(content).startswith(ABSTAIN_PREFIX)


def _is_clarification(content):
    return _strip_wrapping(content).startswith(CLARIFY_PREFIX)


def content_text(content):
    # Responses API returns a LIST of typed blocks (reasoning + text).
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def _citations(text):
    return set(CITE_RE.findall(text if isinstance(text, str) else str(text)))


def _allowed_citations(messages):
    """Citations of the CURRENT turn, read ONLY from ToolMessage.artifact. The tool TEXT is
    untrusted retrieved content: a citation-shaped string inside a document must not
    authorize a citation."""
    allowed = set()
    for m in reversed(messages):
        t = getattr(m, "type", None)
        if t == "human":
            break
        if t == "tool":
            for r in (getattr(m, "artifact", None) or ()):
                if isinstance(r, (list, tuple)) and len(r) == 2:
                    allowed.add((str(r[0]), str(r[1])))
    return allowed


def _citation_retry_request(request, messages):
    return request.override(messages=messages, tool_choice="none")


def _ungrounded(content, allowed):
    cited = _citations(content)
    text = content if isinstance(content, str) else str(content)
    if cited - allowed:
        return True
    return (not cited and bool(text.strip())
            and not _is_abstention(text) and not _is_clarification(text))


def _turn_started(messages):
    for m in reversed(messages):
        t = getattr(m, "type", None)
        if t == "human":
            return True
        if t == "ai":
            return False
    return False


def _turn_search_count(messages):
    n = 0
    for m in reversed(messages):
        t = getattr(m, "type", None)
        if t == "human":
            break
        if t == "tool":
            n += 1
    return n


def _not_iterating_docs(messages):
    # >=2 searches this turn with <2 distinct `doc` values = blind reformulations.
    n_search, docs = 0, set()
    for m in reversed(messages):
        t = getattr(m, "type", None)
        if t == "human":
            break
        if t == "ai":
            for tc in (getattr(m, "tool_calls", None) or []):
                if tc.get("name") == "search":
                    n_search += 1
                    d = (tc.get("args") or {}).get("doc")
                    if d:
                        docs.add(str(d).strip().lower())
    return n_search >= 2 and len(docs) < 2


def _strip_citations(content):
    # A tag is only valid inside its own turn (the gate allowlists current-turn retrieval).
    return CITE_RE.sub("", content_text(content)).strip()


def _compact_turn(turn):
    # Returns the original objects when already compact -> identity no-op for the caller.
    finals = [m for m in turn
              if getattr(m, "type", None) == "ai" and not getattr(m, "tool_calls", None)]
    if not finals:
        return turn[:1]                       # answerless turn -> keep the question
    last = finals[-1]
    text = _strip_citations(last.content)
    if len(turn) == 2 and text == last.content:
        return turn
    return [turn[0], last.model_copy(update={"content": text})]


def _normalize_messages(msgs, window):
    """(1) no unanswered tool call survives — an open pair 400s the whole thread; (2) every
    completed turn collapses to question + final answer, tags stripped (a remembered tag
    fails the gate); the current turn passes untouched; (3) window, starting on a human."""
    humans = [i for i, m in enumerate(msgs) if getattr(m, "type", None) == "human"]
    if humans:
        bounds = humans + [len(msgs)]
        kept = msgs[:humans[0]]
        for a, b in zip(bounds, bounds[1:]):
            turn = msgs[a:b]
            answered = {
                m.tool_call_id for m in turn
                if getattr(m, "type", None) == "tool" and getattr(m, "tool_call_id", None)
            }
            open_call = any(
                getattr(m, "type", None) == "ai" and getattr(m, "tool_calls", None)
                and any(tc["id"] not in answered for tc in m.tool_calls)
                for m in turn
            )
            if open_call:
                continue
            if b != len(msgs):
                turn = _compact_turn(turn)
            kept = kept + turn
        msgs = kept
    tail = msgs[-window:] if len(msgs) > window else msgs
    while tail and getattr(tail[0], "type", None) != "human":
        tail = tail[1:]
    return tail


_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name):
    with open(os.path.join(_PROMPTS_DIR, name), encoding="utf-8") as f:
        return f.read().strip()


SYSTEM = _load_prompt("system.md")


def build_agent(checkpointer=None, store=None):
    from langchain.tools import tool, ToolRuntime
    from langchain.chat_models import init_chat_model
    from langchain.agents import create_agent
    from langchain.agents.middleware import before_model, wrap_model_call
    from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    @tool(response_format="content_and_artifact")
    def search(query: str, runtime: ToolRuntime[Ctx], doc: str | None = None) -> tuple[str, list]:
        """Search the document corpus; returns relevant passages with citations.

        doc: optional narrowing filter — a document label or a part of a file name as shown
        in passage headers (e.g. "charter", "otchet", "IFRS"). Use it when the question is
        tied to a specific document, when the user says where to look, or when a previous
        search returned passages from the wrong documents. Omit for a broad search."""
        tenant = runtime.context.tenant_id
        sources = note = None
        if doc:
            sources, universe = resolve_doc(tenant, doc)
            if not sources:
                # Unresolved hint -> widen instead of failing, and show the corpus map.
                names = "; ".join(_doc_line(s, info) for s, info in sorted(universe.items()))
                note = (f'Фильтр doc="{doc}" не совпал ни с одним документом — поиск выполнен '
                        f"по всем. Документы корпуса: {names}.")
        hits = _search(query, tenant_id=tenant, top_k=TOP_K_RERANK, sources=sources)
        refs = [(h["doc_id"], str(h["chunk"])) for h in hits]   # trusted allowlist channel
        out = _format_context(hits) if hits else "No relevant passages found."
        return (f"{note}\n\n{out}" if note else out, refs)

    @before_model
    def normalize_history(state, runtime):
        # Full replacement so the cleaned log is persisted — a broken turn can't poison the
        # thread. Identity check, not length: compaction can swap objects without changing count.
        msgs = state["messages"]
        tail = _normalize_messages(msgs, MEMORY_WINDOW)
        if len(tail) == len(msgs) and all(a is b for a, b in zip(tail, msgs)):
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *tail]}

    @wrap_model_call
    async def grounding(request, handler):
        """1) force `search` on the first model step; 2) past SEARCH_BUDGET ->
        tool_choice="none"; 3) fabricated citation -> hard refusal; 4) uncited substantive
        answer with sources -> ONE retry, then refusal."""
        # Snapshot BEFORE the hint injections below: _allowed_citations scans back to the last
        # human message, so computed post-injection it would return empty and flip a correctly
        # cited answer into a false "fabrication" refusal.
        allowed = _allowed_citations(request.messages)
        if _turn_started(request.messages):
            request = request.override(tool_choice="search")
        elif _turn_search_count(request.messages) >= SEARCH_BUDGET:
            request = request.override(tool_choice="none")
        elif _not_iterating_docs(request.messages):
            # Attach the FULL corpus map: passage headers only show already-retrieved
            # documents, so the model can't iterate what it has never seen. Request-only.
            hint = DOC_HINT
            ctx = getattr(getattr(request, "runtime", None), "context", None)
            if ctx is not None:
                docs = _tenant_docs(ctx.tenant_id)
                if docs:
                    hint += "\nДокументы корпуса:\n" + "\n".join(
                        "- " + _doc_line(s, info) for s, info in sorted(docs.items()))
            request = request.override(messages=[*request.messages,
                                                 HumanMessage(content=hint)])
        response = await handler(request)
        last = response.result[-1]
        if getattr(last, "type", None) != "ai" or getattr(last, "tool_calls", None):
            return response
        text = content_text(last.content)
        if _citations(text) - allowed:
            refusal = AIMessage(content=GROUNDING_REFUSAL, id=getattr(last, "id", None))
            return replace(response, result=[*response.result[:-1], refusal])
        if _ungrounded(text, allowed):
            if allowed:                           # sources WERE retrieved -> one retry for the tags
                retry = _citation_retry_request(
                    request,
                    [*request.messages, last, HumanMessage(content=CITE_RETRY)],
                )
                response = await handler(retry)
                last = response.result[-1]
                if not _ungrounded(content_text(last.content), allowed):
                    return response
            # An answer from zero retrieval can't pass uncited.
            refusal = AIMessage(content=GROUNDING_REFUSAL, id=getattr(last, "id", None))
            return replace(response, result=[*response.result[:-1], refusal])
        return response

    return create_agent(
        # Responses API is REQUIRED: chat/completions 400s on reasoning_effort + function tools.
        init_chat_model(LLM_MODEL, reasoning_effort=LLM_REASONING_EFFORT, use_responses_api=True),
        tools=[search],
        system_prompt=SYSTEM,
        context_schema=Ctx,
        checkpointer=checkpointer,
        store=store,
        middleware=[normalize_history, grounding],
    )


def ask(question, tenant_id=None):
    from config import DEFAULT_TENANT
    agent = build_agent()
    out = asyncio.run(agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        context=Ctx(tenant_id=tenant_id or DEFAULT_TENANT),
    ))
    return content_text(out["messages"][-1].content)


if __name__ == "__main__":
    import sys
    print(ask(sys.argv[1] if len(sys.argv) > 1 else "What was revenue in 2023?"))
