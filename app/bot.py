"""Telegram bot (long-polling): thin channel over the RAG agent."""
import asyncio
import io
import logging
import os
import re
import sys
import tempfile
import time
from itertools import groupby

import telegramify_markdown as tgmd
from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, MessageEntity
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent import Ctx, build_agent, content_text
from models import warmup
from retrieve import fetch_locators
from ingest import ingest_file, _delete_source, set_label, get_labels, get_client
from stt import transcribe, STTError
from db import (get_role, register_document, set_document_status, list_documents,
                delete_document, document_status, prune_seen_updates, reset_stuck_documents)
from sanitize import sanitize_text
from keys import thread_id_for, CITE_RE
from middlewares import DedupMiddleware, TenantMiddleware
from config import (POSTGRES_URL, AGENT_RECURSION_LIMIT, AGENT_TIMEOUT, TELEGRAM_LIMIT,
                    MAX_UPLOAD_BYTES)

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("bot")

# Per-chat busy set. check+add has no await between them -> atomic in the event loop.
# In-memory, single-process — all long-polling allows anyway (a 2nd copy hits Telegram 409).
_busy = set()

BUSY_REPLY = "⏳ Отвечаю на прошлый вопрос — дождись ответа."
CLEAR_BUSY_REPLY = "⏳ Отвечаю на прошлый вопрос — дождись ответа, потом /clear."


def _reserve(tid):
    if tid in _busy:
        return False
    _busy.add(tid)
    return True


def _tenant_busy(tenant_id):
    return any(t.startswith(tenant_id + ":") for t in _busy)


async def _drain_tenant(tenant_id, timeout, poll=0.5):
    deadline = time.monotonic() + timeout
    while _tenant_busy(tenant_id):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(poll)
    return True


# Docling converter/chunker are module-global singletons and not thread-safe.
_ingest_lock = asyncio.Lock()

# Per-tenant ingest gate: while a tenant's document is processed, its questions, /label, /rm
# and further uploads are rejected up front, and the rebuild starts only after the tenant's
# in-flight answers finish (_drain_tenant). Other tenants are unaffected.
_ingesting = set()

WELCOME = (
    "Здравствуйте. Отвечаю на вопросы по загруженным документам — напишите вопрос или пришлите голосовое. "
    "Отвечаю только по тому, что нашёл в базе, со ссылками на источник.\n"
    "Пришлите файл (PDF/docx/txt) — добавлю в корпус; в подписи к файлу можно указать метку.\n"
    "/docs — список документов, /clear — очистить контекст диалога."
)


async def _clear_thread(message, saver, tid):
    # Holding the same claim as a normal turn: no checkpoint can be written mid-deletion.
    if not _reserve(tid):
        await message.answer(CLEAR_BUSY_REPLY)
        return
    try:
        try:
            await saver.adelete_thread(tid)
        except Exception as e:                 # noqa: BLE001
            log.warning("clear failed for %s: %s", tid, e)
            await message.answer("Контекст не очищен. Попробуйте /clear ещё раз.")
            return
        await message.answer("Контекст очищен.")
    finally:
        _busy.discard(tid)


# Rendering: plain text + entity spans, the bot NEVER sends parse_mode — nothing is parsed
# server-side, so broken markup can't get a message rejected. Formatting is expendable, text never is.

_ENTITY_FIELDS = {"type", "offset", "length", "url", "language", "custom_emoji_id"}


def _tg_entities(ents):
    out = [MessageEntity(**{k: v for k, v in e.to_dict().items()
                            if k in _ENTITY_FIELDS and v is not None}) for e in ents]
    return out or None


async def _send(message, text, entities=None):
    # NEVER raises: markup-rejected chunk is resent bare, flood control retries the same
    # chunk once, anything else is logged and the send stops.
    chunks = tgmd.split_entities(text or "…", list(entities or []), TELEGRAM_LIMIT)
    if not chunks:
        chunks = [("…", [])]
    for chunk, ents in chunks:
        try:
            await message.answer(chunk, entities=_tg_entities(ents))
        except TelegramRetryAfter as e:
            try:
                await asyncio.sleep(e.retry_after)
                await message.answer(chunk, entities=_tg_entities(ents))
            except Exception:            # noqa: BLE001
                log.exception("send failed after retry-after")
                return
        except TelegramBadRequest:       # markup/entities rejected -> same text without them
            try:
                await message.answer(chunk)
            except Exception:            # noqa: BLE001
                log.exception("bare resend failed — chunk lost")
                return
        except Exception:                # noqa: BLE001
            log.exception("send failed")
            return


async def _voice_turn(message, tid, get_audio, transcribe_fn, answer_fn):
    # Claim BEFORE the paid STT call; release on every exit that doesn't hand off.
    if not _reserve(tid):
        await message.answer(BUSY_REPLY)
        return
    handed_off = False
    try:
        try:
            question = await transcribe_fn(await get_audio())
        except STTError as e:
            log.warning("stt failed for %s: %s", tid, e)
            await message.answer("Не смог распознать голос. Попробуй ещё раз или напиши текстом.")
            return
        except Exception as e:                 # noqa: BLE001
            log.exception("voice handling error: %s", e)
            await message.answer("Не смог обработать голосовое. Напиши текстом.")
            return
        await _send(message, f"🎤 Распознал: {question}")
        handed_off = True
    finally:
        if not handed_off:
            _busy.discard(tid)
    await answer_fn(question)


_TAGS_ONLY = re.compile(r"(?:\\\[\d+\\\][ \t]*)+")     # a line that is ONLY display tags \[N\]
_TIDY_DUP = re.compile(r"(\\\[\d+\\\])(?:[ \t]*\1)+")  # immediate repeats of the SAME tag


def _tidy_citations(text):
    # A tags-only line is glued onto preceding prose, but detached with a blank line after
    # a table row / code fence (the parser would absorb it).
    lines, out, in_fence = text.split("\n"), [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            out.append(ln)
            continue
        if not in_fence and s and _TAGS_ONLY.fullmatch(s) and out:
            prev = out[-1].rstrip()
            if prev and not prev.lstrip().startswith(("|", "```")):
                out[-1] = prev + " " + s
                continue
            if prev.lstrip().startswith(("|", "```")):
                out.append("")
                out.append(s)
                continue
        out.append(ln)
    return _TIDY_DUP.sub(r"\1", "\n".join(out))


def _clean_source(source):
    name = os.path.splitext(source or "")[0]
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    return name or "документ"


def _locator_lines(p):
    # One field per line; `page` is PDF-only — for txt/docx the line is simply omitted.
    lines = []
    if p.get("label"):
        lines.append(str(p["label"]))
    lines.append(_clean_source(p.get("source")))
    if p.get("section"):
        lines.append(f"§ {p['section']}")
    if p.get("page") is not None:
        lines.append(f"стр. {p['page']}")
    return lines


def _sources_block(text, entities, lines):
    # The card text is data, not markup — odd characters can't inject formatting.
    head = text + "\n\nИсточники:\n"
    block = "\n".join(lines)
    ents = list(entities)
    ents.append(tgmd.MessageEntity(type="expandable_blockquote",
                                   offset=tgmd.utf16_len(head), length=tgmd.utf16_len(block)))
    return head + block, ents


async def _render_answer(answer_md, tenant_id):
    # The display [N] is emitted already escaped (\[N\]) — our own deterministic insert, so
    # it can never parse as a link; latex_escape=False keeps it from being read as math.
    in_order = [(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(answer_md)]
    uniq = list(dict.fromkeys(in_order))
    body_md, lines = answer_md, []
    if uniq:
        locs = await asyncio.to_thread(fetch_locators, tenant_id, set(uniq))  # qdrant client is sync
        num = {ref: i + 1 for i, ref in enumerate(uniq)}
        body_md = _tidy_citations(CITE_RE.sub(
            lambda m: rf"\[{num[(m.group(1), int(m.group(2)))]}\]", answer_md))
        for ref in uniq:
            if lines:
                lines.append("")
            p = locs.get(ref)
            if p:
                card = _locator_lines(p)
                lines.append(f"[{num[ref]}] {card[0]}")
                lines.extend(card[1:])
            else:
                lines.append(f"[{num[ref]}] {ref[0]}#{ref[1]}")
    text, entities = tgmd.convert(body_md, latex_escape=False)
    if lines:
        text, entities = _sources_block(text, entities, lines)
    return text, entities


async def _process_document(pool, message: Message, tenant_id: str, source: str, doc, label=None):
    tmp = None
    try:
        if not await _drain_tenant(tenant_id, AGENT_TIMEOUT + 30):
            raise RuntimeError("активные ответы тенанта не завершились — попробуй позже")
        # Status write INSIDE the try: a Postgres blip must reach except/finally (doc ->
        # failed, gate reopened), not strand tenant_id in _ingesting until a restart.
        await set_document_status(pool, tenant_id, source, "processing")
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(source)[1] or ".bin")
        os.close(fd)
        await message.bot.download(doc, destination=tmp)
        async with _ingest_lock:           # Docling isn't thread-safe
            n = await asyncio.to_thread(ingest_file, tmp, source, tenant_id)
            note = ""
            if label:
                try:
                    await asyncio.to_thread(set_label, get_client(), source, tenant_id, label)
                except ValueError as e:        # a rejected label must not fail the ingested document
                    note = f" Метка не применена: {e}"
            await set_document_status(pool, tenant_id, source, "ready", chunks=n)
            await message.answer(f"Готово: «{source}» добавлен в корпус.{note}")
    except Exception as e:                 # noqa: BLE001 - report, don't crash polling
        log.exception("ingest failed for %s/%s", tenant_id, source)
        await set_document_status(pool, tenant_id, source, "failed", error=str(e)[:500])
        await message.answer(f"Не смог обработать «{source}»: {e}")
    finally:
        _ingesting.discard(tenant_id)      # reopen the gate whatever happened
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


async def _answer(agent, langfuse_handler, message: Message, tenant_id: str, question: str,
                  reserved: bool = False):
    # The voice path reserves before its paid STT call and hands the claim over via
    # reserved=True; the finally below is the single release point.
    tid = thread_id_for(tenant_id, message.chat.id)
    if not reserved and not _reserve(tid):
        await message.answer(BUSY_REPLY)
        return
    try:
        if tenant_id in _ingesting:        # the tenant's corpus is mid-rebuild
            await message.answer("⏳ Идёт загрузка документа — отвечу, когда она закончится.")
            return
        try:
            await message.bot.send_chat_action(message.chat.id, "typing")
        except Exception:                  # noqa: BLE001 - cosmetic
            pass
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config={"configurable": {"thread_id": tid},
                            "recursion_limit": AGENT_RECURSION_LIMIT,
                            "callbacks": [langfuse_handler] if langfuse_handler else []},
                    context=Ctx(tenant_id=tenant_id),
                ),
                timeout=AGENT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("agent timeout for %s", tid)
            await message.answer("Запрос занял слишком долго — попробуй переформулировать.")
            return
        except Exception as e:             # noqa: BLE001 - never crash the handler
            log.exception("agent error for %s: %s", tid, e)
            await message.answer("Что-то пошло не так. Попробуй ещё раз.")
            return
        log.info("answered %s in %.1fs", tid, time.perf_counter() - t0)
        answer = content_text(result["messages"][-1].content)
        try:
            text, entities = await _render_answer(answer, tenant_id)
        except Exception:                  # noqa: BLE001 - rendering is cosmetic, text must reach the user
            log.exception("render failed for %s", tid)
            text, entities = answer, None
        await _send(message, text, entities)
    finally:
        _busy.discard(tid)


async def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]

    # Bot's own queries (dedup, tenant); tuple rows — db.py reads positionally.
    pool = AsyncConnectionPool(POSTGRES_URL, open=False, max_size=10,
                               kwargs={"autocommit": True})
    await pool.open(wait=True, timeout=10)
    stuck = await reset_stuck_documents(pool)
    if stuck:
        log.warning("marked %d stuck document(s) failed (interrupted by restart)", stuck)

    # Separate pool for the checkpointer: langgraph needs dict_row + prepare_threshold=0.
    ckpt_pool = AsyncConnectionPool(
        POSTGRES_URL, open=False, max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await ckpt_pool.open(wait=True, timeout=10)
    saver = AsyncPostgresSaver(ckpt_pool)
    await saver.setup()
    agent = build_agent(checkpointer=saver)
    langfuse_handler = None
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        from langfuse.langchain import CallbackHandler
        langfuse_handler = CallbackHandler()
        log.info("langfuse tracing enabled")
    await asyncio.to_thread(warmup)
    log.info("agent ready")

    dp = Dispatcher()
    dp.update.outer_middleware(DedupMiddleware(pool))
    dp.update.outer_middleware(TenantMiddleware(pool))

    @dp.message(CommandStart())
    async def on_start(message: Message, tenant_id: str):
        await message.answer(WELCOME)

    @dp.message(Command("clear"))
    async def on_clear(message: Message, tenant_id: str):
        tid = thread_id_for(tenant_id, message.chat.id)
        await _clear_thread(message, saver, tid)

    async def _require_admin(message: Message, tenant_id: str, refusal: str):
        user_id = message.from_user.id if message.from_user else None
        if await get_role(pool, tenant_id, user_id) != "admin":
            await message.answer(refusal)
            return None
        return user_id

    @dp.message(F.document)
    async def on_document(message: Message, tenant_id: str):
        user_id = await _require_admin(message, tenant_id, "Загружать документы может только admin тенанта.")
        if user_id is None:
            return
        doc = message.document
        if doc.file_size and doc.file_size > MAX_UPLOAD_BYTES:
            await message.answer("Файл больше 20 МБ — пока не поддерживается.")
            return
        # The file name reaches the LLM prompt and both stores key on it — sanitize ONCE at
        # intake so registry and payload agree on the clean name.
        source = sanitize_text(doc.file_name or f"doc_{doc.file_unique_id}").strip() \
            or f"doc_{doc.file_unique_id}"
        if os.path.splitext(source)[1].lower() not in {".pdf", ".docx", ".txt"}:
            await message.answer("Поддерживаю только PDF, DOCX и TXT.")
            return
        if tenant_id in _ingesting:
            await message.answer("⏳ Уже обрабатываю документ — дождись окончания.")
            return
        _ingesting.add(tenant_id)              # check+add with no await between -> atomic
        try:
            label = (message.caption or "").strip() or None
            await register_document(pool, tenant_id, source, user_id)
            note = f" с меткой «{label}»" if label else ""
            await message.answer(f"Принял «{source}»{note}, обрабатываю…")
            asyncio.create_task(_process_document(pool, message, tenant_id, source, doc, label))
        except Exception:                      # failed before the task took gate ownership
            _ingesting.discard(tenant_id)
            raise

    @dp.message(Command("docs"))
    async def on_docs(message: Message, tenant_id: str):
        rows = await list_documents(pool, tenant_id)
        if not rows:
            await message.answer("Корпус пуст. Пришли документ (PDF/docx/txt) — загружу.")
            return
        labels = await asyncio.to_thread(get_labels, get_client(), tenant_id)
        marks = {"ready": "✓", "processing": "⏳", "pending": "⏳", "failed": "✗"}
        docs = sorted(((s, st, ch, labels.get(s)) for s, st, ch, _at in rows),
                      key=lambda d: (d[3] is None, (d[3] or "").lower()))
        lines, bold = [f"Документы в корпусе ({len(rows)}):"], set()
        for label, grp in groupby(docs, key=lambda d: d[3]):
            grp = list(grp)
            lines.append("")
            bold.add(len(lines))                       # group-title line -> bold entity below
            lines.append(f"{label if label else 'без метки'} — {len(grp)}")
            for source, status, chunks, _lbl in grp:
                tail = "" if chunks else f" — {status}"
                lines.append(f"  {marks.get(status, '?')} {source}{tail}")
        # Plain text + our own bold entities (offsets in UTF-16, as Telegram measures).
        ents, off = [], 0
        for i, ln in enumerate(lines):
            if i in bold:
                ents.append(tgmd.MessageEntity(type="bold", offset=off, length=tgmd.utf16_len(ln)))
            off += tgmd.utf16_len(ln) + 1              # +1 for the joining "\n"
        await _send(message, "\n".join(lines), ents)

    @dp.message(Command("rm"))
    async def on_rm(message: Message, tenant_id: str):
        if await _require_admin(message, tenant_id, "Удалять документы может только admin тенанта.") is None:
            return
        if tenant_id in _ingesting:
            await message.answer("⏳ Идёт загрузка документа — дождись окончания.")
            return
        source = (message.text or "").partition(" ")[2].strip()
        if not source:
            await message.answer("Укажи файл: /rm имя_файла")
            return
        # Qdrant first: a crash between the steps leaves an inert registry row a repeated /rm
        # finishes off; the reverse order left invisible chunks search kept citing.
        await asyncio.to_thread(_delete_source, get_client(), source, tenant_id)
        existed = await delete_document(pool, tenant_id, source)
        await message.answer(f"Удалил «{source}» из корпуса." if existed
                             else f"«{source}» не найден в корпусе.")

    @dp.message(Command("label"))
    async def on_label(message: Message, tenant_id: str):
        # /label <метка> | <файл> — the '|' splits, so both parts may contain spaces.
        if await _require_admin(message, tenant_id, "Менять метки может только admin тенанта.") is None:
            return
        if tenant_id in _ingesting:
            await message.answer("⏳ Идёт загрузка документа — дождись окончания.")
            return
        label, sep, source = (message.text or "").partition(" ")[2].partition("|")
        label, source = label.strip(), source.strip()
        if not sep or not label or not source:
            await message.answer("Формат: /label <метка> | <файл>\n"
                                 "Напр.: /label Годовая отчётность | отчёт 2025.pdf")
            return
        status = await document_status(pool, tenant_id, source)
        if status is None:
            await message.answer(f"«{source}» не найден в корпусе. /docs — список.")
            return
        if status != "ready":                  # no ready chunks -> set_payload would silently no-op
            await message.answer(f"«{source}» не готов (статус: {status}) — метка не применена. /docs — детали.")
            return
        try:
            await asyncio.to_thread(set_label, get_client(), source, tenant_id, label)
        except ValueError as e:
            await message.answer(f"Метка отклонена: {e}")
            return
        await message.answer(f"Метка обновлена: «{source}» → «{label}».")

    @dp.message(F.voice)
    async def on_voice(message: Message, tenant_id: str):
        if tenant_id in _ingesting:            # cheap gate peek before paying for STT
            await message.answer("⏳ Идёт загрузка документа — отвечу, когда она закончится.")
            return
        voice = message.voice
        if voice.file_size and voice.file_size > MAX_UPLOAD_BYTES:
            await message.answer("Голосовое больше 20 МБ — не потяну.")
            return

        async def get_audio():
            try:
                await message.bot.send_chat_action(message.chat.id, "typing")
            except Exception:                  # noqa: BLE001 - cosmetic
                pass
            buf = io.BytesIO()
            await message.bot.download(voice, destination=buf)
            return buf.getvalue()

        tid = thread_id_for(tenant_id, message.chat.id)
        await _voice_turn(message, tid, get_audio, transcribe,
                          lambda q: _answer(agent, langfuse_handler, message, tenant_id, q,
                                            reserved=True))

    @dp.message()
    async def on_message(message: Message, tenant_id: str):
        if not message.text:
            await message.answer("Пока умею текст и голосовые сообщения.")
            return
        await _answer(agent, langfuse_handler, message, tenant_id, message.text)

    bot = Bot(token=token)

    async def _prune_loop():
        while True:
            try:
                n = await prune_seen_updates(pool)
                if n:
                    log.info("pruned %d old seen_updates", n)
            except Exception:              # noqa: BLE001 - housekeeping must not crash polling
                log.exception("seen_updates prune failed")
            await asyncio.sleep(86400)

    asyncio.create_task(_prune_loop())
    log.info("starting polling")
    try:
        await dp.start_polling(bot, tasks_concurrency_limit=20)
    finally:
        if langfuse_handler:                # flush buffered traces before exit
            try:
                from langfuse import get_client as _lf_get_client   # do NOT shadow ingest.get_client
                _lf_get_client().shutdown()
            except Exception:               # noqa: BLE001
                log.exception("langfuse shutdown failed")
        for close in (bot.session.close, pool.close, ckpt_pool.close):
            try:
                await close()
            except Exception:               # noqa: BLE001
                log.exception("cleanup failed")


if __name__ == "__main__":
    asyncio.run(main())
