"""_answer resilience: every outcome answers the user and leaves the chat unlocked."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import bot
from bot import _answer


def _message():
    return SimpleNamespace(chat=SimpleNamespace(id=5), answer=AsyncMock(),
                           bot=SimpleNamespace(send_chat_action=AsyncMock()))


def test_busy_chat_rejected_without_touching_the_claim():
    async def case():
        bot._busy.clear()
        bot._busy.add("t1:5")
        agent = SimpleNamespace(ainvoke=AsyncMock())
        message = _message()
        await _answer(agent, None, message, "t1", "q")
        message.answer.assert_awaited_once_with(bot.BUSY_REPLY)
        agent.ainvoke.assert_not_awaited()
        assert "t1:5" in bot._busy
        bot._busy.clear()

    asyncio.run(case())


def test_ingest_gate_rejects_and_releases_chat():
    async def case():
        bot._busy.clear()
        bot._ingesting.add("t1")
        agent = SimpleNamespace(ainvoke=AsyncMock())
        message = _message()
        try:
            await _answer(agent, None, message, "t1", "q")
        finally:
            bot._ingesting.discard("t1")
        assert "загрузка документа" in message.answer.call_args.args[0]
        agent.ainvoke.assert_not_awaited()
        assert "t1:5" not in bot._busy

    asyncio.run(case())


def test_agent_timeout_reports_and_releases_chat():
    async def case():
        bot._busy.clear()

        async def slow(*args, **kwargs):
            await asyncio.sleep(1)

        message = _message()
        original = bot.AGENT_TIMEOUT
        bot.AGENT_TIMEOUT = 0.05
        try:
            await _answer(SimpleNamespace(ainvoke=slow), None, message, "t1", "q")
        finally:
            bot.AGENT_TIMEOUT = original
        assert "слишком долго" in message.answer.call_args.args[0]
        assert "t1:5" not in bot._busy

    asyncio.run(case())


def test_agent_error_reports_and_releases_chat():
    async def case():
        bot._busy.clear()

        async def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        message = _message()
        await _answer(SimpleNamespace(ainvoke=boom), None, message, "t1", "q")
        assert "Что-то пошло не так" in message.answer.call_args.args[0]
        assert "t1:5" not in bot._busy

    asyncio.run(case())


def test_happy_path_sends_answer_and_releases_chat():
    async def case():
        bot._busy.clear()

        async def ok(*args, **kwargs):
            return {"messages": [SimpleNamespace(content="raw")]}

        message = _message()
        send = AsyncMock()
        originals = (bot.content_text, bot._render_answer, bot._send)
        bot.content_text = lambda content: "answer-md"
        bot._render_answer = AsyncMock(return_value=("rendered", None))
        bot._send = send
        try:
            await _answer(SimpleNamespace(ainvoke=ok), None, message, "t1", "q")
        finally:
            bot.content_text, bot._render_answer, bot._send = originals
        send.assert_awaited_once_with(message, "rendered", None)
        assert "t1:5" not in bot._busy

    asyncio.run(case())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all answer-error tests passed")
