"""The /clear reply must match the checkpointer outcome."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import bot
from bot import _clear_thread


def test_clear_reports_success_after_delete():
    bot._busy.clear()
    message = SimpleNamespace(answer=AsyncMock())
    saver = SimpleNamespace(adelete_thread=AsyncMock())

    asyncio.run(_clear_thread(message, saver, "tenant:chat"))

    saver.adelete_thread.assert_awaited_once_with("tenant:chat")
    message.answer.assert_awaited_once_with("Контекст очищен.")
    assert "tenant:chat" not in bot._busy


def test_clear_never_confirms_success_when_delete_fails():
    bot._busy.clear()
    message = SimpleNamespace(answer=AsyncMock())
    saver = SimpleNamespace(adelete_thread=AsyncMock(side_effect=RuntimeError("db unavailable")))

    asyncio.run(_clear_thread(message, saver, "tenant:chat"))

    saver.adelete_thread.assert_awaited_once_with("tenant:chat")
    message.answer.assert_awaited_once_with("Контекст не очищен. Попробуйте /clear ещё раз.")
    assert "tenant:chat" not in bot._busy


def test_clear_holds_the_chat_until_delete_finishes():
    async def scenario():
        bot._busy.clear()
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_delete(tid):
            started.set()
            await release.wait()

        saver = SimpleNamespace(adelete_thread=AsyncMock(side_effect=slow_delete))
        first = SimpleNamespace(answer=AsyncMock())
        second = SimpleNamespace(answer=AsyncMock())

        task = asyncio.create_task(_clear_thread(first, saver, "tenant:chat"))
        await started.wait()
        assert "tenant:chat" in bot._busy

        await _clear_thread(second, saver, "tenant:chat")
        second.answer.assert_awaited_once_with(bot.CLEAR_BUSY_REPLY)
        saver.adelete_thread.assert_awaited_once_with("tenant:chat")

        release.set()
        await task
        first.answer.assert_awaited_once_with("Контекст очищен.")
        assert "tenant:chat" not in bot._busy

    asyncio.run(scenario())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all clear tests passed")
