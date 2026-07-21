"""The chat is reserved BEFORE the paid STT call and released on every non-handoff exit."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import bot
from bot import _reserve, _voice_turn
from stt import STTError

TID = "tenant:1"


def test_reserve_is_exclusive_until_released():
    bot._busy.discard(TID)
    assert _reserve(TID) is True
    assert _reserve(TID) is False
    bot._busy.discard(TID)
    assert _reserve(TID) is True
    bot._busy.discard(TID)


def test_second_voice_rejected_without_paying_stt():
    async def scenario():
        bot._busy.discard(TID)
        stt_started, stt_release = asyncio.Event(), asyncio.Event()
        stt_calls, answered = [], []

        async def get_audio():
            return b"ogg"

        async def slow_stt(audio):
            stt_calls.append(audio)
            stt_started.set()
            await stt_release.wait()
            return "вопрос"

        async def answer(question):
            answered.append(question)
            bot._busy.discard(TID)             # in prod _answer's finally releases the claim

        first = SimpleNamespace(answer=AsyncMock())
        second = SimpleNamespace(answer=AsyncMock())
        task = asyncio.create_task(_voice_turn(first, TID, get_audio, slow_stt, answer))
        await stt_started.wait()
        await _voice_turn(second, TID, get_audio, slow_stt, answer)   # arrives mid-transcription

        second.answer.assert_awaited_once_with(bot.BUSY_REPLY)
        assert len(stt_calls) == 1             # the loser never paid for STT

        stt_release.set()
        await task
        assert answered == ["вопрос"]
        assert TID not in bot._busy

    asyncio.run(scenario())


def test_stt_failure_releases_the_chat():
    async def scenario():
        bot._busy.discard(TID)

        async def get_audio():
            return b"ogg"

        async def bad_stt(audio):
            raise STTError("scribe down")

        message = SimpleNamespace(answer=AsyncMock())
        answer = AsyncMock()
        await _voice_turn(message, TID, get_audio, bad_stt, answer)

        message.answer.assert_awaited_once()
        answer.assert_not_awaited()
        assert TID not in bot._busy            # released -> the chat is not wedged

    asyncio.run(scenario())


def test_handoff_keeps_the_claim_until_answer():
    async def scenario():
        bot._busy.discard(TID)

        async def get_audio():
            return b"ogg"

        async def stt(audio):
            return "вопрос"

        seen = []

        async def answer(question):
            seen.append(TID in bot._busy)      # the claim must still be held at handoff
            bot._busy.discard(TID)

        await _voice_turn(SimpleNamespace(answer=AsyncMock()), TID, get_audio, stt, answer)
        assert seen == [True]

    asyncio.run(scenario())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all voice-reserve tests passed")
