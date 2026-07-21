import asyncio

import bot
from bot import _tenant_busy, _drain_tenant


def test_tenant_busy_matches_only_own_threads():
    bot._busy.clear()
    bot._busy.add("t1:5")
    assert _tenant_busy("t1") is True
    assert _tenant_busy("t") is False
    assert _tenant_busy("t2") is False
    bot._busy.clear()


def test_drain_waits_until_release():
    async def scenario():
        bot._busy.clear()
        bot._busy.add("t1:5")

        async def release():
            await asyncio.sleep(0.05)
            bot._busy.discard("t1:5")

        task = asyncio.create_task(release())
        ok = await _drain_tenant("t1", timeout=2, poll=0.01)
        await task
        assert ok is True

    asyncio.run(scenario())


def test_drain_times_out_when_turn_never_ends():
    async def scenario():
        bot._busy.clear()
        bot._busy.add("t1:5")
        ok = await _drain_tenant("t1", timeout=0.05, poll=0.01)
        bot._busy.clear()
        assert ok is False

    asyncio.run(scenario())


def test_drain_ignores_other_tenants():
    async def scenario():
        bot._busy.clear()
        bot._busy.add("t2:7")
        ok = await _drain_tenant("t1", timeout=0.05, poll=0.01)
        bot._busy.clear()
        assert ok is True

    asyncio.run(scenario())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all drain tests passed")
