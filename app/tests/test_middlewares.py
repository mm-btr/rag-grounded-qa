"""Access gate: dedup drops retried updates, tenant middleware is closed by default."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import middlewares
from middlewares import DedupMiddleware, TenantMiddleware


def _with_db(is_dup=None, tenant=None, fn=None):
    originals = (middlewares.is_duplicate, middlewares.resolve_tenant)
    middlewares.is_duplicate = AsyncMock(return_value=is_dup)
    middlewares.resolve_tenant = AsyncMock(return_value=tenant)
    try:
        return asyncio.run(fn())
    finally:
        middlewares.is_duplicate, middlewares.resolve_tenant = originals


def test_duplicate_update_is_swallowed_before_handler():
    async def scenario():
        handler = AsyncMock()
        event = SimpleNamespace(update_id=7)
        result = await DedupMiddleware(pool=None)(handler, event, {})
        assert result is None
        handler.assert_not_awaited()

    _with_db(is_dup=True, fn=scenario)


def test_fresh_update_reaches_handler():
    async def scenario():
        handler = AsyncMock(return_value="handled")
        event = SimpleNamespace(update_id=8)
        data = {}
        result = await DedupMiddleware(pool=None)(handler, event, data)
        assert result == "handled"
        handler.assert_awaited_once_with(event, data)

    _with_db(is_dup=False, fn=scenario)


def test_update_without_chat_is_ignored():
    async def scenario():
        handler = AsyncMock()
        result = await TenantMiddleware(pool=None)(handler, SimpleNamespace(), {"event_chat": None})
        assert result is None
        handler.assert_not_awaited()
        middlewares.resolve_tenant.assert_not_awaited()

    _with_db(tenant="t1", fn=scenario)


def test_unknown_chat_is_refused_and_stopped():
    async def scenario():
        handler = AsyncMock()
        msg = SimpleNamespace(answer=AsyncMock())
        event = SimpleNamespace(message=msg, callback_query=None)
        data = {"event_chat": SimpleNamespace(id=404)}
        result = await TenantMiddleware(pool=None)(handler, event, data)
        assert result is None
        handler.assert_not_awaited()
        msg.answer.assert_awaited_once_with("Нет доступа.")
        assert "tenant_id" not in data

    _with_db(tenant=None, fn=scenario)


def test_unknown_chat_without_message_still_stops_quietly():
    async def scenario():
        handler = AsyncMock()
        event = SimpleNamespace(message=None, callback_query=None)
        result = await TenantMiddleware(pool=None)(handler, event, {"event_chat": SimpleNamespace(id=404)})
        assert result is None
        handler.assert_not_awaited()

    _with_db(tenant=None, fn=scenario)


def test_allowed_chat_gets_tenant_and_handler_runs():
    async def scenario():
        handler = AsyncMock(return_value="ok")
        event = SimpleNamespace(message=None, callback_query=None)
        data = {"event_chat": SimpleNamespace(id=5)}
        result = await TenantMiddleware(pool=None)(handler, event, data)
        assert result == "ok"
        assert data["tenant_id"] == "t1"
        handler.assert_awaited_once_with(event, data)

    _with_db(tenant="t1", fn=scenario)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all middleware tests passed")
