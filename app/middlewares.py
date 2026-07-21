"""aiogram outer middlewares. Order matters: Dedup first (cheap), then Tenant (access gate)."""
import logging

from aiogram import BaseMiddleware
from db import is_duplicate, resolve_tenant

log = logging.getLogger("bot.middleware")


class DedupMiddleware(BaseMiddleware):
    def __init__(self, pool):
        self.pool = pool

    async def __call__(self, handler, event, data):
        if await is_duplicate(self.pool, event.update_id):
            return None
        return await handler(event, data)


class TenantMiddleware(BaseMiddleware):
    def __init__(self, pool):
        self.pool = pool

    async def __call__(self, handler, event, data):
        chat = data.get("event_chat")
        if chat is None:                          # service update with no chat -> ignore
            return None
        tenant_id = await resolve_tenant(self.pool, chat.id)
        if tenant_id is None:
            log.info("access denied for chat_id=%s", chat.id)
            msg = event.message or (event.callback_query and event.callback_query.message)
            if msg is not None:
                await msg.answer("Нет доступа.")
            return None                           # closed by default
        data["tenant_id"] = tenant_id
        return await handler(event, data)
