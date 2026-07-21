"""Postgres helpers for the bot (psycopg3 async pool)."""


async def is_duplicate(pool, update_id):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "INSERT INTO seen_updates (update_id) VALUES (%s) "
            "ON CONFLICT (update_id) DO NOTHING",
            (update_id,),
        )
        return cur.rowcount == 0          # 0 rows inserted -> conflict -> duplicate


async def prune_seen_updates(pool, keep_days=1):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM seen_updates WHERE seen_at < now() - make_interval(days => %s)",
            (keep_days,),
        )
        return cur.rowcount


async def resolve_tenant(pool, chat_id):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT c.tenant_id FROM chats c "
            "JOIN tenants t ON t.tenant_id = c.tenant_id "
            "WHERE c.chat_id = %s AND c.is_active = true AND t.is_active = true",
            (chat_id,),
        )
        row = await cur.fetchone()
        return row[0] if row else None


async def get_role(pool, tenant_id, user_id):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT role FROM tenant_roles WHERE tenant_id = %s AND user_id = %s",
            (tenant_id, user_id),
        )
        row = await cur.fetchone()
        return row[0] if row else None


async def register_document(pool, tenant_id, source, user_id):
    # The label is NOT stored here — it lives only in the Qdrant payload.
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO documents (tenant_id, source, uploaded_by, status) "
            "VALUES (%s, %s, %s, 'pending') "
            "ON CONFLICT (tenant_id, source) DO UPDATE SET "
            "uploaded_by = EXCLUDED.uploaded_by, uploaded_at = now(), "
            "status = 'pending', chunks = NULL, error = NULL",
            (tenant_id, source, user_id),
        )


async def document_status(pool, tenant_id, source):
    """Status or None."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT status FROM documents WHERE tenant_id = %s AND source = %s",
            (tenant_id, source),
        )
        row = await cur.fetchone()
        return row[0] if row else None


async def reset_stuck_documents(pool):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "UPDATE documents SET status = 'failed', error = 'прерван перезапуском бота' "
            "WHERE status IN ('pending', 'processing')",
        )
        return cur.rowcount


async def set_document_status(pool, tenant_id, source, status, chunks=None, error=None):
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE documents SET status = %s, chunks = %s, error = %s "
            "WHERE tenant_id = %s AND source = %s",
            (status, chunks, error, tenant_id, source),
        )


async def list_documents(pool, tenant_id):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT source, status, chunks, uploaded_at FROM documents "
            "WHERE tenant_id = %s ORDER BY uploaded_at DESC",
            (tenant_id,),
        )
        return await cur.fetchall()


async def delete_document(pool, tenant_id, source):
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM documents WHERE tenant_id = %s AND source = %s",
            (tenant_id, source),
        )
        return cur.rowcount > 0
