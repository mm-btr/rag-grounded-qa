"""Registry wrappers: dedup semantics, kill-switch predicate, status writes, startup sweep scope."""
import asyncio

import db


class FakeCursor:
    def __init__(self, rowcount=0, row=None, rows=None):
        self.rowcount = rowcount
        self._row = row
        self._rows = rows or []

    async def fetchone(self):
        return self._row

    async def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.calls = []

    async def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return self._cursor


class FakePool:
    def __init__(self, cursor):
        self.conn = FakeConn(cursor)

    def connection(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _run(coro):
    return asyncio.run(coro)


def test_is_duplicate_true_only_on_conflict():
    fresh = FakePool(FakeCursor(rowcount=1))
    dup = FakePool(FakeCursor(rowcount=0))
    assert _run(db.is_duplicate(fresh, 1)) is False
    assert _run(db.is_duplicate(dup, 1)) is True


def test_resolve_tenant_requires_both_kill_switches():
    pool = FakePool(FakeCursor(row=("t1",)))
    assert _run(db.resolve_tenant(pool, 5)) == "t1"
    sql = pool.conn.calls[0][0]
    assert "c.is_active = true" in sql
    assert "t.is_active = true" in sql


def test_resolve_tenant_none_when_not_allowed():
    pool = FakePool(FakeCursor(row=None))
    assert _run(db.resolve_tenant(pool, 5)) is None


def test_register_document_resets_row_to_pending():
    pool = FakePool(FakeCursor())
    _run(db.register_document(pool, "t1", "a.pdf", 42))
    sql, params = pool.conn.calls[0]
    assert "ON CONFLICT (tenant_id, source)" in sql
    assert "status = 'pending'" in sql
    assert "error = NULL" in sql
    assert params == ("t1", "a.pdf", 42)


def test_set_document_status_param_order():
    pool = FakePool(FakeCursor())
    _run(db.set_document_status(pool, "t1", "a.pdf", "failed", error="boom"))
    _, params = pool.conn.calls[0]
    assert params == ("failed", None, "boom", "t1", "a.pdf")


def test_startup_sweep_targets_only_stuck_statuses():
    pool = FakePool(FakeCursor(rowcount=2))
    assert _run(db.reset_stuck_documents(pool)) == 2
    sql = pool.conn.calls[0][0]
    assert "IN ('pending', 'processing')" in sql
    assert "status = 'failed'" in sql
    assert "'ready'" not in sql


def test_document_status_none_when_absent():
    assert _run(db.document_status(FakePool(FakeCursor(row=None)), "t1", "a.pdf")) is None
    assert _run(db.document_status(FakePool(FakeCursor(row=("ready",))), "t1", "a.pdf")) == "ready"


def test_delete_document_reports_existence():
    assert _run(db.delete_document(FakePool(FakeCursor(rowcount=1)), "t1", "a.pdf")) is True
    assert _run(db.delete_document(FakePool(FakeCursor(rowcount=0)), "t1", "a.pdf")) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all db-registry tests passed")
