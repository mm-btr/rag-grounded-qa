"""_process_document failure paths: the document ends `failed` and the tenant gate reopens."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import bot
from bot import _process_document


def _message():
    return SimpleNamespace(bot=SimpleNamespace(download=AsyncMock()), answer=AsyncMock())


def _run(case, **patches):
    originals = {name: getattr(bot, name) for name in patches}
    for name, value in patches.items():
        setattr(bot, name, value)
    bot._ingesting.add("t1")
    try:
        asyncio.run(case())
    finally:
        for name, value in originals.items():
            setattr(bot, name, value)
        bot._ingesting.discard("t1")


def test_parse_failure_marks_failed_and_reopens_gate():
    status = AsyncMock()

    def broken_ingest(tmp, source, tenant_id):
        raise ValueError("кривой файл")

    async def case():
        message = _message()
        await _process_document(object(), message, "t1", "a.pdf", doc=object())
        statuses = [c.args[3] for c in status.call_args_list]
        assert statuses == ["processing", "failed"]
        assert "кривой файл" in status.call_args_list[1].kwargs["error"]
        assert "t1" not in bot._ingesting
        assert "Не смог обработать" in message.answer.call_args.args[0]

    _run(case, _drain_tenant=AsyncMock(return_value=True),
         set_document_status=status, ingest_file=broken_ingest)


def test_status_write_failure_still_reopens_gate():
    status = AsyncMock(side_effect=[RuntimeError("db blip"), None])

    async def case():
        message = _message()
        await _process_document(object(), message, "t1", "a.pdf", doc=object())
        assert status.call_args_list[1].args[3] == "failed"
        assert "t1" not in bot._ingesting
        message.answer.assert_awaited()

    _run(case, _drain_tenant=AsyncMock(return_value=True), set_document_status=status)


def test_drain_timeout_fails_document_without_ingest():
    status = AsyncMock()
    ingest = AsyncMock()

    async def case():
        message = _message()
        await _process_document(object(), message, "t1", "a.pdf", doc=object())
        assert [c.args[3] for c in status.call_args_list] == ["failed"]
        assert "активные ответы" in status.call_args.kwargs["error"]
        ingest.assert_not_called()
        assert "t1" not in bot._ingesting

    _run(case, _drain_tenant=AsyncMock(return_value=False),
         set_document_status=status, ingest_file=ingest)


def test_rejected_label_does_not_fail_the_document():
    status = AsyncMock()

    def rejected_label(client, source, tenant_id, label):
        raise ValueError("Подозрение на инъекцию в метке")

    async def case():
        message = _message()
        await _process_document(object(), message, "t1", "a.pdf", doc=object(), label="метка")
        ready = status.call_args_list[1]
        assert ready.args[3] == "ready" and ready.kwargs["chunks"] == 3
        assert "Метка не применена" in message.answer.call_args.args[0]
        assert "t1" not in bot._ingesting

    _run(case, _drain_tenant=AsyncMock(return_value=True), set_document_status=status,
         ingest_file=lambda tmp, source, tenant_id: 3,
         set_label=rejected_label, get_client=lambda: None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all ingest-error tests passed")
