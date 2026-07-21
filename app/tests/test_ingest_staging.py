"""A failed multi-batch ingest never publishes its already-written prefix."""
import ingest


class FakeClient:
    def __init__(self, fail_upsert=None):
        self.fail_upsert = fail_upsert
        self.upserts = []
        self.activations = []

    def delete(self, **kwargs):
        pass

    def upsert(self, **kwargs):
        call = len(self.upserts) + 1
        if call == self.fail_upsert:
            raise RuntimeError("second batch failed")
        self.upserts.append(kwargs["points"])

    def set_payload(self, **kwargs):
        self.activations.append(kwargs)


def _items(count=3):
    return [f"chunk {i}" for i in range(count)]


def _run_with(client, fn):
    originals = {
        "get_client": ingest.get_client,
        "ensure_collection": ingest.ensure_collection,
        "_sanitize_and_scan": ingest._sanitize_and_scan,
        "embed_texts": ingest.embed_texts,
        "INGEST_BATCH_SIZE": ingest.INGEST_BATCH_SIZE,
    }
    ingest.get_client = lambda: client
    ingest.ensure_collection = lambda _client: None
    ingest._sanitize_and_scan = lambda items, source: None
    ingest.embed_texts = lambda texts: [
        {"dense": [float(i)], "sparse": {i: 1.0}} for i, _ in enumerate(texts)
    ]
    ingest.INGEST_BATCH_SIZE = 2
    try:
        return fn()
    finally:
        for name, value in originals.items():
            setattr(ingest, name, value)


def test_failed_second_batch_never_activates_first_batch():
    client = FakeClient(fail_upsert=2)
    try:
        _run_with(client, lambda: ingest.ingest_chunks(_items(), "doc.txt", "tenant"))
    except RuntimeError as exc:
        assert str(exc) == "second batch failed"
    else:
        raise AssertionError("expected the second batch to fail")
    assert len(client.upserts) == 1
    assert all(p.payload["ingest_ready"] is False for p in client.upserts[0])
    assert client.activations == []


def test_success_activates_only_after_all_batches():
    client = FakeClient()
    n = _run_with(client, lambda: ingest.ingest_chunks(_items(), "doc.txt", "tenant"))
    assert n == 3 and [len(batch) for batch in client.upserts] == [2, 1]
    assert all(p.payload["ingest_ready"] is False
               for batch in client.upserts for p in batch)
    assert len(client.activations) == 1
    assert client.activations[0]["payload"] == {"ingest_ready": True}
    assert client.activations[0]["wait"] is True


def test_read_filter_excludes_only_explicit_staging_points():
    flt = ingest._read_filter([ingest._tenant_match("tenant")])
    assert len(flt.must) == 1 and len(flt.must_not) == 1
    assert flt.must_not[0].key == "ingest_ready"
    assert flt.must_not[0].match.value is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all ingest-staging tests passed")
