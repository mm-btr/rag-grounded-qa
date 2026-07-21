"""Ingestion: text -> chunks -> dense+sparse embeddings -> Qdrant (tenant-scoped)."""
import hashlib
import uuid
from functools import lru_cache
from qdrant_client import QdrantClient, models
from config import (
    QDRANT_URL, COLLECTION, DENSE_DIM,
    INGEST_BATCH_SIZE, HNSW_M, HNSW_PAYLOAD_M, INJECTION_THRESHOLD,
)
from models import embed_texts, injection_score
from sanitize import sanitize_text


_INGEST_READY = "ingest_ready"


@lru_cache(maxsize=1)
def get_client():
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client):
    """Create the collection + indexes if absent. Idempotent, never destructive."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
        # m=0 disables the global graph; payload_m builds per-tenant subgraphs.
        hnsw_config=models.HnswConfigDiff(m=HNSW_M, payload_m=HNSW_PAYLOAD_M),
    )
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="tenant_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True
        ),
    )
    # source index: scoped delete on re-ingest.
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _tenant_match(tenant_id):
    """The tenant-isolation predicate — ONE definition of the partition key."""
    return models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))


def _read_filter(must):
    """Read only published points. Missing `ingest_ready` means legacy-ready, so the
    existing corpus needs no backfill; only explicit False marks a staging write."""
    return models.Filter(
        must=list(must),
        must_not=[models.FieldCondition(
            key=_INGEST_READY, match=models.MatchValue(value=False)
        )],
    )


def delete_tenant(client, tenant_id):
    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(must=[_tenant_match(tenant_id)])),
    )


def doc_id(source):
    # Plain hex: an odd file name (spaces, '#', brackets) can't break the citation regex.
    return hashlib.sha1(source.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def point_id(tenant_id, did, idx):
    # Deterministic -> stable on re-ingest; hex doc_id keeps the '/' join collision-free.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}/{did}/{idx}"))


def _source_filter(tenant_id, source):
    return models.Filter(must=[
        _tenant_match(tenant_id),
        models.FieldCondition(key="source", match=models.MatchValue(value=source)),
    ])


def _read_source_filter(tenant_id, source):
    return _read_filter(_source_filter(tenant_id, source).must)


def _delete_source(client, source, tenant_id):
    # Full delete of the old version, then a fresh insert — version mixing is impossible.
    client.delete(collection_name=COLLECTION,
                  points_selector=models.FilterSelector(filter=_source_filter(tenant_id, source)))


def _activate_source(client, source, tenant_id):
    """Publish a fully written document in one payload operation; until this succeeds,
    every new point stays excluded by `_read_filter`."""
    client.set_payload(
        collection_name=COLLECTION,
        payload={_INGEST_READY: True},
        points=_source_filter(tenant_id, source),
        wait=True,
    )


# Label/descr live ONLY in the Qdrant payload and are written ONLY here (payload-only, no re-embed).

def _guard_meta(value, what):
    # label/descr reach the LLM prompt — same untrusted surface as chunk text.
    value = sanitize_text(value)
    score = injection_score(value)
    if score >= INJECTION_THRESHOLD:
        raise ValueError(f"Подозрение на инъекцию в {what} (score {score:.2f}). Отклонено.")
    return value


def set_label(client, source, tenant_id, label):
    label = _guard_meta(label, "метке")
    client.set_payload(collection_name=COLLECTION, payload={"label": label},
                       points=_source_filter(tenant_id, source))


def get_label(client, source, tenant_id):
    recs, _ = client.scroll(collection_name=COLLECTION,
                            scroll_filter=_read_source_filter(tenant_id, source),
                            with_payload=["label"], limit=1)
    return recs[0].payload.get("label") if recs else None


def set_descr(client, source, tenant_id, descr):
    descr = _guard_meta(descr, "описании")
    client.set_payload(collection_name=COLLECTION, payload={"descr": descr},
                       points=_source_filter(tenant_id, source))


def get_descr(client, source, tenant_id):
    recs, _ = client.scroll(collection_name=COLLECTION,
                            scroll_filter=_read_source_filter(tenant_id, source),
                            with_payload=["descr"], limit=1)
    return recs[0].payload.get("descr") if recs else None


def get_labels(client, tenant_id):
    out, offset = {}, None
    tflt = _read_filter([_tenant_match(tenant_id)])
    while True:
        recs, offset = client.scroll(collection_name=COLLECTION, scroll_filter=tflt,
                                     with_payload=["source", "label"], limit=256, offset=offset)
        for r in recs:
            p = r.payload or {}
            if p.get("source"):
                out[p["source"]] = p.get("label")
        if offset is None:
            break
    return out


def _scan_injections(items, source=None):
    # The prompt surface is wider than chunk text: section headings and the file name reach
    # the passage header, so both are scanned too.
    for i, it in enumerate(items):
        score = injection_score(it["text"])
        if score >= INJECTION_THRESHOLD:
            raise ValueError(
                f"Подозрение на инъекцию во фрагменте {i + 1} (score {score:.2f}). Документ отклонён."
            )
    metas = {it["section"] for it in items if it.get("section")}
    if source:
        metas.add(source)
    for m in metas:
        score = injection_score(m)
        if score >= INJECTION_THRESHOLD:
            raise ValueError(
                f"Подозрение на инъекцию в метаданных документа («{m[:80]}», score {score:.2f}). Документ отклонён."
            )


def _sanitize_and_scan(items, source):
    """Runs BEFORE the old version is deleted — a malicious or broken new file must never
    destroy the version it was meant to replace."""
    for it in items:
        it["text"] = sanitize_text(it["text"])
        it["embed_text"] = sanitize_text(it["embed_text"])
        if it.get("section"):
            it["section"] = sanitize_text(it["section"])
    _scan_injections(items, source)


def _upsert_chunks(client, items, source, tenant_id):
    """Chunks are staged with `ingest_ready=False` and without a label; the caller carries
    metadata, then publishes via `_activate_source` — a half-written batch is invisible."""
    if not items:
        return 0
    did = doc_id(source)
    total = 0
    for bstart in range(0, len(items), INGEST_BATCH_SIZE):
        batch = items[bstart:bstart + INGEST_BATCH_SIZE]
        embs = embed_texts([it["embed_text"] for it in batch])
        points = []
        for j, (it, emb) in enumerate(zip(batch, embs)):
            idx = bstart + j
            vector = {"dense": emb["dense"]}
            sp = emb["sparse"]
            if sp:                       # skip empty sparse vector (degenerate chunk)
                vector["sparse"] = models.SparseVector(
                    indices=list(sp.keys()), values=list(sp.values())
                )
            points.append(models.PointStruct(
                id=point_id(tenant_id, did, idx),
                vector=vector,
                payload={
                    "text": it["text"], "source": source, "doc_id": did,
                    "chunk": idx, "tenant_id": tenant_id,
                    _INGEST_READY: False,
                    "page": it.get("page"), "section": it.get("section"),
                },
            ))
        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
    return total


def ingest_chunks(chunks, source, tenant_id):
    """Pre-split chunks (tests/seed; the production path is `ingest_file`)."""
    client = get_client()
    ensure_collection(client)
    items = [
        {"text": c, "embed_text": c, "page": None, "section": None}
        for c in chunks
    ]
    if not items:
        raise ValueError("Нет чанков для индексации (пустой вход).")
    _sanitize_and_scan(items, source)
    _delete_source(client, source, tenant_id)
    n = _upsert_chunks(client, items, source, tenant_id)
    _activate_source(client, source, tenant_id)
    return n


def ingest_file(path, source, tenant_id):
    """Re-ingest = full delete of the old version, then a fresh insert; everything fallible
    (parse, sanitize, scan) runs BEFORE the delete."""
    from parse import parse_and_chunk          # lazy: docling is heavy
    client = get_client()
    ensure_collection(client)
    prev_label = get_label(client, source, tenant_id)
    prev_descr = get_descr(client, source, tenant_id)
    items = [
        {
            "text": c["text"], "embed_text": c["embed_text"],
            "page": c["page"], "section": c["section"],
        }
        for c in parse_and_chunk(path)
    ]
    if not items:
        raise ValueError(
            "Документ не дал текста для индексации (пустой файл или скан без текстового слоя)."
        )
    _sanitize_and_scan(items, source)
    _delete_source(client, source, tenant_id)
    n = _upsert_chunks(client, items, source, tenant_id)
    if prev_label is not None:
        set_label(client, source, tenant_id, prev_label)
    if prev_descr is not None:
        set_descr(client, source, tenant_id, prev_descr)
    _activate_source(client, source, tenant_id)
    return n
