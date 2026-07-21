"""Integration: citations, staged visibility and shorter re-ingest against real Qdrant.
Needs Qdrant + embedding models. Self-cleaning tenant.
Run: PYTHONPATH=/app PYTHONIOENCODING=utf-8 python tests/test_ingest_cite.py
"""
from qdrant_client import models

from ingest import ingest_chunks, doc_id, delete_tenant, get_client
from retrieve import search, fetch_locators

T = "__test_ingest__"


def _count(c, source):
    return c.count("docs", count_filter=models.Filter(must=[
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=T)),
        models.FieldCondition(key="source", match=models.MatchValue(value=source)),
    ]), exact=True).count


def _doc_filter(source):
    return models.Filter(must=[
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=T)),
        models.FieldCondition(key="source", match=models.MatchValue(value=source)),
    ])


def main():
    c = get_client()
    delete_tenant(c, T)
    src = "тест [файл] #1.txt"                       # deliberately citation-hostile name
    chunks = [
        "Возврат товара надлежащего качества возможен в течение четырнадцати дней.",
        "Для возврата товар должен сохранить товарный вид и потребительские свойства.",
        "Возврат осуществляется при наличии чека или иного подтверждения оплаты.",
        "Деньги возвращаются в течение десяти дней с момента обращения покупателя.",
    ]
    n = ingest_chunks(chunks, src, T)
    did = doc_id(src)
    assert n >= 2, n
    print(f"ingested {n} chunks; doc_id={did} (source has spaces/[/#)")

    hits = search("возврат товара", tenant_id=T, top_k=2)
    assert hits and hits[0]["doc_id"] == did, hits[0].get("doc_id")
    print(f"citation id from search = {hits[0]['doc_id']}  (hex, regex-safe)")

    c.set_payload("docs", {"ingest_ready": False}, points=_doc_filter(src), wait=True)
    assert search("возврат товара", tenant_id=T, top_k=2) == []
    print("explicit staging is invisible to search")

    c.delete_payload("docs", ["ingest_ready"], points=_doc_filter(src), wait=True)
    legacy_hits = search("возврат товара", tenant_id=T, top_k=2)
    assert legacy_hits and legacy_hits[0]["doc_id"] == did
    print("legacy points without ingest_ready remain visible")

    locs = fetch_locators(T, [(did, legacy_hits[0]["chunk"])])
    p = locs[(did, legacy_hits[0]["chunk"])]
    assert p["source"] == src, p.get("source")
    print(f"resolved doc_id -> source = {p['source']}")

    n2 = ingest_chunks(["Один короткий абзац про возврат."], src, T)   # SHORTER re-ingest
    total = _count(c, src)
    assert total == n2, (total, n2)
    print(f"re-ingest shorter: {n2} chunks, qdrant total = {total} (no orphan tail)")

    delete_tenant(c, T)
    print("OK: doc_id citations odd-name-safe + atomic re-ingest")


if __name__ == "__main__":
    main()
