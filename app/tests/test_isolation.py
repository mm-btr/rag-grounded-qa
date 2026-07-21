"""Integration: tenant isolation — a search scoped to tenant A must NEVER surface tenant B's
data, and vice versa. The core multi-tenant guarantee. Needs Qdrant + models. Self-cleaning.

Run: PYTHONPATH=/app PYTHONIOENCODING=utf-8 python tests/test_isolation.py
"""
from ingest import ingest_chunks, delete_tenant, get_client
from retrieve import search

A, B = "__iso_a__", "__iso_b__"


def main():
    c = get_client()
    for t in (A, B):
        delete_tenant(c, t)
    ingest_chunks(["Секрет тенанта А: красный код возврата товара."], "a.txt", A)
    ingest_chunks(["Секрет тенанта Б: синий код гарантии товара."], "b.txt", B)

    ha = search("код товара", tenant_id=A, top_k=5)
    hb = search("код товара", tenant_id=B, top_k=5)
    assert ha and all(h["source"] == "a.txt" for h in ha), [h["source"] for h in ha]
    assert hb and all(h["source"] == "b.txt" for h in hb), [h["source"] for h in hb]

    for t in (A, B):
        delete_tenant(c, t)
    print("OK: tenant isolation — A sees only A, B sees only B")


if __name__ == "__main__":
    main()
