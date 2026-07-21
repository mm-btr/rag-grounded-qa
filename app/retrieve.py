"""Retrieval (the `search` tool): query -> hybrid (dense+sparse, RRF) -> rerank -> top-k.
The tenant filter is applied to EACH prefetch — the only place it isolates candidates."""
from qdrant_client import models
from config import COLLECTION, TOP_K_RETRIEVE, TOP_K_RERANK
from models import embed_texts, get_reranker
from ingest import point_id, get_client, _tenant_match, _read_filter


def _tenant_filter(tenant_id):
    return _read_filter([_tenant_match(tenant_id)])


def _tenant_docs(tenant_id):
    client = get_client()
    docs, off = {}, None
    while True:
        pts, off = client.scroll(collection_name=COLLECTION, scroll_filter=_tenant_filter(tenant_id),
                                 limit=512, offset=off, with_payload=["source", "label", "descr"])
        for p in pts:
            pl = p.payload or {}
            src = pl.get("source")
            if src:
                docs[src] = {"label": pl.get("label"), "descr": pl.get("descr")}
        if off is None:
            return docs


def _match_docs(docs, needle):
    """Label substring -> ALL docs carrying the matched labels; filename substring ->
    exactly ONE match; anything else -> None (unresolved)."""
    n = (needle or "").strip().lower()
    if not n:
        return None
    by_label = [s for s, info in docs.items()
                if info.get("label") and n in str(info["label"]).lower()]
    if by_label:
        return by_label
    by_file = [s for s in docs if n in s.lower()]
    return by_file if len(by_file) == 1 else None


def resolve_doc(tenant_id, doc):
    """Unresolved -> the caller searches unfiltered and shows the universe: widen, don't fail."""
    docs = _tenant_docs(tenant_id)
    return _match_docs(docs, doc), docs


def search(query, tenant_id, top_k=TOP_K_RERANK, sources=None):
    """`sources` narrows ON TOP of the tenant filter, which is always applied."""
    client = get_client()
    q = embed_texts([query])[0]
    sp = q["sparse"]
    must = [_tenant_match(tenant_id)]
    if sources:
        must.append(models.FieldCondition(key="source", match=models.MatchAny(any=list(sources))))
    flt = _read_filter(must)
    hits = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=q["dense"], using="dense", filter=flt, limit=TOP_K_RETRIEVE),
            models.Prefetch(
                query=models.SparseVector(indices=list(sp.keys()), values=list(sp.values())),
                using="sparse", filter=flt, limit=TOP_K_RETRIEVE,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=flt,            # belt-and-suspenders at the fusion stage
        limit=TOP_K_RETRIEVE,
        with_payload=True,
    ).points
    if not hits:
        return []
    reranker = get_reranker()
    pairs = [[query, h.payload["text"]] for h in hits]
    scores = [float(s) for s in reranker.predict(pairs)]
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{
        "text": h.payload["text"],
        "source": h.payload.get("source"),
        "doc_id": h.payload.get("doc_id"),
        "label": h.payload.get("label"),
        "chunk": h.payload.get("chunk"),
        "page": h.payload.get("page"),
        "section": h.payload.get("section"),
        "score": s,
    } for h, s in ranked]


def fetch_locators(tenant_id, refs):
    refs = list(refs)
    if not refs:
        return {}
    client = get_client()
    recs = client.retrieve(
        collection_name=COLLECTION,
        ids=[point_id(tenant_id, did, ch) for did, ch in refs],
        with_payload=True,
    )
    out = {}
    for r in recs:
        p = r.payload or {}
        if p.get("ingest_ready", True) is False:  # direct id lookup must respect staging too
            continue
        out[(p.get("doc_id"), p.get("chunk"))] = p
    return out
