"""Published-corpus dumps: Qdrant -> eval/corpus/ — corpus_flat.json (machine, check_gold.py reads it)
and per-document corpus_<source>.txt (eye-reading). Re-run after ANY corpus change — the
dumps go stale silently. Run inside the app image: python eval/dump_corpus.py [--tenant]."""
import argparse
import json
import os
import re
import sys

from qdrant_client import QdrantClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "app"))

from ingest import _read_filter, _tenant_match

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def fetch_points(url, collection, tenant):
    client = QdrantClient(url=url)
    flt = _read_filter([_tenant_match(tenant)])
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection_name=collection, scroll_filter=flt,
                                      limit=256, offset=offset, with_payload=True)
        points.extend(p.payload for p in batch)
        if offset is None:
            return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default=os.environ.get("DEFAULT_TENANT", "default"))
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://qdrant:6333"))
    ap.add_argument("--collection", default=os.environ.get("COLLECTION", "docs"))
    args = ap.parse_args()

    rows = fetch_points(args.url, args.collection, args.tenant)
    rows.sort(key=lambda p: (p.get("source") or "", p.get("chunk") or 0))
    os.makedirs(OUT_DIR, exist_ok=True)

    # Machine form: one chunk per line (insertion order = source/chunk) — diff-friendly.
    flat = {f"{p['doc_id']}#{p['chunk']}": p.get("text") or "" for p in rows}
    flat_path = os.path.join(OUT_DIR, "corpus_flat.json")
    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=1)
    print(f"written: {flat_path} ({len(flat)} chunks, tenant '{args.tenant}')")

    # Eye form: a txt per source document.
    by_src = {}
    for p in rows:
        by_src.setdefault(p.get("source") or "_unknown_", []).append(p)
    for src, pts in sorted(by_src.items()):
        slug = re.sub(r"[^\w.-]+", "_", os.path.splitext(src)[0])[:60]
        path = os.path.join(OUT_DIR, f"corpus_{slug}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for p in pts:
                head = f"##### {p['doc_id']}#{p['chunk']}"
                extras = []
                if p.get("page") is not None:
                    extras.append(f"page {p['page']}")
                if p.get("section"):
                    extras.append(f"section {p['section']}")
                if extras:
                    head += " (" + ", ".join(extras) + ")"
                f.write(head + "\n" + (p.get("text") or "") + "\n\n")
        print(f"written: {path} ({len(pts)} chunks)")


if __name__ == "__main__":
    main()
