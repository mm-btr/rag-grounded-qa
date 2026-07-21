"""Docling parsing + structure-aware chunking (PDF/docx/txt), CPU-only, OCR off.
Tables are serialized as markdown grids and never share a chunk with prose."""
from config import EMBED_MODEL, CHUNK_MAX_TOKENS

_converter = None
_chunker = None
_tokenizer = None


def _build():
    global _converter, _chunker, _tokenizer
    if _converter is not None:
        return
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    pdf = PdfPipelineOptions()
    pdf.do_ocr = False
    pdf.generate_page_images = False
    pdf.do_table_structure = True
    pdf.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)

    class _MarkdownTables(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())

    _converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf)}
    )
    _tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL),
        max_tokens=CHUNK_MAX_TOKENS,
    )
    _chunker = HybridChunker(
        tokenizer=_tokenizer,
        merge_peers=False,                   # prose re-merged in _merge_prose; tables stay alone
        repeat_table_header=True,            # a split table keeps column headers in every part
        serializer_provider=_MarkdownTables(),
    )


def _merge_prose(rows, count_tokens, max_tokens):
    # A table never merges with anything, so its vector stays undiluted.
    out = []
    for r in rows:
        prev = out[-1] if out else None
        if (
            prev is not None
            and not r["is_table"] and not prev["is_table"]
            and r["section"] == prev["section"]
            and count_tokens(prev["embed_text"] + "\n" + r["text"]) <= max_tokens
        ):
            prev["text"] += "\n" + r["text"]
            prev["embed_text"] += "\n" + r["text"]   # headings prefix already sits in prev
            if prev["page"] is None:
                prev["page"] = r["page"]
            continue
        out.append(dict(r))
    return out


def parse_and_chunk(path):
    """-> [{text, embed_text, page, section}]; embed_text is heading-contextualized."""
    _build()
    from docling_core.types.doc import DocItemLabel

    doc = _converter.convert(path).document
    rows = []
    for chunk in _chunker.chunk(dl_doc=doc):
        pages = sorted({p.page_no for it in chunk.meta.doc_items for p in it.prov})
        headings = chunk.meta.headings or []
        rows.append({
            "text": chunk.text,
            "embed_text": _chunker.contextualize(chunk),
            "page": pages[0] if pages else None,
            "section": " > ".join(headings) if headings else None,
            "is_table": any(getattr(it, "label", None) == DocItemLabel.TABLE
                            for it in chunk.meta.doc_items),
        })
    rows = _merge_prose(rows, _tokenizer.count_tokens, CHUNK_MAX_TOKENS)
    return [{k: r[k] for k in ("text", "embed_text", "page", "section")} for r in rows]
