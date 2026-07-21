"""Lazy singletons for the heavy models (load once per process)."""
from functools import lru_cache
from config import EMBED_MODEL, RERANK_MODEL, RERANK_MAX_SEQ, INJECTION_GUARD_MODEL


_INJECTION_MAX_TOKENS = 512
_INJECTION_OVERLAP = 64
_BENIGN_LABELS = {"BENIGN", "LABEL_0", "SAFE", "NEGATIVE"}


@lru_cache(maxsize=1)
def get_embedder():
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(EMBED_MODEL, use_fp16=False)  # CPU -> fp16 off


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    # max_length caps the query+passage pair; the 8192 default is far past our <=512-token chunks.
    return CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_SEQ)


def embed_texts(texts):
    model = get_embedder()
    out = model.encode(
        texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    return [
        {
            "dense": out["dense_vecs"][i].tolist(),
            "sparse": {int(k): float(v) for k, v in out["lexical_weights"][i].items()},
        }
        for i in range(len(texts))
    ]


def warmup():
    # The first user query must not pay model load + cold torch forward.
    embed_texts(["warmup"])
    get_reranker().predict([["warmup", "warmup"]])


@lru_cache(maxsize=1)
def get_injection_guard():
    from transformers import pipeline
    return pipeline("text-classification", model=INJECTION_GUARD_MODEL, top_k=None)


def _injection_windows(text, guard):
    # Overlapping windows over the guard's own tokens: an instruction spanning a boundary
    # stays visible in at least one window.
    tokenizer = guard.tokenizer
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    special = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = _INJECTION_MAX_TOKENS - special
    if capacity <= _INJECTION_OVERLAP:
        raise RuntimeError("Prompt Guard tokenizer leaves no usable window")
    if len(token_ids) <= capacity:
        return [text]
    step = capacity - _INJECTION_OVERLAP
    windows, start = [], 0
    while True:
        windows.append(tokenizer.decode(
            token_ids[start:start + capacity], skip_special_tokens=True
        ))
        if start + capacity >= len(token_ids):
            return windows
        start += step


def _score_groups(raw):
    if not raw:
        return []
    return [raw] if isinstance(raw[0], dict) else raw


def injection_score(text):
    """Injection probability [0,1]; the maximum non-benign score across windows wins."""
    guard = get_injection_guard()
    windows = _injection_windows(text, guard)
    inputs = windows[0] if len(windows) == 1 else windows
    raw = guard(inputs, truncation=True, max_length=_INJECTION_MAX_TOKENS)
    return max(
        (float(s["score"]) for scores in _score_groups(raw) for s in scores
         if s["label"].upper() not in _BENIGN_LABELS),
        default=0.0,
    )
