import os

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION", "docs")

EMBED_MODEL = "BAAI/bge-m3"          # dense + sparse (ColBERT head off)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_DIM = 1024

CHUNK_MAX_TOKENS = 512               # Docling HybridChunker window, in BGE-M3 tokens
INGEST_BATCH_SIZE = 64

TOP_K_RETRIEVE = 10                  # candidates to rerank (cross-encoder cost is linear in pairs)
TOP_K_RERANK = 5
RERANK_MAX_SEQ = 512                 # cap for the query+passage pair

# Set at create time (changing later = full reindex).
HNSW_M = 0
HNSW_PAYLOAD_M = 16
DEFAULT_TENANT = os.environ.get("DEFAULT_TENANT", "default")

LLM_MODEL = os.environ.get("LLM_MODEL", "openai:gpt-5.4-mini")
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "medium")  # measured: high over-searches
                                     # negatives instead of refusing
MEMORY_WINDOW = int(os.environ.get("MEMORY_WINDOW", "50"))


def _postgres_url():
    url = os.environ.get("POSTGRES_URL")
    if url:
        return url
    from urllib.parse import quote
    user = quote(os.environ.get("POSTGRES_USER", "postgres"), safe="")
    password = quote(os.environ.get("POSTGRES_PASSWORD", "postgres"), safe="")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "ragdb")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


POSTGRES_URL = _postgres_url()

SEARCH_BUDGET = int(os.environ.get("SEARCH_BUDGET", "5"))                   # searches/turn
AGENT_RECURSION_LIMIT = int(os.environ.get("AGENT_RECURSION_LIMIT", "30"))  # safety net above SEARCH_BUDGET
AGENT_TIMEOUT = float(os.environ.get("AGENT_TIMEOUT", "600"))               # safety net vs a true hang
TELEGRAM_LIMIT = 4096
MAX_UPLOAD_BYTES = 20 * 1024 * 1024                                         # Telegram getFile cap

INJECTION_GUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
INJECTION_THRESHOLD = float(os.environ.get("INJECTION_THRESHOLD", "0.9"))   # high -> fewer false rejections
                                     # of legit docs

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
STT_MODEL = os.environ.get("STT_MODEL", "scribe_v2")   # language auto-detected, not pinned
STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
STT_TIMEOUT = float(os.environ.get("STT_TIMEOUT", "120"))
