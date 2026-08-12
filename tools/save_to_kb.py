"""
Ingest documents into Qdrant knowledge base.

Replaces ChromaDB with Qdrant as the storage backend.
Public interface is identical:
    save_to_kb(text, source, metadata) → bool

Chunking strategy:
    224 content-token children with 32-token overlap, measured with the
    all-MiniLM-L6-v2 tokenizer.  The model adds two special tokens to each
    embedded child, keeping every vector input within its 256-token limit.

Embedding model: all-MiniLM-L6-v2 (384 dimensions)
    Must stay in sync with query_kb.py — same model, same dimension.

Collection creation:
    Created automatically on first call if it doesn't exist.
    Uses cosine distance — consistent with how all-MiniLM-L6-v2 was trained.

Point IDs:
    Qdrant requires integer or UUID point IDs.
    Use uuid.uuid4() — no collision risk, no coordination needed.
"""

import os
import uuid

from dotenv import load_dotenv
from tools.query_kb import invalidate_bm25
load_dotenv()
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Module level singletons

_client: QdrantClient | None = None
_encoder: SentenceTransformer | None = None
CHUNK_SIZE = 224
CHUNK_OVERLAP = 32

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _client = QdrantClient(url=url)

    return _client


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "machinist_evergreen")


def _ensure_collection() -> None:
    """
    Create the Qdrant collection if it doesn't exist.
    Safe to call on every ingest - no-ops if already exists.

    Vector config:
        size=384    — all-MiniLM-L6-v2 output dimension
        distance=Cosine — correct for normalized sentence embeddings

    """
    client = _get_client()
    collection = _collection_name()

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"[save_to_kb] Created Qdrant collection: {collection}")



def _chunk_text(text: str, encoder: SentenceTransformer | None = None) -> list[str]:
    """
    Split text into MiniLM-tokenizer-aligned overlapping child chunks.

    Offset mapping retains source text rather than decoding token IDs, which
    makes overlapping child text deterministic for post-retrieval expansion.
    """
    if not text.strip():
        return []

    encoder = encoder or _get_encoder()
    tokenizer = encoder.tokenizer
    special_token_overhead = tokenizer.num_special_tokens_to_add(pair=False)
    max_content_tokens = encoder.max_seq_length - special_token_overhead
    content_token_limit = min(CHUNK_SIZE, max_content_tokens)
    if content_token_limit <= CHUNK_OVERLAP:
        raise ValueError("Chunk overlap must be smaller than the content-token limit")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = [offset for offset in encoded["offset_mapping"] if offset[1] > offset[0]]
    chunks = []
    start = 0

    while start < len(offsets):
        # A final span no longer than the overlap is already contained in the
        # preceding child; indexing it would add only a duplicate sibling.
        if chunks and len(offsets) - start <= CHUNK_OVERLAP:
            break
        end = min(start + content_token_limit, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunks.append(text[char_start:char_end])
        start += content_token_limit - CHUNK_OVERLAP

    return chunks

def save_to_kb(text: str, source: str, metadata: dict | None = None) -> bool:
    """
    Chunk and ingest a text document into the Qdrant.

    Args:
        text: Full text to ingest (markdown or plain text)
        source: Identifier for this document (e.g. filename or URL)
        metadata: Optional dict stored as Qdrant payload alongside each chunk

    Returns:
        True on success, False on failure.

    What gets stored in each Qdrant point:
        vector:  384-dim embedding of the chunk text
        payload: {text, source, chunk_index, filename, type, ...metadata}

    Failure modes:
        - Qdrant not running: exception caught, returns False
        - Empty text: returns False immediately
        - Embedding failure: returns False
    """
    try:
        encoder = _get_encoder()
        chunks = _chunk_text(text, encoder)
        if not chunks:
            print(f"[save_to_kb] No chunks produced for source: {source}")
            return False
        _ensure_collection()
        client = _get_client()
        collection = _collection_name()
        embeddings = encoder.encode(chunks, show_progress_bar=False)

        base_payload = {"source": source}
        if metadata:
            base_payload.update(metadata)

        points: list[PointStruct] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = {
                **base_payload,
                "text": chunk,
                "chunk_index": i,
            }
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload,
            ))
        # Upsert in batches of 100 to avoid request size limits
        batch_size = 100
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start:batch_start + batch_size]
            client.upsert(collection_name=collection, points=batch)

        print(f"[save_to_kb] Ingested {len(chunks)} chunks from: {source}")
        invalidate_bm25()
        return True


    except Exception as e:
        print(f"[save_to_kb] ERROR: {e}")
        return False
