"""
tools/save_to_kb.py
-------------------
Ingest approved articles back into ChromaDB.
Called after HITL approval — this is how the KB self-improves.

Chunking strategy:
- 400-token chunks with 50-token overlap
- Overlap prevents losing context at chunk boundaries
  (a sentence split across a chunk boundary would lose its meaning otherwise)
- tiktoken used for accurate token counting (not character counting)
"""

import os
import uuid
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tools.query_kb import invalidate_bm25
load_dotenv()


_collection = None
_tokenizer = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 400    # tokens
CHUNK_OVERLAP = 50  # tokens

def _get_collection():
    global _collection
    if _collection is None:
        db_path = os.getenv("CHROMA_DB_PATH", "./kb/chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION", "machinist_evergreen")

        client = chromadb.PersistentClient(path=db_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef
        )
    return _collection


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping token chunks.
    Returns list of text strings, each ~CHUNK_SIZE tokens.
    """
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

def save_to_kb(text: str, source: str, metadata: dict | None = None) -> bool:
    """
    Chunk and ingest a text document into the KB.

    Args:
        text: Full text to ingest (markdown or plain text)
        source: Identifier for this document (e.g. filename or URL)
        metadata: Optional dict of extra metadata stored alongside the chunk

    Returns:
        True on success, False on failure.
    """

    collection = _get_collection()
    chunks = _chunk_text(text)

    if not chunks:
        print(f"[save_to_kb] No chunks produced for source: {source}")
        return False

    base_meta = {"source": source}
    if metadata:
        base_meta.update(metadata)

    try:
        ids = [f"{source}-chunk-{i}-{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
        metas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metas,
        )
        print(f"[save_to_kb] Ingested {len(chunks)} chunks from: {source}")
        invalidate_bm25()
        return True

    except Exception as e:
        print(f"[save_to_kb] ERROR: {e}")
        return False