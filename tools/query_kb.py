"""
Qdrant-backed knowledge base query with BM25 hybrid retrieval.

Replaces ChromaDB with Qdrant as the dense retrieval backend.
Public interface is identical to the ChromaDB version:
    query_kb(query, n_results) → list[dict]
    invalidate_bm25()

Retrieval strategy: two-stage fusion
    1. Dense retrieval via Qdrant (cosine similarity on all-MiniLM-L6-v2 embeddings)
    2. BM25 retrieval over in-memory index rebuilt from Qdrant's stored payload
    3. Reciprocal Rank Fusion (k=60) — merges ranked lists without score normalization

Why hybrid matters here:
    Dense-only misses exact technical terms. "interrupt()" in a seed doc won't reliably
    surface when the query says "human approval checkpoint" — embeddings compress meaning
    and lose token precision. BM25 catches these gaps.

Why RRF instead of score averaging:
    BM25 scores are unbounded; cosine distances are 0–2. Averaging them is meaningless.
    RRF uses only rank position — each doc gets 1/(k+rank) from each ranker, summed.
    k=60 is standard — it prevents rank-1 from dominating the fusion.

BM25 lifecycle:
    Built lazily from Qdrant's stored documents on first query.
    Invalidated by save_to_kb when new documents are added.
    Rebuilt automatically on next query. Never persisted to disk.

Why Qdrant over ChromaDB:
    - Runs in Docker — local dev environment matches production exactly
    - Payload filtering — filter by topic, doc type, date without reloading all docs
    - Scroll API — paginate large collections without loading everything into memory
    - Concurrent writes are safe — ChromaDB's PersistentClient has known issues
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from observability.logger import get_logger

log = get_logger("query_kb")


# Module-level singletons
_client: QdrantClient | None = None
_encoder: SentenceTransformer | None = None
_bm25 = None
_bm25_docs: list[str] = []
_bm25_metas: list[dict] = []

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _client = QdrantClient(url=url)
    return _client


def _get_encoder() -> SentenceTransformer:
    """
    Load all-MiniLM-L6-v2 once and reuse.
    Same model used in save_to_kb.py (Must be in sync).
    Embedding dimension: 384.
    """
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "machinist_evergreen")


def _collection_exists() -> bool:
    """Return True if _collection exists and has at least one point."""
    client = _get_client()
    try:
        info = client.get_collection(_collection_name())
        return info.points_count > 0
    except Exception:
        return False


def _count() -> int:
    """Return total no. of points in the collection. 0 if collection is missing."""
    client = _get_client()
    try:
        info = client.get_collection(_collection_name())
        return info.points_count
    except Exception:
        return 0


# BM25

def _build_bm25() -> None:
    """
    Build BM25 index from all documents currently in Qdrant.

    Uses Qdrant's scroll API to page through all points.
    Scroll is the correct approach for full-collection reads —
    it handles any collection size without loading everything at once.

    Why not store the BM25 index on disk?
        The index is a pure function of the stored documents.
        Rebuilding is cheap at current collection size (~160 chunks).
        Persisting adds complexity with no benefit until >10k chunks.
    """
    global _bm25, _bm25_docs, _bm25_metas
    from rank_bm25 import BM25Okapi

    client = _get_client()
    collection = _collection_name()

    if not _collection_exists():
        _bm25 = None
        _bm25_docs = []
        _bm25_metas = []
        return

    # Scroll through all points, page_size=100
    # with_payload=True to get text and metadata
    # with_vectors=False - we only need text for BM25

    all_docs: list[str] = []
    all_metas: list[dict] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            payload = point.payload or {}
            text = payload.get("text", "")
            if text:
                all_docs.append(text)
                all_metas.append({
                    "source": payload.get("source", "unknown"),
                    "chunk_index": payload.get("chunk_index", 0),
                })
        if next_offset is None:
            break
        offset = next_offset


    _bm25_docs = all_docs
    _bm25_metas = all_metas

    if not all_docs:
        _bm25 = None
        return

    # Tokenize
    # A proper tokenizer (spaCy, NLTK) would add latency with marginal gain here.
    tokenized = [doc.lower().split() for doc in all_docs]
    _bm25 = BM25Okapi(tokenized)



def invalidate_bm25() -> None:
    """
    Called by save_to_kb after adding new documents.
    Forces BM25 rebuild on next query.
    """
    global _bm25
    _bm25 = None



def _bm25_query(query: str, n_results: int, *, include_scores: bool = False) -> list[dict]:
    """
    Run BM25 retrieval over the in-memory index.
    Returns top-n results as {text, source, distance} dicts.
    distance is set to 0.0 — BM25 doesn't produce distances, only scores.
    `include_scores` is reserved for evaluator diagnostics; normal hybrid retrieval
    deliberately retains its existing public result shape.
    """
    global _bm25, _bm25_docs, _bm25_metas

    if _bm25 is None:
        _build_bm25()

    if _bm25 is None:   # still None = collection empty
        return []

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    # Get top-n indices by score
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:n_results]

    results = []
    for idx in top_indices:
        if scores[idx] == 0.0:
            # BM25 score of 0 = no term overlap at all — not worth including
            continue
        result = {
            "text": _bm25_docs[idx],
            "source": _bm25_metas[idx].get("source", "unknown"),
            "chunk_index": _bm25_metas[idx].get("chunk_index", 0),
            "distance": 0.0,
        }
        if include_scores:
            result["bm25_score"] = float(scores[idx])
        results.append(result)

    return results



def _reciprocal_rank_fusion(
                                dense_results: list[dict],
                                bm25_results: list[dict],
                                k: int = 60,
                                n: int = 5,
                            ) -> list[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    Args:
        dense_results: Ranked list from Qdrant DB (best first)
        bm25_results:  Ranked list from BM25 (best first)
        k:             RRF damping constant (60 is standard)
        n:             Number of results to return

    How it works:
        Each document gets score = sum of 1/(k + rank) across both rankers.
        Rank is 0-indexed. A document ranked #1 by dense and #3 by BM25 gets:
            1/(60+0) + 1/(60+2) = 0.01667 + 0.01613 = 0.03280
        A document ranked #1 by dense only gets:
            1/(60+0) = 0.01667
        Fusion rewards documents that appear highly in BOTH lists.

    Returns:
        Top-n dicts with keys: text, source, distance, rrf_score
    """
    scores: dict[str, float] = {}
    # Index dense results by text content (text is the stable identity — no IDs here)
    # Why text and not ID? query() doesn't return IDs by default; get() does.
    # Using text as key is safe because chunks are unique within a collection.
    dense_index = {r["text"]: r for r in dense_results}
    bm25_index = {r["text"]: r for r in bm25_results}

    all_texts = set(dense_index) | set(bm25_index)

    for text in all_texts:
        score = 0.0
        if text in dense_index:
            rank = dense_results.index(dense_index[text])
            score += 1.0 / (k + rank)
        if text in bm25_index:
            rank = bm25_results.index(bm25_index[text])
            score += 1.0 / (k + rank)
        scores[text] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    output: list[dict] = []
    for text, rrf_score in ranked[:n]:
        base = dense_index.get(text) or bm25_index.get(text)
        output.append({
            "text": text,
            "source": base["source"],
            "chunk_index": base.get("chunk_index", 0),
            "distance": base.get("distance", 0.0),   # keep original distance for logging
            "rrf_score": round(rrf_score, 5),
        })

    return output

def _dense_query(query: str, n_results: int) -> list[dict] | None:
    """Return raw dense candidates, or None when dense retrieval fails."""
    encoder = _get_encoder()
    client = _get_client()
    collection = _collection_name()
    try:
        query_vector = encoder.encode(query).tolist()
        search_results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=n_results,
            with_payload=True,
        )
        dense_results: list[dict] = []
        for hit in search_results:
            payload = hit.payload or {}
            # Qdrant returns score (cosine similarity, higher = better)
            # Convert to distance convention (lower = better) for RRF compatibility:
            # distance = 1 - score (cosine similarity ∈ [-1,1], distance ∈ [0,2])
            distance = round(1.0 - hit.score, 4)
            dense_results.append({
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "chunk_index": payload.get("chunk_index", 0),
                "distance": distance,
                "dense_similarity": round(float(hit.score), 4),
            })
    except Exception as e:
        log.error("query_kb.search_error", error=str(e))
        return None

    return dense_results


def _without_dense_diagnostics(results: list[dict]) -> list[dict]:
    """Keep query_kb()'s existing result shape when dense-only fallback is used."""
    return [
        {key: value for key, value in result.items() if key != "dense_similarity"}
        for result in results
    ]


def _retrieve_components(query: str, n_results: int, *, include_bm25_scores: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """Retrieve hybrid results plus raw dense/BM25 components for diagnostics."""
    if not _collection_exists():
        return [], [], []

    fetch_n = min(n_results * 2, _count())
    dense_results = _dense_query(query, fetch_n)
    if dense_results is None:
        return [], [], []

    # BM25 retrieval
    try:
        bm25_results = _bm25_query(
            query,
            n_results=fetch_n,
            include_scores=include_bm25_scores,
        )
    except ImportError:
        log.warning("query_kb.bm25_unavailable",
                    note="rank_bm25 not installed — falling back to dense-only")
        return _without_dense_diagnostics(dense_results[:n_results]), dense_results, []
    except Exception as e:
        log.error("query_kb.bm25_error", error=str(e),
                  note="falling back to dense-only")
        return _without_dense_diagnostics(dense_results[:n_results]), dense_results, []

    if not bm25_results:
        return _without_dense_diagnostics(dense_results[:n_results]), dense_results, []

    hybrid_results = _reciprocal_rank_fusion(dense_results, bm25_results, k=60, n=n_results)
    return hybrid_results, dense_results, bm25_results


# Public interface

def query_kb(query: str, n_results: int = 5) -> list[dict]:
    """Hybrid Qdrant dense + BM25 retrieval, fused via unchanged RRF(k=60)."""
    hybrid_results, _, _ = _retrieve_components(query, n_results)
    return hybrid_results


def query_kb_diagnostics(query: str, n_results: int = 5) -> dict:
    """Return raw ranker signals for evaluator diagnostics without changing query_kb().

    The mixed RRF result's `distance` is not an OOS confidence score: BM25-only
    candidates carry a synthetic 0.0 distance. This private diagnostic surface
    exposes ranker-native values so calibration can be evaluated separately.
    """
    hybrid_results, dense_results, bm25_results = _retrieve_components(
        query,
        n_results,
        include_bm25_scores=True,
    )
    dense_top = dense_results[0] if dense_results else None
    bm25_top = bm25_results[0] if bm25_results else None

    return {
        "dense_top1_distance": dense_top.get("distance") if dense_top else None,
        "dense_top1_similarity": dense_top.get("dense_similarity") if dense_top else None,
        "bm25_top_score": bm25_top.get("bm25_score") if bm25_top else None,
        "hybrid_top1_rrf_score": hybrid_results[0].get("rrf_score") if hybrid_results else None,
        "hybrid_results": [
            {
                "rank": rank,
                "source": result.get("source"),
                "chunk_index": result.get("chunk_index"),
                "rrf_score": result.get("rrf_score"),
            }
            for rank, result in enumerate(hybrid_results, start=1)
        ],
    }


def warmup() -> dict:
    """
       Eagerly load the encoder and build the BM25 index before the pipeline runs.

       Why: both are one-time per-process costs (~8s model load, ~4s torch import
       happens at module import). Paying them inside the first retrieve_node call
       pollutes latency_ms.retrieve_kb, which must measure steady-state query cost.
       Called by main.py before graph invocation. Returns timings for logging.
       """
    import time
    t0 = time.perf_counter()
    _get_encoder()
    t1 = time.perf_counter()
    if _bm25 is None:
        _build_bm25()
    t2 = time.perf_counter()
    return {
        "encoder_load_ms": int((t1-t0) * 1000),
        "bm25_build_ms": int((t2-t1) * 1000),
    }














