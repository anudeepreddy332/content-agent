"""
ChromaDB local knowledge base query with BM25 hybrid retrieval.

Retrieval strategy: two-stage fusion
    1. Dense retrieval (ChromaDB cosine similarity) — captures semantic meaning
    2. BM25 retrieval — captures exact term matches (critical for precise technical terms)
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
    Built lazily at first query from ChromaDB's stored documents.
    Invalidated (set to None) by save_to_kb when new documents are added.
    Rebuilt automatically on next query. Never persisted to disk — always rebuilt from source.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

_collection = None
_bm25 = None
_bm25_docs = []
_bm25_metas = []

def _get_collection():
    global _collection
    if _collection is None:
        db_path = os.getenv("CHROMA_DB_PATH", "./kb/chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION", "machinist_evergreen")

        client = chromadb.PersistentClient(path=db_path)

        # Local embedding model — free, fast, no API calls
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # get_or_create: safe to call even if collection doesn't exist yet
        _collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"description": "Evergreen AI/ML concepts for themachinist.org"},
        )

    return _collection



def _build_bm25():
    """
    Build BM25 index from all documents currently in ChromaDB.

    Why rebuild from ChromaDB instead of from disk?
        ChromaDB is the source of truth — it may contain documents added via
        save_to_kb (approved articles) that are not in seed_docs/. Building
        from disk would miss those.

    Failure modes:
        - rank_bm25 not installed: raises ImportError — caught in query_kb
        - ChromaDB empty: returns None — query_kb handles this gracefully
        - ChromaDB fetch fails: raises, caught in query_kb
    """
    global _bm25, _bm25_docs, _bm25_metas
    from rank_bm25 import BM25Okapi

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        _bm25 = None
        _bm25_docs = []
        _bm25_metas = []
        return

    # Fetch ALL documents from ChromaDB in one call
    # Why not paginate? At 20 seed docs × ~8 chunks each = ~160 chunks.
    # Even at 1000 chunks, this fits comfortably in memory.
    # Revisit pagination only when collection exceeds ~10k chunks.

    all_docs = collection.get(include=["documents", "metadatas"])

    _bm25_docs = all_docs["documents"]
    _bm25_metas = all_docs["metadatas"]

    # Tokenize
    # A proper tokenizer (spaCy, NLTK) would add latency with marginal gain here.
    tokenized = [doc.lower().split() for doc in _bm25_docs]
    _bm25 = BM25Okapi(tokenized)



def invalidate_bm25():
    """
    Called by save_to_kb after adding new documents.
    Forces rebuild on next query.
    """
    global _bm25
    _bm25 = None


def _reciprocal_rank_fusion(
                                dense_results: list[dict],
                                bm25_results: list[dict],
                                k: int = 60,
                                n: int = 5,
                            ) -> list[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    Args:
        dense_results: Ranked list from ChromaDB (best first)
        bm25_results:  Ranked list from BM25 (best first)
        k:             RRF damping constant (60 is standard)
        n:             Number of results to return

    How it works:
        Each document gets score = sum of 1/(k + rank) across all rankers.
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
            "distance": base.get("distance", 0.0),   # keep original distance for logging
            "rrf_score": round(rrf_score, 5),
        })

    return output


def _bm25_query(query: str, n_results: int) -> list[dict]:
    """
    Run BM25 retrieval over the in-memory index.
    Returns top-n results as {text, source, distance} dicts.
    distance is set to 0.0 — BM25 doesn't produce distances, only scores.
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
        results.append({
            "text": _bm25_docs[idx],
            "source": _bm25_metas[idx].get("source", "unknown"),
            "distance": 0.0,
        })

    return results


def query_kb(query: str, n_results: int = 5) -> list[dict]:
    """
    Hybrid query: dense retrieval + BM25, fused via RRF.

    Args:
        query: Natural language query
        n_results: Number of results to return after fusion

    Returns:
        List of {text, source, distance, rrf_score} dicts, best first.
        Returns empty list if KB has no documents.

    Failure modes:
        - rank_bm25 not installed: falls back to dense-only, logs warning
        - BM25 build fails: falls back to dense-only
        - Collection empty: returns []
        - ChromaDB query fails: returns [], logs error
    """
    collection = _get_collection()

    # Guard: if KB is empty, return gracefully
    if collection.count() == 0:
        return []

    # Dense retrieval

    try:
        dense_raw = collection.query(
            query_texts=[query],
            n_results=min(n_results * 2, collection.count()),   # fetch 2x, fusion trims to n
            include=["documents", "metadatas", "distances"],
        )
        dense_results = []
        docs = dense_raw["documents"][0]
        metas = dense_raw["metadatas"][0]
        dists = dense_raw["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            dense_results.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": round(dist, 4),
            })

    except Exception as e:
        print(f"[query_kb] Dense retrieval ERROR: {e}")
        return []


    # BM 25 retrieval
    try:
        bm25_results = _bm25_query(query, n_results=n_results * 2)
    except ImportError:
        print("[query_kb] WARNING: rank_bm25 not installed — falling back to dense-only")
        return dense_results[:n_results]
    except Exception as e:
        print(f"[query_kb] BM25 ERROR: {e} — falling back to dense-only")
        return dense_results[:n_results]

    # Fusion
    if not bm25_results:
        # BM25 found nothing (all scores 0) — dense only
        return dense_results[:n_results]

    return _reciprocal_rank_fusion(dense_results, bm25_results, k=60, n=n_results)
