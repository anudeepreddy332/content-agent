"""
tools/query_kb.py
-----------------
ChromaDB local knowledge base query.
Returns evergreen AI/ML concept chunks for the retrieve_node.

Why local KB vs always hitting Tavily:
- Zero API cost for evergreen concepts (definitions, formulas, theory)
- No latency variance — local disk read is deterministic
- Tavily is for current facts; KB is for foundational depth
- Every approved article gets ingested back → KB improves with use
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

_collection = None

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


def query_kb(query: str, n_results: int = 5) -> list[dict]:
    """
    Query the local ChromaDB knowledge base.

    Args:
        query: Natural language query
        n_results: Number of chunks to return (default 5)

    Returns:
        List of {text, source, distance} dicts.
        distance is ChromaDB cosine distance — lower is more similar.
        Returns empty list if KB has no documents yet.
    """
    collection = _get_collection()

    # Guard: if KB is empty, return gracefully
    if collection.count() == 0:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": round(dist, 4),
            })

        return output
    except Exception as e:
        print(f"[query_kb] ERROR: {e}")
        return []

