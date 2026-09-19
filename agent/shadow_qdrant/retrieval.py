"""Shadow Qdrant dense + BM25 hybrid retrieval with CSWP expansion/packing."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from rank_bm25 import BM25Okapi

from agent.drafter_packed_evidence import (
    DRAFTER_PACKED_EVIDENCE_V1,
    packed_identity_fingerprint,
    serialize_drafter_packed_evidence_v1,
)
from agent.retrieval.chunks import EvalChunk
from agent.retrieval.encoder import get_encoder
from agent.retrieval.expansion import (
    PACK_BUDGET_CL100K,
    assert_seed_preservation_invariant,
    cl100k,
    dedupe_groups,
    eval_chunks_from_manifest,
    expand_seeds,
    flatten,
    pack_units_seed_first,
)
from agent.retrieval.fusion import (
    CANDIDATE_K,
    RRF_CONSTANT,
    rank_bm25,
    reciprocal_rank_fusion,
)
from agent.shadow_qdrant.constants import PRODUCTION_INDEX_DIR
from agent.shadow_qdrant.index import load_index_manifest, load_shadow_corpus

DENSE_TOP_K = 20
SEED_TOP_K = 5


class ShadowRetrievalError(RuntimeError):
    """Shadow Qdrant retrieval failed."""


def clear_shadow_retrieval_cache() -> None:
    _shadow_index_state.cache_clear()


@lru_cache(maxsize=4)
def _shadow_index_state(client_key: str, index_dir_str: str):
    del client_key  # cache key only; client passed per call
    index_dir = Path(index_dir_str)
    shadow_manifest = load_index_manifest(index_dir)
    representation, units, by_source = load_shadow_corpus(index_dir)
    chunks = eval_chunks_from_manifest(representation)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    texts = [chunk.retrieval_text for chunk in chunks]
    bm25 = BM25Okapi([text.lower().split() for text in texts])
    ordinal_by_id = {chunk.chunk_id: chunk.ordinal for chunk in chunks}
    return shadow_manifest, units, by_source, chunks, chunks_by_id, bm25, ordinal_by_id


def _rank_dense_qdrant(
    client: QdrantClient,
    collection: str,
    query_embedding: np.ndarray,
    chunks: list[EvalChunk],
    *,
    limit: int = DENSE_TOP_K,
) -> list[dict[str, Any]]:
    hits = client.search(
        collection_name=collection,
        query_vector=query_embedding.tolist(),
        limit=limit,
        with_payload=["chunk_id"],
        search_params=SearchParams(exact=True),
    )
    ranked: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        chunk_id = hit.payload["chunk_id"]
        chunk = next(c for c in chunks if c.chunk_id == chunk_id)
        ranked.append(
            {
                "chunk_id": chunk_id,
                "source": chunk.source,
                "chunk_index": chunk.ordinal,
                "native_score": round(float(hit.score), 8),
                "rank": rank,
            }
        )
    return ranked


def _hybrid_top5_shadow(
    client: QdrantClient,
    query: str,
    *,
    index_dir: Path = PRODUCTION_INDEX_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    client_key = str(id(client))
    (
        shadow_manifest,
        units,
        by_source,
        chunks,
        chunks_by_id,
        bm25,
        ordinal_by_id,
    ) = _shadow_index_state(client_key, str(index_dir))

    encoder = get_encoder()
    query_embedding = np.asarray(encoder.encode(query), dtype=np.float32)
    dense = _rank_dense_qdrant(
        client,
        shadow_manifest["collection_name"],
        query_embedding,
        chunks,
        limit=DENSE_TOP_K,
    )
    bm25_rows = rank_bm25(query, bm25, chunks)
    hybrid = reciprocal_rank_fusion(
        dense,
        bm25_rows,
        candidate_k=CANDIDATE_K,
        rrf_constant=RRF_CONSTANT,
    )

    seed_texts = [chunks_by_id[row["chunk_id"]].retrieval_text for row in hybrid[:SEED_TOP_K]]
    seed_embeddings = np.asarray(encoder.encode(seed_texts), dtype=np.float32)
    seeds: list[dict[str, Any]] = []
    for index, row in enumerate(hybrid[:SEED_TOP_K]):
        seeds.append(
            {
                "rank": row["rank"],
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "rrf_score": row["rrf_score"],
                "dense_rank": row.get("dense_rank"),
                "bm25_rank": row.get("bm25_rank"),
                "distance": round(
                    1.0 - float(np.dot(seed_embeddings[index], query_embedding)),
                    4,
                ),
            }
        )

    return seeds, hybrid, dense


def retrieve_shadow_kb(
    client: QdrantClient,
    query: str,
    *,
    index_dir: Path = PRODUCTION_INDEX_DIR,
    n_seeds: int = 5,
) -> dict[str, Any]:
    """Shadow retrieval: Qdrant exact dense + BM25 → RRF → expand → seed-first pack."""

    if n_seeds != 5:
        raise ShadowRetrievalError("shadow contract requires top-5 seeds")

    client_key = str(id(client))
    shadow_manifest, units, by_source, _chunks, _chunks_by_id, _bm25, _ord = (
        _shadow_index_state(client_key, str(index_dir))
    )

    retrieval_seeds, hybrid, dense = _hybrid_top5_shadow(
        client, query, index_dir=index_dir
    )
    groups = expand_seeds(retrieval_seeds, units, by_source)
    expanded_groups, _dup_ids, _additional = dedupe_groups(groups)
    expanded = flatten(expanded_groups)
    encoding = cl100k()
    packed, skipped, used, exhausted, pack_stats = pack_units_seed_first(
        expanded, units, PACK_BUDGET_CL100K, encoding
    )
    assert_seed_preservation_invariant(
        expanded, units, PACK_BUDGET_CL100K, encoding
    )
    serialized = serialize_drafter_packed_evidence_v1(packed, units)
    fingerprint = packed_identity_fingerprint(serialized)

    return {
        "retrieval_seeds": retrieval_seeds,
        "dense_top20": dense,
        "hybrid": hybrid,
        "packed_rows": packed,
        "serialized_groups": serialized,
        "packed_fingerprint": fingerprint,
        "pack_stats": pack_stats,
        "used_cl100k": used,
        "skipped": skipped,
        "exhausted": exhausted,
        "collection_name": shadow_manifest["collection_name"],
        "qualified_contract": DRAFTER_PACKED_EVIDENCE_V1,
        "provider_calls": 0,
        "qdrant_reads": 1,
        "qdrant_writes": 0,
    }
