"""Dense/BM25 ranking and reciprocal rank fusion over CSWP units."""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from agent.retrieval.chunks import EvalChunk

RRF_CONSTANT = 60
CANDIDATE_K = 20


def rank_dense(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: list[EvalChunk],
) -> list[dict[str, Any]]:
    scores = np.asarray(chunk_embeddings @ query_embedding, dtype=np.float64)
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index].chunk_id),
    )
    return [
        {
            "chunk_id": chunks[index].chunk_id,
            "source": chunks[index].source,
            "chunk_index": chunks[index].ordinal,
            "native_score": round(float(scores[index]), 8),
            "rank": rank,
        }
        for rank, index in enumerate(order, start=1)
    ]


def rank_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: list[EvalChunk],
) -> list[dict[str, Any]]:
    scores = bm25.get_scores(query.lower().split())
    order = np.argsort(scores)[::-1]
    ranked: list[dict[str, Any]] = []
    for raw_index in order:
        index = int(raw_index)
        score = float(scores[index])
        if score == 0.0:
            continue
        chunk = chunks[index]
        ranked.append(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "chunk_index": chunk.ordinal,
                "native_score": round(score, 8),
                "rank": len(ranked) + 1,
            }
        )
    return ranked


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    bm25: list[dict[str, Any]],
    *,
    candidate_k: int = CANDIDATE_K,
    rrf_constant: int = RRF_CONSTANT,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for channel, rows in (("dense", dense[:candidate_k]), ("bm25", bm25[:candidate_k])):
        for zero_rank, row in enumerate(rows):
            entry = by_id.setdefault(
                row["chunk_id"],
                {
                    "chunk_id": row["chunk_id"],
                    "source": row["source"],
                    "chunk_index": row["chunk_index"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                },
            )
            entry["rrf_score"] += 1.0 / (rrf_constant + zero_rank)
            entry[f"{channel}_rank"] = zero_rank + 1
    ranked = sorted(
        by_id.values(),
        key=lambda row: (-row["rrf_score"], row["chunk_id"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["rrf_score"] = round(row["rrf_score"], 8)
    return ranked
