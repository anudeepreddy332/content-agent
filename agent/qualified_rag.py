"""Production Contiguous Structural Window Packing (CSWP) qualified KB retrieval.

Hybrid dense (all-MiniLM-L6-v2) + BM25 (raw lexical tokenization) over frozen
CSWP retrieval units, Reciprocal Rank Fusion (RRF) top-5 seeds, bounded ±1
neighbor expansion, and seed-first packing under a 2000 cl100k_base evidence
budget. Does not read or write production Qdrant.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from agent.drafter_packed_evidence import (
    DRAFTER_PACKED_EVIDENCE_V1,
    packed_identity_fingerprint,
    serialize_drafter_packed_evidence_v1,
)
from observability.logger import get_logger
from scripts import phase5a2_shadow_ab as ab
from scripts import phase5b1_candidate_c as candidate_c
from scripts.phase5a0_baseline import (
    CANDIDATE_K,
    RRF_CONSTANT,
    reciprocal_rank_fusion,
)

log = get_logger("qualified_rag")

ROOT = Path(__file__).resolve().parent.parent
CSWP_MANIFEST = ROOT / "reports/phase5/phase5a2e/candidate_cswp_manifest.json"
CSWP_FINGERPRINT = (
    "439ccdf81e2aff1bc0bf3448734cb71f774bc8d3e7619dd1e4b51b2685b79bb3"
)
PACK_BUDGET_CL100K = candidate_c.PACK_BUDGET_CL100K
ENCODER_MODEL = "all-MiniLM-L6-v2"


class QualifiedRAGError(RuntimeError):
    """Qualified CSWP retrieval/pre/post contract failed."""


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    from scripts.phase5a0_baseline import resolve_local_model_snapshot

    snapshot = resolve_local_model_snapshot()
    return SentenceTransformer(str(snapshot), local_files_only=True)


def _source_offsets(source_path: str) -> tuple[int, ...]:
    return ab._source_offsets((ROOT / source_path).read_text(encoding="utf-8"))


def _eval_chunks_from_manifest(manifest: dict[str, Any]) -> list[ab.EvalChunk]:
    offsets = {
        unit["source_path"]: _source_offsets(unit["source_path"])
        for unit in manifest["children"]
    }
    chunks: list[ab.EvalChunk] = []
    for ordinal, unit in enumerate(manifest["children"]):
        intervals = tuple(
            interval
            for span in unit["source_spans"]
            if (
                interval := ab._normalize_interval(
                    span["source_char_start"],
                    span["source_char_end"],
                    *offsets[unit["source_path"]],
                )
            )
            is not None
        )
        chunks.append(
            ab.EvalChunk(
                ordinal=ordinal,
                chunk_id=unit["chunk_id"],
                source=Path(unit["source_path"]).stem,
                source_path=unit["source_path"],
                retrieval_text=unit["retrieval_text"],
                source_intervals=intervals,
                embedding_content_token_count=unit["embedding_content_token_count"],
            )
        )
    return chunks


@lru_cache(maxsize=1)
def _index():
    manifest = json.loads(CSWP_MANIFEST.read_text(encoding="utf-8"))
    if ab.sha256_json(manifest) != CSWP_FINGERPRINT:
        raise QualifiedRAGError("CSWP manifest fingerprint drift")
    _manifest_loaded, units, by_source = candidate_c.load_units()
    chunks = _eval_chunks_from_manifest(manifest)
    if len(chunks) != 159:
        raise QualifiedRAGError("CSWP unit count drift")
    encoder = _encoder()
    texts = [chunk.retrieval_text for chunk in chunks]
    embeddings = np.asarray(encoder.encode(texts), dtype=np.float32)
    bm25 = BM25Okapi([text.lower().split() for text in texts])
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return manifest, units, by_source, chunks, chunks_by_id, embeddings, bm25


def _rank_dense(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: list[ab.EvalChunk],
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


def _rank_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: list[ab.EvalChunk],
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


def _hybrid_top5(query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _manifest, _units, _by_source, chunks, chunks_by_id, embeddings, bm25 = _index()
    encoder = _encoder()
    query_embedding = np.asarray(encoder.encode(query), dtype=np.float32)
    dense = _rank_dense(query_embedding, embeddings, chunks)
    bm25_rows = _rank_bm25(query, bm25, chunks)
    hybrid = reciprocal_rank_fusion(
        dense,
        bm25_rows,
        candidate_k=CANDIDATE_K,
        rrf_constant=RRF_CONSTANT,
    )
    seeds: list[dict[str, Any]] = []
    for row in hybrid[:5]:
        chunk = chunks_by_id[row["chunk_id"]]
        seeds.append(
            {
                "rank": row["rank"],
                "chunk_id": row["chunk_id"],
                "source": row["source"],
                "rrf_score": row["rrf_score"],
                "dense_rank": row.get("dense_rank"),
                "bm25_rank": row.get("bm25_rank"),
                "distance": round(
                    1.0 - float(np.dot(embeddings[chunk.ordinal], query_embedding)),
                    4,
                ),
            }
        )
    return seeds, hybrid


def _packed_rows_to_kb_results(
    packed_rows: list[dict[str, Any]],
    units: dict[str, dict[str, Any]],
    retrieval_seeds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    serialized = serialize_drafter_packed_evidence_v1(packed_rows, units)
    seed_meta = {seed["rank"]: seed for seed in retrieval_seeds}
    kb_results: list[dict[str, Any]] = []
    packed_order = 0
    for group in serialized:
        for member in group["members"]:
            row = next(item for item in packed_rows if item["chunk_id"] == member["chunk_id"])
            seed = seed_meta.get(row["seed_rank"], {})
            kb_results.append(
                {
                    "text": member["serialized_text"],
                    "source": Path(member["source_path"]).stem,
                    "chunk_index": packed_order,
                    "chunk_id": member["chunk_id"],
                    "seed_rank": member["seed_rank"],
                    "seed_chunk_id": member["seed_chunk_id"],
                    "relation": member["relation"],
                    "source_intervals": member["source_intervals"],
                    "source_path": member["source_path"],
                    "document_id": member["document_id"],
                    "document_version": member["document_version"],
                    "source_sha256": member["source_sha256"],
                    "packed_order": packed_order,
                    "cl100k_tokens": member.get("cl100k_tokens"),
                    "serialized_text_sha256": member["serialized_text_sha256"],
                    "qualified_contract": DRAFTER_PACKED_EVIDENCE_V1,
                    "distance": seed.get("distance", 0.0),
                    "rrf_score": seed.get("rrf_score"),
                }
            )
            packed_order += 1
    return kb_results


def is_qualified_kb(kb_results: list[dict[str, Any]] | None) -> bool:
    rows = kb_results or []
    return bool(rows) and rows[0].get("qualified_contract") == DRAFTER_PACKED_EVIDENCE_V1


def build_drafter_kb_context(kb_results: list[dict[str, Any]]) -> str:
    """Deterministic full-text KB block for draft_node — no top-3 gate or char clip."""

    parts: list[str] = []
    for item in kb_results:
        header = (
            f"[KB seed_rank={item['seed_rank']} relation={item['relation']}] "
            f"{item['source']}"
        )
        parts.append(f"{header}\n{item['text']}")
    return "\n\n".join(parts)


def retrieve_qualified_kb(query: str, *, n_seeds: int = 5) -> dict[str, Any]:
    """Retrieve, expand, seed-first pack, and serialize production KB evidence."""

    if n_seeds != 5:
        raise QualifiedRAGError("qualified contract requires top-5 seeds")

    _manifest, units, by_source, _chunks, _chunks_by_id, _embeddings, _bm25 = _index()
    retrieval_seeds, _hybrid = _hybrid_top5(query)
    groups = candidate_c.expand_seeds(retrieval_seeds, units, by_source)
    expanded_groups, _dup_ids, _additional = candidate_c.dedupe_groups(groups)
    expanded = candidate_c.flatten(expanded_groups)
    encoding = candidate_c.cl100k()
    packed, skipped, used, exhausted, pack_stats = candidate_c.pack_units_seed_first(
        expanded, units, PACK_BUDGET_CL100K, encoding
    )
    candidate_c.assert_seed_preservation_invariant(
        expanded, units, PACK_BUDGET_CL100K, encoding
    )
    serialized = serialize_drafter_packed_evidence_v1(packed, units)
    kb_results = _packed_rows_to_kb_results(packed, units, retrieval_seeds)
    fingerprint = packed_identity_fingerprint(serialized)
    log.info(
        "qualified_rag.complete",
        seeds=len(retrieval_seeds),
        packed_units=len(packed),
        skipped=len(skipped),
        used_cl100k=used,
        exhausted=exhausted,
        fingerprint=fingerprint,
    )
    return {
        "kb_results": kb_results,
        "retrieval_seeds": retrieval_seeds,
        "packed_rows": packed,
        "serialized_groups": serialized,
        "packed_fingerprint": fingerprint,
        "pack_stats": pack_stats,
        "used_cl100k": used,
        "provider_calls": 0,
        "qdrant_reads": 0,
        "qdrant_writes": 0,
    }


def warmup() -> dict[str, int]:
    """Eagerly load CSWP index and MiniLM encoder before pipeline retrieve."""

    import time

    t0 = time.perf_counter()
    _index()
    t1 = time.perf_counter()
    _encoder()
    t2 = time.perf_counter()
    return {
        "cswp_index_ms": int((t1 - t0) * 1000),
        "encoder_load_ms": int((t2 - t1) * 1000),
    }
