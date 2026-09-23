"""Production Contiguous Structural Window Packing (CSWP) qualified KB retrieval.

Hybrid dense (all-MiniLM-L6-v2) + BM25 (raw lexical tokenization) over frozen
CSWP retrieval units, Reciprocal Rank Fusion (RRF) top-5 seeds, bounded ±1
neighbor expansion, and seed-first packing under a 2000 cl100k_base evidence
budget. Does not read or write production Qdrant.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from agent.cswp.constants import (
    CURRENT_CORPUS_REPRESENTATION_FINGERPRINT,
    PRODUCTION_INDEX_DIR,
)
from agent.cswp.loader import load_production_index, production_index_available
from agent.drafter_packed_evidence import (
    DRAFTER_PACKED_EVIDENCE_V1,
    packed_identity_fingerprint,
    serialize_drafter_packed_evidence_v1,
)
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
    rank_dense,
    reciprocal_rank_fusion,
)
from agent.shadow_qdrant.identity import collection_content_fingerprint
from agent.shadow_qdrant.index import load_index_manifest
from agent.shadow_qdrant.payload import build_point_payload
from observability.logger import get_logger

log = get_logger("qualified_rag")

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_CSWP_INDEX = PRODUCTION_INDEX_DIR


class QualifiedRAGError(RuntimeError):
    """Qualified CSWP retrieval/pre/post contract failed."""


@lru_cache(maxsize=1)
def _index():
    if not production_index_available(PRODUCTION_CSWP_INDEX):
        raise QualifiedRAGError("production CSWP index missing")
    manifest, units, by_source = load_production_index(PRODUCTION_CSWP_INDEX)
    chunks = eval_chunks_from_manifest(manifest, ROOT)
    if len(chunks) != 159:
        raise QualifiedRAGError("CSWP unit count drift")
    encoder = get_encoder()
    texts = [chunk.retrieval_text for chunk in chunks]
    embeddings = np.asarray(encoder.encode(texts), dtype=np.float32)
    bm25 = BM25Okapi([text.lower().split() for text in texts])
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return manifest, units, by_source, chunks, chunks_by_id, embeddings, bm25


def _hybrid_top5(query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _manifest, _units, _by_source, chunks, chunks_by_id, embeddings, bm25 = _index()
    encoder = get_encoder()
    query_embedding = np.asarray(encoder.encode(query), dtype=np.float32)
    dense = rank_dense(query_embedding, embeddings, chunks)
    bm25_rows = rank_bm25(query, bm25, chunks)
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


def _runtime_identity(units: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Derive local identity from the same canonical payload contract as serving."""

    shadow_manifest = load_index_manifest(PRODUCTION_CSWP_INDEX)
    representation_fingerprint = CURRENT_CORPUS_REPRESENTATION_FINGERPRINT
    if representation_fingerprint != shadow_manifest["representation_fingerprint"]:
        raise QualifiedRAGError("local/Qdrant representation fingerprint drift")
    compiler_version = shadow_manifest["cswp_compiler_version"]
    payloads = [
        build_point_payload(unit, compiler_version=compiler_version)
        for unit in units.values()
    ]
    return {
        "source_corpus_fingerprint": shadow_manifest["source_corpus_fingerprint"],
        "index_fingerprint": collection_content_fingerprint(
            payloads,
            representation_fingerprint=representation_fingerprint,
            compiler_version=compiler_version,
        ),
        "minilm_model_id": shadow_manifest["minilm_model_id"],
        "minilm_model_revision": shadow_manifest["minilm_model_revision"],
        "qualified_contract": DRAFTER_PACKED_EVIDENCE_V1,
    }


def packed_rows_to_kb_results(
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
            row = next(
                item for item in packed_rows if item["chunk_id"] == member["chunk_id"]
            )
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
    return (
        bool(rows) and rows[0].get("qualified_contract") == DRAFTER_PACKED_EVIDENCE_V1
    )


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
    groups = expand_seeds(retrieval_seeds, units, by_source)
    expanded_groups, _dup_ids, _additional = dedupe_groups(groups)
    expanded = flatten(expanded_groups)
    encoding = cl100k()
    packed, skipped, used, exhausted, pack_stats = pack_units_seed_first(
        expanded, units, PACK_BUDGET_CL100K, encoding
    )
    assert_seed_preservation_invariant(expanded, units, PACK_BUDGET_CL100K, encoding)
    serialized = serialize_drafter_packed_evidence_v1(packed, units)
    kb_results = packed_rows_to_kb_results(packed, units, retrieval_seeds)
    expanded_rows = packed_rows_to_kb_results(expanded, units, retrieval_seeds)
    fingerprint = packed_identity_fingerprint(serialized)
    runtime_identity = _runtime_identity(units)
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
        "expanded_rows": expanded_rows,
        "retrieval_seeds": retrieval_seeds,
        "packed_rows": packed,
        "serialized_groups": serialized,
        "packed_fingerprint": fingerprint,
        "pack_stats": pack_stats,
        "used_cl100k": used,
        "provider_calls": 0,
        "qdrant_reads": 0,
        "qdrant_writes": 0,
        **runtime_identity,
    }


def warmup() -> dict[str, int]:
    """Eagerly load CSWP index and MiniLM encoder before pipeline retrieve."""

    import time

    t0 = time.perf_counter()
    _index()
    t1 = time.perf_counter()
    get_encoder()
    t2 = time.perf_counter()
    return {
        "cswp_index_ms": int((t1 - t0) * 1000),
        "encoder_load_ms": int((t2 - t1) * 1000),
    }
