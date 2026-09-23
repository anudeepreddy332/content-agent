"""Durable Qdrant CSWP serving backend with payload hydration."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    SearchParams,
)
from rank_bm25 import BM25Okapi

from agent.drafter_packed_evidence import (
    DRAFTER_PACKED_EVIDENCE_V1,
    packed_identity_fingerprint,
    serialize_drafter_packed_evidence_v1,
)
from agent.qualified_rag import packed_rows_to_kb_results
from agent.retrieval.chunks import EvalChunk
from agent.retrieval.encoder import (
    EMBEDDING_DIMENSION,
    ENCODER_MODEL,
    MODEL_REVISION,
    get_encoder,
)
from agent.retrieval.expansion import (
    PACK_BUDGET_CL100K,
    assert_seed_preservation_invariant,
    cl100k,
    dedupe_groups,
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
from agent.shadow_qdrant.constants import (
    BLOCKED_COLLECTIONS,
    PAYLOAD_SCHEMA_VERSION,
    PRODUCTION_INDEX_DIR,
    SERVING_ALIAS,
    SHADOW_MANIFEST_FILE,
)
from agent.shadow_qdrant.identity import collection_content_fingerprint
from agent.shadow_qdrant.payload import validate_point_payload
from observability.logger import get_logger

log = get_logger("kb_backend.qdrant")

BACKEND_NAME = "cswp_qdrant"
DENSE_TOP_K = 20
SEED_TOP_K = 5


class QdrantServingError(RuntimeError):
    """Qdrant CSWP serving backend failed."""


def clear_serving_cache() -> None:
    _serving_state.cache_clear()


def _qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333")


def _serving_client() -> QdrantClient:
    return QdrantClient(url=_qdrant_url())


def payload_to_unit(payload: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a CSWP unit dict from a Qdrant point payload."""

    return {
        "chunk_id": payload["chunk_id"],
        "document_id": payload["document_id"],
        "document_version": payload["document_version"],
        "source_path": payload["source_path"],
        "source_sha256": payload["source_sha256"],
        "retrieval_text": payload["retrieval_text"],
        "source_spans": payload["source_spans"],
        "heading_path": payload.get("heading_path", []),
        "structural_segments": payload.get("structural_segments", []),
        "embedding_content_token_count": payload["embedding_content_token_count"],
        "reading_order_ordinal": payload["reading_order_ordinal"],
        "previous_chunk_id": payload.get("previous_chunk_id"),
        "next_chunk_id": payload.get("next_chunk_id"),
        "retrieval_char_start": payload["retrieval_char_start"],
        "retrieval_char_end": payload["retrieval_char_end"],
        "packing_version": payload.get("packing_version"),
    }


def eval_chunks_from_units(units_list: list[dict[str, Any]]) -> list[EvalChunk]:
    """Build rankable chunks from hydrated payload units (no local units.jsonl)."""

    ordered = sorted(units_list, key=lambda unit: unit["reading_order_ordinal"])
    chunks: list[EvalChunk] = []
    for ordinal, unit in enumerate(ordered):
        intervals = tuple(
            (span["source_char_start"], span["source_char_end"])
            for span in unit["source_spans"]
        )
        chunks.append(
            EvalChunk(
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


def _build_by_source(
    units_list: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units_list:
        by_source[unit["source_path"]].append(unit)
    for rows in by_source.values():
        rows.sort(key=lambda row: row["retrieval_char_start"])
    return dict(by_source)


def _resolve_serving_collection(client: QdrantClient) -> str:
    """Resolve the serving target via alias only — never a physical collection name."""

    aliases = client.get_aliases().aliases
    for alias in aliases:
        if alias.alias_name == SERVING_ALIAS:
            collection = alias.collection_name
            if collection in BLOCKED_COLLECTIONS:
                raise QdrantServingError(
                    f"serving alias {SERVING_ALIAS!r} points at blocked collection {collection!r}"
                )
            return collection

    physical_names = {collection.name for collection in client.get_collections().collections}
    if SERVING_ALIAS in physical_names:
        raise QdrantServingError(
            f"serving name {SERVING_ALIAS!r} exists as a physical collection without alias; "
            "refusing implicit fallback"
        )

    raise QdrantServingError(
        f"serving alias {SERVING_ALIAS!r} not found at {_qdrant_url()}"
    )


def _validate_collection_config(info: Any) -> None:
    vectors = info.config.params.vectors
    if vectors.size != EMBEDDING_DIMENSION:
        raise QdrantServingError(
            f"serving vector size {vectors.size} != required {EMBEDDING_DIMENSION}"
        )
    if vectors.distance != Distance.COSINE:
        raise QdrantServingError(
            f"serving distance {vectors.distance!r} != required Cosine"
        )


def _load_expected_manifest() -> dict[str, Any]:
    manifest_path = PRODUCTION_INDEX_DIR / SHADOW_MANIFEST_FILE
    if not manifest_path.is_file():
        raise QdrantServingError("qdrant shadow manifest missing for serving validation")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_manifest_contract(
    manifest: dict[str, Any],
    *,
    collection: str,
    point_count: int,
) -> str:
    expected_fp = manifest.get("index_fingerprint")
    expected_collection = manifest.get("collection_name")
    expected_points = manifest.get("point_count")
    expected_schema = manifest.get("payload_schema_version")
    expected_compiler = manifest.get("cswp_compiler_version")
    expected_minilm_id = manifest.get("minilm_model_id")
    expected_minilm_rev = manifest.get("minilm_model_revision")
    expected_vector_size = manifest.get("vector_size")
    expected_distance = manifest.get("distance")

    if not expected_fp or not expected_collection:
        raise QdrantServingError("shadow manifest missing index_fingerprint or collection_name")
    if expected_schema != PAYLOAD_SCHEMA_VERSION:
        raise QdrantServingError(
            f"manifest payload_schema_version {expected_schema!r} != pin {PAYLOAD_SCHEMA_VERSION!r}"
        )
    if expected_minilm_id != ENCODER_MODEL:
        raise QdrantServingError(
            f"manifest minilm_model_id {expected_minilm_id!r} != pin {ENCODER_MODEL!r}"
        )
    if expected_minilm_rev != MODEL_REVISION:
        raise QdrantServingError(
            f"manifest minilm_model_revision {expected_minilm_rev!r} != pin {MODEL_REVISION!r}"
        )
    if expected_vector_size != EMBEDDING_DIMENSION:
        raise QdrantServingError(
            f"manifest vector_size {expected_vector_size} != required {EMBEDDING_DIMENSION}"
        )
    if expected_distance != "cosine":
        raise QdrantServingError(
            f"manifest distance {expected_distance!r} != required 'cosine'"
        )
    if collection != expected_collection:
        raise QdrantServingError(
            f"serving collection {collection!r} != manifest {expected_collection!r}"
        )
    if expected_points is not None and point_count != expected_points:
        raise QdrantServingError(
            f"serving point count {point_count} != manifest {expected_points}"
        )
    if not expected_compiler:
        raise QdrantServingError("shadow manifest missing cswp_compiler_version")
    return expected_fp


def _scroll_payloads(
    client: QdrantClient,
    collection: str,
    *,
    expected_schema_version: str,
    expected_compiler_version: str,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            if point.payload is None:
                raise QdrantServingError(f"point {point.id} missing payload")
            try:
                validate_point_payload(
                    point.payload,
                    expected_schema_version=expected_schema_version,
                    expected_compiler_version=expected_compiler_version,
                )
            except ValueError as exc:
                raise QdrantServingError(f"invalid serving payload for point {point.id}: {exc}") from exc
            payloads.append(point.payload)
        if offset is None:
            break
    if not payloads:
        raise QdrantServingError(f"serving collection {collection!r} is empty")
    return payloads


def _validate_live_payload_topology(
    payloads: list[dict[str, Any]], *, expected_point_count: int
) -> dict[str, dict[str, Any]]:
    """Validate cross-point content and the frozen same-document ±1 topology."""

    if len(payloads) != expected_point_count:
        raise QdrantServingError(
            f"hydrated payload count {len(payloads)} != expected {expected_point_count}"
        )

    units: dict[str, dict[str, Any]] = {}
    document_versions: dict[str, str] = {}
    version_documents: dict[str, str] = {}
    document_sources: dict[str, tuple[str, str]] = {}
    ordinals: set[int] = set()
    for payload in payloads:
        chunk_id = payload["chunk_id"]
        if chunk_id in units:
            raise QdrantServingError(f"duplicate chunk_id in serving collection: {chunk_id!r}")
        ordinal = payload["reading_order_ordinal"]
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise QdrantServingError(f"invalid reading_order_ordinal for {chunk_id!r}")
        if ordinal in ordinals:
            raise QdrantServingError(f"duplicate reading_order_ordinal: {ordinal}")
        ordinals.add(ordinal)
        document_id = payload["document_id"]
        document_version = payload["document_version"]
        source = (payload["source_path"], payload["source_sha256"])
        if document_versions.setdefault(document_id, document_version) != document_version:
            raise QdrantServingError(
                f"document_id {document_id!r} has conflicting document versions"
            )
        if version_documents.setdefault(document_version, document_id) != document_id:
            raise QdrantServingError(
                f"document_version {document_version!r} belongs to multiple document IDs"
            )
        if document_sources.setdefault(document_id, source) != source:
            raise QdrantServingError(
                f"document_id {document_id!r} has inconsistent source identity"
            )
        units[chunk_id] = payload

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in units.values():
        by_document[payload["document_id"]].append(payload)
    for document_id, rows in by_document.items():
        rows.sort(key=lambda row: row["reading_order_ordinal"])
        for index, row in enumerate(rows):
            expected_previous = rows[index - 1]["chunk_id"] if index else None
            expected_next = rows[index + 1]["chunk_id"] if index + 1 < len(rows) else None
            if row["previous_chunk_id"] != expected_previous:
                raise QdrantServingError(
                    f"broken previous_chunk_id for {row['chunk_id']!r} in {document_id!r}"
                )
            if row["next_chunk_id"] != expected_next:
                raise QdrantServingError(
                    f"broken next_chunk_id for {row['chunk_id']!r} in {document_id!r}"
                )
    return units


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
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ranked: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        chunk_id = hit.payload["chunk_id"]
        chunk = chunks_by_id[chunk_id]
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


def _hybrid_top5(
    client: QdrantClient,
    query: str,
    *,
    collection: str,
    chunks: list[EvalChunk],
    chunks_by_id: dict[str, EvalChunk],
    bm25: BM25Okapi,
    stage_telemetry: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    encoder = get_encoder()

    t0 = time.perf_counter()
    query_embedding = np.asarray(encoder.encode(query), dtype=np.float32)
    dense = _rank_dense_qdrant(client, collection, query_embedding, chunks, limit=DENSE_TOP_K)
    stage_telemetry["dense_ms"] = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    bm25_rows = rank_bm25(query, bm25, chunks)
    stage_telemetry["bm25_ms"] = int((time.perf_counter() - t1) * 1000)

    t2 = time.perf_counter()
    hybrid = reciprocal_rank_fusion(
        dense,
        bm25_rows,
        candidate_k=CANDIDATE_K,
        rrf_constant=RRF_CONSTANT,
    )
    stage_telemetry["fusion_ms"] = int((time.perf_counter() - t2) * 1000)

    seed_texts = [
        chunks_by_id[row["chunk_id"]].retrieval_text for row in hybrid[:SEED_TOP_K]
    ]
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
    return seeds, hybrid


@lru_cache(maxsize=1)
def _serving_state(client_key: str):
    del client_key
    t0 = time.perf_counter()
    client = _serving_client()
    collection = _resolve_serving_collection(client)
    info = client.get_collection(collection)
    _validate_collection_config(info)
    manifest = _load_expected_manifest()
    index_fingerprint = _validate_manifest_contract(
        manifest,
        collection=collection,
        point_count=info.points_count,
    )
    payloads = _scroll_payloads(
        client,
        collection,
        expected_schema_version=manifest["payload_schema_version"],
        expected_compiler_version=manifest["cswp_compiler_version"],
    )
    payload_by_chunk_id = _validate_live_payload_topology(
        payloads, expected_point_count=info.points_count
    )
    live_fingerprint = collection_content_fingerprint(
        payloads,
        representation_fingerprint=manifest["representation_fingerprint"],
        compiler_version=manifest["cswp_compiler_version"],
    )
    if live_fingerprint != index_fingerprint:
        raise QdrantServingError(
            "live collection fingerprint mismatch: serving payload identity does not match "
            "the qualified index"
        )
    units_list = [payload_to_unit(payload) for payload in payload_by_chunk_id.values()]
    units = {unit["chunk_id"]: unit for unit in units_list}
    by_source = _build_by_source(units_list)
    chunks = eval_chunks_from_units(units_list)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    texts = [chunk.retrieval_text for chunk in chunks]
    bm25 = BM25Okapi([text.lower().split() for text in texts])
    hydrate_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "client": client,
        "collection": collection,
        "units": units,
        "by_source": by_source,
        "chunks": chunks,
        "chunks_by_id": chunks_by_id,
        "bm25": bm25,
        "point_count": info.points_count,
        "index_fingerprint": index_fingerprint,
        "live_collection_fingerprint": live_fingerprint,
        "manifest": manifest,
        "hydrate_ms": hydrate_ms,
    }


def validate_startup() -> dict[str, Any]:
    """Fail-closed startup validation for the Qdrant serving backend."""

    state = _serving_state(str(id(_serving_client())))
    manifest = state["manifest"]
    return {
        "kb_backend": BACKEND_NAME,
        "serving_alias": SERVING_ALIAS,
        "collection_name": state["collection"],
        "point_count": state["point_count"],
        "index_fingerprint": state["index_fingerprint"],
        "live_collection_fingerprint": state["live_collection_fingerprint"],
        "qdrant_url": _qdrant_url(),
        "vector_size": EMBEDDING_DIMENSION,
        "distance": "cosine",
        "payload_schema_version": manifest["payload_schema_version"],
        "cswp_compiler_version": manifest["cswp_compiler_version"],
        "minilm_model_id": manifest["minilm_model_id"],
        "minilm_model_revision": manifest["minilm_model_revision"],
        "startup_validated": True,
        "hydrate_ms": state["hydrate_ms"],
    }


def retrieve(query: str, *, n_seeds: int = 5) -> dict[str, Any]:
    if n_seeds != 5:
        raise QdrantServingError("qualified contract requires top-5 seeds")

    t0 = time.perf_counter()
    state = _serving_state(str(id(_serving_client())))
    client = state["client"]
    collection = state["collection"]
    units = state["units"]
    by_source = state["by_source"]
    chunks = state["chunks"]
    chunks_by_id = state["chunks_by_id"]
    bm25 = state["bm25"]

    stage_telemetry: dict[str, int] = {"hydrate_ms": state["hydrate_ms"]}
    retrieval_seeds, _hybrid = _hybrid_top5(
        client,
        query,
        collection=collection,
        chunks=chunks,
        chunks_by_id=chunks_by_id,
        bm25=bm25,
        stage_telemetry=stage_telemetry,
    )

    t_expand = time.perf_counter()
    groups = expand_seeds(retrieval_seeds, units, by_source)
    expanded_groups, _dup_ids, _additional = dedupe_groups(groups)
    expanded = flatten(expanded_groups)
    stage_telemetry["expand_ms"] = int((time.perf_counter() - t_expand) * 1000)

    t_pack = time.perf_counter()
    encoding = cl100k()
    packed, skipped, used, exhausted, pack_stats = pack_units_seed_first(
        expanded, units, PACK_BUDGET_CL100K, encoding
    )
    assert_seed_preservation_invariant(expanded, units, PACK_BUDGET_CL100K, encoding)
    serialized = serialize_drafter_packed_evidence_v1(packed, units)
    kb_results = packed_rows_to_kb_results(packed, units, retrieval_seeds)
    expanded_rows = packed_rows_to_kb_results(expanded, units, retrieval_seeds)
    fingerprint = packed_identity_fingerprint(serialized)
    stage_telemetry["pack_ms"] = int((time.perf_counter() - t_pack) * 1000)
    stage_telemetry["total_ms"] = int((time.perf_counter() - t0) * 1000)

    log.info(
        "kb_backend.qdrant.complete",
        seeds=len(retrieval_seeds),
        packed_units=len(packed),
        collection=collection,
        fingerprint=fingerprint,
        stage_telemetry=stage_telemetry,
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
        "kb_backend": BACKEND_NAME,
        "serving_alias": SERVING_ALIAS,
        "collection_name": collection,
        "point_count": state["point_count"],
        "index_fingerprint": state["index_fingerprint"],
        "live_collection_fingerprint": state["live_collection_fingerprint"],
        "source_corpus_fingerprint": state["manifest"]["source_corpus_fingerprint"],
        "minilm_model_id": state["manifest"]["minilm_model_id"],
        "minilm_model_revision": state["manifest"]["minilm_model_revision"],
        "qualified_contract": DRAFTER_PACKED_EVIDENCE_V1,
        "payload_schema_version": state["manifest"]["payload_schema_version"],
        "cswp_compiler_version": state["manifest"]["cswp_compiler_version"],
        "stage_telemetry": stage_telemetry,
        "provider_calls": 0,
        "qdrant_reads": 1,
        "qdrant_writes": 0,
    }


def warmup() -> dict[str, int | str]:
    import time as _time

    t0 = _time.perf_counter()
    info = validate_startup()
    t1 = _time.perf_counter()
    get_encoder()
    t2 = _time.perf_counter()
    return {
        "kb_backend": BACKEND_NAME,
        "serving_alias": SERVING_ALIAS,
        "collection_name": info["collection_name"],
        "point_count": info["point_count"],
        "index_fingerprint": info["index_fingerprint"],
        "startup_validated": info["startup_validated"],
        "cswp_hydrate_ms": int((t1 - t0) * 1000),
        "encoder_load_ms": int((t2 - t1) * 1000),
    }


def create_serving_alias(
    client: QdrantClient,
    collection_name: str,
    *,
    alias_name: str = SERVING_ALIAS,
) -> None:
    """Atomically repoint the serving alias at a versioned collection."""

    if collection_name in BLOCKED_COLLECTIONS:
        raise QdrantServingError(
            f"refusing alias to blocked collection: {collection_name}"
        )

    operations: list[Any] = []
    existing_aliases = {alias.alias_name for alias in client.get_aliases().aliases}
    if alias_name in existing_aliases:
        operations.append(
            DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name))
        )
    operations.append(
        CreateAliasOperation(
            create_alias=CreateAlias(
                collection_name=collection_name,
                alias_name=alias_name,
            )
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)
    clear_serving_cache()
