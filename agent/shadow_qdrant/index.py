"""Deterministic shadow Qdrant index lifecycle for production CSWP units."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from agent.cswp.constants import MANIFEST_FILE, UNITS_FILE
from agent.cswp.identity import sha256_json
from agent.cswp.loader import load_production_index, load_units_jsonl
from agent.retrieval.encoder import (
    EMBEDDING_DIMENSION,
    ENCODER_MODEL,
    MODEL_REVISION,
    get_encoder,
)
from agent.shadow_qdrant.constants import (
    BLOCKED_COLLECTIONS,
    PAYLOAD_SCHEMA_VERSION,
    POINT_ID_NAMESPACE,
    PRODUCTION_COLLECTION,
    PRODUCTION_INDEX_DIR as INDEX_DIR,
    SHADOW_MANIFEST_FILE,
    shadow_collection_name,
)
from agent.shadow_qdrant.payload import build_point_payload

PRODUCTION_COLLECTION_GUARD = PRODUCTION_COLLECTION


class ShadowIndexError(RuntimeError):
    """Shadow Qdrant index build or lifecycle contract failed."""


def point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def _assert_safe_collection(name: str) -> None:
    if name in BLOCKED_COLLECTIONS:
        raise ShadowIndexError(f"refusing to mutate production collection: {name}")


def _index_fingerprint(
    cswp_manifest: dict[str, Any], units: list[dict[str, Any]]
) -> str:
    payload = {
        "representation_fingerprint": cswp_manifest["representation_fingerprint"],
        "compiler_version": cswp_manifest["compiler_version"],
        "unit_count": len(units),
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "encoder_model": ENCODER_MODEL,
        "encoder_revision": MODEL_REVISION,
        "ordered_chunk_ids": [unit["chunk_id"] for unit in units],
    }
    return sha256_json(payload)


def _index_digest16(index_fingerprint: str) -> str:
    return index_fingerprint[:16]


def _embed_units(units: list[dict[str, Any]]) -> np.ndarray:
    encoder = get_encoder()
    texts = [unit["retrieval_text"] for unit in units]
    return np.asarray(encoder.encode(texts), dtype=np.float32)


def _build_points(
    units: list[dict[str, Any]], embeddings: np.ndarray, compiler_version: str
) -> list[PointStruct]:
    points: list[PointStruct] = []
    for unit, vector in zip(units, embeddings, strict=True):
        points.append(
            PointStruct(
                id=point_id_for_chunk(unit["chunk_id"]),
                vector=vector.tolist(),
                payload=build_point_payload(unit, compiler_version=compiler_version),
            )
        )
    return points


def _write_manifest(
    index_dir: Path,
    *,
    cswp_manifest: dict[str, Any],
    collection_name: str,
    point_count: int,
    index_fingerprint: str,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "cswp_shadow_qdrant_manifest_v1",
        "source_corpus_fingerprint": cswp_manifest["baseline_corpus_fingerprint"],
        "representation_fingerprint": cswp_manifest["representation_fingerprint"],
        "cswp_compiler_version": cswp_manifest["compiler_version"],
        "minilm_model_id": ENCODER_MODEL,
        "minilm_model_revision": MODEL_REVISION,
        "collection_name": collection_name,
        "point_count": point_count,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "index_fingerprint": index_fingerprint,
        "point_id_namespace": str(POINT_ID_NAMESPACE),
        "vector_size": EMBEDDING_DIMENSION,
        "distance": "cosine",
        "exact_search": True,
        "serving_alias_active": False,
        "production_collection_mutated": False,
    }
    path = index_dir / SHADOW_MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_index_manifest(index_dir: Path = INDEX_DIR) -> dict[str, Any]:
    path = index_dir / SHADOW_MANIFEST_FILE
    if not path.is_file():
        raise ShadowIndexError("shadow Qdrant manifest missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_units_ordered(index_dir: Path = INDEX_DIR) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = index_dir / MANIFEST_FILE
    units_path = index_dir / UNITS_FILE
    if not manifest_path.is_file() or not units_path.is_file():
        raise ShadowIndexError("production CSWP index missing")
    cswp_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = load_units_jsonl(units_path)
    return cswp_manifest, units


def full_rebuild(
    client: QdrantClient,
    *,
    index_dir: Path = INDEX_DIR,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """Drop and rebuild the versioned shadow collection from production CSWP index."""

    cswp_manifest, units = _load_units_ordered(index_dir)
    if len(units) != cswp_manifest.get("unit_count"):
        raise ShadowIndexError("CSWP unit count mismatch")
    index_fingerprint = _index_fingerprint(cswp_manifest, units)
    collection = shadow_collection_name(_index_digest16(index_fingerprint))
    _assert_safe_collection(collection)

    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        if not replace_existing:
            raise ShadowIndexError(f"shadow collection already exists: {collection}")
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )

    embeddings = _embed_units(units)
    points = _build_points(units, embeddings, cswp_manifest["compiler_version"])
    client.upsert(collection_name=collection, points=points)

    info = client.get_collection(collection)
    if info.points_count != len(units):
        raise ShadowIndexError(
            f"point count mismatch: expected {len(units)} got {info.points_count}"
        )

    return _write_manifest(
        index_dir,
        cswp_manifest=cswp_manifest,
        collection_name=collection,
        point_count=len(units),
        index_fingerprint=index_fingerprint,
    )


def upsert_document(
    client: QdrantClient,
    document_units: list[dict[str, Any]],
    *,
    index_dir: Path = INDEX_DIR,
    compiler_version: str | None = None,
) -> int:
    """Upsert all units for one document; caller must delete stale versions first."""

    shadow_manifest = load_index_manifest(index_dir)
    collection = shadow_manifest["collection_name"]
    _assert_safe_collection(collection)
    version = compiler_version or shadow_manifest["cswp_compiler_version"]
    embeddings = _embed_units(document_units)
    points = _build_points(document_units, embeddings, version)
    client.upsert(collection_name=collection, points=points)
    return len(points)


def delete_document(
    client: QdrantClient,
    document_id: str,
    *,
    index_dir: Path = INDEX_DIR,
) -> int:
    """Remove all shadow points for a document_id."""

    shadow_manifest = load_index_manifest(index_dir)
    collection = shadow_manifest["collection_name"]
    _assert_safe_collection(collection)
    selector = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )
    before = client.count(collection_name=collection, count_filter=selector).count
    client.delete(collection_name=collection, points_selector=selector)
    after = client.count(collection_name=collection, count_filter=selector).count
    return before - after


def delete_document_version(
    client: QdrantClient,
    document_version: str,
    *,
    index_dir: Path = INDEX_DIR,
) -> int:
    """Remove all shadow points for a specific document_version."""

    shadow_manifest = load_index_manifest(index_dir)
    collection = shadow_manifest["collection_name"]
    _assert_safe_collection(collection)
    selector = Filter(
        must=[
            FieldCondition(
                key="document_version", match=MatchValue(value=document_version)
            )
        ]
    )
    before = client.count(collection_name=collection, count_filter=selector).count
    client.delete(collection_name=collection, points_selector=selector)
    after = client.count(collection_name=collection, count_filter=selector).count
    return before - after


def units_for_document(
    units: list[dict[str, Any]], source_path: str
) -> list[dict[str, Any]]:
    return [unit for unit in units if unit["source_path"] == source_path]


def update_document(
    client: QdrantClient,
    old_document_version: str,
    new_units: list[dict[str, Any]],
    *,
    index_dir: Path = INDEX_DIR,
    compiler_version: str | None = None,
) -> dict[str, int]:
    """Replace one document version: delete old points, upsert new units."""

    deleted = delete_document_version(client, old_document_version, index_dir=index_dir)
    inserted = upsert_document(
        client,
        new_units,
        index_dir=index_dir,
        compiler_version=compiler_version,
    )
    return {"deleted": deleted, "inserted": inserted}


def load_shadow_corpus(index_dir: Path = INDEX_DIR) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load production CSWP index for shadow retrieval (no reports/ authority)."""

    representation, units, by_source = load_production_index(index_dir)
    return representation, units, by_source
