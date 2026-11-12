"""Canonical content identity for qualified CSWP Qdrant collections.

This intentionally derives the serving identity from the hydrated point
payloads.  It is shared by the index writer and serving validator so a
manifest fingerprint cannot stand in for the collection's actual contents.
"""

from __future__ import annotations

from typing import Any

from agent.cswp.identity import sha256_json, sha
from agent.retrieval.encoder import EMBEDDING_DIMENSION, ENCODER_MODEL, MODEL_REVISION
from agent.shadow_qdrant.constants import PAYLOAD_SCHEMA_VERSION


def canonical_payload_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the order-independent, content-bearing identity of one point."""

    return {
        "chunk_id": payload["chunk_id"],
        "document_id": payload["document_id"],
        "document_version": payload["document_version"],
        "source_path": payload["source_path"],
        "source_sha256": payload["source_sha256"],
        "retrieval_text_sha256": sha(payload["retrieval_text"]),
        "source_spans": payload["source_spans"],
        "heading_path": payload["heading_path"],
        "structural_segments": payload["structural_segments"],
        "embedding_content_token_count": payload["embedding_content_token_count"],
        "reading_order_ordinal": payload["reading_order_ordinal"],
        "previous_chunk_id": payload["previous_chunk_id"],
        "next_chunk_id": payload["next_chunk_id"],
        "retrieval_char_start": payload["retrieval_char_start"],
        "retrieval_char_end": payload["retrieval_char_end"],
        "packing_version": payload.get("packing_version"),
        "payload_schema_version": payload["payload_schema_version"],
        "compiler_version": payload["compiler_version"],
    }


def collection_content_fingerprint(
    payloads: list[dict[str, Any]],
    *,
    representation_fingerprint: str,
    compiler_version: str,
    vector_size: int = EMBEDDING_DIMENSION,
    distance: str = "cosine",
) -> str:
    """Hash canonical serving payload content independently of Qdrant scroll order."""

    records = sorted(
        (canonical_payload_record(payload) for payload in payloads),
        key=lambda record: record["chunk_id"],
    )
    return sha256_json(
        {
            "schema_version": "cswp_live_collection_identity_v1",
            "representation_fingerprint": representation_fingerprint,
            "point_count": len(records),
            "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "compiler_version": compiler_version,
            "encoder_model": ENCODER_MODEL,
            "encoder_revision": MODEL_REVISION,
            "vector_size": vector_size,
            "distance": distance,
            "points": records,
        }
    )
