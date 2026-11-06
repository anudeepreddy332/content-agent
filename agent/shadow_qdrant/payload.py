"""Qdrant payload contract for CSWP shadow points."""

from __future__ import annotations

from typing import Any

from agent.shadow_qdrant.constants import PAYLOAD_SCHEMA_VERSION

REQUIRED_PAYLOAD_FIELDS = frozenset(
    {
        "payload_schema_version",
        "chunk_id",
        "document_id",
        "document_version",
        "source_path",
        "source_sha256",
        "retrieval_text",
        "source_spans",
        "heading_path",
        "structural_segments",
        "embedding_content_token_count",
        "reading_order_ordinal",
        "previous_chunk_id",
        "next_chunk_id",
        "compiler_version",
        "retrieval_char_start",
        "retrieval_char_end",
    }
)


def validate_point_payload(
    payload: dict[str, Any],
    *,
    expected_schema_version: str,
    expected_compiler_version: str,
) -> None:
    """Fail closed if a Qdrant point payload violates the CSWP serving contract."""

    missing = REQUIRED_PAYLOAD_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"payload missing required fields: {sorted(missing)}")
    if payload["payload_schema_version"] != expected_schema_version:
        raise ValueError(
            "payload_schema_version mismatch: "
            f"expected {expected_schema_version!r} got {payload['payload_schema_version']!r}"
        )
    if payload["compiler_version"] != expected_compiler_version:
        raise ValueError(
            "compiler_version mismatch: "
            f"expected {expected_compiler_version!r} got {payload['compiler_version']!r}"
        )


def build_point_payload(unit: dict[str, Any], *, compiler_version: str) -> dict[str, Any]:
    """Persist metadata required for durable retrieval and ±1 expansion."""

    return {
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "chunk_id": unit["chunk_id"],
        "document_id": unit["document_id"],
        "document_version": unit["document_version"],
        "source_path": unit["source_path"],
        "source_sha256": unit["source_sha256"],
        "retrieval_text": unit["retrieval_text"],
        "source_spans": unit["source_spans"],
        "heading_path": unit.get("heading_path", []),
        "structural_segments": unit.get("structural_segments", []),
        "embedding_content_token_count": unit["embedding_content_token_count"],
        "reading_order_ordinal": unit["reading_order_ordinal"],
        "previous_chunk_id": unit.get("previous_chunk_id"),
        "next_chunk_id": unit.get("next_chunk_id"),
        "compiler_version": compiler_version,
        "packing_version": unit.get("packing_version"),
        "retrieval_char_start": unit["retrieval_char_start"],
        "retrieval_char_end": unit["retrieval_char_end"],
    }
