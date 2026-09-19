"""Qdrant payload contract for CSWP shadow points."""

from __future__ import annotations

from typing import Any

from agent.shadow_qdrant.constants import PAYLOAD_SCHEMA_VERSION


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
