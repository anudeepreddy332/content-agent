"""Durable shadow Qdrant index for CSWP retrieval parity (non-serving)."""

from agent.shadow_qdrant.index import (
    ShadowIndexError,
    delete_document,
    full_rebuild,
    load_index_manifest,
    point_id_for_chunk,
    upsert_document,
)
from agent.shadow_qdrant.retrieval import retrieve_shadow_kb

__all__ = [
    "ShadowIndexError",
    "delete_document",
    "full_rebuild",
    "load_index_manifest",
    "point_id_for_chunk",
    "retrieve_shadow_kb",
    "upsert_document",
]
