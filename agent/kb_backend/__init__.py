"""Explicit CSWP retrieval backends: local in-memory or durable Qdrant serving."""

from agent.kb_backend.dispatch import (
    get_kb_backend,
    retrieve_kb,
    warmup,
)

__all__ = ["get_kb_backend", "retrieve_kb", "warmup"]
