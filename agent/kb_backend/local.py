"""Local in-memory CSWP backend (developer default)."""

from __future__ import annotations

from typing import Any

from agent.qualified_rag import retrieve_qualified_kb, warmup as local_warmup

BACKEND_NAME = "cswp_local"


def retrieve(query: str, *, n_seeds: int = 5) -> dict[str, Any]:
    result = retrieve_qualified_kb(query, n_seeds=n_seeds)
    result["kb_backend"] = BACKEND_NAME
    return result


def warmup() -> dict[str, int | str]:
    timings = local_warmup()
    timings["kb_backend"] = BACKEND_NAME
    return timings
