"""Unified KB retrieval dispatch for the selected backend."""

from __future__ import annotations

from typing import Any

from agent.kb_backend import local as local_backend
from agent.kb_backend import qdrant_serving as qdrant_backend
from agent.kb_backend.selector import resolve_kb_backend

_BACKENDS = {
    "cswp_local": local_backend,
    "cswp_qdrant": qdrant_backend,
}


def get_kb_backend(raw: str | None = None) -> str:
    return resolve_kb_backend(raw)


def _module_for(backend: str):
    module = _BACKENDS.get(backend)
    if module is None:
        raise RuntimeError(f"no implementation for KB_BACKEND={backend!r}")
    return module


def retrieve_kb(query: str, *, n_seeds: int = 5) -> dict[str, Any]:
    """Retrieve KB evidence via the configured backend; fail closed on errors."""

    backend = get_kb_backend()
    try:
        return _module_for(backend).retrieve(query, n_seeds=n_seeds)
    except Exception as exc:
        raise RuntimeError(f"KB backend {backend!r} retrieval failed: {exc}") from exc


def warmup() -> dict[str, int | str]:
    """Warm the configured backend before pipeline retrieve."""

    backend = get_kb_backend()
    return _module_for(backend).warmup()
