"""Fail-closed KB_BACKEND selector."""

from __future__ import annotations

import os
import sys

VALID_BACKENDS = frozenset({"cswp_local", "cswp_qdrant"})
DEFAULT_BACKEND = "cswp_local"


class KBBackendConfigError(RuntimeError):
    """KB_BACKEND is missing, unknown, or inconsistent."""


def resolve_kb_backend(raw: str | None = None) -> str:
    """Resolve KB_BACKEND with no auto-detect and no silent fallback."""

    value = (raw if raw is not None else os.getenv("KB_BACKEND", "")).strip()
    if not value:
        return DEFAULT_BACKEND
    if value in VALID_BACKENDS:
        return value
    raise KBBackendConfigError(
        f"unknown KB_BACKEND={value!r}; expected one of {sorted(VALID_BACKENDS)} "
        f"or unset for developer default {DEFAULT_BACKEND!r}"
    )


def abort_on_invalid_kb_backend() -> str:
    """Resolve backend or terminate the process at startup."""

    try:
        return resolve_kb_backend()
    except KBBackendConfigError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
