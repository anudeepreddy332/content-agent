"""Benchmark-only web snapshot adapter loaded through ``sitecustomize``.

This module is copied beside a generated ``sitecustomize.py`` and placed on an
arm subprocess's ``PYTHONPATH``.  It deliberately does not alter normal
production imports or Tavily cache behavior.
"""

import copy
import json
import os
from pathlib import Path


SNAPSHOT_ENV = "CONTENT_AGENT_FROZEN_WEB_SNAPSHOT"


class FrozenWebEvidenceError(RuntimeError):
    """A benchmark arm tried to use unavailable or malformed frozen evidence."""


_snapshot_path: Path | None = None
_snapshot: dict | None = None


def _load_snapshot() -> dict:
    global _snapshot_path, _snapshot
    configured = os.environ.get(SNAPSHOT_ENV)
    if not configured:
        raise FrozenWebEvidenceError(f"{SNAPSHOT_ENV} is required for benchmark snapshot mode")
    path = Path(configured)
    if _snapshot_path == path and _snapshot is not None:
        return _snapshot
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenWebEvidenceError(f"frozen web snapshot is unreadable: {path}") from error
    if data.get("schema_version") != 1 or not isinstance(data.get("queries"), dict):
        raise FrozenWebEvidenceError("frozen web snapshot has an invalid schema")
    _snapshot_path = path
    _snapshot = data
    return data


def frozen_web_search(query: str, max_results: int = 5, force_refresh: bool = False) -> list[dict]:
    """Return only the recorded response for this exact query.

    ``force_refresh`` is accepted for compatibility with ``retrieve_node`` but
    intentionally never authorizes a live Tavily call during a benchmark.
    """
    entry = _load_snapshot()["queries"].get(query)
    if not isinstance(entry, dict):
        raise FrozenWebEvidenceError(f"frozen web evidence missing exact query: {query!r}")
    if entry.get("max_results") != max_results or not isinstance(entry.get("results"), list):
        raise FrozenWebEvidenceError(f"frozen web evidence is invalid for query: {query!r}")
    return copy.deepcopy(entry["results"])


def install_frozen_web_search() -> None:
    """Patch the arm's Tavily wrapper only when snapshot mode is explicitly set."""
    if not os.environ.get(SNAPSHOT_ENV):
        return
    from tools import web_search

    web_search.web_search = frozen_web_search
