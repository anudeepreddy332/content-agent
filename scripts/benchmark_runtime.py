"""Benchmark-only web snapshot adapter loaded through ``sitecustomize``.

This module is copied beside a generated ``sitecustomize.py`` and placed on an
arm subprocess's ``PYTHONPATH``.  It deliberately does not alter normal
production imports or Tavily cache behavior.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path


SNAPSHOT_ENV = "CONTENT_AGENT_FROZEN_WEB_SNAPSHOT"
SNAPSHOT_HASH_ENV = "CONTENT_AGENT_FROZEN_WEB_SNAPSHOT_HASH"
CONSUMPTION_ENV = "CONTENT_AGENT_FROZEN_WEB_CONSUMPTION"
_FAILURE_PREFIX = "BENCHMARK_FROZEN_WEB_FAILURE:"


class FrozenWebEvidenceError(RuntimeError):
    """A benchmark arm tried to use unavailable or malformed frozen evidence."""

    def __init__(self, message: str):
        super().__init__(f"{_FAILURE_PREFIX} {message}")


_snapshot_path: Path | None = None
_snapshot: dict | None = None
_snapshot_hash: str | None = None
_consumed_queries: set[str] = set()


def _snapshot_digest(snapshot: dict) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_results(query: str, results: object) -> list[dict]:
    if not isinstance(results, list) or not results:
        raise FrozenWebEvidenceError(f"frozen web evidence has no usable results for query: {query!r}")
    for result in results:
        if not isinstance(result, dict):
            raise FrozenWebEvidenceError(f"frozen web evidence has a non-object result for query: {query!r}")
        if not all(isinstance(result.get(field), str) and result[field].strip()
                   for field in ("title", "url", "content")):
            raise FrozenWebEvidenceError(f"frozen web evidence has an unusable result for query: {query!r}")
        score = result.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise FrozenWebEvidenceError(f"frozen web evidence has a non-finite score for query: {query!r}")
    return results


def _load_snapshot() -> dict:
    global _snapshot_path, _snapshot, _snapshot_hash, _consumed_queries
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
    digest = _snapshot_digest(data)
    expected_digest = os.environ.get(SNAPSHOT_HASH_ENV)
    if expected_digest and digest != expected_digest:
        raise FrozenWebEvidenceError("frozen web snapshot hash does not match the benchmark environment")
    _snapshot_path = path
    _snapshot = data
    _snapshot_hash = digest
    _consumed_queries = set()
    return data


def frozen_web_search(query: str, max_results: int = 5, force_refresh: bool = False) -> list[dict]:
    """Return only the recorded response for this exact query.

    ``force_refresh`` is accepted for compatibility with ``retrieve_node`` but
    intentionally never authorizes a live Tavily call during a benchmark.
    """
    entry = _load_snapshot()["queries"].get(query)
    if not isinstance(entry, dict):
        raise FrozenWebEvidenceError(f"frozen web evidence missing exact query: {query!r}")
    if entry.get("max_results") != max_results:
        raise FrozenWebEvidenceError(f"frozen web evidence is invalid for query: {query!r}")
    results = _validate_results(query, entry.get("results"))
    _consumed_queries.add(query)
    _persist_consumption()
    return copy.deepcopy(results)


def frozen_web_consumption() -> dict:
    """Return the snapshot identity and exact query set used by this process."""
    _load_snapshot()
    return {"snapshot_hash": _snapshot_hash, "queries": sorted(_consumed_queries)}


def _persist_consumption() -> None:
    """Publish benchmark-only evidence that this child consumed the snapshot."""
    configured = os.environ.get(CONSUMPTION_ENV)
    if not configured:
        return
    path = Path(configured)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(frozen_web_consumption(), sort_keys=True), encoding="utf-8")
    os.replace(temporary_path, path)


def install_frozen_web_search() -> None:
    """Patch the arm's Tavily wrapper only when snapshot mode is explicitly set."""
    if not os.environ.get(SNAPSHOT_ENV):
        return
    from tools import web_search

    web_search.web_search = frozen_web_search
    web_search.benchmark_frozen_web_consumption = frozen_web_consumption


def install_benchmark_guards() -> None:
    """Make frozen-evidence failures fatal in either architecture checkout.

    The baseline checkout predates the benchmark harness and its retrieval node
    intentionally treats Tavily errors as warnings.  This benchmark-only hook
    preserves that production behavior while making a snapshot failure invalid
    for the detached benchmark process.
    """
    if not os.environ.get(SNAPSHOT_ENV):
        return

    from agent import graph, nodes

    nodes.web_search = frozen_web_search
    original = nodes.retrieve_node
    if getattr(original, "_benchmark_snapshot_guard", False):
        return

    def guarded_retrieve_node(state):
        result = original(state)
        for error in result.get("error_log", []):
            if _FAILURE_PREFIX in str(error):
                raise FrozenWebEvidenceError(str(error))
        return result

    guarded_retrieve_node._benchmark_snapshot_guard = True
    nodes.retrieve_node = guarded_retrieve_node
    graph.retrieve_node = guarded_retrieve_node
