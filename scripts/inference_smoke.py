"""Clean-environment inference smoke: pinned MiniLM + one qualified KB retrieval.

No LLM or provider calls. Requires `scripts/provision_minilm_snapshot.py`
to have populated the pinned snapshot first. Intended for CI and fresh-machine
runtime checks after dependency preparation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agent.kb_backend import retrieve_kb, warmup  # noqa: E402
from agent.kb_backend.selector import resolve_kb_backend  # noqa: E402
from agent.qualified_rag import is_qualified_kb  # noqa: E402
from scripts.phase5a0_baseline import (  # noqa: E402
    MODEL_REVISION,
    resolve_local_model_snapshot,
)


SMOKE_QUERY = "support vector machine margin"


def inference_smoke() -> list[str]:
    failures: list[str] = []
    try:
        snapshot = resolve_local_model_snapshot()
    except Exception as exc:  # noqa: BLE001 — smoke must report all setup failures
        failures.append(f"pinned MiniLM snapshot unavailable: {exc}")
        return failures

    if snapshot.name != MODEL_REVISION:
        failures.append(
            f"snapshot revision mismatch: {snapshot.name} != {MODEL_REVISION}"
        )

    backend = resolve_kb_backend()
    try:
        timings = warmup()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"KB backend {backend!r} warmup failed: {exc}")
        return failures

    if timings.get("kb_backend") != backend:
        failures.append(f"warmup backend mismatch: {timings!r}")
    if "encoder_load_ms" not in timings:
        failures.append(f"warmup missing encoder_load_ms: {timings!r}")

    try:
        result = retrieve_kb(SMOKE_QUERY, n_seeds=5)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"qualified KB retrieval failed: {exc}")
        return failures

    kb = result.get("kb_results") or []
    if not kb:
        failures.append("qualified KB returned zero results")
    elif not is_qualified_kb(kb):
        failures.append("KB results missing qualified contract markers")
    elif not kb[0].get("text"):
        failures.append("top KB result has empty text")

    if not result.get("packed_fingerprint"):
        failures.append("packed fingerprint missing from retrieval result")

    return failures


def main() -> int:
    failures = inference_smoke()
    if failures:
        print("INFERENCE SMOKE FAIL", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("INFERENCE SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
