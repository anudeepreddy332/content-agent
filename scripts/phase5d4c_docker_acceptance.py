"""Isolated real-Docker Qdrant v1.9.2 acceptance for Phase 5D4C.

Runs in a fresh process without CSWP sys.addaudithook offline guard so
loopback to the isolated container is permitted.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.cswp.constants import PRODUCTION_INDEX_DIR, ROOT as REPO_ROOT
from agent.cswp.loader import load_units_jsonl
from agent.kb_backend import retrieve_kb, warmup
from agent.kb_backend.qdrant_serving import (
    BACKEND_NAME as QDRANT_BACKEND,
    clear_serving_cache,
    create_serving_alias,
    validate_startup,
)
from agent.qualified_rag import retrieve_qualified_kb
from agent.retrieval.encoder import resolve_local_model_snapshot
from agent.retrieval.expansion import eval_chunks_from_manifest
from agent.retrieval.metrics import evidence_recall
from agent.shadow_qdrant.constants import QDRANT_IMAGE
from agent.shadow_qdrant.index import full_rebuild
from scripts.phase5a0_baseline import load_sources_and_evidence
from scripts.retrieval_golden_v2 import load_oracle, validate_oracle


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _docker_available() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"],
        capture_output=True,
        check=False,
    ).returncode == 0


@contextmanager
def _docker_qdrant(port: int, container_name: str):
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
    proc = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{port}:6333",
            QDRANT_IMAGE,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    container_id = proc.stdout.strip()
    url = f"http://127.0.0.1:{port}"
    from qdrant_client import QdrantClient

    deadline = time.time() + 60
    ready = False
    while time.time() < deadline:
        try:
            client = QdrantClient(url=url, timeout=5)
            client.get_collections()
            ready = True
            break
        except Exception:
            time.sleep(0.5)
    if not ready:
        subprocess.run(["docker", "rm", "-f", container_name], check=False)
        raise RuntimeError(f"Qdrant container {container_name} failed to become ready")
    with urlopen(url, timeout=5) as response:  # noqa: S310 - isolated loopback container
        qdrant_version = json.load(response)["version"]
    if qdrant_version != "1.9.2":
        raise RuntimeError(
            f"Qdrant server version {qdrant_version!r} does not match required '1.9.2'"
        )
    try:
        yield url, QdrantClient(url=url), qdrant_version
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], check=False)
        del container_id


def _parity_report(client) -> dict:
    import os

    os.environ["KB_BACKEND"] = "cswp_qdrant"
    clear_serving_cache()

    oracle = load_oracle()
    validate_oracle(oracle)
    _, _, evidence = load_sources_and_evidence(oracle)
    spans_by = defaultdict(list)
    for span in evidence:
        spans_by[span.query_id].append(span)

    manifest = json.loads(
        (PRODUCTION_INDEX_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    units = load_units_jsonl(PRODUCTION_INDEX_DIR / "units.jsonl")
    chunks = eval_chunks_from_manifest(
        {
            "children": units,
            "packing_version": manifest["packing_version"],
            "contract_sha256": manifest["contract_sha256"],
            "frozen_B_fingerprint": manifest["frozen_B_fingerprint"],
        },
        REPO_ROOT,
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    gating = [q for q in oracle["queries"] if q["gating_eligible"]]
    seed_mismatches = []
    expanded_mismatches = []
    packed_id_mismatches = []
    packed_fingerprint_mismatches = []
    packed_regressions = []
    baseline_packed = []
    qdrant_packed = []

    for query in gating:
        baseline = retrieve_qualified_kb(query["query"], n_seeds=5)
        shadow = retrieve_kb(query["query"], n_seeds=5)
        spans = spans_by[query["query_id"]]

        base_seeds = [row["chunk_id"] for row in baseline["retrieval_seeds"]]
        qdrant_seeds = [row["chunk_id"] for row in shadow["retrieval_seeds"]]
        if base_seeds != qdrant_seeds:
            seed_mismatches.append(query["query_id"])
        if [row["chunk_id"] for row in baseline["expanded_rows"]] != [
            row["chunk_id"] for row in shadow["expanded_rows"]
        ]:
            expanded_mismatches.append(query["query_id"])

        base_packed_ids = [row["chunk_id"] for row in baseline["packed_rows"]]
        qdrant_packed_ids = [row["chunk_id"] for row in shadow["packed_rows"]]
        base_recall = evidence_recall(
            spans,
            [{"chunk_id": cid} for cid in base_packed_ids],
            chunks_by_id,
            len(base_packed_ids),
        )
        q_recall = evidence_recall(
            spans,
            [{"chunk_id": cid} for cid in qdrant_packed_ids],
            chunks_by_id,
            len(qdrant_packed_ids),
        )
        baseline_packed.append(base_recall)
        qdrant_packed.append(q_recall)
        if base_packed_ids != qdrant_packed_ids:
            packed_id_mismatches.append(query["query_id"])
        if baseline["packed_fingerprint"] != shadow["packed_fingerprint"]:
            packed_fingerprint_mismatches.append(query["query_id"])
        if q_recall < base_recall:
            packed_regressions.append(query["query_id"])

    return {
        "seed_mismatches": seed_mismatches,
        "expanded_mismatches": expanded_mismatches,
        "packed_id_mismatches": packed_id_mismatches,
        "packed_fingerprint_mismatches": packed_fingerprint_mismatches,
        "packed_regressions": packed_regressions,
        "local_packed_recall": round(sum(baseline_packed) / len(baseline_packed), 8),
        "qdrant_packed_recall": round(sum(qdrant_packed) / len(qdrant_packed), 8),
        "gating_count": len(gating),
    }


def main() -> int:
    if not _docker_available():
        print("BLOCKED: Docker unavailable")
        return 2

    resolve_local_model_snapshot()
    port = _free_port()
    container_name = f"content-agent-5d4c-{port}"
    index_dir = ROOT / ".phase5d4c_docker_index"
    index_dir.mkdir(exist_ok=True)
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    import os

    from agent.kb_backend import qdrant_serving

    with _docker_qdrant(port, container_name) as (url, client, qdrant_version):
        os.environ["QDRANT_URL"] = url
        os.environ["KB_BACKEND"] = "cswp_qdrant"
        qdrant_serving._serving_client = lambda: client  # type: ignore[method-assign]
        clear_serving_cache()

        manifest = full_rebuild(client, index_dir=index_dir)
        create_serving_alias(client, manifest["collection_name"])
        clear_serving_cache()

        info = validate_startup()
        assert info["kb_backend"] == QDRANT_BACKEND
        assert info["startup_validated"] is True
        assert info["point_count"] == 159
        assert info["live_collection_fingerprint"] == manifest["index_fingerprint"]

        timings = warmup()
        assert timings["kb_backend"] == QDRANT_BACKEND
        assert timings["startup_validated"] is True

        report = _parity_report(client)
        report["qdrant_version"] = qdrant_version
        report["live_collection_fingerprint"] = info["live_collection_fingerprint"]
        print(json.dumps(report, indent=2))
        if report["gating_count"] != 33:
            return 1
        if (
            report["seed_mismatches"]
            or report["expanded_mismatches"]
            or report["packed_id_mismatches"]
            or report["packed_fingerprint_mismatches"]
            or report["packed_regressions"]
        ):
            return 1
        if report["local_packed_recall"] != 0.93939394:
            return 1
        if report["qdrant_packed_recall"] != 0.93939394:
            return 1
    print("PHASE-5D4C-DOCKER-ACCEPTANCE-PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
