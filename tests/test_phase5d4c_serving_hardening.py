"""Phase 5D4C Qdrant serving contract hardening + real Docker acceptance."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import defaultdict

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from agent.cswp.constants import PRODUCTION_INDEX_DIR, ROOT
from agent.cswp.loader import load_units_jsonl
from agent.kb_backend import retrieve_kb
from agent.kb_backend.qdrant_serving import (
    clear_serving_cache,
    create_serving_alias,
    validate_startup,
)
from agent.qualified_rag import retrieve_qualified_kb
from agent.retrieval.encoder import MODEL_REVISION, resolve_local_model_snapshot
from agent.retrieval.expansion import eval_chunks_from_manifest
from agent.retrieval.metrics import evidence_recall
from agent.shadow_qdrant.constants import SERVING_ALIAS
from agent.shadow_qdrant.index import full_rebuild
from agent.shadow_qdrant.retrieval import clear_shadow_retrieval_cache
from scripts.phase5a0_baseline import load_sources_and_evidence
from scripts.retrieval_golden_v2 import load_oracle, validate_oracle


def _docker_available() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"],
        capture_output=True,
        check=False,
    ).returncode == 0


@pytest.fixture(scope="module")
def minilm_ready():
    """Conftest offline guard allows loopback for in-memory Qdrant."""
    resolve_local_model_snapshot()
    return MODEL_REVISION


@pytest.fixture
def memory_client():
    return QdrantClient(":memory:")


@pytest.fixture
def built_index(memory_client, minilm_ready, tmp_path, monkeypatch):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = full_rebuild(memory_client, index_dir=index_dir)
    create_serving_alias(memory_client, manifest["collection_name"])
    monkeypatch.setattr(
        "agent.kb_backend.qdrant_serving._serving_client",
        lambda: memory_client,
    )
    clear_shadow_retrieval_cache()
    clear_serving_cache()
    return memory_client, index_dir, manifest


def test_rejects_physical_collection_without_alias(
    memory_client, minilm_ready, tmp_path, monkeypatch
):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = full_rebuild(memory_client, index_dir=index_dir)
    memory_client.create_collection(
        collection_name=SERVING_ALIAS,
        vectors_config=VectorParams(size=manifest["vector_size"], distance=Distance.COSINE),
    )
    monkeypatch.setattr(
        "agent.kb_backend.qdrant_serving._serving_client",
        lambda: memory_client,
    )
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()

    with pytest.raises(RuntimeError, match="physical collection without alias"):
        retrieve_kb("support vector machine margin", n_seeds=5)


def test_index_fingerprint_mandatory_without_env_flag(built_index, monkeypatch):
    _client, _index_dir, _manifest = built_index
    monkeypatch.delenv("KB_CSWP_REQUIRE_INDEX_FINGERPRINT", raising=False)
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()
    info = validate_startup()
    assert info["index_fingerprint"]
    assert info["startup_validated"] is True
    assert info["point_count"] == 159


def test_startup_validation_contract_fields(built_index, monkeypatch):
    _client, _index_dir, _manifest = built_index
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()
    info = validate_startup()
    assert info["vector_size"] == 384
    assert info["distance"] == "cosine"
    assert info["payload_schema_version"] == "cswp_shadow_payload_v1"
    assert info["cswp_compiler_version"] == "phase5d2-cswp-compiler-v1"
    assert info["minilm_model_revision"] == MODEL_REVISION
    assert info["minilm_model_id"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_wrong_vector_size_aborts_startup(
    memory_client, minilm_ready, tmp_path, monkeypatch
):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = full_rebuild(memory_client, index_dir=index_dir)
    bad = f"{manifest['collection_name']}_bad_dim"
    memory_client.create_collection(
        collection_name=bad,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE),
    )
    create_serving_alias(memory_client, bad)
    monkeypatch.setattr(
        "agent.kb_backend.qdrant_serving._serving_client",
        lambda: memory_client,
    )
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()

    with pytest.raises(RuntimeError, match="vector size"):
        validate_startup()


def test_atomic_alias_swap_is_single_transaction(
    memory_client, minilm_ready, tmp_path, monkeypatch
):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = full_rebuild(memory_client, index_dir=index_dir)
    collection = manifest["collection_name"]
    decoy = f"{collection}_decoy"
    memory_client.create_collection(
        collection_name=decoy,
        vectors_config=VectorParams(size=manifest["vector_size"], distance=Distance.COSINE),
    )

    calls: list[list] = []
    original = memory_client.update_collection_aliases

    def _capture(*, change_aliases_operations):
        calls.append(change_aliases_operations)
        return original(change_aliases_operations=change_aliases_operations)

    monkeypatch.setattr(memory_client, "update_collection_aliases", _capture)

    create_serving_alias(memory_client, collection)
    assert len(calls) == 1
    assert len(calls[0]) == 1

    create_serving_alias(memory_client, decoy)
    assert len(calls) == 2
    assert len(calls[1]) == 2
    assert calls[1][0].delete_alias.alias_name == SERVING_ALIAS
    assert calls[1][1].create_alias.collection_name == decoy


def test_retrieve_includes_stage_telemetry(built_index, monkeypatch):
    _client, _index_dir, _manifest = built_index
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()
    result = retrieve_kb("gradient descent learning rate", n_seeds=5)
    telemetry = result["stage_telemetry"]
    for key in ("hydrate_ms", "dense_ms", "bm25_ms", "fusion_ms", "expand_ms", "pack_ms", "total_ms"):
        assert key in telemetry
        assert telemetry[key] >= 0


def _parity_report(client, monkeypatch):
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
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
        ROOT,
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    gating = [q for q in oracle["queries"] if q["gating_eligible"]]
    rrf_mismatches = []
    packed_regressions = []
    baseline_packed = []
    qdrant_packed = []

    for query in gating:
        baseline = retrieve_qualified_kb(query["query"], n_seeds=5)
        shadow = retrieve_kb(query["query"], n_seeds=5)
        spans = spans_by[query["query_id"]]

        base_seeds = [s["chunk_id"] for s in baseline["retrieval_seeds"]]
        qdrant_seeds = [s["chunk_id"] for s in shadow["retrieval_seeds"]]
        if base_seeds != qdrant_seeds:
            rrf_mismatches.append(query["query_id"])

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
        if q_recall < base_recall:
            packed_regressions.append(query["query_id"])

    return {
        "rrf_mismatches": rrf_mismatches,
        "packed_regressions": packed_regressions,
        "local_packed_recall": round(sum(baseline_packed) / len(baseline_packed), 8),
        "qdrant_packed_recall": round(sum(qdrant_packed) / len(qdrant_packed), 8),
        "gating_count": len(gating),
    }


def test_parity_33_gating_queries_in_memory(built_index, monkeypatch):
    client, _index_dir, _manifest = built_index
    report = _parity_report(client, monkeypatch)
    assert report["gating_count"] == 33
    assert report["rrf_mismatches"] == []
    assert report["packed_regressions"] == []
    assert report["local_packed_recall"] == 0.93939394
    assert report["qdrant_packed_recall"] == 0.93939394


@pytest.mark.skipif(not _docker_available(), reason="Docker unavailable")
def test_docker_qdrant_v192_33_query_parity():
    """Fresh subprocess avoids CSWP sys.addaudithook from other test modules."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/phase5d4c_docker_acceptance.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PHASE-5D4C-DOCKER-ACCEPTANCE-PASS" in result.stdout


def test_env_example_does_not_document_optional_fingerprint():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "KB_CSWP_REQUIRE_INDEX_FINGERPRINT" not in example
