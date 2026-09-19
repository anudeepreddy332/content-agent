"""Phase 5D4B explicit CSWP serving backend gates."""

from __future__ import annotations

import json
from collections import defaultdict

import pytest
from qdrant_client import QdrantClient

from agent.cswp.constants import PRODUCTION_INDEX_DIR, ROOT
from agent.cswp.loader import load_units_jsonl
from agent.cswp.offline import install_offline_guard
from agent.kb_backend import retrieve_kb, warmup
from agent.kb_backend.qdrant_serving import (
    BACKEND_NAME as QDRANT_BACKEND,
    clear_serving_cache,
    create_serving_alias,
    payload_to_unit,
)
from agent.kb_backend.selector import KBBackendConfigError, resolve_kb_backend
from agent.qualified_rag import retrieve_qualified_kb
from agent.retrieval.encoder import MODEL_REVISION, resolve_local_model_snapshot
from agent.retrieval.metrics import evidence_recall
from agent.shadow_qdrant.constants import PRODUCTION_COLLECTION, SERVING_ALIAS
from agent.shadow_qdrant.index import full_rebuild
from agent.shadow_qdrant.retrieval import clear_shadow_retrieval_cache
from scripts.phase5a0_baseline import load_sources_and_evidence
from scripts.retrieval_golden_v2 import load_oracle, validate_oracle


@pytest.fixture(scope="module")
def offline():
    install_offline_guard()


@pytest.fixture(scope="module")
def minilm_ready(offline):
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


def test_default_backend_is_local(monkeypatch):
    monkeypatch.delenv("KB_BACKEND", raising=False)
    assert resolve_kb_backend() == "cswp_local"


@pytest.mark.parametrize("bad", ["memory", "qdrant", "auto", "unknown"])
def test_unknown_backend_aborts(bad, monkeypatch):
    monkeypatch.setenv("KB_BACKEND", bad)
    with pytest.raises(KBBackendConfigError):
        resolve_kb_backend()


def test_qdrant_backend_requires_serving_alias(memory_client, minilm_ready, tmp_path, monkeypatch):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = full_rebuild(memory_client, index_dir=index_dir)
    monkeypatch.setattr(
        "agent.kb_backend.qdrant_serving._serving_client",
        lambda: memory_client,
    )
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()

    with pytest.raises(RuntimeError, match="serving alias"):
        retrieve_kb("support vector machine margin", n_seeds=5)

    create_serving_alias(memory_client, manifest["collection_name"])
    clear_serving_cache()
    result = retrieve_kb("support vector machine margin", n_seeds=5)
    assert result["kb_backend"] == QDRANT_BACKEND
    assert result["collection_name"] == manifest["collection_name"]
    assert result["serving_alias"] == SERVING_ALIAS
    assert PRODUCTION_COLLECTION not in result["collection_name"]


def test_payload_hydration_without_local_units_jsonl(built_index, monkeypatch):
    _client, index_dir, _manifest = built_index
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()

    # Corrupt local units.jsonl — Qdrant path must still match qualified local baseline
    units_path = index_dir / "units.jsonl"
    units_path.write_text('{"chunk_id":"bogus"}\n', encoding="utf-8")

    oracle = load_oracle()
    query = next(q for q in oracle["queries"] if q["gating_eligible"])["query"]
    local = retrieve_qualified_kb(query, n_seeds=5)
    qdrant = retrieve_kb(query, n_seeds=5)
    assert [r["chunk_id"] for r in local["packed_rows"]] == [
        r["chunk_id"] for r in qdrant["packed_rows"]
    ]


def test_alias_swap_and_rollback(memory_client, minilm_ready, tmp_path, monkeypatch):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(
        "agent.kb_backend.qdrant_serving._serving_client",
        lambda: memory_client,
    )
    manifest = full_rebuild(memory_client, index_dir=index_dir)
    collection = manifest["collection_name"]

    # Temporary decoy collection to exercise alias repointing
    decoy = f"{collection}_decoy"
    memory_client.create_collection(
        collection_name=decoy,
        vectors_config={
            "size": manifest["vector_size"],
            "distance": "Cosine",
        },
    )
    create_serving_alias(memory_client, decoy)
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()

    with pytest.raises(RuntimeError, match="empty"):
        retrieve_kb("gradient descent learning rate", n_seeds=5)

    create_serving_alias(memory_client, collection)
    clear_serving_cache()
    before = retrieve_kb("gradient descent learning rate", n_seeds=5)

    create_serving_alias(memory_client, decoy)
    clear_serving_cache()
    with pytest.raises(RuntimeError):
        retrieve_kb("gradient descent learning rate", n_seeds=5)

    create_serving_alias(memory_client, collection)
    clear_serving_cache()
    after_rollback = retrieve_kb("gradient descent learning rate", n_seeds=5)
    assert after_rollback["collection_name"] == collection
    assert before["packed_fingerprint"] == after_rollback["packed_fingerprint"]


def _parity_report(_client, monkeypatch):
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
    from agent.retrieval.expansion import eval_chunks_from_manifest

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
    packed_regressions = []
    baseline_packed = []
    qdrant_packed = []

    for query in gating:
        baseline = retrieve_qualified_kb(query["query"], n_seeds=5)
        shadow = retrieve_kb(query["query"], n_seeds=5)
        spans = spans_by[query["query_id"]]

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
        "packed_regressions": packed_regressions,
        "local_packed_recall": round(sum(baseline_packed) / len(baseline_packed), 8),
        "qdrant_packed_recall": round(sum(qdrant_packed) / len(qdrant_packed), 8),
        "gating_count": len(gating),
    }


def test_parity_33_gating_queries(built_index, monkeypatch):
    _client, _index_dir, _manifest = built_index
    report = _parity_report(_client, monkeypatch)
    assert report["gating_count"] == 33
    assert report["packed_regressions"] == []
    assert report["local_packed_recall"] == 0.93939394
    assert report["qdrant_packed_recall"] == 0.93939394


def test_qdrant_backend_does_not_import_legacy_query_kb():
    source = (ROOT / "agent/kb_backend/qdrant_serving.py").read_text(encoding="utf-8")
    assert "tools.query_kb" not in source
    assert "tools.save_to_kb" not in source
    assert "QDRANT_COLLECTION" not in source


def test_local_warmup_keys(minilm_ready, monkeypatch):
    monkeypatch.delenv("KB_BACKEND", raising=False)
    timings = warmup()
    assert timings["kb_backend"] == "cswp_local"
    assert "encoder_load_ms" in timings
    assert "cswp_index_ms" in timings


def test_qdrant_warmup_keys(built_index, monkeypatch):
    _client, _index_dir, _manifest = built_index
    monkeypatch.setenv("KB_BACKEND", "cswp_qdrant")
    clear_serving_cache()
    timings = warmup()
    assert timings["kb_backend"] == QDRANT_BACKEND
    assert timings["serving_alias"] == SERVING_ALIAS
    assert timings["point_count"] == 159
    assert "cswp_hydrate_ms" in timings


def test_payload_to_unit_roundtrip(built_index):
    _client, index_dir, _manifest = built_index
    unit = load_units_jsonl(index_dir / "units.jsonl")[0]
    from agent.shadow_qdrant.payload import build_point_payload

    payload = build_point_payload(unit, compiler_version="test")
    restored = payload_to_unit(payload)
    for key in (
        "chunk_id",
        "document_id",
        "retrieval_text",
        "previous_chunk_id",
        "next_chunk_id",
        "reading_order_ordinal",
    ):
        assert restored[key] == unit[key]
