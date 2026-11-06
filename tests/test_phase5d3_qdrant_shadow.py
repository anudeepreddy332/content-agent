"""Phase 5D3 durable shadow Qdrant index parity gates."""

from __future__ import annotations

import copy
import json
import uuid
from collections import defaultdict
import numpy as np
import pytest
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from agent.cswp.constants import PRODUCTION_INDEX_DIR, ROOT
from agent.cswp.loader import load_units_jsonl
from agent.cswp.offline import install_offline_guard
from agent.qualified_rag import retrieve_qualified_kb
from agent.retrieval.encoder import MODEL_REVISION, resolve_local_model_snapshot
from agent.retrieval.expansion import eval_chunks_from_manifest
from agent.retrieval.fusion import rank_bm25, rank_dense
from agent.retrieval.metrics import evidence_recall
from agent.shadow_qdrant.constants import (
    PAYLOAD_SCHEMA_VERSION,
    POINT_ID_NAMESPACE,
    PRODUCTION_COLLECTION,
    SERVING_ALIAS,
)
from agent.shadow_qdrant.index import (
    delete_document,
    full_rebuild,
    load_index_manifest,
    point_id_for_chunk,
    update_document,
    upsert_document,
)
from agent.shadow_qdrant.payload import build_point_payload
from agent.shadow_qdrant.retrieval import clear_shadow_retrieval_cache, retrieve_shadow_kb
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
def shadow_built(memory_client, minilm_ready, tmp_path):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = full_rebuild(memory_client, index_dir=index_dir)
    clear_shadow_retrieval_cache()
    return memory_client, index_dir, manifest


def test_deterministic_point_ids(shadow_built):
    _client, index_dir, _manifest = shadow_built
    units = load_units_jsonl(index_dir / "units.jsonl")
    for unit in units:
        expected = str(uuid.uuid5(POINT_ID_NAMESPACE, unit["chunk_id"]))
        assert point_id_for_chunk(unit["chunk_id"]) == expected


def test_payload_complete(shadow_built):
    _client, index_dir, shadow_manifest = shadow_built
    units = load_units_jsonl(index_dir / "units.jsonl")
    unit = units[0]
    payload = build_point_payload(unit, compiler_version=shadow_manifest["cswp_compiler_version"])
    required = {
        "chunk_id",
        "document_id",
        "document_version",
        "source_path",
        "source_sha256",
        "retrieval_text",
        "source_spans",
        "heading_path",
        "structural_segments",
        "embedding_content_token_count",
        "reading_order_ordinal",
        "previous_chunk_id",
        "next_chunk_id",
        "compiler_version",
        "payload_schema_version",
    }
    assert required.issubset(payload.keys())
    assert payload["payload_schema_version"] == PAYLOAD_SCHEMA_VERSION


def test_neighbor_references(shadow_built):
    _client, index_dir, _manifest = shadow_built
    units = load_units_jsonl(index_dir / "units.jsonl")
    by_id = {unit["chunk_id"]: unit for unit in units}
    for unit in units:
        if unit["previous_chunk_id"] is not None:
            prev = by_id[unit["previous_chunk_id"]]
            assert prev["next_chunk_id"] == unit["chunk_id"]
        if unit["next_chunk_id"] is not None:
            nxt = by_id[unit["next_chunk_id"]]
            assert nxt["previous_chunk_id"] == unit["chunk_id"]


def test_minilm_revision_pinned(minilm_ready):
    assert minilm_ready == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def test_rebuild_deterministic(memory_client, minilm_ready, tmp_path):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    first = full_rebuild(memory_client, index_dir=index_dir)
    collection = first["collection_name"]
    memory_client.delete_collection(collection)
    (index_dir / "qdrant_shadow_manifest.json").unlink()
    second = full_rebuild(memory_client, index_dir=index_dir)
    assert first["index_fingerprint"] == second["index_fingerprint"]
    assert first["point_count"] == second["point_count"] == 159


def test_update_delete_lifecycle(memory_client, minilm_ready, tmp_path):
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        src = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    full_rebuild(memory_client, index_dir=index_dir)
    units = load_units_jsonl(index_dir / "units.jsonl")
    sample_path = units[0]["source_path"]
    doc_units = [u for u in units if u["source_path"] == sample_path]
    doc_id = doc_units[0]["document_id"]
    doc_version = doc_units[0]["document_version"]

    deleted = delete_document(memory_client, doc_id, index_dir=index_dir)
    assert deleted == len(doc_units)
    remaining = memory_client.get_collection(load_index_manifest(index_dir)["collection_name"])
    assert remaining.points_count == 159 - len(doc_units)

    upsert_document(memory_client, doc_units, index_dir=index_dir)
    restored = memory_client.get_collection(load_index_manifest(index_dir)["collection_name"])
    assert restored.points_count == 159

    new_version = doc_version + ":next"
    new_units = [copy.deepcopy(unit) for unit in doc_units]
    for unit in new_units:
        unit["document_version"] = new_version

    stats = update_document(
        memory_client,
        doc_version,
        new_units,
        index_dir=index_dir,
    )
    assert stats["deleted"] == len(doc_units)
    assert stats["inserted"] == len(new_units)

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    old_version_count = memory_client.count(
        collection_name=load_index_manifest(index_dir)["collection_name"],
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="document_version", match=MatchValue(value=doc_version)
                )
            ]
        ),
    ).count
    assert old_version_count == 0


def _parity_report(client, index_dir):
    oracle = load_oracle()
    validate_oracle(oracle)
    _, _, evidence = load_sources_and_evidence(oracle)
    spans_by = defaultdict(list)
    for span in evidence:
        spans_by[span.query_id].append(span)

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    representation = {
        "children": load_units_jsonl(index_dir / "units.jsonl"),
        **{k: manifest[k] for k in ("packing_version", "contract_sha256", "frozen_B_fingerprint")},
    }
    chunks = eval_chunks_from_manifest(
        {
            "children": representation["children"],
            "packing_version": manifest["packing_version"],
            "contract_sha256": manifest["contract_sha256"],
            "frozen_B_fingerprint": manifest["frozen_B_fingerprint"],
        },
        ROOT,
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    texts = [chunk.retrieval_text for chunk in chunks]
    bm25 = BM25Okapi([text.lower().split() for text in texts])

    from agent.retrieval.encoder import get_encoder

    encoder = get_encoder()
    embeddings = np.asarray(encoder.encode(texts), dtype=np.float32)

    gating = [q for q in oracle["queries"] if q["gating_eligible"]]
    dense_mismatches = []
    bm25_mismatches = []
    rrf_mismatches = []
    packed_regressions = []
    improved = []
    baseline_packed = []
    shadow_packed = []

    for query in gating:
        qid = query["query_id"]
        spans = spans_by[qid]
        baseline = retrieve_qualified_kb(query["query"], n_seeds=5)
        shadow = retrieve_shadow_kb(client, query["query"], index_dir=index_dir, n_seeds=5)

        query_embedding = np.asarray(encoder.encode(query["query"]), dtype=np.float32)
        mem_dense = rank_dense(query_embedding, embeddings, chunks)[:20]
        mem_bm25 = rank_bm25(query["query"], bm25, chunks)
        if [r["chunk_id"] for r in mem_dense] != [r["chunk_id"] for r in shadow["dense_top20"]]:
            dense_mismatches.append(qid)
        if [r["chunk_id"] for r in mem_bm25] != [
            r["chunk_id"]
            for r in rank_bm25(query["query"], bm25, chunks)
        ]:
            bm25_mismatches.append(qid)

        base_seeds = [s["chunk_id"] for s in baseline["retrieval_seeds"]]
        shadow_seeds = [s["chunk_id"] for s in shadow["retrieval_seeds"]]
        if base_seeds != shadow_seeds:
            rrf_mismatches.append(qid)

        base_packed_ids = [row["chunk_id"] for row in baseline["packed_rows"]]
        shadow_packed_ids = [row["chunk_id"] for row in shadow["packed_rows"]]
        base_recall = evidence_recall(
            spans,
            [{"chunk_id": cid} for cid in base_packed_ids],
            chunks_by_id,
            len(base_packed_ids),
        )
        shadow_recall = evidence_recall(
            spans,
            [{"chunk_id": cid} for cid in shadow_packed_ids],
            chunks_by_id,
            len(shadow_packed_ids),
        )
        baseline_packed.append(base_recall)
        shadow_packed.append(shadow_recall)
        if shadow_recall < base_recall:
            packed_regressions.append(qid)
        elif shadow_recall > base_recall:
            improved.append(qid)

    return {
        "dense_mismatches": dense_mismatches,
        "bm25_mismatches": bm25_mismatches,
        "rrf_mismatches": rrf_mismatches,
        "packed_regressions": packed_regressions,
        "improved": improved,
        "baseline_packed_recall": round(sum(baseline_packed) / len(baseline_packed), 8),
        "shadow_packed_recall": round(sum(shadow_packed) / len(shadow_packed), 8),
        "gating_count": len(gating),
    }


def test_parity_33_gating_queries(shadow_built):
    client, index_dir, _manifest = shadow_built
    report = _parity_report(client, index_dir)
    assert report["gating_count"] == 33
    assert report["dense_mismatches"] == []
    assert report["bm25_mismatches"] == []
    assert report["rrf_mismatches"] == []
    assert report["packed_regressions"] == []
    assert report["baseline_packed_recall"] == 0.93939394
    assert report["shadow_packed_recall"] == 0.93939394


def test_production_collection_not_mutated(shadow_built, memory_client):
    _client, index_dir, manifest = shadow_built
    assert manifest["production_collection_mutated"] is False
    assert PRODUCTION_COLLECTION not in {
        collection.name for collection in memory_client.get_collections().collections
    }
    assert manifest["serving_alias_active"] is False
    assert SERVING_ALIAS != manifest["collection_name"]


def test_qualified_rag_has_no_experiment_imports():
    source = (ROOT / "agent/qualified_rag.py").read_text(encoding="utf-8")
    assert "scripts.phase5a2_shadow_ab" not in source
    assert "scripts.phase5b1_candidate_c" not in source
    assert "reports/" not in source
