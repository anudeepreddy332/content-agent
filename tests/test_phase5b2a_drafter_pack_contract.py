"""Phase 5B2A drafter packed-evidence shadow contract gates."""

from agent.drafter_packed_evidence import (
    DRAFTER_PACKED_EVIDENCE_V1,
    LEGACY_DRAFTER_CHAR_LIMIT,
    LEGACY_DRAFTER_K,
    build_packed_seed_groups,
    serialize_drafter_packed_evidence_v1,
    serialize_legacy_drafter_kb,
)
from scripts import phase5a2_shadow_ab as ab
from scripts import phase5b2a_drafter_pack_contract as b2a


def synthetic_unit(chunk_id, source, start, end, text="body"):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-" + source,
        "document_version": "ver-" + source,
        "source_path": "kb/seed_docs/" + source + ".md",
        "source_sha256": "sha-" + source,
        "retrieval_char_start": start,
        "retrieval_char_end": end,
        "heading_path": [],
        "retrieval_text": "Title\n\n" + text,
        "canonical_content": text,
        "source_spans": [
            {"source_char_start": start, "source_char_end": end, "role": "core"}
        ],
        "embedding_content_token_count": 1,
    }


def test_build_packed_seed_groups_preserves_pack_order():
    rows = [
        {"seed_rank": 1, "seed_chunk_id": "a", "chunk_id": "a", "relation": "SEED"},
        {"seed_rank": 1, "seed_chunk_id": "a", "chunk_id": "b", "relation": "NEXT"},
        {"seed_rank": 3, "seed_chunk_id": "c", "chunk_id": "c", "relation": "SEED"},
    ]
    groups = build_packed_seed_groups(rows)
    assert [g["seed_rank"] for g in groups] == [1, 3]
    assert [m["chunk_id"] for g in groups for m in g["members"]] == ["a", "b", "c"]


def test_packed_contract_exposes_all_groups_without_secondary_clip():
    units = {
        "a": synthetic_unit("a", "svm", 0, 10, "a" * 3000),
        "b": synthetic_unit("b", "svm", 10, 20, "b"),
        "c": synthetic_unit("c", "svm", 20, 30, "c" * 3000),
    }
    rows = [
        {
            "seed_rank": 1,
            "seed_chunk_id": "a",
            "chunk_id": "a",
            "relation": "SEED",
            "source_intervals": [(0, 10)],
            "source_path": units["a"]["source_path"],
            "document_id": "d",
            "document_version": "v",
            "source_sha256": "sha",
            "cl100k_tokens": 10,
        },
        {
            "seed_rank": 4,
            "seed_chunk_id": "c",
            "chunk_id": "c",
            "relation": "SEED",
            "source_intervals": [(20, 30)],
            "source_path": units["c"]["source_path"],
            "document_id": "d",
            "document_version": "v",
            "source_sha256": "sha",
            "cl100k_tokens": 10,
        },
    ]
    serialized = serialize_drafter_packed_evidence_v1(rows, units)
    assert len(serialized) == 2
    assert serialized[0]["contract"] == DRAFTER_PACKED_EVIDENCE_V1
    assert serialized[1]["seed_rank"] == 4
    assert b2a.verify_no_top3_gate(serialized, rows)
    assert b2a.verify_no_secondary_clip(serialized)
    assert serialized[0]["members"][0]["serialized_text"] == units["a"]["retrieval_text"]


def test_legacy_contract_applies_top3_and_char_clip():
    units = {
        "a": synthetic_unit("a", "svm", 0, 10, "a" * 3000),
        "b": synthetic_unit("b", "svm", 10, 20, "b"),
        "c": synthetic_unit("c", "svm", 20, 30, "c"),
    }
    rows = [
        {
            "seed_rank": rank,
            "seed_chunk_id": cid,
            "chunk_id": cid,
            "relation": "SEED",
            "source_intervals": [(0, 10)],
            "source_path": units[cid]["source_path"],
            "document_id": "d",
            "document_version": "v",
            "source_sha256": "sha",
            "cl100k_tokens": 1,
        }
        for rank, cid in ((1, "a"), (4, "c"))
    ]
    legacy = serialize_legacy_drafter_kb(rows, units)
    assert len(legacy) == 1
    assert legacy[0]["seed_rank"] == 1
    assert len(legacy[0]["members"][0]["serialized_text"]) == LEGACY_DRAFTER_CHAR_LIMIT
    assert legacy[0]["members"][0]["legacy_char_clip_applied"] is True


def test_frozen_contract_and_upstream_b1_link():
    contract = b2a.read(b2a.CONTRACT)
    b2a.verify_contract(contract)
    b1 = b2a.read(b2a.B1_CONTRACT)
    assert contract["upstream_contract_sha256"] == b1["contract_sha256"]
    assert contract["shadow_contract"]["id"] == DRAFTER_PACKED_EVIDENCE_V1
    assert contract["legacy_control"]["k"] == LEGACY_DRAFTER_K


def test_two_runs_are_byte_identical():
    first = (b2a.OUTPUT / "run-1/results.json").read_bytes()
    second = (b2a.OUTPUT / "run-2/results.json").read_bytes()
    assert first == second
    result = b2a.read(b2a.OUTPUT / "run-1/results.json")
    runtime = b2a.read(b2a.OUTPUT / "run-1/runtime.json")
    assert runtime["result_sha256"] == ab.sha256_json(result)
    assert result["provider_calls"] == result["external_network_calls"] == 0
    assert result["production_drafter_changed"] is False
    assert result["production_retrieval_changed"] is False
    assert result["verifier_changed"] is False


def test_exposure_metrics_and_focus_queries():
    result = b2a.read(b2a.OUTPUT / "results.json")
    gating = [row for row in result["per_query"] if row["gating_eligible"]]
    assert len(gating) == 33
    agg = result["aggregate_gating_33"]["exposure_recall"]
    assert agg["packed"] == 0.92424242
    assert agg["legacy_drafter"] == 0.75757576
    assert agg["packed_contract_drafter"] == 0.92424242
    assert agg["verifier_exposed"] == 0.92424242
    for qid in b2a.KNOWN_LEGACY_LOSSES:
        row = next(r for r in result["per_query"] if r["query_id"] == qid)
        assert row["exposure_recall"]["packed_contract_drafter"] == row["exposure_recall"]["packed"]
        assert row["exposure_recall"]["packed_contract_drafter"] >= row["exposure_recall"]["legacy_drafter"]
    q22 = next(r for r in result["per_query"] if r["query_id"] == "Q22")
    assert q22["exposure_recall"]["packed"] == 0.5
    assert q22["exposure_recall"]["packed_contract_drafter"] == 0.5
    assert q22["vs_packed"]["packed_contract_drafter"]["classification"] == "SAME"
    assert result["quality_gate"]["shadow_contract_ready_for_live_wiring"] is True
    assert not result["provenance_failures"]
