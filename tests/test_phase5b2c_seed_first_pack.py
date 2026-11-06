"""Phase 5B2C seed-first packing + combined exposure gates."""

from scripts import phase5a2_shadow_ab as ab
from scripts import phase5b1_candidate_c as c
from scripts import phase5b2a_drafter_pack_contract as b2a
from scripts import phase5b2c_seed_first_pack as b2c


def test_seed_first_passes_all_seeds_before_neighbors():
    units = {
        "s1": {"retrieval_text": "a" * 100},
        "n1": {"retrieval_text": "b" * 100},
        "s2": {"retrieval_text": "c" * 100},
    }
    rows = [
        {"chunk_id": "n1", "relation": "NEXT", "seed_rank": 1, "seed_chunk_id": "s1"},
        {"chunk_id": "s1", "relation": "SEED", "seed_rank": 1, "seed_chunk_id": "s1"},
        {"chunk_id": "s2", "relation": "SEED", "seed_rank": 2, "seed_chunk_id": "s2"},
    ]
    encoding = c.cl100k()
    packed, skipped, used, exhausted, stats = c.pack_units_seed_first(
        rows, units, 500, encoding
    )
    assert [r["chunk_id"] for r in packed[:2]] == ["s1", "s2"]
    assert stats["seeds_retained"] == 2
    assert stats["neighbors_retained"] == 1
    assert exhausted is False
    c.assert_seed_preservation_invariant(rows, units, 500, encoding)


def test_seed_preservation_invariant_fails_closed_when_seeds_exceed_budget():
    units = {"s1": {"retrieval_text": "word " * 500}}
    rows = [{"chunk_id": "s1", "relation": "SEED", "seed_rank": 1, "seed_chunk_id": "s1"}]
    encoding = c.cl100k()
    try:
        c.pack_units_seed_first(rows, units, 80, encoding)
        raise AssertionError("expected fail closed")
    except c.CandidateCError as exc:
        assert "seed-only cl100k tokens" in str(exc)


def test_no_top3_gate_allows_seed_first_noncontiguous_groups():
    units = {
        "a": {"retrieval_text": "alpha"},
        "b": {"retrieval_text": "beta"},
        "c": {"retrieval_text": "gamma"},
        "d": {"retrieval_text": "delta"},
    }
    rows = [
        {
            "chunk_id": "a",
            "relation": "SEED",
            "seed_rank": 1,
            "seed_chunk_id": "a",
            "source_intervals": [(0, 1)],
            "source_path": "p",
            "document_id": "d",
            "document_version": "v",
            "source_sha256": "s",
        },
        {
            "chunk_id": "b",
            "relation": "SEED",
            "seed_rank": 4,
            "seed_chunk_id": "b",
            "source_intervals": [(0, 1)],
            "source_path": "p",
            "document_id": "d",
            "document_version": "v",
            "source_sha256": "s",
        },
        {
            "chunk_id": "c",
            "relation": "NEXT",
            "seed_rank": 1,
            "seed_chunk_id": "a",
            "source_intervals": [(0, 1)],
            "source_path": "p",
            "document_id": "d",
            "document_version": "v",
            "source_sha256": "s",
        },
    ]
    serialized, _, _ = b2a.packed_evidence_identity(rows, units)
    assert b2a.verify_no_top3_gate(serialized, rows)


def test_frozen_contract_and_deterministic_runs():
    contract = b2c.read(b2c.CONTRACT)
    b2c.verify_contract(contract)
    first = (b2c.OUTPUT / "run-1/results.json").read_bytes()
    second = (b2c.OUTPUT / "run-2/results.json").read_bytes()
    assert first == second
    result = b2c.read(b2c.OUTPUT / "results.json")
    runtime = b2c.read(b2c.OUTPUT / "run-1/runtime.json")
    assert runtime["result_sha256"] == ab.sha256_json(result)
    assert result["provider_calls"] == result["external_network_calls"] == 0


def test_combined_exposure_layers_and_focus_queries():
    result = b2c.read(b2c.OUTPUT / "results.json")
    agg = result["aggregate_gating_33"]["exposure_recall"]
    assert agg["retrieved"] == 0.78787879
    assert agg["expanded"] == 0.93939394
    assert agg["seed_first_packed"] == 0.93939394
    assert agg["packed_contract_drafter"] == 0.93939394
    assert agg["verifier_exposed"] == 0.93939394
    focus = result["focus"]
    assert focus["Q22"]["seed_first_packed"] == 1.0
    assert focus["Q22"]["vs_expanded"] == "SAME"
    assert focus["Q09"]["seed_first_packed"] == 1.0
    assert focus["Q13"]["seed_first_packed"] == 1.0
    for qid in ("Q19", "Q21", "Q30"):
        row = focus[qid]
        assert row["seed_first_packed"] == row["retrieved"] == 0.5
    assert result["quality_gate"]["combined_contract_ready_for_live_wiring"] is True
    assert not result["forensics_summary"]["provenance_failures"]
    assert not result["forensics_summary"]["regressions_vs_expanded"]
    assert not result["forensics_summary"]["seeds_exceeding_2000"]
