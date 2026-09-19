"""Phase 5B3 production qualified RAG integration gates."""

import json

from agent.drafter_packed_evidence import DRAFTER_PACKED_EVIDENCE_V1
from agent.qualified_rag import (
    build_drafter_kb_context,
    is_qualified_kb,
    retrieve_qualified_kb,
)
from agent.semantic_analyzer.contract import build_evidence_manifest
from scripts import phase5b3_production_integration as b3


def test_qualified_kb_uses_cswp_not_legacy_clip():
    result = retrieve_qualified_kb("support vector machine margin", n_seeds=5)
    kb = result["kb_results"]
    assert kb
    assert is_qualified_kb(kb)
    context = build_drafter_kb_context(kb)
    assert "[KB seed_rank=" in context
    assert kb[0]["text"] in context
    assert len(context) >= len(kb[0]["text"])


def test_verifier_manifest_includes_all_packed_units():
    result = retrieve_qualified_kb("gradient descent optimizer", n_seeds=5)
    manifest = build_evidence_manifest(
        request_id="req-test",
        kb_results=result["kb_results"],
    )
    kb_manifest = [row for row in manifest if row["provenance"]["kind"] == "kb"]
    assert len(kb_manifest) == len(result["kb_results"])
    assert kb_manifest[0]["provenance"]["qualified_contract"] == DRAFTER_PACKED_EVIDENCE_V1


def test_cswp_units_respect_minilm_content_limit():
    from scripts import phase5b1_candidate_c as c

    _manifest, units, _by_source = c.load_units()
    for unit in units.values():
        assert unit["embedding_content_token_count"] <= 254
        assert unit["embedding_total_token_count"] <= 256


def test_frozen_contract_and_production_metrics():
    contract = b3.CONTRACT.read_text(encoding="utf-8")
    assert "phase5b3_production_integration_v1" in contract
    result = json.loads((b3.OUTPUT / "results.json").read_text(encoding="utf-8"))
    agg = result["aggregate_gating_33"]["exposure_recall"]
    assert agg["retrieved"] == 0.78787879
    assert agg["expanded"] == 0.93939394
    assert agg["packed"] == 0.93939394
    assert agg["drafter_exposed"] == 0.93939394
    assert agg["verifier_exposed"] == 0.93939394
    assert result["quality_gate"]["production_integration_ready"] is True
    assert not result["seed_identity_mismatches"]
    assert result["focus"]["Q22"]["packed"] == 1.0
    for qid in ("Q19", "Q21", "Q30"):
        assert result["focus"][qid]["packed"] == 0.5


def test_two_runs_are_byte_identical():
    first = (b3.OUTPUT / "run-1/results.json").read_bytes()
    second = (b3.OUTPUT / "run-2/results.json").read_bytes()
    assert first == second
