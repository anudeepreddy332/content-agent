"""Deterministic verifier semantic-contract 3-cell preflight tests. No providers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verifier_semantic_contract_3cell_preflight import (
    ASSET_IDS,
    CANDIDATE_STATUS_BLOCK,
    CELL_IDS,
    CELL_SOURCE_MAP,
    EXPECTED_PROVIDER_CLASSIFICATION,
    EXPECTED_SEMANTIC_P0_PASS,
    HARD_SPEND_CEILING_USD,
    MAX_OUTPUT_TOKENS,
    MAX_PROVIDER_REQUESTS,
    ORIGINAL_STATUS_PARAGRAPH,
    ExecutionAuthorizationError,
    build_all_requests,
    build_approved_execution_config_hash,
    build_frozen_pack,
    build_provider_client_config,
    load_pack,
    preflight_budget,
    provider_execution_authorized,
    run_preflight,
    score_mocked_provider_output,
    score_run_results,
    sha256_text,
    validate_prompt_diff_only_status,
    validate_stage2d_frozen_identity,
    validate_run_cell_catalog,
)
from scripts.evidence_exposure_2d_preflight import (
    load_verify_system as load_production_verify_system,
    parse_verifier_output,
    shadow_runtime_acceptance,
)


def _report() -> dict:
    return run_preflight(load_pack())


def _verdict(claim: str, status: str, *, source_url: str | None = "https://example.test/src") -> dict:
    return {
        "claim": claim,
        "source_url": source_url,
        "confidence": 0.9,
        "status": status,
        "specificity": "substantive",
    }


def _scored_cell(pack: dict, cell_id: str, *, status: str) -> dict:
    asset = next(item for item in pack["assets"] if item["cell_id"] == cell_id)
    claim = asset["semantic_fixture"]["candidates"][0]["text"]
    raw = json.dumps([_verdict(claim, status)])
    return {"cell_id": cell_id, **score_mocked_provider_output(pack, cell_id=cell_id, raw_provider_response=raw)}


def test_exactly_three_cells():
    pack = load_pack()
    assert len(pack["assets"]) == 3
    assert [asset["cell_id"] for asset in pack["assets"]] == list(CELL_IDS)
    assert [asset["asset_id"] for asset in pack["assets"]] == list(ASSET_IDS)


def test_c1_is_p6_complete_identity():
    pack = load_pack()
    asset = pack["assets"][0]
    assert asset["cell_id"] == "C1"
    assert asset["asset_id"] == "P6"
    assert asset["source_cell_id"] == "P6-COMPLETE"
    assert asset["draft_text"] == "Helix gate permits export when mode flag HG-ENABLE is set."
    assert asset["exposure_arm"] == "complete"


def test_c2_is_p7_complete_identity():
    pack = load_pack()
    asset = pack["assets"][1]
    assert asset["cell_id"] == "C2"
    assert asset["asset_id"] == "P7"
    assert asset["source_cell_id"] == "P7-COMPLETE"
    assert asset["draft_text"] == "Cache tier T3 admits writes when isolation level IL-2 holds."
    assert asset["exposure_arm"] == "complete"


def test_c3_is_p1_complete_identity():
    pack = load_pack()
    asset = pack["assets"][2]
    assert asset["cell_id"] == "C3"
    assert asset["asset_id"] == "P1"
    assert asset["source_cell_id"] == "P1-COMPLETE"
    assert asset["draft_text"] == "Quorum latch engages only after seven replica acknowledgements."
    assert asset["exposure_arm"] == "complete"


def test_complete_evidence_hashes_unchanged_from_stage2d():
    frozen = validate_stage2d_frozen_identity(load_pack())
    assert frozen["all_frozen_identity_ok"] is True
    for cell_id in CELL_IDS:
        cell = frozen["cells"][cell_id]
        assert cell["frozen_identity_ok"] is True
        assert cell["source_cell_id"] == CELL_SOURCE_MAP[cell_id]["source_cell_id"]


def test_only_semantic_status_prompt_paragraph_differs():
    identity = validate_prompt_diff_only_status()
    assert identity["valid"] is True
    assert identity["only_status_paragraph_changed"] is True
    original = load_production_verify_system()
    assert ORIGINAL_STATUS_PARAGRAPH in original
    assert ORIGINAL_STATUS_PARAGRAPH not in Path(
        "prompts/verify_system_semantic_contract_candidate.md"
    ).read_text(encoding="utf-8")


def test_candidate_prompt_contains_full_entailment_rule():
    candidate = Path("prompts/verify_system_semantic_contract_candidate.md").read_text(encoding="utf-8")
    assert CANDIDATE_STATUS_BLOCK in candidate
    assert "fully entails every truth-relevant part" in candidate
    assert "Any material contradiction prevents verified" in candidate
    assert "Conditional or qualified evidence cannot verify a stronger unconditional claim" in candidate


def test_contradiction_blocks_verified_in_contract():
    candidate = Path("prompts/verify_system_semantic_contract_candidate.md").read_text(encoding="utf-8")
    assert "Any material contradiction prevents verified" in candidate


def test_qualifier_condition_blocks_verified_in_contract():
    candidate = Path("prompts/verify_system_semantic_contract_candidate.md").read_text(encoding="utf-8")
    assert "conditions, qualifiers" in candidate
    assert "Conditional or qualified evidence cannot verify" in candidate


def test_expected_labels_weak_weak_verified():
    assert EXPECTED_PROVIDER_CLASSIFICATION == {"C1": "weak", "C2": "weak", "C3": "verified"}


def test_material_fv_gate_zero_on_correct_mock_outputs():
    pack = load_pack()
    results = [
        _scored_cell(pack, "C1", status="weak"),
        _scored_cell(pack, "C2", status="weak"),
        _scored_cell(pack, "C3", status="verified"),
    ]
    report = score_run_results(pack, results)
    assert report["material_false_verification_rate.v2_numerator"] == 0
    assert report["overall_disposition"] == "PASS"


def test_automatic_shadow_false_pass_is_diagnostic_only():
    pack = load_pack()
    results = [
        _scored_cell(pack, "C1", status="weak"),
        _scored_cell(pack, "C2", status="weak"),
        _scored_cell(pack, "C3", status="verified"),
    ]
    assert all(result["automatic_false_pass_diagnostic_only"] for result in results)
    report = score_run_results(pack, results)
    assert report["automatic_false_pass_diagnostic_only"] is True
    # All-weak shadow may accept when no unverified rows — must not fail qualification.
    weak_only = [_scored_cell(pack, cell_id, status="weak") for cell_id in CELL_IDS]
    weak_report = score_run_results(pack, weak_only)
    assert weak_report["automatic_semantic_false_pass_rate.v2_numerator"] >= 0
    assert "automatic_semantic_false_pass" not in (
        weak_report.get("invalid_reasons") or []
    )


def test_routing_unchanged_production_prompt_bytes():
    production = load_production_verify_system()
    assert ORIGINAL_STATUS_PARAGRAPH in production
    assert sha256_text(production) == validate_prompt_diff_only_status()["original_prompt_sha256"]


def test_parser_adapter_evaluator_unchanged_imports():
    sample = json.dumps([_verdict("x", "weak")])
    parsed = parse_verifier_output(sample)
    assert parsed[0]["status"] == "weak"
    shadow = shadow_runtime_acceptance(parsed)
    assert "accepted" in shadow


def test_provider_execution_locked_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VERIFIER_SEMANTIC_CONTRACT_EXECUTE", raising=False)
    assert provider_execution_authorized() is False
    report = _report()
    assert report["provider_execution_default_disabled"] is True


def test_provider_execution_requires_explicit_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VERIFIER_SEMANTIC_CONTRACT_EXECUTE", raising=False)
    with pytest.raises(ExecutionAuthorizationError):
        from scripts.verifier_semantic_contract_3cell_preflight import issue_execution_authorization

        issue_execution_authorization(load_pack())


def test_max_future_http_calls_three():
    assert MAX_PROVIDER_REQUESTS == 3
    report = _report()
    assert report["budget"]["max_provider_requests"] == 3
    assert len(build_all_requests(load_pack())) == 3


def test_budget_ceiling_two_cents():
    budget = preflight_budget(load_pack())
    assert budget["hard_ceiling_usd"] == HARD_SPEND_CEILING_USD == 0.02
    assert budget["budget_authorized"] is True
    assert budget["estimated_max_spend_usd"] <= 0.02
    assert budget["char_bound_max_spend_usd"] <= 0.02


def test_frozen_request_config_and_artifact_construction():
    pack = load_pack()
    requests = build_all_requests(pack)
    assert all(req["model_config"]["max_tokens"] == MAX_OUTPUT_TOKENS for req in requests)
    assert all(req["max_attempts"] == 1 for req in requests)
    assert all(req["provider_retries_disabled"] is True for req in requests)
    config_hash = build_approved_execution_config_hash(pack)
    assert len(config_hash) == 64


def test_provider_client_retry_redirect_safety():
    config = build_provider_client_config()
    assert config["max_retries"] == 0
    assert config["follow_redirects"] is False
    assert config["uses_production_llm_call"] is False


def test_semantic_p0_expected_outcomes():
    assert EXPECTED_SEMANTIC_P0_PASS == {"C1": False, "C2": False, "C3": True}


def test_preflight_ready():
    report = _report()
    assert report["preflight_ready"] is True
    assert report["cell_count"] == 3


def test_run_cell_catalog_exactly_three():
    pack = load_pack()
    results = [_scored_cell(pack, cell_id, status=EXPECTED_PROVIDER_CLASSIFICATION[cell_id]) for cell_id in CELL_IDS]
    catalog = validate_run_cell_catalog(results)
    assert catalog["catalog_valid"] is True
    assert catalog["cell_count"] == 3


def test_build_frozen_pack_matches_committed_fixtures():
    built = build_frozen_pack()
    committed = load_pack()
    assert built["pack_id"] == committed["pack_id"]
    assert [a["cell_id"] for a in built["assets"]] == [a["cell_id"] for a in committed["assets"]]
    assert built["prompt_identity"]["original_prompt_sha256"] == committed["prompt_identity"]["original_prompt_sha256"]
    assert built["prompt_identity"]["candidate_prompt_sha256"] == committed["prompt_identity"]["candidate_prompt_sha256"]
