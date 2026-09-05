"""Deterministic Stage 2D evidence-exposure preflight tests. No providers."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evidence_exposure_2d_preflight import (
    ASSET_IDS,
    CELL_IDS,
    HARD_SPEND_CEILING_USD,
    PAIRED_ASSETS,
    EvidenceExposure2DError,
    RunExecutionState,
    build_all_requests,
    build_cell_request,
    build_frozen_pack,
    build_stage2d_client,
    build_stage2d_client_config,
    execute_cell_once,
    execute_provider_if_authorized,
    load_2c_case,
    load_pack,
    map_provider_to_semantic_fixture,
    preflight_budget,
    provider_execution_authorized,
    run_preflight,
    score_mocked_provider_output,
    score_run_results,
    shadow_runtime_acceptance,
    sha256_text,
    validate_cell_request_artifact,
    validate_pack,
    verify_paired_isolation,
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


def _mock_response(raw: str, *, model: str = "deepseek-chat", response_id: str = "resp-1"):
    return SimpleNamespace(
        id=response_id,
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


def test_exact_seven_assets_and_ten_cells():
    pack = load_pack()
    assert len(pack["assets"]) == 7
    assert [asset["asset_id"] for asset in pack["assets"]] == list(ASSET_IDS)
    cell_ids: list[str] = []
    for asset in pack["assets"]:
        cell_ids.extend(cell["cell_id"] for cell in asset["cells"])
    assert cell_ids == list(CELL_IDS)
    assert len(cell_ids) == 10


def test_paired_arm_membership():
    pack = load_pack()
    for asset_id in PAIRED_ASSETS:
        asset = next(item for item in pack["assets"] if item["asset_id"] == asset_id)
        arms = {cell["exposure_arm"] for cell in asset["cells"]}
        assert arms == {"prefix", "complete"}
        assert len(asset["cells"]) == 2
    single_assets = [aid for aid in ASSET_IDS if aid not in PAIRED_ASSETS]
    for asset_id in single_assets:
        asset = next(item for item in pack["assets"] if item["asset_id"] == asset_id)
        assert len(asset["cells"]) == 1
        assert asset["cells"][0]["exposure_arm"] == "complete"


def test_fixed_draft_and_source_identity():
    pack = load_pack()
    built = build_frozen_pack()
    for live, frozen in zip(pack["assets"], built["assets"], strict=True):
        assert live["asset_id"] == frozen["asset_id"]
        assert live["draft_sha256"] == frozen["draft_sha256"]
        assert live["source"]["source_sha256"] == frozen["source"]["source_sha256"]
        assert live["draft_text"] == frozen["draft_text"]
        assert live["source"]["source_text"] == frozen["source"]["source_text"]


def test_duplicate_stage_2d_identities_reject():
    pack = load_pack()
    dup_cell = copy.deepcopy(pack)
    dup_cell["assets"][0]["cells"].append(copy.deepcopy(dup_cell["assets"][0]["cells"][0]))
    with pytest.raises(EvidenceExposure2DError, match="duplicate cell id"):
        validate_pack(dup_cell)

    dup_asset = copy.deepcopy(pack)
    dup_asset["assets"][1]["asset_id"] = dup_asset["assets"][0]["asset_id"]
    with pytest.raises(EvidenceExposure2DError, match="duplicate asset id"):
        validate_pack(dup_asset)

    dup_sem = copy.deepcopy(pack)
    dup_sem["assets"][1]["semantic_fixture"]["gold_atoms"].append(
        copy.deepcopy(dup_sem["assets"][1]["semantic_fixture"]["gold_atoms"][0])
    )
    with pytest.raises(EvidenceExposure2DError, match="duplicate semantic identity"):
        validate_pack(dup_sem)


def test_paired_cells_differ_only_in_exposure():
    paired = verify_paired_isolation(load_pack())
    for asset_id in PAIRED_ASSETS:
        assert paired[asset_id]["isolated"] is True


def test_paired_isolation_violation_blocks_preflight_ready():
    pack = load_pack()
    original_build = build_cell_request

    def broken_build(current_pack: dict, cell_id: str) -> dict:
        request = original_build(current_pack, cell_id)
        if cell_id == "P1-COMPLETE":
            request = dict(request)
            request["draft_sha256"] = "0" * 64
        return request

    with patch("scripts.evidence_exposure_2d_preflight.build_cell_request", side_effect=broken_build):
        report = run_preflight(pack)
    assert report["paired_isolation"]["P1"]["isolated"] is False
    assert report["paired_isolation_ok"] is False
    assert report["preflight_ready"] is False


def test_complete_source_context_contains_required_evidence():
    pack = load_pack()
    for cell_id in CELL_IDS:
        if not cell_id.endswith("COMPLETE"):
            continue
        request = build_cell_request(pack, cell_id)
        asset = next(item for item in pack["assets"] if item["asset_id"] == request["asset_id"])
        source_text = asset["source"]["source_text"]
        if asset["asset_id"] == "P1":
            w02 = load_2c_case("E2C-W02")
            span = w02["requirements"][0]["truth_spans"][0]
            evidence = w02["source_text"][span[0] : span[1]]
            assert evidence in request["exposed_verifier_context"]
            assert evidence in source_text
        elif asset["asset_id"] == "P6":
            w03 = load_2c_case("E2C-W03")
            spans = w03["requirements"][0]["truth_spans"]
            for span in spans:
                evidence = w03["source_text"][span[0] : span[1]]
                assert evidence in request["exposed_verifier_context"]
        elif asset["asset_id"] == "P7":
            k03 = load_2c_case("E2C-K03")
            span = k03["requirements"][0]["truth_spans"][0]
            evidence = k03["source_text"][span[0] : span[1]]
            assert evidence in request["exposed_verifier_context"]


def test_prefix_context_reproduces_known_p1_p6_p7_exposure_loss():
    pack = load_pack()
    mappings = {
        "P1-PREFIX": "E2C-W02",
        "P6-PREFIX": "E2C-W03",
        "P7-PREFIX": "E2C-K03",
    }
    for cell_id, case_id in mappings.items():
        request = build_cell_request(pack, cell_id)
        case = load_2c_case(case_id)
        requirement = case["requirements"][0]
        for span in requirement["truth_spans"]:
            evidence = case["source_text"][span[0] : span[1]]
            if len(requirement["truth_spans"]) == 1:
                assert evidence not in request["exposed_verifier_context"]
            elif span == requirement["truth_spans"][0]:
                assert evidence in request["exposed_verifier_context"]
            else:
                assert evidence not in request["exposed_verifier_context"]


def test_frozen_semantic_expected_dispositions():
    pack = load_pack()
    expected_pass = {"P1": True, "P4": True}
    expected_fail = {"P2", "P3", "P5", "P6", "P7"}
    checks = _report()["semantic_template_checks"]
    for asset_id, should_pass in expected_pass.items():
        assert checks[asset_id]["semantic_pass"] is should_pass
    for asset_id in expected_fail:
        assert checks[asset_id]["semantic_pass"] is False
    for asset in pack["assets"]:
        for cell_id, expected in asset["expected_corrected_semantic_pass"].items():
            if cell_id.endswith("COMPLETE") and asset["asset_id"] in expected_pass:
                assert expected is True
            if asset["asset_id"] in expected_fail:
                assert asset["expected_corrected_semantic_pass"][cell_id] is False


def test_zero_tolerance_gate_definitions():
    pack = load_pack()
    gates = pack["zero_tolerance_gates"]
    assert gates["material_false_verification_rate.v2_numerator"] == 0
    assert gates["automatic_semantic_false_pass_rate.v2_numerator"] == 0


def test_shadow_runtime_adapter_is_read_only():
    high_uvr = [{"status": "unverified"} for _ in range(10)]
    low_uvr = [{"status": "verified"} for _ in range(10)]
    rejected = shadow_runtime_acceptance(high_uvr)
    accepted = shadow_runtime_acceptance(low_uvr)
    assert rejected["accepted"] is False
    assert accepted["accepted"] is True
    assert rejected["uvr"] == pytest.approx(1.0)
    assert accepted["uvr"] == pytest.approx(0.0)


def test_no_provider_call_under_default_execution():
    assert provider_execution_authorized() is False
    with pytest.raises(EvidenceExposure2DError, match="provider execution disabled"):
        execute_provider_if_authorized(load_pack(), "P1-COMPLETE")


def test_no_retry_configuration():
    pack = load_pack()
    assert pack["max_attempts_per_cell"] == 1
    assert pack["provider_retries_disabled"] is True
    request = build_cell_request(pack, "P1-COMPLETE")
    assert request["max_attempts"] == 1
    assert request["provider_retries_disabled"] is True
    config = build_stage2d_client_config()
    assert config["max_retries"] == 0


def test_authorization_flag_false_blocks_execution(monkeypatch):
    monkeypatch.setenv("EVIDENCE_EXPOSURE_2D_EXECUTE", "0")
    assert provider_execution_authorized() is False
    with pytest.raises(EvidenceExposure2DError, match="provider execution disabled"):
        execute_provider_if_authorized(load_pack(), "P1-COMPLETE")


def test_stage2d_client_config_retries_disabled():
    config = build_stage2d_client_config()
    assert config["max_retries"] == 0
    assert config["uses_production_llm_call"] is False
    assert config["uses_tenacity_retry_wrapper"] is False


def test_stage2d_client_uses_zero_retries():
    with patch("scripts.evidence_exposure_2d_preflight.OpenAI") as mock_openai:
        build_stage2d_client(api_key="test-key")
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["max_retries"] == 0


def test_execute_cell_once_never_uses_production_llm_call():
    pack = load_pack()
    request = build_cell_request(pack, "P1-COMPLETE")
    asset = next(item for item in pack["assets"] if item["asset_id"] == "P1")
    raw = json.dumps([_verdict(asset["draft_text"], "verified")])
    call_hook = MagicMock(return_value=_mock_response(raw))
    with patch("agent.nodes._llm_call") as mock_llm_call:
        result = execute_cell_once(pack, request, call_hook=call_hook)
    mock_llm_call.assert_not_called()
    assert call_hook.call_count == 1
    assert result["disposition"] == "PASS"


def test_timeout_marks_cell_invalid_without_second_call():
    pack = load_pack()
    request = build_cell_request(pack, "P1-COMPLETE")
    call_hook = MagicMock(side_effect=APITimeoutError("timeout"))
    result = execute_cell_once(pack, request, call_hook=call_hook)
    assert result["disposition"] == "INVALID"
    assert result["invalid_reason"] == "provider_timeout"
    assert call_hook.call_count == 1


def test_transport_failure_marks_cell_invalid_without_second_call():
    pack = load_pack()
    request = build_cell_request(pack, "P1-COMPLETE")
    call_hook = MagicMock(side_effect=RateLimitError("rate limited", response=MagicMock(), body=None))
    result = execute_cell_once(pack, request, call_hook=call_hook)
    assert result["disposition"] == "INVALID"
    assert result["invalid_reason"] == "provider_api_error"
    assert call_hook.call_count == 1


def test_malformed_provider_json_marks_cell_invalid():
    pack = load_pack()
    request = build_cell_request(pack, "P1-COMPLETE")
    call_hook = MagicMock(return_value=_mock_response("not json"))
    result = execute_cell_once(pack, request, call_hook=call_hook)
    assert result["disposition"] == "INVALID"
    assert result["invalid_reason"] == "provider_parse_error"
    assert call_hook.call_count == 1


def test_returned_model_identity_drift_marks_cell_invalid():
    pack = load_pack()
    request = build_cell_request(pack, "P1-COMPLETE")
    asset = next(item for item in pack["assets"] if item["asset_id"] == "P1")
    raw = json.dumps([_verdict(asset["draft_text"], "verified")])
    state = RunExecutionState()
    state.frozen_returned_model_identity = "deepseek-chat"
    call_hook = MagicMock(return_value=_mock_response(raw, model="deepseek-chat-v2"))
    result = execute_cell_once(pack, request, call_hook=call_hook, run_state=state)
    assert result["disposition"] == "INVALID"
    assert "model identity drift" in result["invalid_reason"]


def test_live_provider_adapter_uses_validated_semantic_evaluator():
    pack = load_pack()
    asset = next(item for item in pack["assets"] if item["asset_id"] == "P6")
    grounding = [_verdict(asset["draft_text"], "verified")]
    fixture = map_provider_to_semantic_fixture(asset, grounding)
    from scripts.evidence_exposure_2d_preflight import evaluate_semantic_oracle

    result = evaluate_semantic_oracle(fixture)
    assert result["oracle"]["semantic_pass"] is False
    assert result["metrics"]["material_false_verification_rate.v2"]["numerator"] == 1


def test_stale_exposed_context_hash_rejects():
    pack = load_pack()
    request = build_cell_request(pack, "P1-COMPLETE")
    request["exposed_context_sha256"] = "0" * 64
    with pytest.raises(EvidenceExposure2DError, match="stale or mismatched exposed-context hash"):
        validate_cell_request_artifact(pack, request)


def test_budget_preflight_requires_both_estimates_under_ceiling():
    budget = preflight_budget(load_pack(), ceiling_usd=HARD_SPEND_CEILING_USD)
    assert budget["hard_ceiling_usd"] == 0.08
    assert budget["budget_authorized"] is True
    assert budget["estimated_budget_authorized"] is True
    assert budget["char_bound_budget_authorized"] is True
    assert budget["estimated_max_spend_usd"] <= HARD_SPEND_CEILING_USD
    assert budget["char_bound_max_spend_usd"] <= HARD_SPEND_CEILING_USD
    assert len(budget["cell_estimates"]) == 10
    assert all("input_characters" in item for item in budget["cell_estimates"])


def test_budget_preflight_refuses_oversized_pack():
    budget = preflight_budget(load_pack(), ceiling_usd=0.000001)
    assert budget["budget_authorized"] is False


def test_deterministic_request_construction():
    pack = load_pack()
    first = build_all_requests(pack)
    second = build_all_requests(pack)
    assert [item["cell_id"] for item in first] == list(CELL_IDS)
    assert first == second
    for request in first:
        assert request["exposed_context_sha256"] == sha256_text(request["exposed_verifier_context"])
        assert request["draft_sha256"] == sha256_text(request["draft_text"])
        assert len(request["messages"]) == 2
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][1]["role"] == "user"


def test_no_secrets_serialized_in_requests():
    pack = load_pack()
    blob = json.dumps(build_all_requests(pack))
    for token in ("API_KEY", "Bearer ", "sk-", "DEEPSEEK_API_KEY", "TAVILY"):
        assert token not in blob


def test_preflight_report_ready_without_provider():
    report = _report()
    assert report["cell_count"] == 10
    assert report["asset_count"] == 7
    assert report["provider_execution_default_disabled"] is True
    assert report["paired_isolation_ok"] is True
    assert report["preflight_ready"] is True


def test_post_response_scorer_zero_tolerance_fails_unsafe_cell():
    pack = load_pack()
    asset = next(item for item in pack["assets"] if item["asset_id"] == "P6")
    raw = json.dumps([_verdict(asset["draft_text"], "verified")])
    scored = score_mocked_provider_output(pack, cell_id="P6-COMPLETE", raw_provider_response=raw)
    assert scored["disposition"] == "FAIL"
    assert "material_false_verification" in scored["qualification_fail_reasons"]

    run = score_run_results(pack, [{"cell_id": "P6-COMPLETE", "asset_id": "P6", **scored}])
    assert run["overall_disposition"] == "FAIL"
    assert run["material_false_verification_rate.v2_numerator"] > 0


def test_post_response_scorer_passes_expected_p1_complete():
    pack = load_pack()
    asset = next(item for item in pack["assets"] if item["asset_id"] == "P1")
    raw = json.dumps([_verdict(asset["draft_text"], "verified")])
    scored = score_mocked_provider_output(pack, cell_id="P1-COMPLETE", raw_provider_response=raw)
    assert scored["disposition"] == "PASS"


def test_p6_prefix_unsafe_verified_promotion_fails_qualification():
    pack = load_pack()
    asset = next(item for item in pack["assets"] if item["asset_id"] == "P6")
    raw = json.dumps([_verdict(asset["draft_text"], "verified")])
    scored = score_mocked_provider_output(pack, cell_id="P6-PREFIX", raw_provider_response=raw)
    assert scored["disposition"] == "FAIL"
    assert "high_risk_prefix_unsafe_outcome" in scored["qualification_fail_reasons"]


def test_cli_preflight_does_not_call_provider():
    pack = load_pack()
    with patch("scripts.evidence_exposure_2d_preflight.build_stage2d_client") as mock_client:
        report = run_preflight(pack)
        mock_client.assert_not_called()
    assert report["preflight_ready"] is True
