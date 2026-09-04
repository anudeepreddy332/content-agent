"""Deterministic Stage 2D evidence-exposure preflight tests. No providers."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evidence_exposure_2d_preflight import (
    ASSET_IDS,
    CELL_IDS,
    HARD_SPEND_CEILING_USD,
    MAX_ATTEMPTS_PER_CELL,
    PAIRED_ASSETS,
    PROVIDER_RETRIES_DISABLED,
    EvidenceExposure2DError,
    build_all_requests,
    build_cell_request,
    build_frozen_pack,
    execute_provider_if_authorized,
    load_2c_case,
    load_pack,
    preflight_budget,
    provider_execution_authorized,
    run_preflight,
    shadow_runtime_acceptance,
    sha256_text,
    validate_pack,
    verify_paired_isolation,
)


def _report() -> dict:
    return run_preflight(load_pack())


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
    assert pack["provider_retries_disabled"] is PROVIDER_RETRIES_DISABLED
    request = build_cell_request(pack, "P1-COMPLETE")
    assert request["max_attempts"] == MAX_ATTEMPTS_PER_CELL
    assert request["provider_retries_disabled"] is True


def test_budget_preflight_authorizes_under_hard_ceiling():
    budget = preflight_budget(load_pack(), ceiling_usd=HARD_SPEND_CEILING_USD)
    assert budget["hard_ceiling_usd"] == 0.08
    assert budget["budget_authorized"] is True
    assert budget["conservative_max_spend_usd"] <= HARD_SPEND_CEILING_USD
    assert len(budget["cell_estimates"]) == 10


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
    assert report["preflight_ready"] is True
