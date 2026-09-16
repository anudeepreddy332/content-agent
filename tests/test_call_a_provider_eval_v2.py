"""Provider-eval oracle v2 rescoring tests. Zero provider calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.claim_inventory import build_claim_inventory, parse_claim_inventory_rows
from scripts.call_a_provider_eval_v2 import (
    EXPECTED_ORACLE_SHA256,
    evaluate_provider_eval_v2_case,
    load_harness_pack,
    load_oracle,
    rescore_live_run_artifact,
)

HARNESS_PATH = Path("evals/fixtures/claim_inventory_golden_v1.json")
ORACLE_PATH = Path("evals/fixtures/call_a_provider_eval_gold_v2.json")
LIVE_ARTIFACT = Path(
    "outputs/call_a_provider_qualification/runs/call_a_run_20260915T161944Z_1c1d81ac/run_artifact.json"
)
EXPECTED_LIVE_DIGEST = "56873347a00a7473c491cea856de9e7188f6b0af86dbd7841157d33d600dfcf5"
HARNESS_SHA = "112b15ca3b79e16c38d0d9ba94063d60268a4a120414242115a55c5227dae006"


@pytest.fixture(scope="module")
def oracle() -> dict:
    return load_oracle()


@pytest.fixture(scope="module")
def harness() -> dict:
    return load_harness_pack()


@pytest.fixture(scope="module")
def live_artifact() -> dict:
    return json.loads(LIVE_ARTIFACT.read_text(encoding="utf-8"))


def _harness_case(harness: dict, case_id: str) -> dict:
    for case in harness["cases"]:
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)


def _oracle_case(oracle: dict, case_id: str) -> dict:
    for case in oracle["cases"]:
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)


def _live_case(live: dict, case_id: str) -> dict:
    for case in live["case_results"]:
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)


def test_harness_golden_v1_unchanged():
    observed = hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest()
    assert observed == HARNESS_SHA
    pack = json.loads(HARNESS_PATH.read_text())
    assert pack["schema"] == "claim_inventory_golden_v1"
    assert len(pack["cases"]) == 16


def test_oracle_v2_frozen_identity():
    observed = hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest()
    assert observed == EXPECTED_ORACLE_SHA256
    oracle = load_oracle()
    assert oracle["schema"] == "call_a_provider_eval_gold_v2"
    assert oracle["harness_fixture_sha256"] == HARNESS_SHA


def test_g05_atomic_decomposition_scores_correctly(oracle, harness, live_artifact):
    live = _live_case(live_artifact, "G05-factual-definition")
    scored = evaluate_provider_eval_v2_case(
        case_eval=_oracle_case(oracle, "G05-factual-definition"),
        harness_case=_harness_case(harness, "G05-factual-definition"),
        inventory=live["inventory"],
        parsed_rows=live["parsed_rows"],
        raw_response=live["raw_provider_response"],
    )
    assert scored["disposition"] == "PASS"
    assert scored["critical_misses"] == []
    assert "A transformer is a neural architecture." not in scored["invented_producer_claims"]


def test_g06_approved_equivalence_scores_correctly(oracle, harness, live_artifact):
    live = _live_case(live_artifact, "G06-editorial-nonfactual")
    scored = evaluate_provider_eval_v2_case(
        case_eval=_oracle_case(oracle, "G06-editorial-nonfactual"),
        harness_case=_harness_case(harness, "G06-editorial-nonfactual"),
        inventory=live["inventory"],
        parsed_rows=live["parsed_rows"],
        raw_response=live["raw_provider_response"],
    )
    factual = [m for m in scored["claim_matches"] if m["claim_type"] == "factual"][0]
    assert factual["covered"] is True
    assert factual["raw_model_material"] == "unknown"
    assert scored["unknown_materiality"] == ["Dropout randomly zeroes activations during training."]


def test_g06_editorial_excluded_from_factual_recall(oracle, harness, live_artifact):
    live = _live_case(live_artifact, "G06-editorial-nonfactual")
    scored = evaluate_provider_eval_v2_case(
        case_eval=_oracle_case(oracle, "G06-editorial-nonfactual"),
        harness_case=_harness_case(harness, "G06-editorial-nonfactual"),
        inventory=live["inventory"],
        parsed_rows=live["parsed_rows"],
        raw_response=live["raw_provider_response"],
    )
    assert scored["factual_eval_total"] == 1
    assert scored["factual_eval_covered"] == 1
    editorial = [m for m in scored["claim_matches"] if m["claim_type"] == "editorial"][0]
    assert editorial["match_rule"] == "inventory_descriptive_only"


def test_g16_live_expected_proposition_uses_draft_wording(oracle, harness, live_artifact):
    live = _live_case(live_artifact, "G16-anchor-failure")
    scored = evaluate_provider_eval_v2_case(
        case_eval=_oracle_case(oracle, "G16-anchor-failure"),
        harness_case=_harness_case(harness, "G16-anchor-failure"),
        inventory=live["inventory"],
        parsed_rows=live["parsed_rows"],
        raw_response=live["raw_provider_response"],
    )
    assert scored["disposition"] == "PASS"
    assert scored["critical_misses"] == []
    trap = _oracle_case(oracle, "G16-anchor-failure")["harness_only_trap"]
    assert trap["synthetic_harness_claim"] == "Momentum always accelerates convergence."


def test_unapproved_paraphrase_not_accepted(oracle, harness):
    case = _harness_case(harness, "G01-simple-factual")
    raw = json.dumps([{
        "claim_text": "Gradient descent iteratively adjusts parameters to reduce the loss.",
        "anchor_quote": "Gradient descent iteratively updates parameters to reduce the loss.",
        "claim_type": "factual",
        "material": True,
        "satisfies_req_ids": [],
        "specificity": "substantive",
        "requires_citation": None,
    }])
    rows = parse_claim_inventory_rows(raw)
    inv = build_claim_inventory(
        run_id="test", iteration=1, draft_markdown=case["draft_markdown"], raw_claims=rows,
    )
    scored = evaluate_provider_eval_v2_case(
        case_eval=_oracle_case(oracle, "G01-simple-factual"),
        harness_case=case,
        inventory=inv,
        parsed_rows=rows,
        raw_response=raw,
    )
    assert scored["critical_misses"]
    assert scored["disposition"] == "FAIL"


def test_unapproved_decomposition_not_credited(oracle, harness):
    case = _harness_case(harness, "G05-factual-definition")
    raw = json.dumps([{
        "claim_text": "A transformer is a neural architecture.",
        "anchor_quote": "A transformer is a neural architecture that uses self-attention to model token interactions.",
        "claim_type": "definition",
        "material": True,
        "satisfies_req_ids": [],
        "specificity": "substantive",
        "requires_citation": None,
    }])
    rows = parse_claim_inventory_rows(raw)
    inv = build_claim_inventory(
        run_id="test", iteration=1, draft_markdown=case["draft_markdown"], raw_claims=rows,
    )
    scored = evaluate_provider_eval_v2_case(
        case_eval=_oracle_case(oracle, "G05-factual-definition"),
        harness_case=case,
        inventory=inv,
        parsed_rows=rows,
        raw_response=raw,
    )
    assert scored["critical_misses"]
    assert scored["disposition"] == "FAIL"


def test_original_live_artifact_unchanged(tmp_path):
    from scripts.call_a_provider_qualification_runner import verify_run_artifact_integrity

    before = json.loads(LIVE_ARTIFACT.read_text(encoding="utf-8"))
    assert before["artifact_digest"] == EXPECTED_LIVE_DIGEST
    assert verify_run_artifact_integrity(before)["valid"] is True
    out = tmp_path / "provider_eval_v2_rescore.json"
    rescore_live_run_artifact(
        run_artifact_path=LIVE_ARTIFACT,
        expected_artifact_digest=EXPECTED_LIVE_DIGEST,
        output_path=out,
    )
    after = json.loads(LIVE_ARTIFACT.read_text(encoding="utf-8"))
    assert after["artifact_digest"] == EXPECTED_LIVE_DIGEST
    assert verify_run_artifact_integrity(after)["valid"] is True
    assert after["overall_disposition"] == before["overall_disposition"]
    assert after["provider_calls"] == before["provider_calls"]


def test_full_rescore_bounded_qualification(tmp_path):
    out = tmp_path / "provider_eval_v2_rescore.json"
    rescore = rescore_live_run_artifact(
        run_artifact_path=LIVE_ARTIFACT,
        expected_artifact_digest=EXPECTED_LIVE_DIGEST,
        output_path=out,
    )
    assert rescore["source_overall_disposition_v1"] == "FAIL"
    assert rescore["overall_disposition_v2"] == "PASS"
    assert rescore["bounded_qualification_v2"] is True
    assert rescore["provider_calls"] == 0
    g05 = next(r for r in rescore["case_results_v2"] if r["case_id"] == "G05-factual-definition")
    g06 = next(r for r in rescore["case_results_v2"] if r["case_id"] == "G06-editorial-nonfactual")
    g16 = next(r for r in rescore["case_results_v2"] if r["case_id"] == "G16-anchor-failure")
    assert g05["source_disposition_v1"] == "FAIL"
    assert g05["disposition"] == "PASS"
    assert g06["source_disposition_v1"] == "FAIL"
    assert g06["disposition"] == "PASS"
    assert g16["source_disposition_v1"] == "FAIL"
    assert g16["disposition"] == "PASS"
