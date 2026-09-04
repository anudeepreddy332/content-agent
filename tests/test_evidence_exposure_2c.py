"""Deterministic Stage 2C evidence-exposure qualification tests. No providers."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_evidence_exposure_2c import (
    ARM_A,
    ARM_B,
    ARM_C,
    CASE_IDS,
    DEFAULT_FIXTURES,
    EvidenceExposure2CError,
    KB_PREFIX_CHARS,
    WEB_PREFIX_CHARS,
    build_frozen_fixtures,
    evaluate_pack,
    expose_arm_a,
    expose_arm_c,
    load_pack,
    merge_windows,
    prefix_limit,
    sha256_text,
    validate_pack,
)


def _report() -> dict:
    return evaluate_pack(load_pack())


def test_fixture_pack_identity_and_hashes():
    pack = load_pack()
    assert pack["pack_id"] == "evidence_exposure_2c"
    assert len(pack["cases"]) == 8
    assert [case["id"] for case in pack["cases"]] == list(CASE_IDS)
    for case in pack["cases"]:
        assert case["source_sha256"] == sha256_text(case["source_text"])
        assert case["claim"]["claim_sha256"] == sha256_text(case["claim"]["text"])
        assert case["source_rank"] == 1


def test_exact_web_and_kb_prefix_cutoffs():
    web_exposed, web_limit, _ = expose_arm_a("web", "a" * 2000)
    kb_exposed, kb_limit, _ = expose_arm_a("kb", "b" * 2500)
    assert web_limit == WEB_PREFIX_CHARS
    assert kb_limit == KB_PREFIX_CHARS
    assert len(web_exposed) == WEB_PREFIX_CHARS
    assert len(kb_exposed) == KB_PREFIX_CHARS
    assert prefix_limit("web") == 1500
    assert prefix_limit("kb") == 2000


def test_late_web_and_kb_cases_lose_evidence_in_arm_a():
    report = _report()
    by_id = {item["case_id"]: item for item in report["results"]}
    for case_id in ("E2C-W02", "E2C-K02", "E2C-W03", "E2C-K03", "E2C-W04"):
        assert by_id[case_id]["arms"][ARM_A]["draft"]["all_requirements_visible"] is False
        assert by_id[case_id]["arms"][ARM_A]["verifier"]["all_requirements_visible"] is False


def test_early_controls_remain_visible_in_arm_a():
    report = _report()
    by_id = {item["case_id"]: item for item in report["results"]}
    for case_id in ("E2C-W01", "E2C-K01", "E2C-K04"):
        assert by_id[case_id]["arms"][ARM_A]["draft"]["all_requirements_visible"] is True


def test_complete_source_and_gold_context_preserve_all_requirements():
    report = _report()
    for result in report["results"]:
        for arm in (ARM_B, ARM_C):
            assert result["arms"][arm]["draft"]["all_requirements_visible"] is True
            assert result["arms"][arm]["verifier"]["all_requirements_visible"] is True
            assert result["arms"][arm]["draft"]["truncation_violations"] == 0


def test_multi_span_qualifier_contradiction_requires_all_windows():
    pack = load_pack()
    case = next(item for item in pack["cases"] if item["id"] == "E2C-W03")
    requirement = case["requirements"][0]
    assert len(requirement["truth_spans"]) == 2
    exposed_a, limit, _ = expose_arm_a("web", case["source_text"])
    assert limit == WEB_PREFIX_CHARS
    first_span = case["source_text"][requirement["truth_spans"][0][0] : requirement["truth_spans"][0][1]]
    second_span = case["source_text"][requirement["truth_spans"][1][0] : requirement["truth_spans"][1][1]]
    assert first_span in exposed_a
    assert second_span not in exposed_a


def test_gold_context_arm_exposes_merged_windows():
    pack = load_pack()
    case = next(item for item in pack["cases"] if item["id"] == "E2C-K03")
    windows = case["requirements"][0]["required_windows"]
    exposed, merged, _ = expose_arm_c(case["source_text"], windows)
    assert merged == merge_windows(windows)
    for start, end in merged:
        assert case["source_text"][start:end] in exposed


def test_stage_2c_gates_and_metrics():
    report = _report()
    assert report["stage_2c_pass"] is True
    metrics = report["metrics"]
    assert metrics["gold_evidence_requirement_recall_at_k.v1"]["value"] == 1.0
    assert metrics["draft_gold_evidence_recall.v1"]["numerator"] == 3
    assert metrics["draft_gold_evidence_recall.v1"]["denominator"] == 8
    assert metrics["relevant_span_truncation_violation_rate.v1"]["numerator"] == 5
    assert metrics["by_arm"][ARM_B]["draft_gold_evidence_recall"]["value"] == 1.0
    assert metrics["by_arm"][ARM_C]["truncation_violation_rate"]["numerator"] == 0


def test_zero_denominator_is_explicit_not_pass():
    with pytest.raises(EvidenceExposure2CError):
        validate_pack(
            {
                "pack_id": "evidence_exposure_2c",
                "schema_version": 1,
                "evaluator_id": "evidence_exposure_2c",
                "stage": "2c",
                "description": "empty",
                "cases": [],
            }
        )


def test_invalid_span_mapping_is_rejected():
    pack = load_pack()
    broken = copy.deepcopy(pack)
    broken["cases"][0]["requirements"][0]["required_windows"] = [[99999, 100000]]
    with pytest.raises(EvidenceExposure2CError):
        validate_pack(broken)


def test_deterministic_repeated_output():
    first = json.dumps(_report(), sort_keys=True)
    second = json.dumps(_report(), sort_keys=True)
    assert first == second


def test_per_case_metrics_present():
    report = _report()
    for result in report["results"]:
        assert result["retrieval_gold_evidence_requirement_recall_at_k"]["value"] == 1.0
        for arm in (ARM_A, ARM_B, ARM_C):
            for consumer in ("draft", "verifier"):
                recall = result["arms"][arm][consumer]["gold_evidence_recall"]
                assert recall["denominator"] >= 1
                assert recall["undefined"] is False


def test_build_frozen_fixtures_matches_committed_json():
    generated = build_frozen_fixtures()
    committed = json.loads(DEFAULT_FIXTURES.read_text(encoding="utf-8"))
    assert generated["cases"][0]["id"] == committed["cases"][0]["id"]
    assert len(generated["cases"]) == len(committed["cases"])


def test_production_source_context_limits_unchanged():
    nodes = Path("agent/nodes.py").read_text(encoding="utf-8")
    assert "WEB_CHARS = 1500" in nodes
    assert "KB_CHARS  = 2000" in nodes
