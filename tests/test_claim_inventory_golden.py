"""Golden-set harness tests for the Phase 4 Slice 2A claim inventory.

Offline, deterministic, ZERO providers. The harness compares the deterministic
production pipeline against hand-labeled gold. NO production extraction-quality
threshold is set or implied (none is calibrated); critical deterministic
invariants only:

- no silently dropped claims;
- anchor failure/ambiguity never treated as success;
- required⇒material override works;
- UNKNOWN materiality remains UNKNOWN.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.claim_inventory import (
    ANCHOR_AMBIGUOUS,
    ANCHOR_FAILED,
    build_claim_inventory,
    inventory_critical_failures,
    parse_claim_inventory_rows,
)
from scripts.evaluate_claim_inventory import evaluate_case, evaluate_pack

FIXTURE = Path(__file__).resolve().parents[1] / "evals/fixtures/claim_inventory_golden_v1.json"


@pytest.fixture(scope="module")
def pack() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(pack) -> dict:
    return evaluate_pack(pack)


def test_golden_set_covers_required_case_kinds(pack):
    tags = {tag for case in pack["cases"] for tag in case.get("tags", [])}
    required = {
        "simple_factual", "multi_atomic", "repeated_claim", "near_duplicate",
        "factual_definition", "editorial", "code_factual", "cross_section",
        "revision_new", "revision_removed", "material_central", "incidental",
        "required_by_brief", "false_nonmaterial_trap", "anchor_ambiguity",
        "anchor_failure",
    }
    assert required <= tags
    assert len(pack["cases"]) >= 16


def test_metrics_match_frozen_expectations(report):
    m = report["metrics"]
    # Every extracted claim matches gold one-to-one in this set; G14 is the one
    # deliberately divergent extractor error (materiality), measured not hidden.
    assert m["factual_claim_recall"] == 1.0
    assert m["factual_claim_precision"] == 1.0
    assert m["factual_recall_denominator"] == 21
    assert m["split_error_count"] == 0
    assert m["merge_error_count"] == 0
    assert m["duplicate_cases"] == 1
    assert m["duplicate_cases_correct"] == 1
    assert m["anchor_binding_accuracy"] == 1.0
    assert m["anchor_binding_denominator"] == 24
    assert m["material_claim_recall_denominator"] == 17
    assert m["material_claim_recall"] == pytest.approx(16 / 17)
    assert m["false_nonmaterial_count"] == 1
    assert m["false_nonmaterial_rate"] == pytest.approx(1 / 17)
    assert m["false_nonmaterial_cases"] == [
        "G14-false-nonmaterial-trap:The scheduler guarantees at-most-once delivery."
    ]
    confusion = m["materiality_confusion"]
    assert confusion["true"]["true"] == 16
    assert confusion["true"]["false"] == 1
    # G04 x2 + G06 editorial + G07 editorial + G10 + G12
    assert confusion["false"]["false"] == 6
    assert confusion["unknown"]["unknown"] == 1


def test_no_silently_dropped_claims(report, pack):
    """Every parsed Call-A row maps to an inventory claim (merge preserves
    identity); nothing is dropped silently anywhere in the pack."""
    for case_result in report["results"]:
        inventory = case_result["inventory"]
        rows = parse_claim_inventory_rows(
            json.dumps(_case_by_id(pack, case_result["case_id"])["call_a_response"])
        )
        assert len(inventory["claims"]) <= len(rows)
        assert inventory["counts"]["total"] == len(inventory["claims"])
        produced_texts = {c["claim_text"] for c in inventory["claims"]}
        for row in rows:
            assert row["claim_text"].strip() in produced_texts


def _case_by_id(report_or_pack, case_id: str) -> dict:
    cases = report_or_pack["cases"] if "cases" in report_or_pack else report_or_pack["results"]
    for case in cases:
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)


def test_anchor_failure_and_ambiguity_never_succeed(report):
    by_id = {r["case_id"]: r for r in report["results"]}
    failed = by_id["G16-anchor-failure"]["inventory"]
    assert failed["claims"][0]["anchor_validity"] == ANCHOR_FAILED
    assert failed["claims"][0]["occurrences"] == []
    assert inventory_critical_failures(failed), "eligible ANCHOR_FAILED claim must be critical"

    ambiguous = by_id["G15-anchor-ambiguity"]["inventory"]
    assert len(ambiguous["claims"]) == 2
    for claim in ambiguous["claims"]:
        assert claim["anchor_validity"] == ANCHOR_AMBIGUOUS
        assert len(claim["occurrences"]) == 2  # occurrences preserved, not dropped
    assert len(inventory_critical_failures(ambiguous)) == 2


def test_required_implies_material_override(report):
    result = _case_by_id(report, "G13-required-by-brief-material")
    claim = result["inventory"]["claims"][0]
    assert claim["material"] is True
    assert claim["materiality_override"] == "required_by_brief"
    assert result["inventory"]["satisfied_req_ids"] == ["REQ-2"]
    assert claim["requires_citation"] is True


def test_unknown_materiality_remains_unknown(report):
    result = _case_by_id(report, "G11-material-central-claim")
    unknown = [c for c in result["inventory"]["claims"] if c["material"] == "unknown"]
    assert len(unknown) == 1
    assert unknown[0]["claim_text"] == "The simulator's dashboard is blue."
    # UNKNOWN fails toward citation for factual claims; never converted to false.
    assert unknown[0]["requires_citation"] is True


def test_revision_cases_new_and_removed_claims(pack):
    for case_id, expected_new, expected_removed in (
        ("G09-new-claim-after-revision", ["Early stopping also helps."], []),
        ("G10-removed-claim", ["Weight decay penalizes large weights."], ["Dropout reduces overfitting."]),
    ):
        case = _case_by_id(pack, case_id)
        prior_rows = parse_claim_inventory_rows(json.dumps(case["prior"]["call_a_response"]))
        prior_inv = build_claim_inventory(
            run_id=f"golden-{case_id}", iteration=1,
            draft_markdown=case["prior"]["draft_markdown"],
            raw_claims=prior_rows,
            brief_requirements=case.get("brief_requirements") or [],
        )
        current_inv = evaluate_case(case)["inventory"]
        assert current_inv["draft_sha256"] != prior_inv["draft_sha256"]
        prior_texts = {c["claim_text"] for c in prior_inv["claims"]}
        current_texts = {c["claim_text"] for c in current_inv["claims"]}
        assert sorted(current_texts - prior_texts) == expected_new
        assert sorted(prior_texts - current_texts) == expected_removed

    # Removed required-linked claim: the requirement identity persists and is
    # now unsatisfied — deleting the claim cannot delete the requirement.
    g10 = evaluate_case(_case_by_id(pack, "G10-removed-claim"))["inventory"]
    assert g10["satisfied_req_ids"] == []
    assert [r["req_id"] for r in g10["brief_requirements"]] == ["REQ-1"]
