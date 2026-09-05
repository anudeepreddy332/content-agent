"""Deterministic offline qualification tests for the hybrid verifier status engine."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hybrid_verifier_status_engine import (
    DeterministicFactVeto,
    adjudicate_hybrid_verifier_observations,
    build_minimal_envelope,
)

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "evals/fixtures/hybrid_verifier_status_offline.json"


def load_offline_pack() -> dict:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def build_case_envelope(case: dict) -> dict:
    claim_span_end = len(case["draft_text"])
    return build_minimal_envelope(
        request_id=case["request_id"],
        schema_version=1,
        draft_text=case["draft_text"],
        claims=[
            {
                "claim_id": case["claim_id"],
                "claim_text": case["draft_text"],
                "claim_span": [0, claim_span_end],
            }
        ],
        evidence_manifest=[
            {
                "evidence_id": case["source_id"],
                "request_id": case["request_id"],
                "source_text": case["source_text"],
                "source_sha256": case["source_sha256"],
                "verifier_visible": True,
            }
        ],
        observations=[copy.deepcopy(case["observation"])],
    )


def minimal_support_observation(
    *,
    claim_id: str = "X.claim.1",
    evidence_id: str = "SRC-1",
    support: bool,
    full_entailment: bool,
    blocker_kind: str | None = None,
) -> dict:
    obs: dict = {
        "claim_id": claim_id,
        "support_spans": [],
        "full_entailment": full_entailment,
        "blockers": [],
    }
    if support:
        obs["support_spans"] = [{"evidence_id": evidence_id, "start": 0, "end": 5}]
    if blocker_kind == "contradiction":
        obs["blockers"] = [
            {
                "kind": "contradiction",
                "evidence_spans": [{"evidence_id": evidence_id, "start": 10, "end": 20}],
                "explanation": "Material conflict with claim.",
            }
        ]
    elif blocker_kind == "limitation":
        obs["blockers"] = [
            {
                "kind": "limitation",
                "evidence_spans": [{"evidence_id": evidence_id, "start": 10, "end": 20}],
                "explanation": "Scope or condition mismatch.",
            }
        ]
    return obs


def build_table_envelope(observation: dict) -> dict:
    source_text = "alpha beta gamma delta epsilon"
    return build_minimal_envelope(
        request_id="REQ-TABLE",
        schema_version=1,
        draft_text="Claim text.",
        claims=[
            {
                "claim_id": observation["claim_id"],
                "claim_text": "Claim text.",
                "claim_span": [0, 10],
            }
        ],
        evidence_manifest=[
            {
                "evidence_id": "SRC-1",
                "request_id": "REQ-TABLE",
                "source_text": source_text,
                "source_sha256": __import__("hashlib").sha256(source_text.encode()).hexdigest(),
                "verifier_visible": True,
            }
        ],
        observations=[observation],
    )


def single_claim_result(envelope: dict) -> dict:
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert len(result.claim_results) == 1
    return result.claim_results[0]


@pytest.fixture
def pack() -> dict:
    return load_offline_pack()


@pytest.mark.parametrize(
    ("case_key", "expected_status"),
    [
        ("P6-COMPLETE", "weak"),
        ("P7-COMPLETE", "weak"),
        ("P1-COMPLETE", "verified"),
    ],
)
def test_frozen_offline_cases(pack: dict, case_key: str, expected_status: str) -> None:
    case = pack["cases"][case_key]
    envelope = build_case_envelope(case)
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert result.validity == "VALID"
    claim = result.claim_results[0]
    assert claim.validity == "VALID"
    assert claim.semantic_status == expected_status


@pytest.mark.parametrize("case_key", ["P6-COMPLETE", "P7-COMPLETE"])
def test_optimistic_entailment_mutation_stays_weak(pack: dict, case_key: str) -> None:
    case = pack["cases"][case_key]
    envelope = build_case_envelope(case)
    envelope["observations"][0]["full_entailment"] = True
    claim = single_claim_result(envelope)
    assert claim.validity == "VALID"
    assert claim.semantic_status == "weak"
    assert "full_entailment_true_with_blockers_downgraded_to_weak" in claim.consistency_diagnostics


@pytest.mark.parametrize(
    ("support", "full_entailment", "blocker_kind", "expected_validity", "expected_status"),
    [
        (False, False, None, "VALID", "unverified"),
        (False, False, "contradiction", "VALID", "unverified"),
        (False, True, None, "INVALID", None),
        (False, True, "contradiction", "INVALID", None),
        (True, False, None, "VALID", "weak"),
        (True, False, "contradiction", "VALID", "weak"),
        (True, True, None, "VALID", "verified"),
        (True, True, "contradiction", "VALID", "weak"),
    ],
)
def test_eight_combination_decision_table_contradiction(
    support: bool,
    full_entailment: bool,
    blocker_kind: str | None,
    expected_validity: str,
    expected_status: str | None,
) -> None:
    observation = minimal_support_observation(
        support=support,
        full_entailment=full_entailment,
        blocker_kind=blocker_kind,
    )
    claim = single_claim_result(build_table_envelope(observation))
    assert claim.validity == expected_validity
    assert claim.semantic_status == expected_status
    if support and full_entailment and blocker_kind:
        assert "full_entailment_true_with_blockers_downgraded_to_weak" in claim.consistency_diagnostics


@pytest.mark.parametrize(
    ("support", "full_entailment", "expected_validity", "expected_status"),
    [
        (True, False, "VALID", "weak"),
        (True, True, "VALID", "weak"),
    ],
)
def test_limitation_blocker_cases(
    support: bool,
    full_entailment: bool,
    expected_validity: str,
    expected_status: str,
) -> None:
    observation = minimal_support_observation(
        support=support,
        full_entailment=full_entailment,
        blocker_kind="limitation",
    )
    claim = single_claim_result(build_table_envelope(observation))
    assert claim.validity == expected_validity
    assert claim.semantic_status == expected_status
    if full_entailment:
        assert "full_entailment_true_with_blockers_downgraded_to_weak" in claim.consistency_diagnostics


def test_no_support_limitation_without_spans_is_valid_unverified() -> None:
    observation = {
        "claim_id": "X.claim.1",
        "support_spans": [],
        "full_entailment": False,
        "blockers": [
            {
                "kind": "limitation",
                "evidence_spans": [],
                "explanation": "Required supporting component absent from evidence.",
            }
        ],
    }
    claim = single_claim_result(build_table_envelope(observation))
    assert claim.validity == "VALID"
    assert claim.semantic_status == "unverified"


def test_unknown_claim_id_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"][0]["claim_id"] = "UNKNOWN.claim"
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert result.validity == "INVALID"
    assert result.claim_results[0].invalid_reasons


def test_missing_claim_row_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"] = []
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert result.validity == "INVALID"
    assert any("missing_claim_row" in reason for reason in result.invalid_reasons)


def test_duplicate_claim_row_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"].append(copy.deepcopy(envelope["observations"][0]))
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert result.validity == "INVALID"
    assert "duplicate_claim_row" in result.claim_results[0].invalid_reasons


def test_conflicting_duplicate_rows_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    conflicting = copy.deepcopy(envelope["observations"][0])
    conflicting["full_entailment"] = True
    envelope["observations"].append(conflicting)
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert result.validity == "INVALID"
    assert "conflicting_duplicate_claim_row" in result.claim_results[0].invalid_reasons


def test_extra_claim_row_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    extra = copy.deepcopy(envelope["observations"][0])
    extra["claim_id"] = "EXTRA.claim"
    envelope["observations"].append(extra)
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert result.validity == "INVALID"


def test_unknown_evidence_id_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"][0]["support_spans"][0]["evidence_id"] = "NO-SUCH"
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"
    assert any("unknown_evidence_id" in reason for reason in claim.invalid_reasons)


def test_stale_evidence_hash_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["evidence_manifest"][0]["source_sha256"] = "0" * 64
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"
    assert any("stale_evidence_hash" in reason for reason in claim.invalid_reasons)


def test_cross_request_evidence_reference_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["evidence_manifest"][0]["request_id"] = "OTHER-REQ"
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"
    assert any("cross_request_evidence_reference" in reason for reason in claim.invalid_reasons)


def test_invisible_span_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["evidence_manifest"][0]["verifier_visible"] = False
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"
    assert any("invisible_evidence" in reason for reason in claim.invalid_reasons)


def test_negative_span_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"][0]["support_spans"][0]["start"] = -1
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"


def test_zero_length_span_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"][0]["support_spans"][0]["end"] = 400
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"


def test_out_of_bounds_span_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"][0]["support_spans"][0]["end"] = 99999
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"


def test_malformed_boolean_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P1-COMPLETE"])
    envelope["observations"][0]["full_entailment"] = "true"
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"
    assert "observation_invalid_full_entailment_type" in claim.invalid_reasons


def test_unknown_blocker_kind_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"][0]["blockers"][0]["kind"] = "scope"
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"


def test_contradiction_without_span_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["observations"][0]["blockers"][0]["evidence_spans"] = []
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"


def test_missing_required_field_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P1-COMPLETE"])
    del envelope["observations"][0]["blockers"]
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"


def test_unknown_top_level_observation_field_invalid(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P1-COMPLETE"])
    envelope["observations"][0]["status"] = "verified"
    claim = single_claim_result(envelope)
    assert claim.validity == "INVALID"
    assert "observation_unknown_field" in claim.invalid_reasons


def test_support_span_ordering_invariant(pack: dict) -> None:
    envelope_a = build_case_envelope(pack["cases"]["P7-COMPLETE"])
    envelope_b = build_case_envelope(pack["cases"]["P7-COMPLETE"])
    support = envelope_b["observations"][0]["support_spans"]
    support.reverse()
    result_a = adjudicate_hybrid_verifier_observations(envelope_a)
    result_b = adjudicate_hybrid_verifier_observations(envelope_b)
    assert result_a.claim_results[0].semantic_status == result_b.claim_results[0].semantic_status


def test_blocker_ordering_invariant(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    observation = envelope["observations"][0]
    observation["blockers"] = [
        {
            "kind": "limitation",
            "evidence_spans": [{"evidence_id": "SRC-P6-W03", "start": 401, "end": 457}],
            "explanation": "Synthetic second blocker for ordering test.",
        },
        observation["blockers"][0],
    ]
    baseline = adjudicate_hybrid_verifier_observations(build_case_envelope(pack["cases"]["P6-COMPLETE"]))
    reordered = adjudicate_hybrid_verifier_observations(envelope)
    assert baseline.claim_results[0].semantic_status == reordered.claim_results[0].semantic_status


def test_evidence_manifest_ordering_invariant(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P1-COMPLETE"])
    manifest = envelope["evidence_manifest"]
    envelope["evidence_manifest"] = list(reversed(manifest))
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert result.claim_results[0].semantic_status == "verified"


def test_blocker_monotonicity_verified_becomes_weak(pack: dict) -> None:
    verified_env = build_case_envelope(pack["cases"]["P1-COMPLETE"])
    verified = single_claim_result(verified_env)
    assert verified.semantic_status == "verified"

    weakened_env = build_case_envelope(pack["cases"]["P1-COMPLETE"])
    weakened_env["observations"][0]["blockers"] = [
        {
            "kind": "contradiction",
            "evidence_spans": [{"evidence_id": "SRC-P1-W02", "start": 1500, "end": 1563}],
            "explanation": "Injected contradiction must downgrade status.",
        }
    ]
    weakened = single_claim_result(weakened_env)
    assert weakened.semantic_status == "weak"


def test_deterministic_fact_veto_extension_point(pack: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.hybrid_verifier_status_engine as engine

    class AlwaysVeto(DeterministicFactVeto):
        def claim_ids(self) -> frozenset[str]:
            return frozenset()

        def vetoes_full_entailment(self, *, claim, observation, envelope) -> bool:
            return True

    monkeypatch.setattr(engine, "DETERMINISTIC_FACT_VETOS", [AlwaysVeto()])
    envelope = build_case_envelope(pack["cases"]["P1-COMPLETE"])
    claim = single_claim_result(envelope)
    assert claim.semantic_status == "weak"
    assert "DETERMINISTIC_FACT_VETO" in claim.reason_codes


def test_observation_schema_has_no_model_status_field(pack: dict) -> None:
    case = pack["cases"]["P6-COMPLETE"]
    observation = case["observation"]
    forbidden = {"status", "materiality", "confidence", "publication_decision"}
    assert forbidden.isdisjoint(observation.keys())


def test_coverage_requires_exactly_one_row_per_claim(pack: dict) -> None:
    envelope = build_case_envelope(pack["cases"]["P6-COMPLETE"])
    envelope["claims"].append(
        {
            "claim_id": "P6.claim.2",
            "claim_text": "Second claim.",
            "claim_span": [0, 13],
        }
    )
    result = adjudicate_hybrid_verifier_observations(envelope)
    assert result.validity == "INVALID"
    assert any("missing_claim_row:P6.claim.2" == reason for reason in result.invalid_reasons)
