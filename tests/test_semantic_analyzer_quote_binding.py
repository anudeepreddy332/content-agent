"""Deterministic exact-quote binding tests for semantic analyzer contract. No providers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hybrid_verifier_status_engine import adjudicate_hybrid_verifier_observations, build_minimal_envelope
from scripts.semantic_analyzer_preprovider_harness import (
    build_case_bundle,
    build_gold_mock_responses,
    get_verified_frozen_case_truth,
    load_fixture_pack,
    qualify_case_response,
    span_observation_to_quote_observation,
)
from scripts.semantic_analyzer_quote_binding import (
    QuoteBindingError,
    bind_quote_against_manifest,
    bind_quote_to_span,
    convert_quote_observations_to_canonical,
    find_exact_quote_matches,
)


@pytest.fixture
def pack() -> dict:
    return load_fixture_pack()


def _manifest(pack: dict, case_id: str = "P6") -> list[dict]:
    return build_case_bundle(pack, case_id)["evidence_manifest"]


def _source_text(pack: dict, case_id: str = "P6") -> str:
    return pack["cases"][f"{case_id}-COMPLETE"]["source_text"]


def _quote_obs(pack: dict, case_id: str) -> dict:
    case = pack["cases"][f"{case_id}-COMPLETE"]
    return span_observation_to_quote_observation(
        source_text=case["source_text"],
        observation=case["observation"],
    )


def _raw_quote(observation: dict) -> str:
    return json.dumps({"observations": [observation]})


P6_SUPPORT = "Helix gate permits export when mode flag HG-ENABLE is set."
P6_CONTRA = "Unless mode flag HG-ENABLE is cleared, export remains blocked."
P7_SUPPORT = "Cache tier T3 admits writes when isolation level IL-2 holds."
P7_LIMITATION = "unless cache mode is disabled via CM-OFF switch."
P1_SUPPORT = "Quorum latch engages only after seven replica acknowledgements."


def test_unique_exact_support_quote_binds_correct_span(pack: dict):
    manifest = _manifest(pack, "P6")
    result = bind_quote_against_manifest(
        evidence_id="SRC-P6-W03",
        quote=P6_SUPPORT,
        evidence_manifest=manifest,
    )
    assert result.start == 400
    assert result.end == 458
    assert result.bound_text == P6_SUPPORT
    assert result.match_count == 1


def test_unique_exact_blocker_quote_binds_correct_span(pack: dict):
    manifest = _manifest(pack, "P6")
    result = bind_quote_against_manifest(
        evidence_id="SRC-P6-W03",
        quote=P6_CONTRA,
        evidence_manifest=manifest,
    )
    assert result.start == 1500
    assert result.end == 1562
    assert result.bound_text == P6_CONTRA


def test_nonexistent_quote_fails_closed(pack: dict):
    manifest = _manifest(pack, "P6")
    with pytest.raises(QuoteBindingError, match="quote_not_found"):
        bind_quote_against_manifest(
            evidence_id="SRC-P6-W03",
            quote="This text does not exist in source.",
            evidence_manifest=manifest,
        )


def test_one_character_mutation_fails_closed(pack: dict):
    manifest = _manifest(pack, "P6")
    mutated = P6_SUPPORT[:-1] + ("X" if P6_SUPPORT[-1] != "X" else "Y")
    with pytest.raises(QuoteBindingError, match="quote_not_found"):
        bind_quote_against_manifest(
            evidence_id="SRC-P6-W03",
            quote=mutated,
            evidence_manifest=manifest,
        )


def test_paraphrased_quote_fails_closed(pack: dict):
    manifest = _manifest(pack, "P6")
    with pytest.raises(QuoteBindingError, match="quote_not_found"):
        bind_quote_against_manifest(
            evidence_id="SRC-P6-W03",
            quote="Helix gate allows export when HG-ENABLE mode flag is set.",
            evidence_manifest=manifest,
        )


def test_wrong_evidence_id_fails_closed(pack: dict):
    manifest = _manifest(pack, "P6")
    with pytest.raises(QuoteBindingError, match="invalid_evidence_id"):
        bind_quote_against_manifest(
            evidence_id="SRC-NO-SUCH",
            quote=P6_SUPPORT,
            evidence_manifest=manifest,
        )


def test_duplicate_exact_quote_in_same_evidence_ambiguous(pack: dict):
    source_text = "alpha beta alpha beta"
    with pytest.raises(QuoteBindingError, match="ambiguous_quote"):
        bind_quote_to_span(evidence_id="E1", quote="alpha", source_text=source_text)


def test_duplicate_quote_across_evidence_ids_scoped_safely(pack: dict):
    manifest = [
        {"evidence_id": "E1", "source_text": "shared phrase here"},
        {"evidence_id": "E2", "source_text": "shared phrase here"},
    ]
    r1 = bind_quote_against_manifest(
        evidence_id="E1", quote="shared phrase", evidence_manifest=manifest
    )
    r2 = bind_quote_against_manifest(
        evidence_id="E2", quote="shared phrase", evidence_manifest=manifest
    )
    assert r1.start == 0 and r1.end == len("shared phrase")
    assert r2.start == 0 and r2.end == len("shared phrase")


def test_empty_quote_rejected(pack: dict):
    manifest = _manifest(pack, "P6")
    with pytest.raises(QuoteBindingError, match="empty_quote"):
        bind_quote_against_manifest(
            evidence_id="SRC-P6-W03",
            quote="",
            evidence_manifest=manifest,
        )


def test_whitespace_normalized_but_not_exact_quote_rejected(pack: dict):
    manifest = _manifest(pack, "P6")
    normalized = "  " + P6_SUPPORT + "  "
    with pytest.raises(QuoteBindingError, match="quote_not_found"):
        bind_quote_against_manifest(
            evidence_id="SRC-P6-W03",
            quote=normalized,
            evidence_manifest=manifest,
        )


def test_unicode_exact_quote_uses_code_point_offsets():
    source_text = "café résumé naïve"
    quote = "résumé"
    matches = find_exact_quote_matches(source_text, quote)
    assert len(matches) == 1
    start, end = matches[0]
    assert source_text[start:end] == quote
    result = bind_quote_to_span(evidence_id="U1", quote=quote, source_text=source_text)
    assert result.bound_text == quote


def test_quote_derived_span_text_equals_quote_exactly(pack: dict):
    manifest = _manifest(pack, "P6")
    canonical = convert_quote_observations_to_canonical(
        [
            {
                "claim_id": "P6.claim.1",
                "support_quotes": [{"evidence_id": "SRC-P6-W03", "quote": P6_SUPPORT}],
                "full_entailment": False,
                "blockers": [],
            }
        ],
        manifest,
    )
    source = _source_text(pack, "P6")
    span = canonical[0]["support_spans"][0]
    assert source[span["start"] : span["end"]] == P6_SUPPORT


def test_incorrect_support_quote_cannot_become_verified(pack: dict):
    obs = _quote_obs(pack, "P6")
    obs["support_quotes"] = [{"evidence_id": "SRC-P6-W03", "quote": P6_SUPPORT + "X"}]
    result = qualify_case_response(
        case_id="P6",
        bundle=build_case_bundle(pack, "P6"),
        raw_response=_raw_quote(obs),
        frozen_truth=get_verified_frozen_case_truth("P6"),
    )
    assert result["disposition"] == "INVALID"
    assert any("quote_binding" in reason for reason in result["invalid_reasons"])


def test_missing_contradiction_quote_cannot_allow_p6_verified(pack: dict):
    obs = _quote_obs(pack, "P6")
    obs["blockers"] = []
    result = qualify_case_response(
        case_id="P6",
        bundle=build_case_bundle(pack, "P6"),
        raw_response=_raw_quote(obs),
        frozen_truth=get_verified_frozen_case_truth("P6"),
    )
    assert result["disposition"] == "FAIL"
    assert result["adjudication"]["semantic_status_by_claim"]["P6.claim.1"] == "weak"
    assert result["semantic_oracle"]["derived_semantic_status"] == "weak"


def test_p6_valid_quote_response_derives_exact_gold_spans(pack: dict, gold_responses: dict):
    result = qualify_case_response(
        case_id="P6",
        bundle=build_case_bundle(pack, "P6"),
        raw_response=gold_responses["P6"],
        frozen_truth=get_verified_frozen_case_truth("P6"),
    )
    spans = result["response_contract"]["observations"][0]
    assert spans["support_spans"] == [{"evidence_id": "SRC-P6-W03", "start": 400, "end": 458}]
    assert spans["blockers"][0]["evidence_spans"] == [
        {"evidence_id": "SRC-P6-W03", "start": 1500, "end": 1562}
    ]
    assert result["disposition"] == "PASS"
    assert result["semantic_oracle"]["derived_semantic_status"] == "weak"


def test_p7_valid_quote_response_derives_exact_gold_spans(pack: dict, gold_responses: dict):
    result = qualify_case_response(
        case_id="P7",
        bundle=build_case_bundle(pack, "P7"),
        raw_response=gold_responses["P7"],
        frozen_truth=get_verified_frozen_case_truth("P7"),
    )
    spans = result["response_contract"]["observations"][0]
    assert spans["support_spans"] == [{"evidence_id": "SRC-P7-K03", "start": 0, "end": 60}]
    assert spans["blockers"][0]["evidence_spans"] == [
        {"evidence_id": "SRC-P7-K03", "start": 2000, "end": 2048}
    ]
    assert result["disposition"] == "PASS"
    assert result["semantic_oracle"]["derived_semantic_status"] == "weak"
    assert spans["full_entailment"] is False


def test_p1_valid_quote_response_derives_exact_gold_spans(pack: dict, gold_responses: dict):
    result = qualify_case_response(
        case_id="P1",
        bundle=build_case_bundle(pack, "P1"),
        raw_response=gold_responses["P1"],
        frozen_truth=get_verified_frozen_case_truth("P1"),
    )
    spans = result["response_contract"]["observations"][0]
    assert spans["support_spans"] == [{"evidence_id": "SRC-P1-W02", "start": 1500, "end": 1563}]
    assert spans["blockers"] == []
    assert spans["full_entailment"] is True
    assert result["disposition"] == "PASS"
    assert result["semantic_oracle"]["derived_semantic_status"] == "verified"


@pytest.fixture
def gold_responses() -> dict[str, str]:
    return build_gold_mock_responses()


def test_p6_quote_binding_status_engine_weak(pack: dict, gold_responses: dict):
    bundle = build_case_bundle(pack, "P6")
    canonical = convert_quote_observations_to_canonical(
        json.loads(gold_responses["P6"])["observations"],
        bundle["evidence_manifest"],
    )
    envelope = build_minimal_envelope(
        request_id=bundle["request_id"],
        schema_version=bundle["schema_version"],
        draft_text=bundle["draft_text"],
        claims=bundle["claims"],
        evidence_manifest=bundle["evidence_manifest"],
        observations=canonical,
    )
    adjudication = adjudicate_hybrid_verifier_observations(envelope)
    assert adjudication.semantic_status_by_claim["P6.claim.1"] == "weak"
