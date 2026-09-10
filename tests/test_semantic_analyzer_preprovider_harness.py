"""Deterministic semantic-analyzer pre-provider qualification harness tests. No providers."""
from __future__ import annotations

import copy
import json
import socket
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.semantic_analyzer_preprovider_harness import (
    ANALYZER_PROMPT_SHA256,
    ANALYZER_SCHEMA_SHA256,
    CASE_ORDER,
    EXPECTED_FIXTURE_SHA256,
    FAILED_HARNESS_CHECKPOINT_SHA,
    REQUIRED_ENGINE_BASELINE_SHA,
    AttemptGovernanceError,
    AttemptLedger,
    ResponseContractError,
    build_analyzer_input,
    build_case_bundle,
    build_frozen_case_truth,
    build_gold_mock_responses,
    build_run_identity_hashes,
    compute_qualification_ready,
    get_implementation_git_sha,
    load_fixture_pack,
    parse_analyzer_response,
    provider_execution_authorized,
    qualify_case_response,
    run_offline_qualification,
    run_preflight,
    sha256_text,
    validate_harness_identities,
    validate_trusted_ingress,
    verify_artifact_integrity,
)


@pytest.fixture
def pack() -> dict:
    return load_fixture_pack()


@pytest.fixture
def gold_responses(pack: dict) -> dict[str, str]:
    return build_gold_mock_responses(pack)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network must remain denied during qualification tests."""

    class GuardedSocket(socket.socket):
        def connect(self, address):  # type: ignore[override]
            raise OSError(f"network denied during qualification tests: {address}")

    monkeypatch.setattr(socket, "socket", GuardedSocket)
    monkeypatch.delenv("SEMANTIC_ANALYZER_EXECUTE", raising=False)


def _bundle(pack: dict, case_id: str = "P6") -> dict:
    return build_case_bundle(pack, case_id)


def _frozen(pack: dict, case_id: str = "P6") -> dict:
    return build_frozen_case_truth(pack, case_id)


def _qualify(
    pack: dict,
    case_id: str,
    raw_response: str,
    *,
    bundle: dict | None = None,
) -> dict:
    return qualify_case_response(
        case_id=case_id,
        bundle=bundle or _bundle(pack, case_id),
        raw_response=raw_response,
        frozen_truth=_frozen(pack, case_id),
    )


def _obs(pack: dict, case_id: str) -> dict:
    return copy.deepcopy(pack["cases"][f"{case_id}-COMPLETE"]["observation"])


def _raw(observation: dict) -> str:
    return json.dumps({"observations": [observation]})


def test_fixture_hash_matches_frozen_expectation():
    raw = Path("evals/fixtures/hybrid_verifier_status_offline.json").read_bytes()
    import hashlib

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_FIXTURE_SHA256


def test_preflight_reports_ready_for_harness_head_not_engine_baseline():
    report = run_preflight()
    harness_sha = get_implementation_git_sha()
    assert report["preflight_ready"] is True
    assert harness_sha != REQUIRED_ENGINE_BASELINE_SHA
    assert report["identity_hashes"]["harness_implementation_sha"] == harness_sha
    assert report["identity_hashes"]["required_engine_baseline_sha"] == REQUIRED_ENGINE_BASELINE_SHA
    assert report["identity_hashes"]["failed_harness_checkpoint_sha"] == FAILED_HARNESS_CHECKPOINT_SHA
    assert report["fixture_sha256"] == EXPECTED_FIXTURE_SHA256


def test_analyzer_input_excludes_gold_hints(pack: dict):
    bundle = _bundle(pack, "P6")
    analyzer_input = build_analyzer_input(bundle)
    serialized = json.dumps(analyzer_input)
    assert "observation" not in serialized
    assert "expected_status" not in serialized
    assert "support_span" not in serialized
    assert "contradiction_span" not in serialized


def test_valid_p6_oracle_passes(pack: dict, gold_responses: dict):
    result = _qualify(pack, "P6", gold_responses["P6"])
    assert result["disposition"] == "PASS"
    assert result["semantic_oracle"]["derived_semantic_status"] == "weak"


def test_valid_p7_oracle_passes(pack: dict, gold_responses: dict):
    result = _qualify(pack, "P7", gold_responses["P7"])
    assert result["disposition"] == "PASS"
    assert result["semantic_oracle"]["derived_semantic_status"] == "weak"


def test_valid_p1_oracle_passes(pack: dict, gold_responses: dict):
    result = _qualify(pack, "P1", gold_responses["P1"])
    assert result["disposition"] == "PASS"
    assert result["semantic_oracle"]["derived_semantic_status"] == "verified"


def test_wrong_p6_support_fails(pack: dict):
    obs = _obs(pack, "P6")
    obs["support_spans"] = [{"evidence_id": "SRC-P6-W03", "start": 401, "end": 457}]
    result = _qualify(pack, "P6", _raw(obs))
    assert result["disposition"] == "FAIL"
    assert "support_span_mismatch" in result["fail_reasons"]


def test_wrong_p6_contradiction_span_fails(pack: dict):
    obs = _obs(pack, "P6")
    obs["blockers"][0]["evidence_spans"] = [{"evidence_id": "SRC-P6-W03", "start": 1501, "end": 1562}]
    result = _qualify(pack, "P6", _raw(obs))
    assert result["disposition"] == "FAIL"


def test_missing_p6_contradiction_fails(pack: dict):
    obs = _obs(pack, "P6")
    obs["blockers"] = []
    result = _qualify(pack, "P6", _raw(obs))
    assert result["disposition"] == "FAIL"
    assert result["adjudication"]["semantic_status_by_claim"]["P6.claim.1"] == "weak"


def test_fabricated_p6_blocker_fails(pack: dict):
    obs = _obs(pack, "P6")
    obs["blockers"].append(
        {
            "kind": "limitation",
            "evidence_spans": [{"evidence_id": "SRC-P6-W03", "start": 401, "end": 457}],
            "explanation": "Unrelated fabricated restriction.",
        }
    )
    result = _qualify(pack, "P6", _raw(obs))
    assert result["disposition"] == "FAIL"


def test_optimistic_p6_entailment_fails(pack: dict):
    obs = _obs(pack, "P6")
    obs["full_entailment"] = True
    result = _qualify(pack, "P6", _raw(obs))
    assert result["disposition"] == "FAIL"
    assert "full_entailment_mismatch" in result["fail_reasons"]


def test_wrong_p7_support_fails(pack: dict):
    obs = _obs(pack, "P7")
    obs["support_spans"] = [{"evidence_id": "SRC-P7-K03", "start": 1, "end": 60}]
    result = _qualify(pack, "P7", _raw(obs))
    assert result["disposition"] == "FAIL"


def test_wrong_p7_limitation_fails(pack: dict):
    obs = _obs(pack, "P7")
    obs["blockers"][0]["evidence_spans"] = [{"evidence_id": "SRC-P7-K03", "start": 2001, "end": 2048}]
    result = _qualify(pack, "P7", _raw(obs))
    assert result["disposition"] == "FAIL"


def test_empty_p7_blocker_span_invalid(pack: dict):
    obs = _obs(pack, "P7")
    obs["blockers"][0]["evidence_spans"] = []
    result = _qualify(pack, "P7", _raw(obs))
    assert result["disposition"] == "INVALID"
    assert "empty_limitation_evidence_span" in result["invalid_reasons"]


def test_optimistic_p7_entailment_fails(pack: dict):
    obs = _obs(pack, "P7")
    obs["full_entailment"] = True
    result = _qualify(pack, "P7", _raw(obs))
    assert result["disposition"] == "FAIL"


def test_p1_wrong_but_in_bounds_support_fails(pack: dict):
    obs = _obs(pack, "P1")
    obs["support_spans"] = [{"evidence_id": "SRC-P1-W02", "start": 1501, "end": 1563}]
    result = _qualify(pack, "P1", _raw(obs))
    assert result["disposition"] == "FAIL"


def test_p1_spurious_blocker_fails(pack: dict):
    obs = _obs(pack, "P1")
    obs["blockers"] = [
        {
            "kind": "limitation",
            "evidence_spans": [{"evidence_id": "SRC-P1-W02", "start": 1500, "end": 1563}],
            "explanation": "Spurious qualifier.",
        }
    ]
    result = _qualify(pack, "P1", _raw(obs))
    assert result["disposition"] == "FAIL"


def test_malformed_response_invalid():
    with pytest.raises(ResponseContractError):
        parse_analyzer_response("{not json")


def test_markdown_fenced_response_rejected(pack: dict, gold_responses: dict):
    fenced = f"```json\n{gold_responses['P1']}\n```"
    with pytest.raises(ResponseContractError, match="markdown_fence_rejected"):
        parse_analyzer_response(fenced)


def test_trailing_prose_invalid(pack: dict, gold_responses: dict):
    with pytest.raises(ResponseContractError, match="trailing_prose"):
        parse_analyzer_response(gold_responses["P1"] + "\nDone.")


def test_unknown_top_level_field_invalid(pack: dict, gold_responses: dict):
    payload = json.loads(gold_responses["P1"])
    payload["status"] = "verified"
    with pytest.raises(ResponseContractError, match="unknown_top_level_field"):
        parse_analyzer_response(json.dumps(payload))


def test_duplicate_json_keys_invalid():
    raw = '{"observations": [{"claim_id": "P1.claim.1", "claim_id": "X", "support_spans": [], "full_entailment": true, "blockers": []}]}'
    with pytest.raises(ResponseContractError, match="duplicate_json_keys"):
        parse_analyzer_response(raw)


def test_string_true_boolean_invalid(pack: dict):
    obs = _obs(pack, "P1")
    obs["full_entailment"] = "true"
    with pytest.raises(ResponseContractError, match="invalid_full_entailment_type"):
        parse_analyzer_response(_raw(obs))


def test_numeric_one_boolean_invalid(pack: dict):
    obs = _obs(pack, "P1")
    obs["full_entailment"] = 1
    with pytest.raises(ResponseContractError, match="invalid_full_entailment_type"):
        parse_analyzer_response(_raw(obs))


def test_unknown_observation_field_invalid(pack: dict):
    obs = _obs(pack, "P1")
    obs["status"] = "verified"
    with pytest.raises(ResponseContractError, match="unknown_field"):
        parse_analyzer_response(_raw(obs))


def test_extra_claim_row_invalid(pack: dict, gold_responses: dict):
    payload = json.loads(gold_responses["P6"])
    extra = copy.deepcopy(payload["observations"][0])
    extra["claim_id"] = "EXTRA.claim"
    payload["observations"].append(extra)
    result = _qualify(pack, "P6", json.dumps(payload))
    assert result["disposition"] == "INVALID"


def test_duplicate_claim_row_invalid(pack: dict, gold_responses: dict):
    payload = json.loads(gold_responses["P6"])
    payload["observations"].append(copy.deepcopy(payload["observations"][0]))
    result = _qualify(pack, "P6", json.dumps(payload))
    assert result["disposition"] == "INVALID"


def test_mutated_draft_with_recomputed_hash_invalid(pack: dict, gold_responses: dict):
    bundle = _bundle(pack, "P6")
    bundle["draft_text"] = bundle["draft_text"] + " mutated"
    bundle["draft_sha256"] = sha256_text(bundle["draft_text"])
    result = _qualify(pack, "P6", gold_responses["P6"], bundle=bundle)
    assert result["disposition"] == "INVALID"
    assert "frozen_draft_text_mismatch" in result["invalid_reasons"]


def test_caller_replacement_hash_cannot_override_frozen_truth(pack: dict, gold_responses: dict):
    bundle = _bundle(pack, "P6")
    bundle["draft_text"] = bundle["draft_text"] + " mutated"
    bundle["draft_sha256"] = sha256_text(bundle["draft_text"])
    ingress = validate_trusted_ingress(
        bundle,
        frozen_truth=_frozen(pack, "P6"),
        caller_draft_sha256=bundle["draft_sha256"],
    )
    assert ingress["valid"] is False
    assert "caller_draft_sha256_overrides_frozen_truth" in ingress["invalid_reasons"]


def test_mutated_claim_with_consistent_span_invalid(pack: dict, gold_responses: dict):
    bundle = _bundle(pack, "P6")
    bundle["claims"][0]["claim_text"] = "Different claim text."
    bundle["claims"][0]["claim_span"] = [0, len("Different claim text.")]
    bundle["draft_text"] = "Different claim text."
    bundle["draft_sha256"] = sha256_text(bundle["draft_text"])
    result = _qualify(pack, "P6", gold_responses["P6"], bundle=bundle)
    assert result["disposition"] == "INVALID"
    assert "frozen_draft_text_mismatch" in result["invalid_reasons"]


def test_mutated_source_with_recomputed_hash_invalid(pack: dict, gold_responses: dict):
    bundle = _bundle(pack, "P6")
    bundle["evidence_manifest"][0]["source_text"] += "x"
    bundle["evidence_manifest"][0]["source_sha256"] = sha256_text(
        bundle["evidence_manifest"][0]["source_text"]
    )
    result = _qualify(pack, "P6", gold_responses["P6"], bundle=bundle)
    assert result["disposition"] == "INVALID"
    assert "frozen_evidence_manifest_mismatch" in result["invalid_reasons"]


def test_invisible_evidence_invalid(pack: dict, gold_responses: dict):
    bundle = _bundle(pack, "P6")
    bundle["evidence_manifest"][0]["verifier_visible"] = False
    result = _qualify(pack, "P6", gold_responses["P6"], bundle=bundle)
    assert result["disposition"] == "INVALID"


def test_invalid_span_invalid(pack: dict, gold_responses: dict):
    obs = _obs(pack, "P6")
    obs["support_spans"][0]["start"] = -1
    result = _qualify(pack, "P6", _raw(obs))
    assert result["disposition"] == "INVALID"


def test_duplicate_evidence_id_invalid(pack: dict, gold_responses: dict):
    bundle = _bundle(pack, "P6")
    duplicate = copy.deepcopy(bundle["evidence_manifest"][0])
    bundle["evidence_manifest"].append(duplicate)
    ingress = validate_trusted_ingress(bundle)
    assert ingress["valid"] is False
    assert "duplicate_evidence_id" in ingress["invalid_reasons"]


def test_invalid_envelope_verified_p1_not_promoted(pack: dict, gold_responses: dict):
    bundle = _bundle(pack, "P1")
    bundle["draft_sha256"] = "0" * 64
    result = _qualify(pack, "P1", gold_responses["P1"], bundle=bundle)
    assert result["disposition"] == "INVALID"
    assert result.get("adjudication") is None


def test_attempt_ledger_order_and_stop_on_fail(pack: dict, gold_responses: dict):
    bad_p6 = json.dumps(
        {
            "observations": [
                {
                    **_obs(pack, "P6"),
                    "blockers": [],
                }
            ]
        }
    )
    artifact = run_offline_qualification(
        pack,
        response_provider={
            "P6": bad_p6,
            "P7": gold_responses["P7"],
            "P1": gold_responses["P1"],
        },
    )
    assert artifact["overall_disposition"] == "FAIL"
    assert artifact["pass_count"] == 0
    assert artifact["fail_count"] == 1
    dispositions = {row["case_id"]: row["disposition"] for row in artifact["case_results"]}
    assert dispositions["P6"] == "FAIL"
    assert dispositions["P7"] == "NOT_RUN"
    assert dispositions["P1"] == "NOT_RUN"
    assert artifact["attempt_ledger"]["attempt_count"] == 1


def test_fourth_attempt_denied():
    ledger = AttemptLedger(run_id="test")
    for case_id in CASE_ORDER:
        ledger.record_attempt(case_id=case_id, disposition="PASS")
    with pytest.raises(AttemptGovernanceError, match="fourth_attempt_denied"):
        ledger.authorize_attempt("P6")


def test_retry_replay_denied():
    ledger = AttemptLedger(run_id="test")
    ledger.record_attempt(case_id="P6", disposition="PASS")
    with pytest.raises(AttemptGovernanceError, match="duplicate_retry"):
        ledger.record_attempt(case_id="P6", disposition="PASS")
    with pytest.raises(AttemptGovernanceError, match="replay_retry_denied"):
        ledger.reject_replay_attempt("P6")


def test_unauthorized_out_of_order_attempt():
    ledger = AttemptLedger(run_id="test")
    with pytest.raises(AttemptGovernanceError, match="out_of_order_attempt"):
        ledger.authorize_attempt("P1")


def test_gold_mock_run_passes_three_of_three(pack: dict, gold_responses: dict):
    artifact = run_offline_qualification(pack, response_provider=gold_responses)
    assert artifact["overall_disposition"] == "PASS"
    assert artifact["pass_count"] == 3
    assert artifact["invalid_count"] == 0
    assert artifact["provider_calls"] == 0
    assert artifact["qualification_ready"] is True
    assert artifact["artifact_integrity"]["valid"] is True


def test_stale_pass_artifact_rejected(pack: dict, gold_responses: dict):
    artifact = run_offline_qualification(pack, response_provider=gold_responses)
    artifact["identity_hashes"]["harness_implementation_sha"] = "0" * 40
    integrity = verify_artifact_integrity(artifact)
    assert integrity["valid"] is False
    assert "stale_harness_implementation_sha" in integrity["invalid_reasons"]


def test_corrupt_pass_artifact_digest_rejected(pack: dict, gold_responses: dict):
    artifact = run_offline_qualification(pack, response_provider=gold_responses)
    artifact["pass_count"] = 99
    integrity = verify_artifact_integrity(artifact)
    assert integrity["valid"] is False
    assert "corrupt_artifact_digest" in integrity["invalid_reasons"]


def test_failed_artifact_not_overwritten_by_integrity_check(pack: dict, gold_responses: dict):
    bad = run_offline_qualification(
        pack,
        response_provider={"P6": _raw({**_obs(pack, "P6"), "blockers": []})},
    )
    assert bad["overall_disposition"] == "FAIL"
    integrity = verify_artifact_integrity(bad)
    assert "pass_with_non_pass_case" not in integrity["invalid_reasons"]


def test_provider_execution_flag_blocks_offline_run(pack: dict, gold_responses: dict, monkeypatch):
    monkeypatch.setenv("SEMANTIC_ANALYZER_EXECUTE", "1")
    with pytest.raises(Exception, match="provider execution flag set"):
        run_offline_qualification(pack, response_provider=gold_responses)


def test_identity_hashes_include_schema_and_prompt():
    hashes = build_run_identity_hashes(harness_implementation_sha=get_implementation_git_sha())
    assert hashes["schema_sha256"] == ANALYZER_SCHEMA_SHA256
    assert hashes["prompt_sha256"] == ANALYZER_PROMPT_SHA256
    assert hashes["required_engine_baseline_sha"] == REQUIRED_ENGINE_BASELINE_SHA


def test_correct_harness_head_and_engine_baseline_allowed():
    result = validate_harness_identities()
    assert result["valid"] is True
    assert result["harness_implementation_sha"] != REQUIRED_ENGINE_BASELINE_SHA


def test_wrong_harness_head_engine_baseline_only_fails():
    result = validate_harness_identities(harness_implementation_sha=REQUIRED_ENGINE_BASELINE_SHA)
    assert result["valid"] is False
    assert "harness_sha_equals_engine_baseline_only" in result["invalid_reasons"]


def test_wrong_engine_baseline_not_ancestor_fails():
    result = validate_harness_identities(harness_implementation_sha="0" * 40)
    assert result["valid"] is False
    assert "engine_baseline_not_ancestor_of_harness" in result["invalid_reasons"]


def test_engine_baseline_substitution_cannot_make_preflight_ready():
    report = run_preflight()
    assert report["identity_validation"]["harness_implementation_sha"] != REQUIRED_ENGINE_BASELINE_SHA
    substituted = validate_harness_identities(harness_implementation_sha=REQUIRED_ENGINE_BASELINE_SHA)
    assert substituted["valid"] is False


def test_ledger_restore_clears_false_stopped_flag():
    payload = {
        "run_id": "tampered",
        "stopped": False,
        "stop_reason": None,
        "records": [
            {
                "case_id": "P6",
                "attempt_index": 1,
                "authorized": True,
                "recorded_at": "t0",
                "disposition": "FAIL",
            }
        ],
    }
    with pytest.raises(AttemptGovernanceError, match="ledger_stopped_flag_contradicts_history"):
        AttemptLedger.from_dict(payload)


def test_ledger_restore_rejects_attempts_after_terminal_fail():
    payload = {
        "run_id": "tampered",
        "stopped": False,
        "records": [
            {
                "case_id": "P6",
                "attempt_index": 1,
                "authorized": True,
                "recorded_at": "t0",
                "disposition": "FAIL",
            },
            {
                "case_id": "P7",
                "attempt_index": 2,
                "authorized": True,
                "recorded_at": "t1",
                "disposition": "PASS",
            },
        ],
    }
    with pytest.raises(AttemptGovernanceError, match="ledger_attempts_after_terminal_disposition"):
        AttemptLedger.from_dict(payload)


def test_ledger_restore_rejects_out_of_order_history():
    payload = {
        "run_id": "tampered",
        "records": [
            {
                "case_id": "P7",
                "attempt_index": 1,
                "authorized": True,
                "recorded_at": "t0",
                "disposition": "PASS",
            }
        ],
    }
    with pytest.raises(AttemptGovernanceError, match="ledger_order_violation"):
        AttemptLedger.from_dict(payload)


def test_qualification_ready_false_when_integrity_invalid(pack: dict, gold_responses: dict):
    artifact = run_offline_qualification(pack, response_provider=gold_responses)
    artifact["identity_hashes"]["harness_implementation_sha"] = "0" * 40
    artifact["artifact_integrity"] = verify_artifact_integrity(artifact)
    assert artifact["artifact_integrity"]["valid"] is False
    assert compute_qualification_ready(artifact) is False


def test_qualification_ready_false_when_digest_corrupt(pack: dict, gold_responses: dict):
    artifact = run_offline_qualification(pack, response_provider=gold_responses)
    artifact["pass_count"] = 99
    artifact["artifact_integrity"] = verify_artifact_integrity(artifact)
    assert compute_qualification_ready(artifact) is False


def test_qualification_ready_false_on_fail_run(pack: dict):
    artifact = run_offline_qualification(
        pack,
        response_provider={"P6": _raw({**_obs(pack, "P6"), "blockers": []})},
    )
    assert artifact["qualification_ready"] is False


def test_provider_execution_default_disabled():
    assert provider_execution_authorized() is False
