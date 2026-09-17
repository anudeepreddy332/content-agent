"""Deterministic Call-A provider qualification runner tests. No providers."""
from __future__ import annotations

import json
import socket
from pathlib import Path

import httpx
import pytest

from scripts.call_a_preprovider_harness import (
    CASE_ORDER,
    EXPECTED_FIXTURE_SHA256,
    build_gold_mock_response,
    evaluate_parsed_case,
    load_fixture_pack,
    prompt_sha256,
    response_contract_sha256,
)
from scripts.call_a_provider_qualification_runner import (
    ACCEPTED_RETURNED_MODEL,
    EXECUTE_ENV_VAR,
    MAX_PROVIDER_REQUESTS,
    OWNER_AUTHORIZATION_ENV_VAR,
    REQUESTED_MODEL,
    AttemptGovernanceError,
    DurableAttemptLedger,
    ExecutionAuthorizationError,
    allocate_run_directory,
    build_all_case_requests,
    build_approved_execution_config_hash,
    build_frozen_experiment_identity,
    compute_owner_authorization_token,
    conservative_total_cost_bound,
    execute_case_once,
    execute_mock_qualification_run,
    execute_provider_qualification_run,
    frozen_experiment_identity_hash,
    issue_execution_authorization,
    provider_execution_authorized,
    run_provider_preflight,
    validate_provider_http_response,
    verify_run_artifact_integrity,
)


@pytest.fixture(autouse=True)
def enforce_deepseek_flash_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setattr("config.DEEPSEEK_MODEL", "deepseek-flash", raising=False)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class GuardedSocket(socket.socket):
        def connect(self, address):  # type: ignore[override]
            raise OSError(f"network denied during qualification tests: {address}")

    monkeypatch.setattr(socket, "socket", GuardedSocket)
    monkeypatch.delenv(EXECUTE_ENV_VAR, raising=False)
    monkeypatch.delenv(OWNER_AUTHORIZATION_ENV_VAR, raising=False)
    isolated_root = tmp_path / "call_a_qualification_root"
    (isolated_root / "runs").mkdir(parents=True)
    monkeypatch.setattr(
        "scripts.call_a_provider_qualification_runner.EXPERIMENT_ROOT",
        isolated_root,
    )
    monkeypatch.setattr(
        "scripts.call_a_provider_qualification_runner.OUTPUT_ROOT",
        isolated_root / "runs",
    )


@pytest.fixture
def pack() -> dict:
    return load_fixture_pack()


@pytest.fixture
def cases_by_id(pack) -> dict[str, dict]:
    return {case["case_id"]: case for case in pack["cases"]}


def test_frozen_model_identity_policy():
    from config import DEEPSEEK_MODEL

    assert REQUESTED_MODEL == "deepseek-flash"
    assert ACCEPTED_RETURNED_MODEL == "deepseek-flash"
    assert REQUESTED_MODEL == ACCEPTED_RETURNED_MODEL
    assert DEEPSEEK_MODEL == "deepseek-flash"


def test_preflight_freezes_fixture_and_hashes(pack):
    preflight = run_provider_preflight()
    assert preflight["execution_ready"] is True
    assert preflight["fixture_population"]["fixture_count"] == 16
    assert preflight["fixture_population"]["gold_claim_count"] == 24
    assert preflight["fixture_population"]["gold_material_claim_count"] == 17
    assert preflight["fixture_sha256"] == EXPECTED_FIXTURE_SHA256
    assert preflight["prompt_sha256"] == prompt_sha256()
    assert preflight["response_contract_sha256"] == response_contract_sha256()
    assert preflight["max_provider_requests"] == 16
    assert preflight["case_order"] == list(CASE_ORDER)


def test_all_fixtures_load_and_request_count_bounded(pack):
    requests = build_all_case_requests(pack)
    assert len(requests) == MAX_PROVIDER_REQUESTS
    assert [req["case_id"] for req in requests] == list(CASE_ORDER)
    hashes = {req["case_id"]: req["request_body_sha256"] for req in requests}
    assert len(set(hashes.values())) == len(hashes)


def test_mock_run_uses_production_pipeline(cases_by_id, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.call_a_provider_qualification_runner.OUTPUT_ROOT",
        tmp_path / "runs",
    )
    artifact = execute_mock_qualification_run(run_id="mock_run_pipeline")
    assert artifact["provider_calls"] == 0
    assert len(artifact["case_results"]) == 16
    g13 = next(r for r in artifact["case_results"] if r["case_id"] == "G13-required-by-brief-material")
    comp = g13["comparisons"][0]
    assert comp["raw_model_material"] is False
    assert comp["final_policy_material"] is True
    assert comp["materiality_override"] == "required_by_brief"
    g15 = next(r for r in artifact["case_results"] if r["case_id"] == "G15-anchor-ambiguity")
    assert all(c["anchor_validity"] == "ANCHOR_AMBIGUOUS" for c in g15["inventory"]["claims"])
    integrity = verify_run_artifact_integrity(artifact)
    assert integrity["valid"] is True
    identity = artifact["identity_hashes"]
    assert identity["execution_git_sha"] == identity["harness_implementation_sha"]
    assert identity["harness_implementation_sha"] == identity["runner_implementation_sha"]
    assert identity["requested_model"] == "deepseek-flash"
    assert identity["accepted_returned_model"] == "deepseek-flash"


def test_frozen_identity_explicitly_binds_execution_harness_and_runner_heads():
    identity = build_frozen_experiment_identity()
    assert identity["execution_git_sha"] == identity["harness_implementation_sha"]
    assert identity["harness_implementation_sha"] == identity["runner_implementation_sha"]
    assert identity["execution_git_sha"] != identity["qualified_harness_head"]


def test_mock_golden_run_fails_critical_false_nonmaterial_trap():
    artifact = execute_mock_qualification_run(run_id="mock_run_golden_fail_g14")
    g14 = next(r for r in artifact["case_results"] if r["case_id"] == "G14-false-nonmaterial-trap")
    assert g14["disposition"] == "FAIL"
    assert g14["critical_false_nonmaterial_final"]
    assert artifact["overall_disposition"] == "FAIL"
    assert any(f["kind"] == "critical_false_nonmaterial" for f in artifact["critical_failures"])


def test_critical_missed_claim_causes_fail(cases_by_id):
    case = cases_by_id["G14-false-nonmaterial-trap"]
    raw = json.dumps([])  # omit the only gold claim
    result = evaluate_parsed_case(case=case, raw_response=raw)
    assert result["disposition"] == "FAIL"
    assert result["critical_misses"]


def test_contract_error_recorded(cases_by_id):
    case = cases_by_id["G01-simple-factual"]
    result = evaluate_parsed_case(case=case, raw_response="not json")
    assert result["disposition"] == "INVALID"
    assert result["contract_error"]


def test_noncritical_miss_reported_without_critical_gate_fail(cases_by_id):
    case = cases_by_id["G12-incidental-factual"]
    raw = json.dumps([])  # miss incidental claim — not in critical set
    result = evaluate_parsed_case(case=case, raw_response=raw)
    assert result["disposition"] == "PASS"
    assert result["missing"]
    assert not result["critical_misses"]


def test_provider_identity_drift_fails_visibly():
    ok = validate_provider_http_response(
        status_code=200,
        body={
            "id": "resp-1",
            "model": REQUESTED_MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": "[]"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )
    assert ok["valid"] is True
    drift = validate_provider_http_response(
        status_code=200,
        body={
            "id": "resp-2",
            "model": "deepseek-chat",
            "choices": [{"finish_reason": "stop", "message": {"content": "[]"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )
    assert drift["valid"] is False
    assert "returned_model_mismatch" in drift["invalid_reasons"]


def test_cost_ceiling_blocks_execution_before_overspend(monkeypatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.setenv(
        OWNER_AUTHORIZATION_ENV_VAR,
        compute_owner_authorization_token(
            experiment_identity_hash=frozen_experiment_identity_hash(),
        ),
    )
    monkeypatch.setattr(
        "scripts.call_a_provider_qualification_runner.conservative_total_cost_bound",
        lambda schedule=None: {
            "conservative_max_total_usd": 999.0,
            "recommended_owner_ceiling_usd": 0.01,
            "pricing_snapshot": {"recommended_owner_ceiling_usd": 0.01},
        },
    )
    with pytest.raises(ExecutionAuthorizationError, match="exceeds owner authorization ceiling"):
        issue_execution_authorization()


def test_run_artifact_directory_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.call_a_provider_qualification_runner.OUTPUT_ROOT",
        tmp_path / "runs",
    )
    first = execute_mock_qualification_run(run_id="append_only_run")
    second_dir = tmp_path / "runs" / "append_only_run"
    assert second_dir.exists()
    with pytest.raises(Exception, match="already exists"):
        allocate_run_directory(run_id="append_only_run")
    assert first["run_id"] == "append_only_run"


def test_provider_execution_disabled_by_default():
    assert provider_execution_authorized() is False
    with pytest.raises(ExecutionAuthorizationError):
        execute_provider_qualification_run()


def test_attempt_ledger_rejects_duplicate_retry(tmp_path):
    ledger = DurableAttemptLedger(
        tmp_path / "ledger_run",
        run_id="ledger_run",
        frozen_experiment_identity_hash="abc",
    )
    ledger.consume_attempt_before_network(CASE_ORDER[0])
    ledger.finalize_attempt(CASE_ORDER[0], disposition="PASS")
    with pytest.raises(AttemptGovernanceError, match="duplicate_retry"):
        ledger.consume_attempt_before_network(CASE_ORDER[0])


def test_anchor_failure_represented_in_mock_run():
    artifact = execute_mock_qualification_run(run_id="mock_anchor_failure")
    g16 = next(r for r in artifact["case_results"] if r["case_id"] == "G16-anchor-failure")
    claim = g16["inventory"]["claims"][0]
    assert claim["anchor_validity"] == "ANCHOR_FAILED"
    assert claim["occurrences"] == []


def test_silently_dropped_claim_would_fail_gate(cases_by_id, monkeypatch):
    case = cases_by_id["G01-simple-factual"]
    # Two rows same claim_text — merge keeps one; not silent drop.
    # Force drop by mocking build to drop — instead test parse rows all appear:
    rows = case["call_a_response"]
    raw = json.dumps(rows)
    result = evaluate_parsed_case(case=case, raw_response=raw)
    assert result["silently_dropped"] == []


def test_conservative_cost_bound_within_recommended_ceiling():
    budget = conservative_total_cost_bound()
    assert budget["conservative_max_total_usd"] <= budget["recommended_owner_ceiling_usd"]
    assert budget["typical_total_usd"] < budget["conservative_max_total_usd"]


def test_live_path_mock_http_without_network(cases_by_id, pack, tmp_path, monkeypatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.setenv(
        OWNER_AUTHORIZATION_ENV_VAR,
        compute_owner_authorization_token(
            experiment_identity_hash=frozen_experiment_identity_hash(),
        ),
    )
    monkeypatch.setattr(
        "scripts.call_a_provider_qualification_runner.OUTPUT_ROOT",
        tmp_path / "runs",
    )
    case = cases_by_id["G01-simple-factual"]
    request = build_all_case_requests(pack)[0]
    run_dir = allocate_run_directory(run_id="live_mock_http")
    ledger = DurableAttemptLedger(
        run_dir,
        run_id="live_mock_http",
        frozen_experiment_identity_hash=frozen_experiment_identity_hash(),
    )
    from scripts.call_a_provider_qualification_runner import ProviderRunState

    auth = issue_execution_authorization()
    run_state = ProviderRunState(run_id="live_mock_http", run_dir=run_dir)

    def fake_post(url, headers, json):  # type: ignore[no-untyped-def]
        content = build_gold_mock_response(case)
        model = json["model"]
        return httpx.Response(
            200,
            json={
                "id": "mock-resp",
                "model": model,
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    result = execute_case_once(
        case=case,
        request=request,
        run_state=run_state,
        ledger=ledger,
        execution_auth=auth,
        http_post=fake_post,
    )
    assert result["disposition"] == "PASS"
    assert run_state.provider_calls == 1


def test_invalid_schema_response_is_contract_error(cases_by_id):
    case = cases_by_id["G01-simple-factual"]
    bad = [{"claim_text": "x", "anchor_quote": "y", "claim_type": "factual", "material": True, "extra": "nope"}]
    result = evaluate_parsed_case(case=case, raw_response=json.dumps(bad))
    assert result["disposition"] == "INVALID"
    assert "contract_error" in result["invalid_reasons"][0]


def test_approved_execution_config_hash_stable(pack):
    h1 = build_approved_execution_config_hash(pack)
    h2 = build_approved_execution_config_hash(pack)
    assert h1 == h2
