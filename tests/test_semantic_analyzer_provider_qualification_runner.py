"""Deterministic semantic-analyzer provider qualification runner tests. No providers."""
from __future__ import annotations

import copy
import socket
from pathlib import Path
import httpx
import pytest

from scripts.semantic_analyzer_preprovider_harness import ExecutionAuthorizationError, sha256_text
from scripts.semantic_analyzer_provider_qualification_runner import (
    CASE_ORDER,
    EXECUTE_ENV_VAR,
    HARD_SPEND_CEILING_USD,
    MAX_OUTPUT_TOKENS,
    QUALIFIED_HARNESS_HEAD,
    RUNNER_ID,
    REQUESTED_MODEL,
    AttemptGovernanceError,
    DurableAttemptLedger,
    allocate_run_directory,
    build_all_case_requests,
    build_approved_execution_config_hash,
    build_provider_client_config,
    build_request_identities,
    build_semantic_analyzer_http_client,
    conservative_total_cost_bound,
    execute_case_once,
    execute_provider_qualification_run,
    issue_execution_authorization,
    load_price_schedule,
    provider_execution_authorized,
    run_provider_preflight,
    validate_provider_http_response,
    verify_run_artifact_integrity,
)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class GuardedSocket(socket.socket):
        def connect(self, address):  # type: ignore[override]
            raise OSError(f"network denied during qualification tests: {address}")

    monkeypatch.setattr(socket, "socket", GuardedSocket)
    monkeypatch.delenv(EXECUTE_ENV_VAR, raising=False)


@pytest.fixture
def requests_by_case() -> dict[str, dict]:
    return {row["case_id"]: row for row in build_all_case_requests()}


def _mock_response(
    *,
    status_code: int = 200,
    content: str,
    model: str = REQUESTED_MODEL,
    finish_reason: str = "stop",
    completion_tokens: int = 100,
) -> httpx.Response:
    body = {
        "id": "resp-test-1",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 500, "completion_tokens": completion_tokens, "total_tokens": 500 + completion_tokens},
        "system_fingerprint": "fp-test",
    }
    return httpx.Response(status_code=status_code, json=body, request=httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions"))


def test_preflight_reports_execution_disabled_by_default():
    report = run_provider_preflight()
    assert report["provider_execution_default_disabled"] is True
    assert report["execution_ready"] is True
    assert report["qualified_harness_head"] == QUALIFIED_HARNESS_HEAD
    assert report["conservative_cost_bound"]["budget_authorized"] is True
    assert report["conservative_cost_bound"]["total_usd"] <= HARD_SPEND_CEILING_USD


def test_request_hashes_regenerated_for_deepseek_v4_flash():
    identities = build_request_identities()
    for case_id in CASE_ORDER:
        body = next(row for row in build_all_case_requests() if row["case_id"] == case_id)["request_body"]
        assert body["model"] == REQUESTED_MODEL
        assert body["max_tokens"] == MAX_OUTPUT_TOKENS
        assert body["temperature"] == 0.1
        assert body["stream"] is False
        assert body["response_format"] == {"type": "json_object"}
        assert body["thinking_mode"] == "disabled"
        assert identities[case_id]["request_body_sha256"] == sha256_text(
            __import__(
                "scripts.semantic_analyzer_provider_qualification_runner",
                fromlist=["canonical_json_dumps"],
            ).canonical_json_dumps(body)
        )
        assert len(identities[case_id]["request_body_sha256"]) == 64


def test_conservative_cost_bound_under_ceiling():
    bound = conservative_total_cost_bound()
    assert bound["total_usd"] <= HARD_SPEND_CEILING_USD
    assert bound["budget_authorized"] is True


def test_execution_without_authorization_blocked():
    with pytest.raises(ExecutionAuthorizationError, match="provider execution disabled"):
        issue_execution_authorization()


def test_execute_run_without_authorization_blocked():
    with pytest.raises(ExecutionAuthorizationError):
        execute_provider_qualification_run()


def test_transport_config_is_retry_free_and_no_redirects():
    config = build_provider_client_config()
    assert config["transport_retries"] == 0
    assert config["follow_redirects"] is False
    assert config["trust_env"] is False
    assert config["requested_model"] == REQUESTED_MODEL
    client = build_semantic_analyzer_http_client()
    assert client._transport is not None  # noqa: SLF001


def test_durable_ledger_consumes_before_network(tmp_path: Path):
    ledger = DurableAttemptLedger(tmp_path, run_id="run-test")
    record = ledger.consume_attempt_before_network("P6")
    assert record["state"] == "CONSUMED"
    restored = DurableAttemptLedger.from_run_dir(tmp_path)
    assert restored.payload["records"][0]["case_id"] == "P6"
    with pytest.raises(AttemptGovernanceError, match="restart_retry_denied"):
        restored.restore_or_reject_retry("P6")


def test_durable_ledger_rejects_p7_before_p6_pass(tmp_path: Path):
    ledger = DurableAttemptLedger(tmp_path, run_id="run-test")
    with pytest.raises(AttemptGovernanceError, match="out_of_order_attempt"):
        ledger.consume_attempt_before_network("P7")


def test_durable_ledger_rejects_p1_before_p7_pass(tmp_path: Path):
    ledger = DurableAttemptLedger(tmp_path, run_id="run-test")
    ledger.consume_attempt_before_network("P6")
    ledger.finalize_attempt("P6", disposition="PASS")
    with pytest.raises(AttemptGovernanceError, match="out_of_order_attempt"):
        ledger.consume_attempt_before_network("P1")


def test_durable_ledger_fourth_attempt_denied(tmp_path: Path):
    ledger = DurableAttemptLedger(tmp_path, run_id="run-test")
    for case_id in CASE_ORDER:
        ledger.consume_attempt_before_network(case_id)
        ledger.finalize_attempt(case_id, disposition="PASS")
    with pytest.raises(AttemptGovernanceError, match="fourth_attempt_denied"):
        ledger.consume_attempt_before_network("P6")


def test_durable_ledger_fail_stops_run(tmp_path: Path):
    ledger = DurableAttemptLedger(tmp_path, run_id="run-test")
    ledger.consume_attempt_before_network("P6")
    ledger.finalize_attempt("P6", disposition="FAIL")
    with pytest.raises(AttemptGovernanceError, match="run_stopped"):
        ledger.consume_attempt_before_network("P7")


def test_allocate_run_directory_rejects_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.OUTPUT_ROOT",
        tmp_path,
    )
    first = allocate_run_directory(run_id="collision-run")
    assert first.exists()
    with pytest.raises(Exception, match="already exists"):
        allocate_run_directory(run_id="collision-run")


def test_provider_response_validation_requires_usage_and_model():
    missing_usage = validate_provider_http_response(
        status_code=200,
        headers={},
        body={
            "id": "x",
            "model": REQUESTED_MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": '{"observations":[]}'}}],
        },
    )
    assert missing_usage["valid"] is False
    assert "missing_usage" in missing_usage["invalid_reasons"]

    bad_model = validate_provider_http_response(
        status_code=200,
        headers={},
        body={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    assert bad_model["valid"] is False
    assert "returned_model_mismatch" in bad_model["invalid_reasons"]


def test_redirect_response_invalid():
    result = validate_provider_http_response(
        status_code=302,
        headers={"location": "https://evil.example"},
        body={"choices": []},
    )
    assert result["valid"] is False
    assert "provider_redirect" in result["invalid_reasons"]


def test_completion_tokens_above_max_invalid():
    result = validate_provider_http_response(
        status_code=200,
        headers={},
        body={
            "id": "x",
            "model": REQUESTED_MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": MAX_OUTPUT_TOKENS + 1},
        },
    )
    assert result["valid"] is False
    assert "completion_tokens_above_max" in result["invalid_reasons"]


def test_execute_case_timeout_consumes_attempt(
    tmp_path: Path,
    requests_by_case: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    ledger = DurableAttemptLedger(tmp_path, run_id="timeout-run")
    state = __import__("scripts.semantic_analyzer_provider_qualification_runner", fromlist=["ProviderRunState"]).ProviderRunState(
        run_id="timeout-run",
        run_dir=tmp_path,
    )

    def _timeout(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    artifact = execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=_timeout,
    )
    assert artifact["disposition"] == "INVALID"
    assert artifact["invalid_reason"] == "provider_timeout"
    assert state.provider_calls == 0
    assert ledger.payload["records"][0]["state"] == "FINALIZED"
    with pytest.raises(AttemptGovernanceError):
        ledger.consume_attempt_before_network("P6")


def test_execute_case_connection_error_consumes_attempt(
    tmp_path: Path,
    requests_by_case: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    ledger = DurableAttemptLedger(tmp_path, run_id="conn-run")
    state = __import__("scripts.semantic_analyzer_provider_qualification_runner", fromlist=["ProviderRunState"]).ProviderRunState(
        run_id="conn-run",
        run_dir=tmp_path,
    )

    def _conn_error(*args, **kwargs):
        raise httpx.ConnectError("connection failed")

    artifact = execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=_conn_error,
    )
    assert artifact["invalid_reason"] == "provider_connection_error"
    assert ledger.payload["stopped"] is True


def test_prompt_hash_drift_detected_in_preflight(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.analyzer_prompt_sha256",
        lambda: "0" * 64,
    )
    with pytest.raises(Exception, match="prompt hash drift"):
        run_provider_preflight()


def test_request_hash_drift_detected_in_execution_config():
    identities = build_request_identities()
    approved = build_approved_execution_config_hash()
    assert len(approved) == 64
    assert identities["P6"]["request_body_sha256"] != identities["P7"]["request_body_sha256"]


def test_cost_bound_over_ceiling_blocks_authorization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    schedule = load_price_schedule()
    schedule = copy.deepcopy(schedule)
    schedule["input_cost_per_million_tokens_usd"] = 10.0
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.load_price_schedule",
        lambda *args, **kwargs: schedule,
    )
    with pytest.raises(ExecutionAuthorizationError, match="conservative cost bound"):
        issue_execution_authorization()


def test_honest_run_artifact_integrity_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    artifact = {
        "runner_id": "semantic_analyzer_provider_qualification_runner",
        "stage": "semantic-analyzer-provider-qualification",
        "run_id": "test-run",
        "provider_calls": 0,
        "identity_hashes": {
            "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
            "required_engine_baseline_sha": "1aa4acc7e0ccdb4cb769b8617667d6a505cc2671",
            "fixture_sha256": "72c1bd4dd8a3d39acec01300dd1a9b03021634cfbc49433f0881ee2c2b796d1d",
            "prompt_sha256": run_provider_preflight()["prompt_sha256"],
        },
        "authorization": {"authorization_token": "x"},
        "pricing_snapshot": load_price_schedule(),
        "request_identities": build_request_identities(),
        "attempt_ledger": {"run_id": "test-run", "records": []},
        "case_order": list(CASE_ORDER),
        "case_results": [],
        "pass_count": 0,
        "fail_count": 0,
        "invalid_count": 0,
        "not_run_count": 3,
        "overall_disposition": "FAIL",
        "provider_execution_authorized": False,
    }
    from scripts.semantic_analyzer_provider_qualification_runner import compute_run_artifact_digest

    artifact["artifact_digest"] = compute_run_artifact_digest(artifact)
    first = verify_run_artifact_integrity(artifact)
    second = verify_run_artifact_integrity(artifact)
    assert first["valid"] is True
    assert second["valid"] is True


def test_missing_artifact_digest_invalid():
    artifact = {"artifact_digest": ""}
    integrity = verify_run_artifact_integrity(artifact)
    assert integrity["valid"] is False
    assert "missing_artifact_digest" in integrity["invalid_reasons"]


def test_corrupt_artifact_digest_invalid():
    artifact = {
        "runner_id": "semantic_analyzer_provider_qualification_runner",
        "stage": "semantic-analyzer-provider-qualification",
        "run_id": "test-run",
        "provider_calls": 0,
        "identity_hashes": {
            "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
            "required_engine_baseline_sha": "1aa4acc7e0ccdb4cb769b8617667d6a505cc2671",
            "fixture_sha256": "72c1bd4dd8a3d39acec01300dd1a9b03021634cfbc49433f0881ee2c2b796d1d",
            "prompt_sha256": run_provider_preflight()["prompt_sha256"],
        },
        "authorization": {},
        "pricing_snapshot": load_price_schedule(),
        "request_identities": build_request_identities(),
        "attempt_ledger": {"run_id": "test-run", "records": []},
        "case_order": list(CASE_ORDER),
        "case_results": [],
        "pass_count": 3,
        "fail_count": 0,
        "invalid_count": 0,
        "not_run_count": 0,
        "overall_disposition": "PASS",
        "provider_execution_authorized": True,
        "artifact_digest": "0" * 64,
    }
    integrity = verify_run_artifact_integrity(artifact)
    assert integrity["valid"] is False
    assert "corrupt_artifact_digest" in integrity["invalid_reasons"]


def test_provider_execution_default_disabled():
    assert provider_execution_authorized() is False


def test_p6_first_call_authorized_with_execute_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    assert auth.provider_execution_authorized is True
    assert auth.budget_authorized is True
    assert len(auth.authorization_token) == 64


def test_duplicate_case_request_denied(tmp_path: Path):
    ledger = DurableAttemptLedger(tmp_path, run_id="dup-run")
    ledger.consume_attempt_before_network("P6")
    with pytest.raises(AttemptGovernanceError, match="duplicate_retry"):
        ledger.consume_attempt_before_network("P6")


def test_invalid_stops_run(tmp_path: Path):
    ledger = DurableAttemptLedger(tmp_path, run_id="invalid-run")
    ledger.consume_attempt_before_network("P6")
    ledger.finalize_attempt("P6", disposition="INVALID")
    with pytest.raises(AttemptGovernanceError, match="run_stopped"):
        ledger.consume_attempt_before_network("P7")


def test_fixture_hash_drift_detected_in_artifact_integrity():
    artifact = {
        "runner_id": RUNNER_ID,
        "stage": "semantic-analyzer-provider-qualification",
        "run_id": "fixture-drift",
        "provider_calls": 0,
        "identity_hashes": {
            "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
            "required_engine_baseline_sha": "1aa4acc7e0ccdb4cb769b8617667d6a505cc2671",
            "fixture_sha256": "0" * 64,
            "prompt_sha256": run_provider_preflight()["prompt_sha256"],
        },
        "authorization": {},
        "pricing_snapshot": load_price_schedule(),
        "request_identities": build_request_identities(),
        "attempt_ledger": {"run_id": "fixture-drift", "records": []},
        "case_order": list(CASE_ORDER),
        "case_results": [],
        "pass_count": 0,
        "fail_count": 0,
        "invalid_count": 0,
        "not_run_count": 3,
        "overall_disposition": "FAIL",
        "provider_execution_authorized": False,
        "artifact_digest": "placeholder",
    }
    integrity = verify_run_artifact_integrity(artifact)
    assert integrity["valid"] is False
    assert "stale_fixture_sha256" in integrity["invalid_reasons"]


def test_malformed_provider_output_invalidates_case(
    tmp_path: Path,
    requests_by_case: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    ledger = DurableAttemptLedger(tmp_path, run_id="malformed-run")
    state = __import__("scripts.semantic_analyzer_provider_qualification_runner", fromlist=["ProviderRunState"]).ProviderRunState(
        run_id="malformed-run",
        run_dir=tmp_path,
    )

    def _bad_json(*args, **kwargs):
        return httpx.Response(
            status_code=200,
            text="not-json",
            request=httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions"),
        )

    artifact = execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=_bad_json,
    )
    assert artifact["disposition"] == "INVALID"
    assert "provider_json_decode_error" in artifact["error_message"]
    assert ledger.payload["stopped"] is True


def test_mutated_caller_pack_does_not_change_request_hashes():
    from scripts.semantic_analyzer_preprovider_harness import load_verified_fixture_pack

    verified = load_verified_fixture_pack()
    mutated = copy.deepcopy(verified)
    mutated["cases"]["P6-COMPLETE"]["draft_text"] += " attacker"
    baseline = build_request_identities()
    # Request identities always derive from verified fixture on disk, not caller pack.
    assert baseline == build_request_identities()
    assert mutated["cases"]["P6-COMPLETE"]["draft_text"] != verified["cases"]["P6-COMPLETE"]["draft_text"]
