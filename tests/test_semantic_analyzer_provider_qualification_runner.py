"""Deterministic semantic-analyzer provider qualification runner tests. No providers."""
from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
import httpx
import pytest

from scripts.semantic_analyzer_preprovider_harness import ExecutionAuthorizationError, sha256_text
from scripts.semantic_analyzer_preprovider_harness import build_gold_mock_responses
from scripts.semantic_analyzer_provider_qualification_runner import (
    ACCEPTED_RETURNED_MODEL,
    CASE_ORDER,
    EXECUTE_ENV_VAR,
    HARD_SPEND_CEILING_USD,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_AUTHORIZED,
    LIFECYCLE_SUCCESSOR_PROPOSED,
    LIFECYCLE_TERMINAL,
    MAX_OUTPUT_TOKENS,
    OWNER_AUTHORIZATION_ENV_VAR,
    QUALIFIED_HARNESS_HEAD,
    REGISTRY_VERSION,
    RUNNER_ID,
    REQUESTED_MODEL,
    AttemptGovernanceError,
    DurableAttemptLedger,
    FrozenExperimentRegistry,
    allocate_run_directory,
    build_all_case_requests,
    build_approved_execution_config_hash,
    build_provider_client_config,
    build_request_identities,
    build_semantic_analyzer_http_client,
    calculate_observed_cost_usd,
    canonical_json_dumps,
    compute_owner_authorization_token,
    conservative_total_cost_bound,
    execute_case_once,
    execute_provider_qualification_run,
    find_governed_experiment_evidence,
    find_persisted_terminal_experiments,
    frozen_experiment_identity_hash,
    issue_execution_authorization,
    load_price_schedule,
    prepare_successor_experiment,
    provider_execution_authorized,
    run_provider_preflight,
    validate_provider_http_response,
    verify_run_artifact_integrity,
)

FROZEN_REQUEST_HASHES = {
    "P6": "d25fb7c7588c69eaffd58a54ff4840ec315ea33d41e0f78e23e4883629dbe0bb",
    "P7": "c3459b946a3a46a819010851c2cb25c5b4941a7e23002f1c75bd626bd213cac6",
    "P1": "bdf8a520444a8cdf7d5227ec57c2d3dbcc66b9dc6e25ac04493b4b664c5b7dca",
}
CONSUMED_FIRST_LIVE_IDENTITY_HASH = (
    "654e5f3438858970f19caa530fe6a634c77092fdb0f4a906a8a405a081e2ac9b"
)
CONSUMED_FIRST_LIVE_AUTH_TOKEN = (
    "b32bde1fc6cc1c451f8c7add9db31f2be82c80ac603115e744288d87982b028b"
)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class GuardedSocket(socket.socket):
        def connect(self, address):  # type: ignore[override]
            raise OSError(f"network denied during qualification tests: {address}")

    monkeypatch.setattr(socket, "socket", GuardedSocket)
    monkeypatch.delenv(EXECUTE_ENV_VAR, raising=False)
    monkeypatch.delenv(OWNER_AUTHORIZATION_ENV_VAR, raising=False)
    isolated_root = tmp_path / "isolated_qualification_root"
    isolated_root.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.EXPERIMENT_ROOT",
        isolated_root,
    )
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.OUTPUT_ROOT",
        isolated_root / "runs",
    )
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.EXPERIMENT_REGISTRY_PATH",
        isolated_root / "frozen_experiment_registry.json",
    )


@pytest.fixture
def requests_by_case() -> dict[str, dict]:
    return {row["case_id"]: row for row in build_all_case_requests()}


@pytest.fixture
def experiment_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "semantic_analyzer_provider_qualification"
    runs = root / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.EXPERIMENT_ROOT",
        root,
    )
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.OUTPUT_ROOT",
        runs,
    )
    monkeypatch.setattr(
        "scripts.semantic_analyzer_provider_qualification_runner.EXPERIMENT_REGISTRY_PATH",
        root / "frozen_experiment_registry.json",
    )
    return root


def _registry_path(experiment_root: Path) -> Path:
    return experiment_root / "frozen_experiment_registry.json"


def _owner_token(identity_hash: str | None = None) -> str:
    return compute_owner_authorization_token(
        experiment_identity_hash=identity_hash or frozen_experiment_identity_hash(),
    )


def _seed_disk_governed_evidence(
    experiment_root: Path,
    *,
    identity_hash: str = CONSUMED_FIRST_LIVE_IDENTITY_HASH,
    run_id: str = "semantic_analyzer_run_ed35656b9a7c",
    disposition: str = "INVALID",
) -> Path:
    run_dir = experiment_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = DurableAttemptLedger(
        run_dir,
        run_id=run_id,
        frozen_experiment_identity_hash=identity_hash,
    )
    ledger.consume_attempt_before_network("P6")
    ledger.finalize_attempt("P6", disposition=disposition, detail={"error": "test"})
    artifact = {
        "run_id": run_id,
        "overall_disposition": disposition,
        "identity_hashes": {"frozen_experiment_identity_hash": identity_hash},
    }
    (run_dir / "run_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    return run_dir


def _write_empty_v2_registry(experiment_root: Path) -> None:
    _write_v2_registry(experiment_root, history=[], successor=None)


def _write_v2_registry(
    experiment_root: Path,
    *,
    history: list[dict],
    successor: dict | None,
) -> None:
    registry_path = _registry_path(experiment_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        canonical_json_dumps(
            {
                "registry_version": REGISTRY_VERSION,
                "history": history,
                "successor": successor,
            }
        ),
        encoding="utf-8",
    )


def _successor_proposal(
    *,
    identity_hash: str | None = None,
    run_id: str | None = None,
    lifecycle: str = LIFECYCLE_SUCCESSOR_PROPOSED,
) -> dict:
    return {
        "frozen_experiment_identity_hash": identity_hash or frozen_experiment_identity_hash(),
        "lifecycle": lifecycle,
        "proposed_at": "2026-09-10T12:00:00+00:00",
        "owner_authorization_token": None,
        "authorized_at": None,
        "run_id": run_id,
        "run_dir": f"runs/{run_id}" if run_id else None,
        "execution_session_token": None,
    }


def _terminal_history_row(
    *,
    identity_hash: str = CONSUMED_FIRST_LIVE_IDENTITY_HASH,
    run_id: str = "semantic_analyzer_run_ed35656b9a7c",
    lifecycle: str = LIFECYCLE_TERMINAL,
) -> dict:
    return {
        "run_id": run_id,
        "frozen_experiment_identity_hash": identity_hash,
        "legacy_execution_reference_token": CONSUMED_FIRST_LIVE_AUTH_TOKEN,
        "owner_authorization_token": CONSUMED_FIRST_LIVE_AUTH_TOKEN,
        "run_dir": f"runs/{run_id}",
        "established_at": "2026-09-10T07:44:29.642905+00:00",
        "terminal_at": "2026-09-10T07:44:31.058286+00:00",
        "lifecycle": lifecycle,
        "terminal_disposition": "INVALID",
    }


def _seed_v2_terminal_predecessor(
    experiment_root: Path,
    *,
    identity_hash: str = CONSUMED_FIRST_LIVE_IDENTITY_HASH,
    run_id: str = "semantic_analyzer_run_ed35656b9a7c",
) -> None:
    _seed_disk_governed_evidence(
        experiment_root,
        identity_hash=identity_hash,
        run_id=run_id,
    )
    registry_path = _registry_path(experiment_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry_version": REGISTRY_VERSION,
        "history": [
            {
                "run_id": run_id,
                "frozen_experiment_identity_hash": identity_hash,
                "legacy_execution_reference_token": CONSUMED_FIRST_LIVE_AUTH_TOKEN,
                "owner_authorization_token": CONSUMED_FIRST_LIVE_AUTH_TOKEN,
                "run_dir": f"runs/{run_id}",
                "established_at": "2026-09-10T07:44:29.642905+00:00",
                "terminal_at": "2026-09-10T07:44:31.058286+00:00",
                "lifecycle": LIFECYCLE_TERMINAL,
                "terminal_disposition": "INVALID",
            }
        ],
        "successor": None,
    }
    registry_path.write_text(canonical_json_dumps(payload), encoding="utf-8")


def _mock_response(
    *,
    status_code: int = 200,
    content: str,
    model: str = ACCEPTED_RETURNED_MODEL,
    finish_reason: str = "stop",
    completion_tokens: int = 100,
    prompt_tokens: int = 500,
) -> httpx.Response:
    body = {
        "id": "resp-test-1",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
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
        assert body["thinking"] == {"type": "disabled"}
        assert "thinking_mode" not in body
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
            "model": ACCEPTED_RETURNED_MODEL,
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


def test_frozen_model_pair_accepts_deepseek_flash_response():
    result = validate_provider_http_response(
        status_code=200,
        headers={},
        body={
            "id": "x",
            "model": ACCEPTED_RETURNED_MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": '{"observations":[]}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    assert result["valid"] is True
    assert result["requested_model"] == REQUESTED_MODEL
    assert result["accepted_returned_model"] == ACCEPTED_RETURNED_MODEL
    assert result["observed_returned_model"] == ACCEPTED_RETURNED_MODEL


def test_returned_deepseek_v4_flash_invalid_under_frozen_response_contract():
    result = validate_provider_http_response(
        status_code=200,
        headers={},
        body={
            "id": "x",
            "model": REQUESTED_MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    assert result["valid"] is False
    assert "returned_model_mismatch" in result["invalid_reasons"]


def test_arbitrary_returned_deepseek_model_invalid():
    result = validate_provider_http_response(
        status_code=200,
        headers={},
        body={
            "id": "x",
            "model": "deepseek-reasoner",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    assert result["valid"] is False
    assert "returned_model_mismatch" in result["invalid_reasons"]


def test_empty_returned_model_invalid():
    result = validate_provider_http_response(
        status_code=200,
        headers={},
        body={
            "id": "x",
            "model": "",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    assert result["valid"] is False
    assert "missing_returned_model" in result["invalid_reasons"]


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
            "model": ACCEPTED_RETURNED_MODEL,
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


def test_request_hash_drift_fails_closed():
    identities = build_request_identities()
    approved = build_approved_execution_config_hash()
    tampered = canonical_json_dumps(
        {
            "requested_model": REQUESTED_MODEL,
            "request_identities": {
                **identities,
                "P6": {**identities["P6"], "request_body_sha256": "0" * 64},
            },
        }
    )
    assert approved != sha256_text(tampered)


def _resume_registry(experiment_root: Path, auth_token: str):
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    return registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth_token,
        run_id="different-run-id",
    )


def test_fresh_experiment_allows_p6_once(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    run_dir, ledger, created = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="fresh-run",
    )
    assert created is True
    ledger.consume_attempt_before_network("P6")
    assert ledger.payload["records"][0]["case_id"] == "P6"
    assert run_dir.exists()


def test_p6_consumed_crash_restart_denies_retry(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    _, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="crash-run",
    )
    ledger.consume_attempt_before_network("P6")
    with pytest.raises(AttemptGovernanceError, match="experiment_terminal"):
        _resume_registry(experiment_root, auth.authorization_token)


def test_p6_timeout_restart_no_replacement(experiment_root: Path, monkeypatch: pytest.MonkeyPatch, requests_by_case: dict):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    run_dir, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="timeout-restart",
    )
    state = __import__(
        "scripts.semantic_analyzer_provider_qualification_runner",
        fromlist=["ProviderRunState"],
    ).ProviderRunState(run_id=run_dir.name, run_dir=run_dir)

    def _timeout(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=_timeout,
    )
    with pytest.raises(AttemptGovernanceError, match="experiment_terminal"):
        _resume_registry(experiment_root, auth.authorization_token)


def test_p6_connection_failure_restart_no_replacement(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    requests_by_case: dict,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    run_dir, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="conn-restart",
    )
    state = __import__(
        "scripts.semantic_analyzer_provider_qualification_runner",
        fromlist=["ProviderRunState"],
    ).ProviderRunState(run_id=run_dir.name, run_dir=run_dir)

    def _conn_error(*args, **kwargs):
        raise httpx.ConnectError("connection failed")

    execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=_conn_error,
    )
    with pytest.raises(AttemptGovernanceError, match="experiment_terminal"):
        _resume_registry(experiment_root, auth.authorization_token)


def test_p6_pass_restart_allows_p7_not_p6(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    _, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="p6-pass-run",
    )
    ledger.consume_attempt_before_network("P6")
    ledger.finalize_attempt("P6", disposition="PASS")
    _, ledger2, created = _resume_registry(experiment_root, auth.authorization_token)
    assert created is False
    assert ledger2.next_case_id() == "P7"
    with pytest.raises(AttemptGovernanceError, match="duplicate_retry"):
        ledger2.consume_attempt_before_network("P6")


def test_p6_p7_pass_restart_allows_p1(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    _, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="p6-p7-pass-run",
    )
    for case_id in ("P6", "P7"):
        ledger.consume_attempt_before_network(case_id)
        ledger.finalize_attempt(case_id, disposition="PASS")
    _, ledger2, _ = _resume_registry(experiment_root, auth.authorization_token)
    assert ledger2.next_case_id() == "P1"


def test_p6_fail_restart_blocks_p7_and_new_experiment(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    _, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="p6-fail-run",
    )
    ledger.consume_attempt_before_network("P6")
    ledger.finalize_attempt("P6", disposition="FAIL")
    registry.mark_terminal(ledger=ledger, disposition="FAIL")
    with pytest.raises(AttemptGovernanceError, match="experiment_terminal"):
        _resume_registry(experiment_root, auth.authorization_token)


def test_p6_invalid_restart_blocks_p7(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    _, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="p6-invalid-run",
    )
    ledger.consume_attempt_before_network("P6")
    ledger.finalize_attempt("P6", disposition="INVALID")
    registry.mark_terminal(ledger=ledger, disposition="INVALID")
    with pytest.raises(AttemptGovernanceError, match="experiment_terminal"):
        _resume_registry(experiment_root, auth.authorization_token)


def test_completed_experiment_rerun_denied(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    _, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="completed-run",
    )
    for case_id in CASE_ORDER:
        ledger.consume_attempt_before_network(case_id)
        ledger.finalize_attempt(case_id, disposition="PASS")
    registry.mark_terminal(ledger=ledger, disposition="PASS")
    with pytest.raises(AttemptGovernanceError, match="experiment_terminal"):
        _resume_registry(experiment_root, auth.authorization_token)


def test_different_run_id_cannot_reset_allowance(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    run_dir_a, ledger_a, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="run-alpha",
    )
    ledger_a.consume_attempt_before_network("P6")
    ledger_a.finalize_attempt("P6", disposition="PASS")
    run_dir_b, ledger_b, created = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="run-beta",
    )
    assert created is False
    assert run_dir_b == run_dir_a
    assert ledger_b.payload["records"][0]["case_id"] == "P6"
    assert ledger_b.next_case_id() == "P7"


def test_orphaned_ledger_without_registry_fails_closed(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    run_dir = allocate_run_directory(run_id="orphan-run")
    ledger = DurableAttemptLedger(
        run_dir,
        run_id=run_dir.name,
        frozen_experiment_identity_hash=frozen_experiment_identity_hash(),
    )
    ledger.consume_attempt_before_network("P6")
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    with pytest.raises(AttemptGovernanceError, match="registry_history_missing"):
        registry.establish_or_resume(
            expected_identity_hash=frozen_experiment_identity_hash(),
            authorization_token=auth.authorization_token,
            run_id="fresh-after-delete",
        )


def test_conflicting_experiment_identity_blocked(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    registry = FrozenExperimentRegistry(_registry_path(experiment_root))
    registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        run_id="identity-run",
    )
    with pytest.raises(AttemptGovernanceError, match="conflicting_experiment_identity"):
        registry.establish_or_resume(
            expected_identity_hash="0" * 64,
            authorization_token=auth.authorization_token,
            run_id="identity-run-2",
        )


def test_outbound_requested_model_change_invalid(
    tmp_path: Path,
    requests_by_case: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    ledger = DurableAttemptLedger(tmp_path, run_id="outbound-model-run")
    state = __import__(
        "scripts.semantic_analyzer_provider_qualification_runner",
        fromlist=["ProviderRunState"],
    ).ProviderRunState(run_id="outbound-model-run", run_dir=tmp_path)
    tampered = copy.deepcopy(requests_by_case["P6"])
    tampered["request_body"] = copy.deepcopy(tampered["request_body"])
    tampered["request_body"]["model"] = "deepseek-chat"
    artifact = execute_case_once(
        request=tampered,
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=lambda *args, **kwargs: _mock_response(content="{}"),
    )
    assert artifact["disposition"] == "INVALID"
    assert "outbound_requested_model_mismatch" in artifact["error_message"]
    assert state.provider_calls == 0


def test_model_mismatch_with_usage_records_incurred_cost(
    tmp_path: Path,
    requests_by_case: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    ledger = DurableAttemptLedger(tmp_path, run_id="cost-invalid-run")
    state = __import__(
        "scripts.semantic_analyzer_provider_qualification_runner",
        fromlist=["ProviderRunState"],
    ).ProviderRunState(run_id="cost-invalid-run", run_dir=tmp_path)
    schedule = load_price_schedule()
    usage = {"prompt_tokens": 599, "completion_tokens": 124, "total_tokens": 723}

    def _mismatch(*args, **kwargs):
        return _mock_response(
            content='{"observations":[]}',
            model=REQUESTED_MODEL,
            prompt_tokens=599,
            completion_tokens=124,
        )

    artifact = execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=_mismatch,
    )
    expected_cost, _ = calculate_observed_cost_usd(usage, schedule)
    assert artifact["disposition"] == "INVALID"
    assert "returned_model_mismatch" in artifact["error_message"]
    assert artifact["observed_cost_recorded"] is True
    assert artifact["actual_cost_usd"] == expected_cost
    assert expected_cost > 0
    assert state.total_cost_usd == expected_cost


def test_semantic_fail_records_incurred_cost(
    tmp_path: Path,
    requests_by_case: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    ledger = DurableAttemptLedger(tmp_path, run_id="cost-fail-run")
    state = __import__(
        "scripts.semantic_analyzer_provider_qualification_runner",
        fromlist=["ProviderRunState"],
    ).ProviderRunState(run_id="cost-fail-run", run_dir=tmp_path)
    schedule = load_price_schedule()
    bad_content = json.dumps(
        {
            "observations": [
                {
                    "claim_id": "P6.claim.1",
                    "support_spans": [
                        {"evidence_id": "SRC-P6-W03", "start": 0, "end": 10}
                    ],
                    "full_entailment": False,
                    "blockers": [
                        {
                            "kind": "contradiction",
                            "evidence_spans": [
                                {"evidence_id": "SRC-P6-W03", "start": 1500, "end": 1562}
                            ],
                            "explanation": "Wrong support span; oracle should FAIL not INVALID.",
                        }
                    ],
                }
            ]
        }
    )

    artifact = execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=lambda *args, **kwargs: _mock_response(content=bad_content),
    )
    expected_cost, _ = calculate_observed_cost_usd(
        {"prompt_tokens": 500, "completion_tokens": 100},
        schedule,
    )
    assert artifact["disposition"] == "FAIL"
    assert artifact["actual_cost_usd"] == expected_cost
    assert state.total_cost_usd == expected_cost


def test_pass_records_incurred_cost(
    tmp_path: Path,
    requests_by_case: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    ledger = DurableAttemptLedger(tmp_path, run_id="cost-pass-run")
    state = __import__(
        "scripts.semantic_analyzer_provider_qualification_runner",
        fromlist=["ProviderRunState"],
    ).ProviderRunState(run_id="cost-pass-run", run_dir=tmp_path)
    schedule = load_price_schedule()
    gold = build_gold_mock_responses()["P6"]

    artifact = execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=lambda *args, **kwargs: _mock_response(content=gold),
    )
    expected_cost, _ = calculate_observed_cost_usd(
        {"prompt_tokens": 500, "completion_tokens": 100},
        schedule,
    )
    assert artifact["disposition"] == "PASS"
    assert artifact["actual_cost_usd"] == expected_cost
    assert state.total_cost_usd == expected_cost


def test_missing_trustworthy_usage_does_not_fabricate_cost(
    tmp_path: Path,
    requests_by_case: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    ledger = DurableAttemptLedger(tmp_path, run_id="no-cost-run")
    state = __import__(
        "scripts.semantic_analyzer_provider_qualification_runner",
        fromlist=["ProviderRunState"],
    ).ProviderRunState(run_id="no-cost-run", run_dir=tmp_path)

    def _no_usage(*args, **kwargs):
        body = {
            "id": "resp-test-1",
            "model": ACCEPTED_RETURNED_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "not-json"},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions"),
        )

    artifact = execute_case_once(
        request=requests_by_case["P6"],
        run_state=state,
        durable_ledger=ledger,
        execution_auth=auth,
        http_post=_no_usage,
    )
    assert artifact["disposition"] == "INVALID"
    assert "actual_cost_usd" not in artifact
    assert artifact.get("observed_cost_unavailable") == "missing_usage"
    assert state.total_cost_usd == 0.0


def test_consumed_first_live_experiment_cannot_reopen(experiment_root: Path):
    registry_path = _registry_path(experiment_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        canonical_json_dumps(
            {
                "frozen_experiment_identity_hash": CONSUMED_FIRST_LIVE_IDENTITY_HASH,
                "authorization_token": CONSUMED_FIRST_LIVE_AUTH_TOKEN,
                "established_at": "2026-09-10T07:44:29.642905+00:00",
                "run_dir": "runs/semantic_analyzer_run_ed35656b9a7c",
                "run_id": "semantic_analyzer_run_ed35656b9a7c",
                "terminal": True,
                "terminal_disposition": "INVALID",
                "terminal_at": "2026-09-10T07:44:31.058286+00:00",
            }
        ),
        encoding="utf-8",
    )
    registry = FrozenExperimentRegistry(registry_path)
    with pytest.raises(AttemptGovernanceError, match="experiment_terminal:INVALID"):
        registry.establish_or_resume(
            expected_identity_hash=CONSUMED_FIRST_LIVE_IDENTITY_HASH,
            authorization_token=CONSUMED_FIRST_LIVE_AUTH_TOKEN,
            run_id="semantic_analyzer_run_ed35656b9a7c",
        )


def test_old_authorization_cannot_authorize_new_experiment_identity(experiment_root: Path):
    _seed_v2_terminal_predecessor(experiment_root)
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    registry.prepare_successor(expected_identity_hash=frozen_experiment_identity_hash())
    with pytest.raises(ExecutionAuthorizationError, match="owner_authorization_identity_mismatch"):
        registry.validate_and_persist_owner_authorization(
            CONSUMED_FIRST_LIVE_AUTH_TOKEN,
            frozen_experiment_identity_hash(),
        )


def test_new_experiment_requires_fresh_authorization(experiment_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(EXECUTE_ENV_VAR, raising=False)
    with pytest.raises(ExecutionAuthorizationError, match="provider execution disabled"):
        issue_execution_authorization()
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    auth = issue_execution_authorization()
    assert auth.provider_execution_authorized is True
    assert auth.authorization_token != CONSUMED_FIRST_LIVE_AUTH_TOKEN


def test_p6_p7_p1_request_hashes_unchanged_after_identity_correction():
    identities = build_request_identities()
    for case_id, expected in FROZEN_REQUEST_HASHES.items():
        assert identities[case_id]["request_body_sha256"] == expected


def test_prepare_successor_does_not_authorize_execution(experiment_root: Path):
    _seed_v2_terminal_predecessor(experiment_root)
    result = prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    assert result["execution_authorized"] is False
    assert result["owner_authorization_required"] is True
    assert result["lifecycle"] == LIFECYCLE_SUCCESSOR_PROPOSED
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    successor = registry.get_successor()
    assert successor is not None
    assert successor.get("owner_authorization_token") is None


def test_leftover_execute_flag_cannot_authorize_successor(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.delenv(OWNER_AUTHORIZATION_ENV_VAR, raising=False)
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    with pytest.raises(ExecutionAuthorizationError, match="owner authorization missing"):
        issue_execution_authorization(registry=registry)


def test_owner_authorization_without_execute_flag_blocked(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    monkeypatch.delenv(EXECUTE_ENV_VAR, raising=False)
    monkeypatch.setenv(OWNER_AUTHORIZATION_ENV_VAR, _owner_token())
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    with pytest.raises(ExecutionAuthorizationError, match="provider execution disabled"):
        issue_execution_authorization(registry=registry)


def test_identity_bound_owner_authorization_enables_execution_gate(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.setenv(OWNER_AUTHORIZATION_ENV_VAR, _owner_token())
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    auth = issue_execution_authorization(registry=registry)
    assert auth.owner_authorization_token == _owner_token()
    run_dir, ledger, created = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        owner_authorization_token=auth.owner_authorization_token,
        run_id="successor-run",
    )
    assert created is True
    assert run_dir.exists()
    assert ledger.payload["records"] == []


def test_registry_deletion_with_terminal_artifacts_fails_closed(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = experiment_root / "runs" / "terminal-artifact-run"
    run_dir.mkdir(parents=True)
    artifact = {
        "run_id": "terminal-artifact-run",
        "overall_disposition": "INVALID",
        "identity_hashes": {"frozen_experiment_identity_hash": CONSUMED_FIRST_LIVE_IDENTITY_HASH},
    }
    (run_dir / "run_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    with pytest.raises(AttemptGovernanceError, match="registry_history_missing"):
        registry.ensure_loaded_fail_closed_on_missing_history()


def test_archived_predecessor_remains_in_history_after_successor_prepare(experiment_root: Path):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    payload = json.loads(_registry_path(experiment_root).read_text(encoding="utf-8"))
    assert len(payload["history"]) == 1
    assert payload["history"][0]["lifecycle"] == LIFECYCLE_TERMINAL
    assert payload["history"][0]["run_id"] == "semantic_analyzer_run_ed35656b9a7c"
    assert payload["successor"]["lifecycle"] == LIFECYCLE_SUCCESSOR_PROPOSED


def test_completed_successor_cannot_reuse_owner_authorization(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    owner_token = _owner_token()
    monkeypatch.setenv(OWNER_AUTHORIZATION_ENV_VAR, owner_token)
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    auth = issue_execution_authorization(registry=registry)
    run_dir, ledger, _ = registry.establish_or_resume(
        expected_identity_hash=frozen_experiment_identity_hash(),
        authorization_token=auth.authorization_token,
        owner_authorization_token=auth.owner_authorization_token,
        run_id="successor-complete",
    )
    ledger.consume_attempt_before_network("P6")
    ledger.finalize_attempt("P6", disposition="INVALID")
    registry.mark_terminal(ledger=ledger, disposition="INVALID")
    with pytest.raises(AttemptGovernanceError, match="experiment_terminal:INVALID"):
        registry.establish_or_resume(
            expected_identity_hash=frozen_experiment_identity_hash(),
            authorization_token=auth.authorization_token,
            owner_authorization_token=owner_token,
            run_id="third-experiment",
        )


def test_changed_runner_sha_invalidates_owner_authorization(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    stale_token = compute_owner_authorization_token(
        experiment_identity_hash=frozen_experiment_identity_hash(),
        runner_implementation_sha="0" * 40,
    )
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    with pytest.raises(ExecutionAuthorizationError, match="owner_authorization_identity_mismatch"):
        registry.validate_and_persist_owner_authorization(
            stale_token,
            frozen_experiment_identity_hash(),
        )


def test_find_persisted_terminal_experiments_reconstructs_predecessor(experiment_root: Path):
    run_dir = experiment_root / "runs" / "reconstruct-run"
    run_dir.mkdir(parents=True)
    artifact = {
        "run_id": "reconstruct-run",
        "overall_disposition": "INVALID",
        "identity_hashes": {"frozen_experiment_identity_hash": CONSUMED_FIRST_LIVE_IDENTITY_HASH},
    }
    (run_dir / "run_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    found = find_persisted_terminal_experiments(
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    assert len(found) == 1
    assert found[0]["run_id"] == "reconstruct-run"


def test_empty_v2_registry_with_disk_predecessor_blocks(experiment_root: Path):
    _write_empty_v2_registry(experiment_root)
    _seed_disk_governed_evidence(experiment_root)
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    with pytest.raises(AttemptGovernanceError, match="registry_history_incomplete"):
        registry.ensure_loaded_fail_closed_on_missing_history()


def test_empty_v2_registry_with_differing_identity_ledger_blocks(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_empty_v2_registry(experiment_root)
    _seed_disk_governed_evidence(
        experiment_root,
        identity_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        run_id="other-identity-run",
    )
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    with pytest.raises(AttemptGovernanceError, match="registry_history_incomplete"):
        issue_execution_authorization(registry=registry)


def test_v2_history_omitting_disk_predecessor_blocks(experiment_root: Path):
    _seed_disk_governed_evidence(experiment_root)
    _write_empty_v2_registry(experiment_root)
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    evidence = find_governed_experiment_evidence(
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    assert len(evidence) == 1
    with pytest.raises(AttemptGovernanceError, match="registry_history_incomplete"):
        registry.prepare_successor(expected_identity_hash=frozen_experiment_identity_hash())


def test_mismatched_predecessor_linkage_identity_drift_blocks(experiment_root: Path):
    _seed_disk_governed_evidence(experiment_root)
    registry_path = _registry_path(experiment_root)
    registry_path.write_text(
        canonical_json_dumps(
            {
                "registry_version": REGISTRY_VERSION,
                "history": [
                    {
                        "run_id": "semantic_analyzer_run_ed35656b9a7c",
                        "frozen_experiment_identity_hash": "0" * 64,
                        "run_dir": "runs/semantic_analyzer_run_ed35656b9a7c",
                        "lifecycle": LIFECYCLE_TERMINAL,
                        "terminal_disposition": "INVALID",
                    }
                ],
                "successor": None,
            }
        ),
        encoding="utf-8",
    )
    registry = FrozenExperimentRegistry(registry_path, experiment_root=experiment_root)
    with pytest.raises(AttemptGovernanceError, match="registry_history_linkage_mismatch"):
        registry.ensure_loaded_fail_closed_on_missing_history()


def test_authoritative_predecessor_history_allows_prepare_still_requires_owner_auth(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.delenv(OWNER_AUTHORIZATION_ENV_VAR, raising=False)
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    with pytest.raises(ExecutionAuthorizationError, match="owner authorization missing"):
        issue_execution_authorization(registry=registry)


def test_leftover_execute_flag_cannot_bypass_empty_history_attack(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_empty_v2_registry(experiment_root)
    _seed_disk_governed_evidence(experiment_root)
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.delenv(OWNER_AUTHORIZATION_ENV_VAR, raising=False)
    registry = FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
    )
    with pytest.raises(AttemptGovernanceError, match="registry_history_incomplete"):
        issue_execution_authorization(registry=registry)


def _load_registry(experiment_root: Path) -> FrozenExperimentRegistry:
    return FrozenExperimentRegistry(
        _registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )


def test_grok_successor_run_id_alias_attack_blocked(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_id = "semantic_analyzer_run_ed35656b9a7c"
    run_dir = _seed_disk_governed_evidence(experiment_root, run_id=run_id)
    historical_artifact = (run_dir / "run_artifact.json").read_text(encoding="utf-8")
    historical_ledger = (run_dir / DurableAttemptLedger.LEDGER_FILENAME).read_text(
        encoding="utf-8"
    )
    successor_identity = frozen_experiment_identity_hash()
    assert successor_identity != CONSUMED_FIRST_LIVE_IDENTITY_HASH
    _write_v2_registry(
        experiment_root,
        history=[],
        successor=_successor_proposal(identity_hash=successor_identity, run_id=run_id),
    )
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.delenv(OWNER_AUTHORIZATION_ENV_VAR, raising=False)
    posts: list[tuple] = []

    def _capture_post(*args, **kwargs):
        posts.append((args, kwargs))
        return _mock_response(content="{}")

    with pytest.raises(AttemptGovernanceError, match="registry_history_role_mismatch"):
        execute_provider_qualification_run(http_post=_capture_post)
    assert posts == []

    registry = _load_registry(experiment_root)
    with pytest.raises(AttemptGovernanceError, match="registry_history_role_mismatch"):
        issue_execution_authorization(registry=registry)
    with pytest.raises(AttemptGovernanceError, match="registry_history_role_mismatch"):
        registry.establish_or_resume(
            expected_identity_hash=successor_identity,
            authorization_token="unauthorized",
            run_id="alias-attack-run",
        )

    payload = json.loads(_registry_path(experiment_root).read_text(encoding="utf-8"))
    assert payload["history"] == []
    assert payload["successor"]["lifecycle"] == LIFECYCLE_SUCCESSOR_PROPOSED
    assert payload["successor"]["lifecycle"] != LIFECYCLE_ACTIVE
    assert payload["successor"].get("owner_authorization_token") is None
    assert payload["successor"]["run_id"] == run_id
    assert (run_dir / "run_artifact.json").read_text(encoding="utf-8") == historical_artifact
    assert (run_dir / DurableAttemptLedger.LEDGER_FILENAME).read_text(
        encoding="utf-8"
    ) == historical_ledger


def test_terminal_disk_run_represented_only_as_successor_blocks(experiment_root: Path):
    _seed_disk_governed_evidence(experiment_root)
    _write_v2_registry(
        experiment_root,
        history=[],
        successor=_successor_proposal(run_id="semantic_analyzer_run_ed35656b9a7c"),
    )
    with pytest.raises(AttemptGovernanceError, match="registry_history_role_mismatch"):
        _load_registry(experiment_root).ensure_loaded_fail_closed_on_missing_history()


def test_terminal_disk_run_in_history_with_wrong_identity_blocks(experiment_root: Path):
    _seed_disk_governed_evidence(experiment_root)
    _write_v2_registry(
        experiment_root,
        history=[_terminal_history_row(identity_hash="0" * 64)],
        successor=None,
    )
    with pytest.raises(AttemptGovernanceError, match="registry_history_linkage_mismatch"):
        _load_registry(experiment_root).ensure_loaded_fail_closed_on_missing_history()


def test_terminal_disk_run_in_history_with_non_terminal_lifecycle_blocks(
    experiment_root: Path,
):
    _seed_disk_governed_evidence(experiment_root)
    _write_v2_registry(
        experiment_root,
        history=[_terminal_history_row(lifecycle=LIFECYCLE_ACTIVE)],
        successor=None,
    )
    with pytest.raises(AttemptGovernanceError, match="registry_history_role_mismatch"):
        _load_registry(experiment_root).ensure_loaded_fail_closed_on_missing_history()


def test_terminal_disk_run_in_history_and_successor_blocks(experiment_root: Path):
    _seed_v2_terminal_predecessor(experiment_root)
    history = json.loads(_registry_path(experiment_root).read_text(encoding="utf-8"))[
        "history"
    ]
    _write_v2_registry(
        experiment_root,
        history=history,
        successor=_successor_proposal(run_id="semantic_analyzer_run_ed35656b9a7c"),
    )
    with pytest.raises(
        AttemptGovernanceError,
        match="successor_run_id_collides_with_terminal_history",
    ):
        _load_registry(experiment_root).ensure_loaded_fail_closed_on_missing_history()


def test_successor_run_id_collision_with_terminal_history_blocks(experiment_root: Path):
    _seed_v2_terminal_predecessor(experiment_root)
    history = json.loads(_registry_path(experiment_root).read_text(encoding="utf-8"))[
        "history"
    ]
    _write_v2_registry(
        experiment_root,
        history=history,
        successor=_successor_proposal(run_id="semantic_analyzer_run_ed35656b9a7c"),
    )
    with pytest.raises(
        AttemptGovernanceError,
        match="successor_run_id_collides_with_terminal_history",
    ):
        _load_registry(experiment_root).prepare_successor(
            expected_identity_hash=frozen_experiment_identity_hash()
        )


def test_valid_terminal_history_allows_prepare_but_keeps_successor_unauthorized(
    experiment_root: Path,
):
    _seed_v2_terminal_predecessor(experiment_root)
    result = prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    assert result["execution_authorized"] is False
    assert result["owner_authorization_required"] is True
    payload = json.loads(_registry_path(experiment_root).read_text(encoding="utf-8"))
    assert len(payload["history"]) == 1
    assert payload["history"][0]["lifecycle"] == LIFECYCLE_TERMINAL
    assert payload["history"][0]["run_id"] == "semantic_analyzer_run_ed35656b9a7c"
    assert payload["successor"]["lifecycle"] == LIFECYCLE_SUCCESSOR_PROPOSED
    assert payload["successor"]["run_id"] is None
    assert (
        payload["successor"]["frozen_experiment_identity_hash"]
        != CONSUMED_FIRST_LIVE_IDENTITY_HASH
    )
    assert payload["successor"].get("owner_authorization_token") is None


def test_leftover_execute_flag_with_valid_successor_still_requires_owner_token(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.delenv(OWNER_AUTHORIZATION_ENV_VAR, raising=False)
    with pytest.raises(ExecutionAuthorizationError, match="owner authorization missing"):
        issue_execution_authorization(registry=_load_registry(experiment_root))


def test_identity_bound_owner_token_passes_offline_authorization_gate(
    experiment_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _seed_v2_terminal_predecessor(experiment_root)
    prepare_successor_experiment(
        registry_path=_registry_path(experiment_root),
        experiment_root=experiment_root,
        output_root=experiment_root / "runs",
    )
    monkeypatch.setenv(EXECUTE_ENV_VAR, "1")
    monkeypatch.setenv(OWNER_AUTHORIZATION_ENV_VAR, _owner_token())
    auth = issue_execution_authorization(registry=_load_registry(experiment_root))
    assert auth.owner_authorization_token == _owner_token()
    assert auth.provider_execution_authorized is True
    payload = json.loads(_registry_path(experiment_root).read_text(encoding="utf-8"))
    assert payload["successor"]["lifecycle"] == LIFECYCLE_AUTHORIZED
    assert payload["successor"]["lifecycle"] != LIFECYCLE_ACTIVE
