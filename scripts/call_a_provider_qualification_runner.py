"""Real Call-A provider qualification runner. No provider calls by default."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import httpx
import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.call_a_preprovider_harness import (  # noqa: E402
    CASE_ORDER,
    EXPECTED_FIXTURE_SHA256,
    MAX_ATTEMPTS_PER_CASE,
    MAX_PROVIDER_REQUESTS,
    QUALIFIED_HARNESS_HEAD,
    apply_overall_gate,
    build_all_case_requests,
    build_gold_mock_response,
    build_request_identities,
    call_inventory_code_sha256,
    canonical_json_dumps,
    compute_aggregate_metrics,
    evaluate_parsed_case,
    fixture_population_report,
    get_implementation_git_sha,
    load_fixture_pack,
    prompt_sha256,
    response_contract_sha256,
    sha256_text,
    validate_fixture_structure,
    validate_harness_identities,
)

RUNNER_ID = "call_a_provider_qualification_runner"
STAGE = "call-a-provider-qualification"
EXPERIMENT_ROOT = REPO_ROOT / "outputs" / "call_a_provider_qualification"
OUTPUT_ROOT = EXPERIMENT_ROOT / "runs"
PRICE_SCHEDULE_PATH = REPO_ROOT / "evals" / "fixtures" / "call_a_provider_qualification_price_schedule.json"

EXECUTE_ENV_VAR = "CALL_A_PROVIDER_EXECUTE"
OWNER_AUTHORIZATION_ENV_VAR = "CALL_A_OWNER_AUTHORIZATION"
OWNER_AUTHORIZATION_VERSION = "call-a-owner-v2"

REQUESTED_MODEL = "deepseek-flash"
ACCEPTED_RETURNED_MODEL = "deepseek-flash"

TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 4000
PROVIDER_RETRIES_DISABLED = True
VALID_FINISH_REASONS = frozenset({"stop"})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})

ARTIFACT_DIGEST_FIELDS = (
    "runner_id",
    "stage",
    "run_id",
    "provider_calls",
    "identity_hashes",
    "authorization",
    "pricing_snapshot",
    "request_identities",
    "attempt_ledger",
    "case_order",
    "case_results",
    "aggregate_metrics",
    "critical_failures",
    "overall_gate",
    "pass_count",
    "fail_count",
    "invalid_count",
    "not_run_count",
    "overall_disposition",
    "provider_execution_authorized",
)


class CallAProviderRunnerError(ValueError):
    """Call-A provider qualification runner error."""


class ExecutionAuthorizationError(CallAProviderRunnerError):
    """Provider execution authorization missing or invalid."""


class AttemptGovernanceError(CallAProviderRunnerError):
    """Attempt ledger rejected an unauthorized or duplicate attempt."""


class ProviderResponseValidationError(CallAProviderRunnerError):
    """Provider response failed frozen validation."""


@dataclass(frozen=True)
class ExecutionAuthorization:
    authorization_token: str
    approved_execution_config_hash: str
    execution_git_sha: str
    budget_authorized: bool
    owner_max_spend_usd: float
    provider_execution_authorized: bool


@dataclass
class ProviderRunState:
    run_id: str
    run_dir: Path
    provider_calls: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    frozen_returned_model_identity: str | None = None
    total_cost_usd: float = 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CallAProviderRunnerError(message)


def provider_execution_authorized() -> bool:
    return os.getenv(EXECUTE_ENV_VAR, "").strip() == "1"


def _validate_production_model_config() -> None:
    from config import DEEPSEEK_MODEL

    if DEEPSEEK_MODEL != REQUESTED_MODEL:
        raise CallAProviderRunnerError(
            f"DEEPSEEK_MODEL must be {REQUESTED_MODEL!r} for Call-A qualification; got {DEEPSEEK_MODEL!r}"
        )
    if REQUESTED_MODEL != ACCEPTED_RETURNED_MODEL:
        raise CallAProviderRunnerError("requested and accepted returned model must match exactly")


def _production_model_config() -> dict[str, Any]:
    from config import DEEPSEEK_BASE_URL, LLM_TIMEOUT_S

    _validate_production_model_config()
    return {
        "provider": "deepseek",
        "requested_model": REQUESTED_MODEL,
        "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": None,
        "timeout_s": LLM_TIMEOUT_S,
        "base_url": DEEPSEEK_BASE_URL,
        "production_transport_retries": 3,
        "qualification_transport_retries": 0,
        "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
    }


def load_price_schedule(path: Path | str = PRICE_SCHEDULE_PATH) -> dict[str, Any]:
    schedule = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(schedule["max_output_tokens_per_case"] == MAX_OUTPUT_TOKENS, "price schedule max tokens drift")
    _require(schedule["max_cases"] == MAX_PROVIDER_REQUESTS, "price schedule max cases drift")
    return schedule


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def conservative_case_cost_usd(request: dict[str, Any], schedule: dict[str, Any]) -> float:
    body = request["request_body"]
    prompt_text = "\n".join(message["content"] for message in body["messages"])
    input_tokens = count_tokens(prompt_text)
    return round(
        input_tokens * schedule["input_cost_per_million_tokens_usd"] / 1_000_000
        + schedule["max_output_tokens_per_case"] * schedule["output_cost_per_million_tokens_usd"] / 1_000_000,
        6,
    )


def conservative_total_cost_bound(schedule: dict[str, Any] | None = None) -> dict[str, Any]:
    schedule = schedule or load_price_schedule()
    pack = load_fixture_pack()
    requests = build_all_case_requests(pack)
    per_case = {req["case_id"]: conservative_case_cost_usd(req, schedule) for req in requests}
    total = round(sum(per_case.values()), 6)
    typical_output_tokens = schedule.get("typical_output_tokens_per_case", 800)
    typical_total = round(
        sum(
            count_tokens("\n".join(m["content"] for m in req["request_body"]["messages"]))
            * schedule["input_cost_per_million_tokens_usd"] / 1_000_000
            + typical_output_tokens * schedule["output_cost_per_million_tokens_usd"] / 1_000_000
            for req in requests
        ),
        6,
    )
    return {
        "per_case_usd": per_case,
        "typical_total_usd": typical_total,
        "conservative_max_total_usd": total,
        "recommended_owner_ceiling_usd": schedule["recommended_owner_ceiling_usd"],
        "hard_spend_ceiling_usd": schedule["hard_spend_ceiling_usd"],
        "budget_authorized_default_ceiling": total <= schedule["recommended_owner_ceiling_usd"],
        "pricing_snapshot": schedule,
    }


def calculate_observed_cost_usd(
    usage: dict[str, Any] | None,
    schedule: dict[str, Any],
) -> tuple[float | None, str | None]:
    if not isinstance(usage, dict):
        return None, "missing_usage"
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
        return None, "invalid_prompt_tokens"
    if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool):
        return None, "invalid_completion_tokens"
    cost = round(
        prompt_tokens * schedule["input_cost_per_million_tokens_usd"] / 1_000_000
        + completion_tokens * schedule["output_cost_per_million_tokens_usd"] / 1_000_000,
        6,
    )
    return cost, None


def build_approved_execution_config_hash(pack: dict[str, Any] | None = None) -> str:
    _validate_production_model_config()
    payload = {
        "requested_model": REQUESTED_MODEL,
        "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": None,
        "prompt_sha256": prompt_sha256(),
        "response_contract_sha256": response_contract_sha256(),
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "request_identities": build_request_identities(pack),
        "case_order": list(CASE_ORDER),
    }
    return sha256_text(canonical_json_dumps(payload))


def build_frozen_experiment_identity(*, runner_implementation_sha: str | None = None) -> dict[str, Any]:
    _validate_production_model_config()
    return {
        "runner_id": RUNNER_ID,
        "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
        "authorization_version": OWNER_AUTHORIZATION_VERSION,
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "prompt_sha256": prompt_sha256(),
        "response_contract_sha256": response_contract_sha256(),
        "claim_inventory_code_sha256": call_inventory_code_sha256(),
        "request_identities": build_request_identities(),
        "requested_model": REQUESTED_MODEL,
        "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
        "approved_execution_config_hash": build_approved_execution_config_hash(),
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "case_order": list(CASE_ORDER),
        "runner_implementation_sha": runner_implementation_sha or get_implementation_git_sha(),
    }


def frozen_experiment_identity_hash(*, runner_implementation_sha: str | None = None) -> str:
    return sha256_text(canonical_json_dumps(build_frozen_experiment_identity(
        runner_implementation_sha=runner_implementation_sha,
    )))


def compute_owner_authorization_token(*, experiment_identity_hash: str) -> str:
    return sha256_text(
        canonical_json_dumps({
            "authorization_version": OWNER_AUTHORIZATION_VERSION,
            "frozen_experiment_identity_hash": experiment_identity_hash,
            "fixture_sha256": EXPECTED_FIXTURE_SHA256,
            "prompt_sha256": prompt_sha256(),
            "response_contract_sha256": response_contract_sha256(),
            "requested_model": REQUESTED_MODEL,
            "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
            "max_provider_requests": MAX_PROVIDER_REQUESTS,
            "case_order": list(CASE_ORDER),
        })
    )


def allocate_run_directory(*, run_id: str | None = None) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = run_id or f"call_a_run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_dir = OUTPUT_ROOT / run_id
    if run_dir.exists():
        raise CallAProviderRunnerError(f"run artifact destination already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class DurableAttemptLedger:
    LEDGER_FILENAME = "durable_attempt_ledger.json"

    def __init__(self, run_dir: Path, *, run_id: str, frozen_experiment_identity_hash: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.path = run_dir / self.LEDGER_FILENAME
        self.payload: dict[str, Any] = {
            "run_id": run_id,
            "frozen_experiment_identity_hash": frozen_experiment_identity_hash,
            "stopped": False,
            "stop_reason": None,
            "attempt_count": 0,
            "max_provider_requests": MAX_PROVIDER_REQUESTS,
            "case_order": list(CASE_ORDER),
            "records": [],
        }

    def _atomic_write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        encoded = canonical_json_dumps(self.payload)
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.path)

    def _validate_order(self, case_id: str) -> None:
        if self.payload.get("stopped"):
            raise AttemptGovernanceError(f"run_stopped:{self.payload.get('stop_reason')}")
        records = self.payload["records"]
        if len(records) >= MAX_PROVIDER_REQUESTS:
            raise AttemptGovernanceError("request_budget_exhausted")
        if any(row["case_id"] == case_id for row in records):
            raise AttemptGovernanceError(f"duplicate_retry:{case_id}")
        expected = CASE_ORDER[len(records)]
        if case_id != expected:
            raise AttemptGovernanceError(f"out_of_order_attempt:expected_{expected}_got_{case_id}")

    def consume_attempt_before_network(self, case_id: str) -> dict[str, Any]:
        self._validate_order(case_id)
        record = {
            "case_id": case_id,
            "attempt_index": len(self.payload["records"]) + 1,
            "state": "CONSUMED",
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        self.payload["records"].append(record)
        self.payload["attempt_count"] = len(self.payload["records"])
        self._atomic_write()
        return record

    def finalize_attempt(self, case_id: str, *, disposition: str, detail: dict[str, Any] | None = None) -> None:
        for record in self.payload["records"]:
            if record["case_id"] == case_id:
                record["state"] = "FINALIZED"
                record["disposition"] = disposition
                record["finalized_at"] = datetime.now(UTC).isoformat()
                if detail:
                    record["detail"] = detail
                break
        else:
            raise AttemptGovernanceError(f"unknown_attempt_case:{case_id}")
        if disposition == "INVALID":
            self.payload["stopped"] = True
            self.payload["stop_reason"] = "invalid"
        self._atomic_write()

    def next_case_id(self) -> str | None:
        if self.payload.get("stopped"):
            return None
        finalized = {
            row["case_id"]
            for row in self.payload["records"]
            if row.get("state") == "FINALIZED"
        }
        for case_id in CASE_ORDER:
            if case_id not in finalized:
                return case_id
        return None


def build_provider_client_config() -> dict[str, Any]:
    model = _production_model_config()
    return {
        **model,
        "endpoint": f"{model['base_url'].rstrip('/')}/chat/completions",
        "transport_retries": 0,
        "follow_redirects": False,
        "trust_env": False,
        "uses_production_llm_call": False,
    }


def build_call_a_http_client(*, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    from config import LLM_TIMEOUT_S

    return httpx.Client(
        transport=transport or httpx.HTTPTransport(retries=0),
        timeout=LLM_TIMEOUT_S,
        follow_redirects=False,
        trust_env=False,
    )


def validate_provider_http_response(
    *,
    status_code: int,
    body: dict[str, Any],
    requested_model: str = REQUESTED_MODEL,
    accepted_returned_model: str = ACCEPTED_RETURNED_MODEL,
) -> dict[str, Any]:
    invalid_reasons: list[str] = []
    if status_code in REDIRECT_STATUS_CODES:
        invalid_reasons.append("provider_redirect")
    if status_code != 200:
        invalid_reasons.append(f"http_status_{status_code}")

    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        invalid_reasons.append("choice_count_not_one")

    choice = choices[0] if isinstance(choices, list) and choices else {}
    finish_reason = choice.get("finish_reason")
    if finish_reason not in VALID_FINISH_REASONS:
        invalid_reasons.append("invalid_finish_reason")

    message = choice.get("message") if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        invalid_reasons.append("empty_assistant_content")

    observed_returned_model = body.get("model")
    if not isinstance(observed_returned_model, str) or not observed_returned_model.strip():
        invalid_reasons.append("missing_returned_model")
    elif observed_returned_model != accepted_returned_model:
        invalid_reasons.append("returned_model_mismatch")

    usage = body.get("usage")
    if not isinstance(usage, dict):
        invalid_reasons.append("missing_usage")

    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "finish_reason": finish_reason,
        "requested_model": requested_model,
        "accepted_returned_model": accepted_returned_model,
        "observed_returned_model": observed_returned_model,
        "provider_response_id": body.get("id"),
        "usage": usage,
        "raw_content": content,
    }


def check_returned_model_drift(run_state: ProviderRunState, observed_model: str | None) -> str | None:
    if not isinstance(observed_model, str) or not observed_model.strip():
        return None
    if run_state.frozen_returned_model_identity is None:
        run_state.frozen_returned_model_identity = observed_model
        return None
    if observed_model != run_state.frozen_returned_model_identity:
        return f"returned_model_drift:{observed_model}!={run_state.frozen_returned_model_identity}"
    return None


def issue_execution_authorization(*, owner_max_spend_usd: float | None = None) -> ExecutionAuthorization:
    if not provider_execution_authorized():
        raise ExecutionAuthorizationError(
            f"provider execution disabled; set {EXECUTE_ENV_VAR}=1 after owner authorization"
        )
    owner_token = os.getenv(OWNER_AUTHORIZATION_ENV_VAR, "").strip()
    if not owner_token:
        raise ExecutionAuthorizationError(
            f"owner authorization missing; set {OWNER_AUTHORIZATION_ENV_VAR} to identity-bound token"
        )
    budget = conservative_total_cost_bound()
    schedule = budget["pricing_snapshot"]
    max_spend = owner_max_spend_usd or schedule["recommended_owner_ceiling_usd"]
    if budget["conservative_max_total_usd"] > max_spend:
        raise ExecutionAuthorizationError("conservative cost bound exceeds owner authorization ceiling")
    identity = validate_harness_identities()
    if not identity["valid"]:
        raise ExecutionAuthorizationError(
            "harness identity invalid: " + ", ".join(identity["invalid_reasons"])
        )
    git_sha = get_implementation_git_sha()
    experiment_hash = frozen_experiment_identity_hash(runner_implementation_sha=git_sha)
    expected_token = compute_owner_authorization_token(experiment_identity_hash=experiment_hash)
    if owner_token != expected_token:
        raise ExecutionAuthorizationError("owner authorization token mismatch")

    approved_hash = build_approved_execution_config_hash()
    token_payload = {
        "approved_execution_config_hash": approved_hash,
        "execution_git_sha": git_sha,
        "frozen_experiment_identity_hash": experiment_hash,
        "owner_max_spend_usd": max_spend,
        "provider_execution_authorized": True,
    }
    return ExecutionAuthorization(
        authorization_token=sha256_text(canonical_json_dumps(token_payload)),
        approved_execution_config_hash=approved_hash,
        execution_git_sha=git_sha,
        budget_authorized=True,
        owner_max_spend_usd=max_spend,
        provider_execution_authorized=True,
    )


def run_provider_preflight() -> dict[str, Any]:
    _validate_production_model_config()
    pack = load_fixture_pack()
    identity = validate_harness_identities()
    fixture_validation = validate_fixture_structure(pack)
    budget = conservative_total_cost_bound()
    model = _production_model_config()
    return {
        "runner_id": RUNNER_ID,
        "stage": STAGE,
        "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
        "identity_validation": identity,
        "fixture_validation": fixture_validation,
        "fixture_population": fixture_population_report(pack),
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "prompt_sha256": prompt_sha256(),
        "response_contract_sha256": response_contract_sha256(),
        "claim_inventory_code_sha256": call_inventory_code_sha256(),
        "request_identities": build_request_identities(pack),
        "conservative_cost_bound": budget,
        "provider_model_config": model,
        "provider_model_identity": {
            "requested_model": REQUESTED_MODEL,
            "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
            "exact_match_required": True,
        },
        "superseded_experiments": [
            {
                "run_id": "call_a_run_20260915T160540Z_4135c531",
                "artifact_digest": "36807aa205b3f1bd45fdf891330df22037a1416ce98279dde8436e07fcc17121",
                "disposition": "INVALID",
                "reason": "returned_model_mismatch:requested_deepseek-chat",
                "authorization_version": "call-a-owner-v1",
            }
        ],
        "owner_authorization_version": OWNER_AUTHORIZATION_VERSION,
        "provider_execution_default_disabled": not provider_execution_authorized(),
        "owner_authorization_env_var": OWNER_AUTHORIZATION_ENV_VAR,
        "execution_ready": identity["valid"] and fixture_validation["valid"],
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "max_attempts_per_case": MAX_ATTEMPTS_PER_CASE,
        "case_order": list(CASE_ORDER),
        "worst_case_production_retry_requests": MAX_PROVIDER_REQUESTS * model["production_transport_retries"],
    }


def execute_mock_qualification_run(
    *,
    response_provider: Callable[[dict[str, Any]], str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Offline mocked provider qualification — ZERO network/provider calls."""
    pack = load_fixture_pack()
    preflight = run_provider_preflight()
    if not preflight["execution_ready"]:
        raise CallAProviderRunnerError("preflight not ready")

    run_dir = allocate_run_directory(run_id=run_id)
    experiment_hash = frozen_experiment_identity_hash()
    ledger = DurableAttemptLedger(run_dir, run_id=run_dir.name, frozen_experiment_identity_hash=experiment_hash)
    requests = {req["case_id"]: req for req in build_all_case_requests(pack)}
    cases_by_id = {case["case_id"]: case for case in pack["cases"]}
    provider = response_provider or (lambda case: build_gold_mock_response(case))

    case_results: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        case = cases_by_id[case_id]
        ledger.consume_attempt_before_network(case_id)
        raw = provider(case)
        evaluated = evaluate_parsed_case(case=case, raw_response=raw)
        evaluated["provider_mode"] = "mock"
        evaluated["request_body_sha256"] = requests[case_id]["request_body_sha256"]
        ledger.finalize_attempt(case_id, disposition=evaluated["disposition"])
        case_results.append(evaluated)

    return _finalize_run_artifact(
        run_dir=run_dir,
        ledger=ledger,
        case_results=case_results,
        provider_calls=0,
        total_cost_usd=0.0,
        provider_execution_authorized=False,
        authorization=None,
        identity_drift=False,
        provider_invalid_count=sum(1 for r in case_results if r.get("disposition") == "INVALID"),
    )


def execute_case_once(
    *,
    case: dict[str, Any],
    request: dict[str, Any],
    run_state: ProviderRunState,
    ledger: DurableAttemptLedger,
    execution_auth: ExecutionAuthorization,
    http_post: Callable[..., httpx.Response] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    case_id = case["case_id"]
    _validate_production_model_config()
    started = time.time()
    artifact: dict[str, Any] = {
        "case_id": case_id,
        "requested_model": REQUESTED_MODEL,
        "request_body_sha256": request["request_body_sha256"],
        "disposition": "INVALID",
        "provider_mode": "live",
    }
    schedule = load_price_schedule()
    try:
        if run_state.total_cost_usd >= execution_auth.owner_max_spend_usd:
            raise ExecutionAuthorizationError("owner_spend_ceiling_exceeded_before_request")
        ledger.consume_attempt_before_network(case_id)
        active_client = client or build_call_a_http_client()
        post = http_post or active_client.post
        config = build_provider_client_config()
        response = post(
            config["endpoint"],
            headers={
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            json=request["request_body"],
        )
        run_state.provider_calls += 1
        body = response.json()
        validation = validate_provider_http_response(
            status_code=response.status_code,
            body=body,
        )
        drift = check_returned_model_drift(run_state, validation.get("observed_returned_model"))
        if drift:
            validation = {
                **validation,
                "invalid_reasons": [*validation["invalid_reasons"], drift],
                "valid": False,
            }
        observed_cost, cost_unavailable = calculate_observed_cost_usd(validation.get("usage"), schedule)
        if observed_cost is not None:
            run_state.total_cost_usd = round(run_state.total_cost_usd + observed_cost, 6)
            if run_state.total_cost_usd > execution_auth.owner_max_spend_usd:
                raise ExecutionAuthorizationError("owner_spend_ceiling_exceeded_after_request")
        if not validation["valid"]:
            raise ProviderResponseValidationError(",".join(validation["invalid_reasons"]))
        evaluated = evaluate_parsed_case(case=case, raw_response=validation["raw_content"] or "")
        evaluated.update({
            "provider_mode": "live",
            "request_body_sha256": request["request_body_sha256"],
            "provider_response_validation": validation,
            "provider_response_id": validation.get("provider_response_id"),
            "returned_model": validation.get("observed_returned_model"),
            "usage": validation.get("usage"),
            "actual_cost_usd": observed_cost,
            "observed_cost_unavailable": cost_unavailable,
            "latency_ms": round((time.time() - started) * 1000, 3),
        })
        ledger.finalize_attempt(case_id, disposition=evaluated["disposition"])
        if evaluated["disposition"] == "INVALID":
            run_state.stopped = True
            run_state.stop_reason = "invalid"
        return evaluated
    except Exception as exc:
        artifact.update({
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "latency_ms": round((time.time() - started) * 1000, 3),
        })
        ledger.finalize_attempt(case_id, disposition="INVALID", detail={"error": str(exc)})
        run_state.stopped = True
        run_state.stop_reason = "invalid"
        return artifact


def execute_provider_qualification_run(
    *,
    run_id: str | None = None,
    http_post: Callable[..., httpx.Response] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    if not provider_execution_authorized():
        raise ExecutionAuthorizationError(
            f"provider execution disabled; set {EXECUTE_ENV_VAR}=1 after owner authorization"
        )
    execution_auth = issue_execution_authorization()
    preflight = run_provider_preflight()
    if not preflight["execution_ready"]:
        raise CallAProviderRunnerError("preflight not ready for execution")

    pack = load_fixture_pack()
    run_dir = allocate_run_directory(run_id=run_id)
    experiment_hash = frozen_experiment_identity_hash(runner_implementation_sha=execution_auth.execution_git_sha)
    ledger = DurableAttemptLedger(run_dir, run_id=run_dir.name, frozen_experiment_identity_hash=experiment_hash)
    run_state = ProviderRunState(run_id=run_dir.name, run_dir=run_dir)
    requests = {req["case_id"]: req for req in build_all_case_requests(pack)}
    cases_by_id = {case["case_id"]: case for case in pack["cases"]}
    case_results: list[dict[str, Any]] = []
    identity_drift = False

    while True:
        next_case = ledger.next_case_id()
        if next_case is None or run_state.stopped:
            break
        result = execute_case_once(
            case=cases_by_id[next_case],
            request=requests[next_case],
            run_state=run_state,
            ledger=ledger,
            execution_auth=execution_auth,
            http_post=http_post,
            client=client,
        )
        if result.get("provider_response_validation", {}).get("invalid_reasons"):
            for reason in result["provider_response_validation"]["invalid_reasons"]:
                if "returned_model" in reason:
                    identity_drift = True
        case_results.append(result)

    return _finalize_run_artifact(
        run_dir=run_dir,
        ledger=ledger,
        case_results=case_results,
        provider_calls=run_state.provider_calls,
        total_cost_usd=run_state.total_cost_usd,
        provider_execution_authorized=True,
        authorization=execution_auth,
        identity_drift=identity_drift,
        provider_invalid_count=sum(1 for r in case_results if r.get("disposition") == "INVALID"),
    )


def _finalize_run_artifact(
    *,
    run_dir: Path,
    ledger: DurableAttemptLedger,
    case_results: list[dict[str, Any]],
    provider_calls: int,
    total_cost_usd: float,
    provider_execution_authorized: bool,
    authorization: ExecutionAuthorization | None,
    identity_drift: bool,
    provider_invalid_count: int,
) -> dict[str, Any]:
    evaluated_results = [r for r in case_results if r.get("inventory") or r.get("comparisons") is not None]
    aggregate_metrics = compute_aggregate_metrics(evaluated_results)
    overall_gate = apply_overall_gate(
        case_results=evaluated_results,
        provider_invalid_count=provider_invalid_count,
        identity_drift=identity_drift,
    )
    pass_count = sum(1 for r in case_results if r.get("disposition") == "PASS")
    fail_count = sum(1 for r in case_results if r.get("disposition") == "FAIL")
    invalid_count = sum(1 for r in case_results if r.get("disposition") == "INVALID")
    not_run_count = MAX_PROVIDER_REQUESTS - len(case_results)

    artifact: dict[str, Any] = {
        "runner_id": RUNNER_ID,
        "stage": STAGE,
        "run_id": run_dir.name,
        "provider_calls": provider_calls,
        "identity_hashes": {
            "runner_implementation_sha": get_implementation_git_sha(),
            "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
            "fixture_sha256": EXPECTED_FIXTURE_SHA256,
            "prompt_sha256": prompt_sha256(),
            "response_contract_sha256": response_contract_sha256(),
            "claim_inventory_code_sha256": call_inventory_code_sha256(),
            "frozen_experiment_identity_hash": frozen_experiment_identity_hash(),
        },
        "authorization": None if authorization is None else {
            "authorization_token": authorization.authorization_token,
            "approved_execution_config_hash": authorization.approved_execution_config_hash,
            "execution_git_sha": authorization.execution_git_sha,
            "owner_max_spend_usd": authorization.owner_max_spend_usd,
        },
        "pricing_snapshot": load_price_schedule(),
        "request_identities": build_request_identities(),
        "attempt_ledger": ledger.payload,
        "case_order": list(CASE_ORDER),
        "case_results": case_results,
        "aggregate_metrics": aggregate_metrics,
        "critical_failures": aggregate_metrics["critical_failures"],
        "overall_gate": overall_gate,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "invalid_count": invalid_count,
        "not_run_count": not_run_count,
        "overall_disposition": overall_gate["overall_disposition"],
        "total_cost_usd": total_cost_usd,
        "provider_execution_authorized": provider_execution_authorized,
    }
    artifact["artifact_digest"] = sha256_text(
        canonical_json_dumps({name: artifact[name] for name in ARTIFACT_DIGEST_FIELDS if name in artifact})
    )
    run_path = run_dir / "run_artifact.json"
    run_path.write_text(canonical_json_dumps(artifact), encoding="utf-8")
    try:
        artifact["run_bundle_path"] = str(run_path.relative_to(REPO_ROOT))
    except ValueError:
        artifact["run_bundle_path"] = str(run_path)
    return artifact


def build_artifact_digest_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return {name: artifact[name] for name in ARTIFACT_DIGEST_FIELDS if name in artifact}


def compute_run_artifact_digest(artifact: dict[str, Any]) -> str:
    return sha256_text(canonical_json_dumps(build_artifact_digest_payload(artifact)))


def verify_run_artifact_integrity(artifact: dict[str, Any]) -> dict[str, Any]:
    invalid_reasons: list[str] = []
    stored = artifact.get("artifact_digest")
    recomputed = compute_run_artifact_digest(artifact)
    if stored != recomputed:
        invalid_reasons.append("corrupt_artifact_digest")
    identity = artifact.get("identity_hashes") or {}
    if identity.get("qualified_harness_head") != QUALIFIED_HARNESS_HEAD:
        invalid_reasons.append("stale_qualified_harness_head")
    if identity.get("fixture_sha256") != EXPECTED_FIXTURE_SHA256:
        invalid_reasons.append("stale_fixture_sha256")
    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "recomputed_artifact_digest": recomputed,
    }


def main() -> int:
    preflight = run_provider_preflight()
    print(json.dumps(preflight, indent=2, sort_keys=True))
    return 0 if preflight["execution_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
