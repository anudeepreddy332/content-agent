"""Semantic-analyzer provider qualification runner. No provider calls by default."""
from __future__ import annotations

import json
import os
import subprocess
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

from scripts.semantic_analyzer_preprovider_harness import (  # noqa: E402
    ANALYZER_SCHEMA_SHA256,
    CASE_ORDER,
    EXPECTED_FIXTURE_SHA256,
    ExecutionAuthorizationError,
    REQUIRED_ENGINE_BASELINE_SHA,
    AttemptGovernanceError,
    ResponseContractError,
    build_analyzer_input,
    build_case_bundle,
    get_verified_frozen_case_truth,
    load_verified_fixture_pack,
    qualify_case_response,
    sha256_text,
    validate_harness_identities,
)

RUNNER_ID = "semantic_analyzer_provider_qualification_runner"
STAGE = "semantic-analyzer-provider-qualification"
QUALIFIED_HARNESS_HEAD = "47e7f0458e31b2eca0d24ac4f0b791edd20faa89"

PROMPT_PATH = REPO_ROOT / "prompts" / "semantic_analyzer_system.md"
PRICE_SCHEDULE_PATH = (
    REPO_ROOT / "evals" / "fixtures" / "semantic_analyzer_provider_qualification_price_schedule.json"
)
EXPERIMENT_ROOT = REPO_ROOT / "outputs" / "semantic_analyzer_provider_qualification"
OUTPUT_ROOT = EXPERIMENT_ROOT / "runs"
EXPERIMENT_REGISTRY_PATH = EXPERIMENT_ROOT / "frozen_experiment_registry.json"

PROVIDER_NAME = "deepseek"
REQUESTED_MODEL = "deepseek-v4-flash"
# Frozen exact pair from 2026-09-10 first live experiment: request deepseek-v4-flash,
# response model field observed as deepseek-flash. No broad alias acceptance.
ACCEPTED_RETURNED_MODEL = "deepseek-flash"
# Historical runtime evidence from consumed INVALID run semantic_analyzer_run_ed35656b9a7c
# (diagnostic only; not used as a validation requirement).
HISTORICAL_FIRST_LIVE_SYSTEM_FINGERPRINT = "aeb56401ca74e127821c4f9126dcb669"
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 2000
HARD_SPEND_CEILING_USD = 0.02
MAX_PROVIDER_REQUESTS = 3
MAX_ATTEMPTS_PER_CASE = 1
PROVIDER_RETRIES_DISABLED = True
STREAM = False
THINKING = {"type": "disabled"}
RESPONSE_FORMAT = {"type": "json_object"}
PROVIDER_PATH = "semantic_analyzer_provider_direct_https"
EXECUTE_ENV_VAR = "SEMANTIC_ANALYZER_PROVIDER_EXECUTE"
OWNER_AUTHORIZATION_ENV_VAR = "SEMANTIC_ANALYZER_OWNER_AUTHORIZATION"
REGISTRY_VERSION = 2
OWNER_AUTHORIZATION_VERSION = "semantic-analyzer-owner-v1"
LIFECYCLE_TERMINAL = "TERMINAL"
LIFECYCLE_SUCCESSOR_PROPOSED = "SUCCESSOR_PROPOSED"
LIFECYCLE_AUTHORIZED = "AUTHORIZED"
LIFECYCLE_ACTIVE = "ACTIVE"

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
    "pass_count",
    "fail_count",
    "invalid_count",
    "not_run_count",
    "overall_disposition",
    "provider_execution_authorized",
)


class SemanticAnalyzerProviderRunnerError(ValueError):
    """Provider qualification runner configuration or artifact error."""


class ProviderTransportError(SemanticAnalyzerProviderRunnerError):
    """Direct HTTPS transport failure."""


class ProviderResponseValidationError(SemanticAnalyzerProviderRunnerError):
    """Provider response failed frozen validation."""


@dataclass(frozen=True)
class ExecutionAuthorization:
    authorization_token: str
    approved_execution_config_hash: str
    execution_git_sha: str
    budget_authorized: bool
    provider_execution_authorized: bool
    owner_authorization_token: str | None = None


@dataclass
class ProviderRunState:
    run_id: str
    run_dir: Path
    provider_calls: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    frozen_returned_model_identity: str | None = None
    frozen_system_fingerprint: str | None = None
    total_cost_usd: float = 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticAnalyzerProviderRunnerError(message)


def provider_execution_authorized() -> bool:
    return os.getenv(EXECUTE_ENV_VAR, "").strip() == "1"


def get_implementation_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def load_analyzer_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def analyzer_prompt_sha256() -> str:
    return sha256_text(load_analyzer_system_prompt())


def load_price_schedule(path: Path | str = PRICE_SCHEDULE_PATH) -> dict[str, Any]:
    schedule = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(schedule["hard_spend_ceiling_usd"] == HARD_SPEND_CEILING_USD, "price schedule ceiling drift")
    _require(schedule["max_output_tokens_per_case"] == MAX_OUTPUT_TOKENS, "price schedule max tokens drift")
    return schedule


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def build_analyzer_user_message(analyzer_input: dict[str, Any]) -> str:
    body = {
        "request_id": analyzer_input["request_id"],
        "schema_version": analyzer_input["schema_version"],
        "draft_text": analyzer_input["draft_text"],
        "draft_sha256": analyzer_input["draft_sha256"],
        "claims": analyzer_input["claims"],
        "evidence_manifest": analyzer_input["evidence_manifest"],
    }
    return canonical_json_dumps(body)


def build_case_request(case_id: str) -> dict[str, Any]:
    verified_pack = load_verified_fixture_pack()
    bundle = build_case_bundle(verified_pack, case_id)
    analyzer_input = build_analyzer_input(bundle)
    system_prompt = load_analyzer_system_prompt()
    user_message = build_analyzer_user_message(analyzer_input)
    request_body = {
        "model": REQUESTED_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": STREAM,
        "response_format": RESPONSE_FORMAT,
        "thinking": THINKING,
    }
    return {
        "case_id": case_id,
        "request_id": bundle["request_id"],
        "analyzer_input": analyzer_input,
        "bundle": bundle,
        "frozen_truth": get_verified_frozen_case_truth(case_id),
        "messages": request_body["messages"],
        "request_body": request_body,
        "request_body_sha256": sha256_text(canonical_json_dumps(request_body)),
        "prompt_sha256": analyzer_prompt_sha256(),
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "model_config": {
            "provider": PROVIDER_NAME,
            "requested_model": REQUESTED_MODEL,
            "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": STREAM,
            "thinking": THINKING,
            "response_format": RESPONSE_FORMAT,
            "provider_retries_disabled": PROVIDER_RETRIES_DISABLED,
        },
        "exposure_identity": bundle["exposure_identity"],
    }


def build_all_case_requests() -> list[dict[str, Any]]:
    return [build_case_request(case_id) for case_id in CASE_ORDER]


def build_request_identities() -> dict[str, Any]:
    requests = build_all_case_requests()
    return {
        case["case_id"]: {
            "request_body_sha256": case["request_body_sha256"],
            "request_id": case["request_id"],
            "prompt_sha256": case["prompt_sha256"],
            "fixture_sha256": case["fixture_sha256"],
        }
        for case in requests
    }


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
    requests = build_all_case_requests()
    per_case = {req["case_id"]: conservative_case_cost_usd(req, schedule) for req in requests}
    total = round(sum(per_case.values()), 6)
    return {
        "per_case_usd": per_case,
        "total_usd": total,
        "hard_spend_ceiling_usd": schedule["hard_spend_ceiling_usd"],
        "budget_authorized": total <= schedule["hard_spend_ceiling_usd"],
        "pricing_snapshot": schedule,
    }


def calculate_observed_cost_usd(
    usage: dict[str, Any] | None,
    schedule: dict[str, Any],
) -> tuple[float | None, str | None]:
    """Return (cost_usd, unavailable_reason). Never fabricates cost."""
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


def check_system_fingerprint_drift(
    run_state: ProviderRunState,
    observed_fingerprint: str | None,
) -> str | None:
    """Freeze first observed fingerprint per run; report drift on later cases."""
    if not isinstance(observed_fingerprint, str) or not observed_fingerprint.strip():
        return None
    if run_state.frozen_system_fingerprint is None:
        run_state.frozen_system_fingerprint = observed_fingerprint
        return None
    if observed_fingerprint != run_state.frozen_system_fingerprint:
        return (
            f"system_fingerprint_drift:{observed_fingerprint}"
            f"!={run_state.frozen_system_fingerprint}"
        )
    return None


def build_approved_execution_config_hash() -> str:
    payload = {
        "requested_model": REQUESTED_MODEL,
        "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": STREAM,
        "thinking": THINKING,
        "response_format": RESPONSE_FORMAT,
        "prompt_sha256": analyzer_prompt_sha256(),
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "request_identities": build_request_identities(),
        "case_order": list(CASE_ORDER),
    }
    return sha256_text(canonical_json_dumps(payload))


def build_provider_client_config() -> dict[str, Any]:
    from config import DEEPSEEK_BASE_URL, LLM_TIMEOUT_S

    return {
        "provider_path": PROVIDER_PATH,
        "provider": PROVIDER_NAME,
        "base_url": DEEPSEEK_BASE_URL,
        "endpoint": f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
        "timeout_s": LLM_TIMEOUT_S,
        "transport_retries": 0,
        "follow_redirects": False,
        "trust_env": False,
        "stream": STREAM,
        "requested_model": REQUESTED_MODEL,
        "uses_production_llm_call": False,
    }


def build_semantic_analyzer_http_client(
    *,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    from config import LLM_TIMEOUT_S

    return httpx.Client(
        transport=transport or httpx.HTTPTransport(retries=0),
        timeout=LLM_TIMEOUT_S,
        follow_redirects=False,
        trust_env=False,
    )


def allocate_run_directory(*, run_id: str | None = None) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now(UTC).strftime("semantic_analyzer_run_%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / run_id
    if run_dir.exists():
        raise SemanticAnalyzerProviderRunnerError(f"run artifact destination already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class DurableAttemptLedger:
    """Write-ahead durable attempt ledger for live provider execution."""

    LEDGER_FILENAME = "durable_attempt_ledger.json"

    def __init__(
        self,
        run_dir: Path,
        *,
        run_id: str,
        frozen_experiment_identity_hash: str | None = None,
    ) -> None:
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
        if self.path.exists():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))

    def _atomic_write(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        encoded = canonical_json_dumps(self.payload)
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.path)
        dir_fd = os.open(self.run_dir, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _validate_order(self, case_id: str) -> None:
        if self.payload.get("stopped"):
            raise AttemptGovernanceError(f"run_stopped:{self.payload.get('stop_reason')}")
        records = self.payload["records"]
        if len(records) >= MAX_PROVIDER_REQUESTS:
            raise AttemptGovernanceError("fourth_attempt_denied")
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
            "network_permitted_at": datetime.now(UTC).isoformat(),
            "disposition": None,
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
        if disposition in {"FAIL", "INVALID"}:
            self.payload["stopped"] = True
            self.payload["stop_reason"] = disposition.lower()
        self._atomic_write()

    def restore_or_reject_retry(self, case_id: str) -> None:
        for record in self.payload["records"]:
            if record["case_id"] == case_id and record.get("state") == "CONSUMED":
                raise AttemptGovernanceError(f"restart_retry_denied:{case_id}")

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> DurableAttemptLedger:
        path = run_dir / cls.LEDGER_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        ledger = cls(run_dir, run_id=payload["run_id"])
        ledger.payload = payload
        return ledger

    def reconcile_consumed_without_finalize(self) -> bool:
        """Mark experiment INVALID when a CONSUMED attempt lacks a trustworthy finalize."""
        changed = False
        for record in self.payload["records"]:
            if record.get("state") == "CONSUMED":
                record["state"] = "FINALIZED"
                record["disposition"] = "INVALID"
                record["finalized_at"] = datetime.now(UTC).isoformat()
                record["detail"] = {"error": "restart_retry_denied:consumed_without_trustworthy_result"}
                self.payload["stopped"] = True
                self.payload["stop_reason"] = "invalid"
                changed = True
        if changed:
            self._atomic_write()
        return changed

    def next_case_id(self) -> str | None:
        if self.payload.get("stopped"):
            return None
        records = self.payload["records"]
        if len(records) >= MAX_PROVIDER_REQUESTS:
            return None
        finalized = {
            row["case_id"]
            for row in records
            if row.get("state") == "FINALIZED" and row.get("disposition") == "PASS"
        }
        for case_id in CASE_ORDER:
            if case_id not in finalized:
                return case_id
        return None

    def is_terminal(self) -> bool:
        finalized = [
            row
            for row in self.payload["records"]
            if row.get("state") == "FINALIZED"
        ]
        if self.payload.get("stopped"):
            return bool(finalized)
        if len(finalized) == MAX_PROVIDER_REQUESTS and all(
            row.get("disposition") == "PASS" for row in finalized
        ):
            return True
        return False


def build_frozen_experiment_identity(*, runner_implementation_sha: str | None = None) -> dict[str, Any]:
    return {
        "runner_id": RUNNER_ID,
        "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
        "required_engine_baseline_sha": REQUIRED_ENGINE_BASELINE_SHA,
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "schema_sha256": ANALYZER_SCHEMA_SHA256,
        "prompt_sha256": analyzer_prompt_sha256(),
        "request_identities": build_request_identities(),
        "requested_model": REQUESTED_MODEL,
        "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
        "approved_execution_config_hash": build_approved_execution_config_hash(),
        "hard_spend_ceiling_usd": HARD_SPEND_CEILING_USD,
        "case_order": list(CASE_ORDER),
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "runner_implementation_sha": runner_implementation_sha or get_implementation_git_sha(),
    }


def frozen_experiment_identity_hash(*, runner_implementation_sha: str | None = None) -> str:
    return sha256_text(canonical_json_dumps(build_frozen_experiment_identity(
        runner_implementation_sha=runner_implementation_sha,
    )))


def build_owner_authorization_binding(
    *,
    experiment_identity_hash: str,
    runner_implementation_sha: str,
) -> dict[str, Any]:
    return {
        "authorization_version": OWNER_AUTHORIZATION_VERSION,
        "frozen_experiment_identity_hash": experiment_identity_hash,
        "runner_implementation_sha": runner_implementation_sha,
        "requested_model": REQUESTED_MODEL,
        "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "schema_sha256": ANALYZER_SCHEMA_SHA256,
        "prompt_sha256": analyzer_prompt_sha256(),
        "request_identities": build_request_identities(),
        "hard_spend_ceiling_usd": HARD_SPEND_CEILING_USD,
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "case_order": list(CASE_ORDER),
    }


def compute_owner_authorization_token(
    *,
    experiment_identity_hash: str,
    runner_implementation_sha: str | None = None,
) -> str:
    runner_sha = runner_implementation_sha or get_implementation_git_sha()
    return sha256_text(
        canonical_json_dumps(
            build_owner_authorization_binding(
                experiment_identity_hash=experiment_identity_hash,
                runner_implementation_sha=runner_sha,
            )
        )
    )


def find_persisted_terminal_experiments(
    *,
    output_root: Path | None = None,
    experiment_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Discover terminal run artifacts when registry state may be missing or stale."""
    experiment_root = experiment_root or EXPERIMENT_ROOT
    output_root = output_root or (experiment_root / "runs")
    if not output_root.exists():
        return []
    terminal: list[dict[str, Any]] = []
    for artifact_path in output_root.glob("*/run_artifact.json"):
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        disposition = artifact.get("overall_disposition")
        if disposition not in {"PASS", "FAIL", "INVALID"}:
            continue
        identity = artifact.get("identity_hashes") or {}
        run_dir = artifact_path.parent
        try:
            run_dir_ref = str(run_dir.relative_to(experiment_root))
        except ValueError:
            run_dir_ref = str(run_dir)
        terminal.append(
            {
                "run_id": artifact.get("run_id", run_dir.name),
                "run_dir": run_dir_ref,
                "frozen_experiment_identity_hash": identity.get("frozen_experiment_identity_hash"),
                "terminal_disposition": disposition,
                "artifact_path": str(artifact_path),
            }
        )
    return terminal


def find_orphaned_authoritative_ledgers(
    *,
    output_root: Path | None = None,
    expected_identity_hash: str | None = None,
) -> list[Path]:
    output_root = output_root or OUTPUT_ROOT
    if not output_root.exists():
        return []
    orphaned: list[Path] = []
    for ledger_path in output_root.glob(f"*/{DurableAttemptLedger.LEDGER_FILENAME}"):
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not payload.get("records"):
            continue
        if expected_identity_hash and payload.get("frozen_experiment_identity_hash") not in {
            None,
            expected_identity_hash,
        }:
            continue
        orphaned.append(ledger_path.parent)
    return orphaned


class FrozenExperimentRegistry:
    """Append-preserving experiment history with identity-bound successor authorization."""

    def __init__(
        self,
        registry_path: Path | None = None,
        *,
        experiment_root: Path | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.registry_path = registry_path or EXPERIMENT_REGISTRY_PATH
        self.experiment_root = experiment_root or self.registry_path.parent
        self.output_root = output_root or (self.experiment_root / "runs")
        self.payload: dict[str, Any] | None = None
        if self.registry_path.exists():
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            self.payload = self._normalize_registry_payload(raw)

    @staticmethod
    def _normalize_registry_payload(raw: dict[str, Any]) -> dict[str, Any]:
        if raw.get("registry_version") == REGISTRY_VERSION:
            return raw
        if "history" in raw:
            raise AttemptGovernanceError("corrupt_registry:invalid_history_format")
        if "frozen_experiment_identity_hash" not in raw:
            raise AttemptGovernanceError("corrupt_registry:missing_identity")
        lifecycle = LIFECYCLE_TERMINAL if raw.get("terminal") else LIFECYCLE_ACTIVE
        history_entry = {
            "run_id": raw.get("run_id"),
            "frozen_experiment_identity_hash": raw["frozen_experiment_identity_hash"],
            "legacy_execution_reference_token": raw.get("authorization_token"),
            "owner_authorization_token": raw.get("authorization_token"),
            "run_dir": raw.get("run_dir"),
            "established_at": raw.get("established_at"),
            "terminal_at": raw.get("terminal_at"),
            "lifecycle": lifecycle,
            "terminal_disposition": raw.get("terminal_disposition"),
        }
        successor = None
        if lifecycle == LIFECYCLE_ACTIVE:
            successor = {
                "frozen_experiment_identity_hash": raw["frozen_experiment_identity_hash"],
                "lifecycle": LIFECYCLE_ACTIVE,
                "proposed_at": raw.get("established_at"),
                "owner_authorization_token": None,
                "run_id": raw.get("run_id"),
                "run_dir": raw.get("run_dir"),
                "execution_session_token": raw.get("authorization_token"),
            }
            history_entry = None
        payload: dict[str, Any] = {
            "registry_version": REGISTRY_VERSION,
            "history": [history_entry] if history_entry else [],
            "successor": successor,
        }
        return payload

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.registry_path.with_suffix(".tmp")
        encoded = canonical_json_dumps(payload)
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.registry_path)
        dir_fd = os.open(self.registry_path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        self.payload = payload

    @staticmethod
    def _resolve_run_dir(stored: str, experiment_root: Path) -> Path:
        path = Path(stored)
        if path.is_absolute():
            return path
        return experiment_root / stored

    def ensure_loaded_fail_closed_on_missing_history(self) -> None:
        if self.payload is not None:
            self._validate_registry_integrity()
            return
        persisted = find_persisted_terminal_experiments(
            output_root=self.output_root,
            experiment_root=self.experiment_root,
        )
        if persisted:
            raise AttemptGovernanceError("registry_history_missing:terminal_artifacts_remain")
        orphaned = find_orphaned_authoritative_ledgers(output_root=self.output_root)
        if orphaned:
            raise AttemptGovernanceError(
                "orphaned_authoritative_ledger:registry_missing_but_ledger_remains"
            )

    def _validate_registry_integrity(self) -> None:
        if self.payload is None:
            return
        if self.payload.get("registry_version") != REGISTRY_VERSION:
            raise AttemptGovernanceError("corrupt_registry:unsupported_version")
        history = self.payload.get("history")
        if not isinstance(history, list):
            raise AttemptGovernanceError("corrupt_registry:history_not_list")
        successor = self.payload.get("successor")
        if successor is not None and not isinstance(successor, dict):
            raise AttemptGovernanceError("corrupt_registry:successor_not_object")
        active_successors = 0
        if successor and successor.get("lifecycle") in {
            LIFECYCLE_SUCCESSOR_PROPOSED,
            LIFECYCLE_AUTHORIZED,
            LIFECYCLE_ACTIVE,
        }:
            active_successors = 1
        if active_successors > 1:
            raise AttemptGovernanceError("corrupt_registry:multiple_active_successors")

    def has_terminal_predecessor(self) -> bool:
        self.ensure_loaded_fail_closed_on_missing_history()
        if self.payload is None:
            return False
        return any(
            row.get("lifecycle") == LIFECYCLE_TERMINAL for row in self.payload.get("history", [])
        )

    def requires_owner_authorization(self) -> bool:
        return self.has_terminal_predecessor()

    def get_successor(self) -> dict[str, Any] | None:
        if self.payload is None:
            return None
        successor = self.payload.get("successor")
        return successor if isinstance(successor, dict) else None

    def validate_and_persist_owner_authorization(
        self,
        submitted_token: str,
        expected_identity_hash: str,
    ) -> str:
        if not submitted_token.strip():
            raise ExecutionAuthorizationError("owner authorization missing for successor experiment")
        successor = self.get_successor()
        if successor is None:
            raise ExecutionAuthorizationError("successor_not_prepared")
        if successor.get("frozen_experiment_identity_hash") != expected_identity_hash:
            raise ExecutionAuthorizationError("owner_authorization_identity_mismatch")
        expected_token = compute_owner_authorization_token(
            experiment_identity_hash=expected_identity_hash,
        )
        lifecycle = successor.get("lifecycle")
        if lifecycle == LIFECYCLE_SUCCESSOR_PROPOSED:
            if submitted_token != expected_token:
                raise ExecutionAuthorizationError("owner_authorization_identity_mismatch")
            successor["owner_authorization_token"] = submitted_token
            successor["lifecycle"] = LIFECYCLE_AUTHORIZED
            successor["authorized_at"] = datetime.now(UTC).isoformat()
            self._atomic_write(self.payload)  # type: ignore[arg-type]
            return submitted_token
        if lifecycle in {LIFECYCLE_AUTHORIZED, LIFECYCLE_ACTIVE}:
            stored = successor.get("owner_authorization_token")
            if stored != submitted_token:
                raise ExecutionAuthorizationError("owner_authorization_replay_mismatch")
            if submitted_token != expected_token:
                raise ExecutionAuthorizationError("owner_authorization_stale_identity")
            return submitted_token
        raise ExecutionAuthorizationError(f"successor_not_authorizable:{lifecycle}")

    def prepare_successor(self, *, expected_identity_hash: str) -> dict[str, Any]:
        self.ensure_loaded_fail_closed_on_missing_history()
        if not self.has_terminal_predecessor():
            raise AttemptGovernanceError("successor_preparation_requires_terminal_predecessor")
        successor = self.get_successor()
        if successor is not None:
            lifecycle = successor.get("lifecycle")
            if lifecycle in {LIFECYCLE_SUCCESSOR_PROPOSED, LIFECYCLE_AUTHORIZED, LIFECYCLE_ACTIVE}:
                if successor.get("frozen_experiment_identity_hash") == expected_identity_hash:
                    return successor
                raise AttemptGovernanceError("successor_identity_conflict")
            raise AttemptGovernanceError("successor_slot_not_available")
        proposal = {
            "frozen_experiment_identity_hash": expected_identity_hash,
            "lifecycle": LIFECYCLE_SUCCESSOR_PROPOSED,
            "proposed_at": datetime.now(UTC).isoformat(),
            "owner_authorization_token": None,
            "authorized_at": None,
            "run_id": None,
            "run_dir": None,
            "execution_session_token": None,
        }
        if self.payload is None:
            self.payload = {
                "registry_version": REGISTRY_VERSION,
                "history": [],
                "successor": proposal,
            }
        else:
            self.payload["successor"] = proposal
        self._atomic_write(self.payload)
        return proposal

    def _reject_terminal_identity_reuse(self, expected_identity_hash: str) -> None:
        if self.payload is None:
            return
        for row in self.payload.get("history", []):
            if (
                row.get("lifecycle") == LIFECYCLE_TERMINAL
                and row.get("frozen_experiment_identity_hash") == expected_identity_hash
            ):
                raise AttemptGovernanceError(
                    f"experiment_terminal:{row.get('terminal_disposition', 'INVALID')}"
                )

    def establish_or_resume(
        self,
        *,
        expected_identity_hash: str,
        authorization_token: str,
        run_id: str | None = None,
        owner_authorization_token: str | None = None,
    ) -> tuple[Path, DurableAttemptLedger, bool]:
        self.ensure_loaded_fail_closed_on_missing_history()
        self._reject_terminal_identity_reuse(expected_identity_hash)
        if self.requires_owner_authorization():
            if not owner_authorization_token:
                raise AttemptGovernanceError("successor_not_authorized")
            self.validate_and_persist_owner_authorization(
                owner_authorization_token,
                expected_identity_hash,
            )
            successor = self.get_successor()
            if successor is None:
                raise AttemptGovernanceError("successor_not_prepared")
            if successor.get("lifecycle") == LIFECYCLE_AUTHORIZED:
                if successor.get("run_id") and successor.get("run_dir"):
                    run_dir = self._resolve_run_dir(str(successor["run_dir"]), self.experiment_root)
                    ledger = DurableAttemptLedger.from_run_dir(run_dir)
                    successor["lifecycle"] = LIFECYCLE_ACTIVE
                    successor["execution_session_token"] = authorization_token
                    self._atomic_write(self.payload)  # type: ignore[arg-type]
                    return run_dir, ledger, False
                run_dir = allocate_run_directory(
                    run_id=run_id or f"semantic_analyzer_run_{uuid.uuid4().hex[:12]}"
                )
                ledger = DurableAttemptLedger(
                    run_dir,
                    run_id=run_dir.name,
                    frozen_experiment_identity_hash=expected_identity_hash,
                )
                successor["lifecycle"] = LIFECYCLE_ACTIVE
                successor["run_id"] = run_dir.name
                successor["run_dir"] = str(run_dir.relative_to(self.experiment_root))
                successor["execution_session_token"] = authorization_token
                successor["established_at"] = datetime.now(UTC).isoformat()
                self._atomic_write(self.payload)  # type: ignore[arg-type]
                return run_dir, ledger, True
            if successor.get("lifecycle") == LIFECYCLE_ACTIVE:
                run_dir = self._resolve_run_dir(str(successor["run_dir"]), self.experiment_root)
                ledger = DurableAttemptLedger.from_run_dir(run_dir)
                ledger.reconcile_consumed_without_finalize()
                if ledger.is_terminal():
                    self.mark_terminal(ledger=ledger)
                    raise AttemptGovernanceError(
                        f"experiment_terminal:{successor.get('terminal_disposition')}"
                    )
                return run_dir, ledger, False
            if successor.get("lifecycle") == LIFECYCLE_SUCCESSOR_PROPOSED:
                raise AttemptGovernanceError("successor_not_authorized")
            raise AttemptGovernanceError("successor_not_prepared")
        if self.payload is not None:
            successor = self.get_successor()
            if successor and successor.get("lifecycle") == LIFECYCLE_ACTIVE:
                if successor.get("frozen_experiment_identity_hash") != expected_identity_hash:
                    raise AttemptGovernanceError(
                        "conflicting_experiment_identity:requires_new_owner_authorization"
                    )
                run_dir = self._resolve_run_dir(str(successor["run_dir"]), self.experiment_root)
                ledger = DurableAttemptLedger.from_run_dir(run_dir)
                ledger.reconcile_consumed_without_finalize()
                if ledger.is_terminal():
                    self.mark_terminal(ledger=ledger)
                    raise AttemptGovernanceError(
                        f"experiment_terminal:{successor.get('terminal_disposition')}"
                    )
                return run_dir, ledger, False
        orphaned = find_orphaned_authoritative_ledgers(
            output_root=self.output_root,
            expected_identity_hash=expected_identity_hash,
        )
        if orphaned:
            raise AttemptGovernanceError(
                "orphaned_authoritative_ledger:registry_missing_but_ledger_remains"
            )
        run_dir = allocate_run_directory(run_id=run_id or f"semantic_analyzer_run_{uuid.uuid4().hex[:12]}")
        ledger = DurableAttemptLedger(
            run_dir,
            run_id=run_dir.name,
            frozen_experiment_identity_hash=expected_identity_hash,
        )
        run_dir_ref = str(run_dir.relative_to(self.experiment_root))
        successor_record = {
            "frozen_experiment_identity_hash": expected_identity_hash,
            "lifecycle": LIFECYCLE_ACTIVE,
            "proposed_at": datetime.now(UTC).isoformat(),
            "owner_authorization_token": owner_authorization_token,
            "authorized_at": None,
            "run_id": run_dir.name,
            "run_dir": run_dir_ref,
            "execution_session_token": authorization_token,
        }
        payload = {
            "registry_version": REGISTRY_VERSION,
            "history": self.payload.get("history", []) if self.payload else [],
            "successor": successor_record,
        }
        self._atomic_write(payload)
        return run_dir, ledger, True

    def mark_terminal(
        self,
        *,
        ledger: DurableAttemptLedger,
        disposition: str | None = None,
    ) -> None:
        if self.payload is None:
            return
        if disposition is None:
            finalized = [
                row
                for row in ledger.payload["records"]
                if row.get("state") == "FINALIZED"
            ]
            if ledger.payload.get("stopped"):
                reason = str(ledger.payload.get("stop_reason", "invalid")).lower()
                disposition = {"invalid": "INVALID", "fail": "FAIL"}.get(reason, reason.upper())
            elif len(finalized) == MAX_PROVIDER_REQUESTS and all(
                row.get("disposition") == "PASS" for row in finalized
            ):
                disposition = "PASS"
            else:
                disposition = "IN_PROGRESS"
        successor = self.get_successor()
        if successor is None:
            raise AttemptGovernanceError("corrupt_registry:missing_successor_on_terminal")
        terminal_record = {
            **successor,
            "lifecycle": LIFECYCLE_TERMINAL,
            "terminal_disposition": disposition,
            "terminal_at": datetime.now(UTC).isoformat(),
        }
        history = list(self.payload.get("history", []))
        history.append(terminal_record)
        self.payload["history"] = history
        self.payload["successor"] = None
        self._atomic_write(self.payload)


def prepare_successor_experiment(
    *,
    registry_path: Path | None = None,
    experiment_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Prepare a successor identity offline. Does not authorize execution."""
    registry = FrozenExperimentRegistry(
        registry_path,
        experiment_root=experiment_root,
        output_root=output_root,
    )
    identity_hash = frozen_experiment_identity_hash()
    proposal = registry.prepare_successor(expected_identity_hash=identity_hash)
    return {
        "successor_identity_hash": identity_hash,
        "lifecycle": proposal.get("lifecycle"),
        "owner_authorization_required": True,
        "execution_authorized": False,
    }


def validate_provider_http_response(
    *,
    status_code: int,
    headers: dict[str, Any],
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
    else:
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool):
            invalid_reasons.append("invalid_completion_tokens")
        elif completion_tokens > MAX_OUTPUT_TOKENS:
            invalid_reasons.append("completion_tokens_above_max")

    provider_response_id = body.get("id")
    if provider_response_id is not None and not isinstance(provider_response_id, str):
        invalid_reasons.append("invalid_provider_response_id")

    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "finish_reason": finish_reason,
        "requested_model": requested_model,
        "accepted_returned_model": accepted_returned_model,
        "observed_returned_model": observed_returned_model,
        "returned_model": observed_returned_model,
        "provider_response_id": provider_response_id,
        "usage": usage,
        "system_fingerprint": body.get("system_fingerprint"),
        "raw_content": content,
    }


def enrich_case_with_explanation_review(case_result: dict[str, Any]) -> dict[str, Any]:
    oracle = case_result.get("semantic_oracle") or {}
    extracted = oracle.get("extracted_span_text") or {}
    blockers = []
    response_contract = case_result.get("response_contract") or {}
    for observation in response_contract.get("observations", []):
        for blocker in observation.get("blockers", []):
            blockers.append(
                {
                    "kind": blocker.get("kind"),
                    "explanation": blocker.get("explanation"),
                    "evidence_spans": blocker.get("evidence_spans"),
                }
            )
    case_result["principal_reviewer_blocker_explanations"] = blockers
    case_result["extracted_support_span_text"] = extracted.get("support_span_texts", [])
    case_result["extracted_blocker_span_text"] = extracted.get("blocker_span_texts", [])
    case_result["explanation_review_required"] = True
    return case_result


def qualify_provider_case_response(
    *,
    case_id: str,
    request: dict[str, Any],
    raw_response: str,
) -> dict[str, Any]:
    bundle = request["bundle"]
    frozen_truth = request["frozen_truth"]
    case_result = qualify_case_response(
        case_id=case_id,
        bundle=bundle,
        raw_response=raw_response,
        frozen_truth=frozen_truth,
    )
    return enrich_case_with_explanation_review(case_result)


def build_artifact_digest_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return {name: artifact[name] for name in ARTIFACT_DIGEST_FIELDS if name in artifact}


def compute_run_artifact_digest(artifact: dict[str, Any]) -> str:
    return sha256_text(canonical_json_dumps(build_artifact_digest_payload(artifact)))


def verify_run_artifact_integrity(artifact: dict[str, Any]) -> dict[str, Any]:
    invalid_reasons: list[str] = []
    stored_digest = artifact.get("artifact_digest")
    recomputed = compute_run_artifact_digest(artifact)
    if not isinstance(stored_digest, str) or not stored_digest.strip():
        invalid_reasons.append("missing_artifact_digest")
    elif stored_digest != recomputed:
        invalid_reasons.append("corrupt_artifact_digest")
    identity = artifact.get("identity_hashes") or {}
    if identity.get("qualified_harness_head") != QUALIFIED_HARNESS_HEAD:
        invalid_reasons.append("stale_qualified_harness_head")
    if identity.get("required_engine_baseline_sha") != REQUIRED_ENGINE_BASELINE_SHA:
        invalid_reasons.append("stale_engine_baseline_sha")
    if identity.get("fixture_sha256") != EXPECTED_FIXTURE_SHA256:
        invalid_reasons.append("stale_fixture_sha256")
    if identity.get("prompt_sha256") != analyzer_prompt_sha256():
        invalid_reasons.append("stale_prompt_sha256")
    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "recomputed_artifact_digest": recomputed,
    }


def issue_execution_authorization(
    *,
    registry: FrozenExperimentRegistry | None = None,
) -> ExecutionAuthorization:
    if not provider_execution_authorized():
        raise ExecutionAuthorizationError(
            f"provider execution disabled; set {EXECUTE_ENV_VAR}=1 after independent review"
        )
    budget = conservative_total_cost_bound()
    if not budget["budget_authorized"]:
        raise ExecutionAuthorizationError("conservative cost bound exceeds hard spend ceiling")
    identity = validate_harness_identities()
    if not identity["valid"]:
        raise ExecutionAuthorizationError(
            "harness identity invalid: " + ", ".join(identity["invalid_reasons"])
        )
    git_sha = get_implementation_git_sha()
    approved_hash = build_approved_execution_config_hash()
    experiment_identity_hash = frozen_experiment_identity_hash(runner_implementation_sha=git_sha)
    active_registry = registry or FrozenExperimentRegistry()
    active_registry.ensure_loaded_fail_closed_on_missing_history()
    owner_token: str | None = None
    if active_registry.requires_owner_authorization():
        owner_token = os.getenv(OWNER_AUTHORIZATION_ENV_VAR, "").strip()
        if not owner_token:
            raise ExecutionAuthorizationError(
                "owner authorization missing for successor experiment; "
                f"set {OWNER_AUTHORIZATION_ENV_VAR} to an identity-bound token"
            )
        owner_token = active_registry.validate_and_persist_owner_authorization(
            owner_token,
            experiment_identity_hash,
        )
    token_payload = {
        "approved_execution_config_hash": approved_hash,
        "execution_git_sha": git_sha,
        "frozen_experiment_identity_hash": experiment_identity_hash,
        "budget_authorized": budget["budget_authorized"],
        "provider_execution_authorized": True,
        "case_order": list(CASE_ORDER),
        "owner_authorization_token": owner_token,
    }
    return ExecutionAuthorization(
        authorization_token=sha256_text(canonical_json_dumps(token_payload)),
        approved_execution_config_hash=approved_hash,
        execution_git_sha=git_sha,
        budget_authorized=budget["budget_authorized"],
        provider_execution_authorized=True,
        owner_authorization_token=owner_token,
    )


def run_provider_preflight() -> dict[str, Any]:
    identity = validate_harness_identities()
    budget = conservative_total_cost_bound()
    requests = build_all_case_requests()
    prompt_hash = analyzer_prompt_sha256()
    on_disk_prompt_hash = sha256_text(PROMPT_PATH.read_bytes().decode("utf-8"))
    _require(prompt_hash == on_disk_prompt_hash, "prompt hash drift")
    return {
        "runner_id": RUNNER_ID,
        "stage": STAGE,
        "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
        "identity_validation": identity,
        "fixture_sha256": EXPECTED_FIXTURE_SHA256,
        "schema_sha256": ANALYZER_SCHEMA_SHA256,
        "prompt_sha256": prompt_hash,
        "request_identities": build_request_identities(),
        "conservative_cost_bound": budget,
        "provider_client_config": build_provider_client_config(),
        "provider_model_identity": {
            "requested_model": REQUESTED_MODEL,
            "accepted_returned_model": ACCEPTED_RETURNED_MODEL,
            "historical_first_live_system_fingerprint": HISTORICAL_FIRST_LIVE_SYSTEM_FINGERPRINT,
        },
        "provider_execution_default_disabled": not provider_execution_authorized(),
        "owner_authorization_env_var": OWNER_AUTHORIZATION_ENV_VAR,
        "execution_ready": identity["valid"] and budget["budget_authorized"],
        "max_provider_requests": MAX_PROVIDER_REQUESTS,
        "max_attempts_per_case": MAX_ATTEMPTS_PER_CASE,
        "case_order": list(CASE_ORDER),
        "requests_built": [request["case_id"] for request in requests],
    }


def _provider_failure_reason(exc: Exception) -> str:
    if isinstance(exc, ExecutionAuthorizationError):
        return "execution_authorization_error"
    if isinstance(exc, AttemptGovernanceError):
        return "attempt_governance_error"
    if isinstance(exc, ProviderResponseValidationError):
        return "provider_response_validation_error"
    if isinstance(exc, ResponseContractError):
        return "analyzer_response_contract_error"
    if isinstance(exc, httpx.TimeoutException):
        return "provider_timeout"
    if isinstance(exc, httpx.ConnectError):
        return "provider_connection_error"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else None
        if status in REDIRECT_STATUS_CODES:
            return "provider_redirect_error"
        return "provider_http_error"
    return "provider_execution_error"


def execute_case_once(
    *,
    request: dict[str, Any],
    run_state: ProviderRunState,
    durable_ledger: DurableAttemptLedger,
    execution_auth: ExecutionAuthorization,
    http_post: Callable[..., httpx.Response] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    case_id = request["case_id"]
    started = time.time()
    artifact: dict[str, Any] = {
        "case_id": case_id,
        "request_id": request["request_id"],
        "requested_model": REQUESTED_MODEL,
        "request_body_sha256": request["request_body_sha256"],
        "approved_execution_config_hash": execution_auth.approved_execution_config_hash,
        "disposition": "INVALID",
    }
    try:
        durable_ledger.consume_attempt_before_network(case_id)
        outbound_model = request["request_body"].get("model")
        if outbound_model != REQUESTED_MODEL:
            raise ProviderResponseValidationError("outbound_requested_model_mismatch")
        active_client = client or build_semantic_analyzer_http_client()
        post = http_post or active_client.post
        config = build_provider_client_config()
        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        response = post(
            config["endpoint"],
            headers=headers,
            json=request["request_body"],
        )
        run_state.provider_calls += 1
        raw_text = response.text
        artifact["raw_provider_response"] = raw_text
        artifact["http_status_code"] = response.status_code
        artifact["response_headers"] = dict(response.headers)
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderResponseValidationError(f"provider_json_decode_error:{exc.msg}") from exc
        schedule = load_price_schedule()
        usage_for_cost = body.get("usage") if isinstance(body, dict) else None
        observed_cost, cost_unavailable = calculate_observed_cost_usd(usage_for_cost, schedule)
        if observed_cost is not None:
            run_state.total_cost_usd = round(run_state.total_cost_usd + observed_cost, 6)
            artifact["usage"] = usage_for_cost
            artifact["actual_cost_usd"] = observed_cost
            artifact["observed_cost_recorded"] = True
        else:
            artifact["observed_cost_unavailable"] = cost_unavailable
        validation = validate_provider_http_response(
            status_code=response.status_code,
            headers=dict(response.headers),
            body=body,
        )
        fingerprint_drift = check_system_fingerprint_drift(
            run_state,
            validation.get("system_fingerprint"),
        )
        if fingerprint_drift:
            validation = {
                **validation,
                "invalid_reasons": [*validation["invalid_reasons"], "system_fingerprint_drift"],
                "valid": False,
                "system_fingerprint_drift": fingerprint_drift,
            }
        artifact["provider_response_validation"] = validation
        artifact["accepted_returned_model"] = ACCEPTED_RETURNED_MODEL
        artifact["observed_returned_model"] = validation.get("observed_returned_model")
        artifact["system_fingerprint"] = validation.get("system_fingerprint")
        artifact["system_fingerprint_frozen"] = run_state.frozen_system_fingerprint
        if not validation["valid"]:
            raise ProviderResponseValidationError(",".join(validation["invalid_reasons"]))
        raw_content = validation["raw_content"] or ""
        case_result = qualify_provider_case_response(
            case_id=case_id,
            request=request,
            raw_response=raw_content,
        )
        usage = validation.get("usage") or {}
        cost = observed_cost if observed_cost is not None else 0.0
        artifact.update(
            {
                "disposition": case_result["disposition"],
                "invalid_reasons": case_result.get("invalid_reasons", []),
                "fail_reasons": case_result.get("fail_reasons", []),
                "semantic_oracle": case_result.get("semantic_oracle"),
                "adjudication": case_result.get("adjudication"),
                "principal_reviewer_blocker_explanations": case_result.get(
                    "principal_reviewer_blocker_explanations"
                ),
                "explanation_review_required": case_result.get("explanation_review_required", True),
                "provider_response_id": validation.get("provider_response_id"),
                "returned_model": validation.get("observed_returned_model"),
                "finish_reason": validation.get("finish_reason"),
                "usage": usage,
                "actual_cost_usd": cost,
                "latency_ms": round((time.time() - started) * 1000, 3),
            }
        )
        durable_ledger.finalize_attempt(
            case_id,
            disposition=case_result["disposition"],
            detail={"provider_response_id": validation.get("provider_response_id")},
        )
        if case_result["disposition"] in {"FAIL", "INVALID"}:
            run_state.stopped = True
            run_state.stop_reason = case_result["disposition"].lower()
    except Exception as exc:
        artifact.update(
            {
                "invalid_reason": _provider_failure_reason(exc),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "latency_ms": round((time.time() - started) * 1000, 3),
            }
        )
        durable_ledger.finalize_attempt(case_id, disposition="INVALID", detail={"error": str(exc)})
        run_state.stopped = True
        run_state.stop_reason = "invalid"
    return artifact


def execute_provider_qualification_run(
    *,
    run_id: str | None = None,
    http_post: Callable[..., httpx.Response] | None = None,
    client: httpx.Client | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    if not provider_execution_authorized():
        raise ExecutionAuthorizationError(
            f"provider execution disabled; set {EXECUTE_ENV_VAR}=1 after independent review"
        )
    registry = FrozenExperimentRegistry(registry_path)
    execution_auth = issue_execution_authorization(registry=registry)
    preflight = run_provider_preflight()
    if not preflight["execution_ready"]:
        raise SemanticAnalyzerProviderRunnerError("preflight not ready for execution")

    experiment_identity_hash = frozen_experiment_identity_hash(
        runner_implementation_sha=execution_auth.execution_git_sha,
    )
    run_dir, durable_ledger, _created = registry.establish_or_resume(
        expected_identity_hash=experiment_identity_hash,
        authorization_token=execution_auth.authorization_token,
        run_id=run_id,
        owner_authorization_token=execution_auth.owner_authorization_token,
    )
    run_state = ProviderRunState(run_id=run_dir.name, run_dir=run_dir)
    requests = {req["case_id"]: req for req in build_all_case_requests()}
    case_results: list[dict[str, Any]] = []

    while True:
        next_case_id = durable_ledger.next_case_id()
        if next_case_id is None or run_state.stopped:
            break
        case_results.append(
            execute_case_once(
                request=requests[next_case_id],
                run_state=run_state,
                durable_ledger=durable_ledger,
                execution_auth=execution_auth,
                http_post=http_post,
                client=client,
            )
        )

    attempted = {row["case_id"] for row in case_results}
    for case_id in CASE_ORDER:
        if case_id not in attempted:
            case_results.append({"case_id": case_id, "disposition": "NOT_RUN"})

    pass_count = sum(1 for row in case_results if row.get("disposition") == "PASS")
    fail_count = sum(1 for row in case_results if row.get("disposition") == "FAIL")
    invalid_count = sum(1 for row in case_results if row.get("disposition") == "INVALID")
    not_run_count = sum(1 for row in case_results if row.get("disposition") == "NOT_RUN")
    if invalid_count > 0 or fail_count > 0 or pass_count != 3:
        overall = "INVALID" if invalid_count > 0 else "FAIL"
    else:
        overall = "PASS"

    artifact: dict[str, Any] = {
        "runner_id": RUNNER_ID,
        "stage": STAGE,
        "run_id": run_state.run_id,
        "provider_calls": run_state.provider_calls,
        "identity_hashes": {
            "runner_implementation_sha": get_implementation_git_sha(),
            "qualified_harness_head": QUALIFIED_HARNESS_HEAD,
            "required_engine_baseline_sha": REQUIRED_ENGINE_BASELINE_SHA,
            "fixture_sha256": EXPECTED_FIXTURE_SHA256,
            "schema_sha256": ANALYZER_SCHEMA_SHA256,
            "prompt_sha256": analyzer_prompt_sha256(),
            "frozen_experiment_identity_hash": experiment_identity_hash,
        },
        "authorization": {
            "authorization_token": execution_auth.authorization_token,
            "approved_execution_config_hash": execution_auth.approved_execution_config_hash,
            "execution_git_sha": execution_auth.execution_git_sha,
            "owner_authorization_token": execution_auth.owner_authorization_token,
        },
        "pricing_snapshot": load_price_schedule(),
        "request_identities": build_request_identities(),
        "attempt_ledger": durable_ledger.payload,
        "case_order": list(CASE_ORDER),
        "case_results": case_results,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "invalid_count": invalid_count,
        "not_run_count": not_run_count,
        "overall_disposition": overall,
        "total_cost_usd": run_state.total_cost_usd,
        "provider_execution_authorized": True,
    }
    registry.mark_terminal(ledger=durable_ledger, disposition=overall)
    artifact["frozen_experiment_registry"] = registry.payload
    artifact["artifact_digest"] = compute_run_artifact_digest(artifact)
    artifact["artifact_integrity"] = verify_run_artifact_integrity(artifact)
    artifact["qualification_ready"] = (
        overall == "PASS"
        and pass_count == 3
        and invalid_count == 0
        and fail_count == 0
        and artifact["artifact_integrity"]["valid"]
        and isinstance(artifact.get("artifact_digest"), str)
        and bool(artifact["artifact_digest"])
    )
    run_bundle_path = run_dir / "run_artifact.json"
    run_bundle_path.write_text(canonical_json_dumps(artifact), encoding="utf-8")
    artifact["run_bundle_path"] = str(run_bundle_path.relative_to(REPO_ROOT))
    return artifact
