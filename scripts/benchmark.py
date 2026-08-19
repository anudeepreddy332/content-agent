"""Run benchmark topics and fail closed on unverifiable evaluation evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import click

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    GROUNDING_FLOOR,
    KB_N_RESULTS,
    PROMPT_HASHES,
    PROMPT_VERSION,
    QDRANT_COLLECTION,
    QDRANT_EMBEDDING_DIM,
    QDRANT_URL,
    TAVILY_MAX_RESULTS,
    TAVILY_MIN_AVG_SCORE,
)
from scripts.check_telemetry_fields import (  # noqa: E402
    TelemetryValidationError,
    extract_run_id,
    load_exact_telemetry,
    telemetry_validation_error,
)


UVR_THRESHOLD = 0.15
EVIDENCE_SCHEMA_VERSION = "benchmark-evidence-v2"
RELEASE_CONTRACT_PATH = Path("evals/benchmark_release_contract.json")
RELEASE_CONTRACT_REQUIRED_FIELDS = frozenset({
    "schema_version",
    "manifest_id",
    "manifest_path",
    "manifest_sha256",
    "expected_topic_count",
    "ordered_topic_ids",
})
V1_MANIFEST_ID = "content-agent-release-topics-v1"
V1_RELEASE_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "manifest_id": V1_MANIFEST_ID,
    "manifest_path": "evals/topics.json",
    "manifest_sha256": (
        "2e1a502834252f8c7fe7a3b1136efb9f949d3392f493e6e2102838316b0c7b08"
    ),
    "expected_topic_count": 20,
    "ordered_topic_ids": list(range(1, 21)),
}
TOPIC_REQUIRED_FIELDS = ("id", "topic", "card_id", "series", "slug", "category")
TOPIC_STRING_FIELDS = ("topic", "card_id", "series", "slug", "category")
RELEASE_PASS_RESULT_FIELDS = (
    "id",
    "topic",
    "status",
    "run_id",
    "telemetry",
    "verification_status",
    "evaluation_status",
    "uvr",
    "validation_error",
    "subprocess_exit_code",
)
RELEASE_RESULT_FIELDS = RELEASE_PASS_RESULT_FIELDS + ("wall_time_s", "stderr")
EVIDENCE_REQUIRED_FIELDS = (
    "schema_version",
    "timestamp_utc",
    "mode",
    "release_qualification",
    "gate_requested",
    "github_actions_identity",
    "expected_code_sha",
    "preflight_code_identity",
    "final_code_identity",
    "release_contract_identity",
    "selected_manifest_identity",
    "evaluation_configuration",
    "evaluation_config_sha256",
    "ordered_unit_results",
    "aggregate_metrics",
    "gate_failures",
    "evidence_sha256",
)
VALID_MODES = frozenset({"smoke", "release"})
VALID_RELEASE_QUALIFICATIONS = frozenset({"PASS", "FAIL", "NON_RELEASE"})
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_EVALUATION_CONFIG_FIELDS = frozenset({
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL_SHA256",
    "QDRANT_URL_SHA256",
    "QDRANT_COLLECTION",
    "QDRANT_EMBEDDING_DIM",
    "KB_N_RESULTS",
    "TAVILY_MAX_RESULTS",
    "TAVILY_MIN_AVG_SCORE",
    "PROMPT_VERSION",
    "PROMPT_HASHES",
    "GROUNDING_FLOOR",
    "UVR_THRESHOLD",
})
GITHUB_IDENTITY_FIELDS = frozenset({
    "github_actions", "github_ref", "github_sha", "github_run_id",
    "github_run_attempt", "github_workflow_ref",
})
CODE_IDENTITY_FIELDS = frozenset({"head_sha", "staged_clean", "unstaged_clean"})
RELEASE_SELECTED_MANIFEST_FIELDS = frozenset({
    "manifest_id", "manifest_path", "manifest_sha256", "expected_topic_count",
    "ordered_topic_ids", "actual_topic_count", "actual_topic_ids",
})
SECRET_KEY_MARKERS = (
    "apikey", "authorization", "accesstoken", "refreshtoken", "bearertoken",
    "clientsecret", "password", "privatekey", "credential",
)
KNOWN_SECRET_ENVIRONMENT_KEYS = (
    "DEEPSEEK_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY", "API_BEARER_TOKEN",
)


@dataclass(frozen=True)
class TrustedReleaseFinalizationContext:
    """Process-owned inputs used to authorize a release PASS, not payload inputs."""

    expected_code_sha: str
    preflight_github_identity: dict[str, Any]
    final_github_identity: dict[str, Any]
    preflight_code_identity: dict[str, Any]
    final_code_identity: dict[str, Any]
    release_contract_identity: dict[str, Any]
    selected_manifest_identity: dict[str, Any]
    evaluation_configuration: dict[str, Any]
    evaluation_config_sha256: str
    ordered_unit_results: list[dict[str, Any]]
    aggregate_metrics: dict[str, Any]
    gate_failures: list[str]
    timestamp_utc: str
    release_qualification: str


def _fail_preflight(message: str) -> None:
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(2)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim_total(telemetry: dict[str, Any]) -> int:
    return sum(telemetry.get(field, 0) for field in (
        "claims_verified", "claims_weak", "claims_unverified",
    ))


def verification_outcome(telemetry: dict[str, Any], topic: dict[str, Any]) -> dict[str, Any]:
    """Separate verifier completion from whether UVR is meaningful for this topic."""
    status = telemetry.get("verification_status", "unknown")
    total = _claim_total(telemetry)
    if status != "completed":
        return {
            "verification_status": status,
            "evaluation_status": "verification_incomplete",
            "uvr": None,
            "validation_error": f"verification_status={status}",
        }
    if total == 0:
        if topic.get("allow_zero_claims", False):
            return {
                "verification_status": status,
                "evaluation_status": "allowed_zero_claims",
                "uvr": None,
                "validation_error": None,
            }
        return {
            "verification_status": status,
            "evaluation_status": "unscorable_incomplete",
            "uvr": None,
            "validation_error": "zero verdicts without allow_zero_claims",
        }
    return {
        "verification_status": status,
        "evaluation_status": "scorable",
        "uvr": telemetry.get("claims_unverified", 0) / total,
        "validation_error": None,
    }


def _mean(values: list[float], digits: int) -> float | None:
    return round(sum(values) / len(values), digits) if values else None


def _git_rev_parse_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        _fail_preflight(f"git rev-parse HEAD failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _git_diff_clean(*, staged: bool) -> bool:
    cmd = ["git", "diff", "--cached", "--quiet"] if staged else ["git", "diff", "--quiet"]
    return subprocess.run(cmd, check=False).returncode == 0


def _resolve_code_identity() -> dict[str, Any]:
    return {
        "head_sha": _git_rev_parse_head(),
        "staged_clean": _git_diff_clean(staged=True),
        "unstaged_clean": _git_diff_clean(staged=False),
    }


def _github_actions_identity() -> dict[str, str | None]:
    return {
        "github_actions": os.environ.get("GITHUB_ACTIONS"),
        "github_ref": os.environ.get("GITHUB_REF"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
    }


def _validate_github_release_identity(expected_code_sha: str) -> list[str]:
    failures: list[str] = []
    identity = _github_actions_identity()
    if identity["github_actions"] != "true":
        failures.append("GITHUB_ACTIONS must be true for release mode")
    if identity["github_ref"] != "refs/heads/main":
        failures.append("GITHUB_REF must be refs/heads/main for release mode")
    for field in ("github_run_id", "github_run_attempt", "github_workflow_ref"):
        if not identity[field]:
            failures.append(f"{field.upper()} must be nonempty for release mode")
    if identity["github_sha"] != expected_code_sha:
        failures.append("GITHUB_SHA must equal --expected-code-sha")
    return failures


def _validate_code_identity(expected_code_sha: str, identity: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not GIT_SHA_RE.fullmatch(expected_code_sha):
        failures.append("--expected-code-sha must be 40 lowercase hex characters")
    if identity["head_sha"] != expected_code_sha:
        failures.append("checked-out HEAD must equal --expected-code-sha")
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha and identity["head_sha"] != github_sha:
        failures.append("checked-out HEAD must equal GITHUB_SHA")
    if not identity["staged_clean"]:
        failures.append("tracked staged diff must be clean before execution")
    if not identity["unstaged_clean"]:
        failures.append("tracked unstaged diff must be clean before execution")
    return failures


def _validate_v1_contract_immutability(contract: dict[str, Any]) -> None:
    """Reject any contract that does not match frozen V1 semantics exactly."""
    _validate_v1_contract_types(contract, "release contract")
    for field, expected in V1_RELEASE_CONTRACT.items():
        actual = contract.get(field)
        if actual != expected:
            _fail_preflight(
                f"V1 release contract field {field!r} must equal frozen value "
                f"{expected!r}, got {actual!r}",
            )


def _validate_v1_contract_types(contract: Any, label: str) -> dict[str, Any]:
    contract = _require_exact_keys(contract, RELEASE_CONTRACT_REQUIRED_FIELDS, label)
    if _require_true_int(contract["schema_version"], f"{label}.schema_version") != 1:
        raise ValueError(f"{label}.schema_version must be 1")
    _require_nonempty_string(contract["manifest_id"], f"{label}.manifest_id")
    _require_nonempty_string(contract["manifest_path"], f"{label}.manifest_path")
    _validate_sha256_field(contract["manifest_sha256"], f"{label}.manifest_sha256")
    _require_true_int(
        contract["expected_topic_count"], f"{label}.expected_topic_count", positive=True,
    )
    ordered_ids = contract["ordered_topic_ids"]
    if not isinstance(ordered_ids, list) or not ordered_ids:
        raise ValueError(f"{label}.ordered_topic_ids must be a nonempty list")
    for index, topic_id in enumerate(ordered_ids):
        _require_true_int(topic_id, f"{label}.ordered_topic_ids[{index}]", positive=True)
    if len(ordered_ids) != contract["expected_topic_count"]:
        raise ValueError(f"{label}.ordered_topic_ids length must equal expected_topic_count")
    return contract


def load_release_contract(path: Path = RELEASE_CONTRACT_PATH) -> dict[str, Any]:
    if not path.is_file():
        _fail_preflight(f"release contract missing: {path}")
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        _fail_preflight(f"release contract malformed: {error}")
    try:
        _validate_v1_contract_immutability(contract)
    except ValueError as error:
        _fail_preflight(str(error))
    return contract


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_exact_keys(value: Any, expected: frozenset[str] | tuple[str, ...], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected_keys = set(expected)
    missing = sorted(expected_keys - set(value))
    extra = sorted(set(value) - expected_keys)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


def _require_true_int(value: Any, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be a true integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    value = _require_true_int(value, label)
    if value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_finite_number(
    value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None,
) -> float | int:
    if not _is_numeric(value):
        raise ValueError(f"{label} must be a finite non-bool number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return value


def _frozen_v1_manifest_path() -> Path:
    return Path(__file__).resolve().parent.parent / V1_RELEASE_CONTRACT["manifest_path"]


def _frozen_v1_topic_names_by_id() -> dict[int, str]:
    path = _frozen_v1_manifest_path()
    if _sha256_file(path) != V1_RELEASE_CONTRACT["manifest_sha256"]:
        raise ValueError("frozen V1 manifest digest mismatch")
    topics = json.loads(path.read_text(encoding="utf-8"))
    return {topic["id"]: topic["topic"] for topic in topics}


def _validate_manifest_topics(topics: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    seen_ids: set[int] = set()
    seen_topics: set[str] = set()
    seen_slugs: set[str] = set()
    for index, topic in enumerate(topics):
        if not isinstance(topic, dict):
            failures.append(f"manifest topic at index {index} is not an object")
            continue
        missing = [field for field in TOPIC_REQUIRED_FIELDS if field not in topic]
        if missing:
            failures.append(f"manifest topic at index {index} missing fields: {missing}")
        topic_id = topic.get("id")
        if not _is_positive_int(topic_id):
            failures.append(f"manifest topic at index {index} id must be a positive integer")
        elif topic_id in seen_ids:
            failures.append(f"duplicate manifest topic id: {topic_id}")
        else:
            seen_ids.add(topic_id)
        for field in TOPIC_STRING_FIELDS:
            value = topic.get(field)
            if not isinstance(value, str) or not value.strip():
                failures.append(
                    f"manifest topic at index {index} field {field!r} must be a nonempty string",
                )
                continue
            if field == "topic":
                if value in seen_topics:
                    failures.append(f"duplicate manifest topic name: {value}")
                seen_topics.add(value)
            elif field == "slug":
                if value in seen_slugs:
                    failures.append(f"duplicate manifest slug: {value}")
                seen_slugs.add(value)
    return failures


def _validate_manifest_document(topics: Any, contract: dict[str, Any]) -> list[str]:
    """Schema, uniqueness, count, and order — independent of the V1 digest gate."""
    if not isinstance(topics, list) or not topics:
        return ["manifest must be a nonempty JSON array"]
    failures = _validate_manifest_topics(topics)
    if failures:
        return failures
    topic_ids = [topic["id"] for topic in topics]
    if len(topics) != contract["expected_topic_count"]:
        failures.append(
            f"manifest topic count mismatch: expected {contract['expected_topic_count']}, got {len(topics)}",
        )
    if topic_ids != contract["ordered_topic_ids"]:
        failures.append("manifest topic IDs do not match release contract order")
    return failures


def load_validated_manifest(contract: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(contract["manifest_path"])
    if not manifest_path.is_file():
        _fail_preflight(f"manifest missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != contract["manifest_sha256"]:
        _fail_preflight(
            f"manifest digest mismatch: expected {contract['manifest_sha256']}, got {manifest_sha256}",
        )
    try:
        topics = json.loads(manifest_bytes.decode("utf-8"))
    except json.JSONDecodeError as error:
        _fail_preflight(f"manifest malformed: {error}")
    manifest_failures = _validate_manifest_document(topics, contract)
    if manifest_failures:
        _fail_preflight("; ".join(manifest_failures))
    topic_ids = [topic["id"] for topic in topics]
    identity = {
        "manifest_id": contract["manifest_id"],
        "manifest_path": contract["manifest_path"],
        "manifest_sha256": manifest_sha256,
        "expected_topic_count": contract["expected_topic_count"],
        "ordered_topic_ids": list(contract["ordered_topic_ids"]),
        "actual_topic_count": len(topics),
        "actual_topic_ids": topic_ids,
    }
    return topics, identity


def resolve_evaluation_config() -> tuple[dict[str, Any], str]:
    config = {
        "DEEPSEEK_MODEL": DEEPSEEK_MODEL,
        "DEEPSEEK_BASE_URL_SHA256": _sha256_text(DEEPSEEK_BASE_URL),
        "QDRANT_URL_SHA256": _sha256_text(QDRANT_URL),
        "QDRANT_COLLECTION": QDRANT_COLLECTION,
        "QDRANT_EMBEDDING_DIM": QDRANT_EMBEDDING_DIM,
        "KB_N_RESULTS": KB_N_RESULTS,
        "TAVILY_MAX_RESULTS": TAVILY_MAX_RESULTS,
        "TAVILY_MIN_AVG_SCORE": TAVILY_MIN_AVG_SCORE,
        "PROMPT_VERSION": PROMPT_VERSION,
        "PROMPT_HASHES": dict(sorted(PROMPT_HASHES.items())),
        "GROUNDING_FLOOR": GROUNDING_FLOOR,
        "UVR_THRESHOLD": UVR_THRESHOLD,
    }
    digest = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    return config, digest


def _normalized_evidence_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _known_runtime_secrets() -> tuple[str, ...]:
    return tuple(
        value for name in KNOWN_SECRET_ENVIRONMENT_KEYS
        if (value := os.environ.get(name))
    )


def _validate_no_secrets(value: Any, *, path: str = "evidence") -> None:
    """Bounded schema/known-secret exclusion; it is not unknown-secret discovery."""
    secrets = _known_runtime_secrets()

    def visit(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key)
                normalized = _normalized_evidence_key(key_text)
                if any(marker in normalized for marker in SECRET_KEY_MARKERS):
                    raise ValueError(f"secret-bearing evidence key rejected at {item_path}")
                visit(child, f"{item_path}.{key_text}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")
            return
        if not isinstance(item, str):
            return
        if any(secret in item for secret in secrets):
            raise ValueError(f"known runtime secret rejected at {item_path}")
        parsed = urlsplit(item)
        if parsed.scheme and (parsed.username is not None or parsed.password is not None):
            raise ValueError(f"credential-bearing URL rejected at {item_path}")
        if parsed.scheme:
            for query_key, _ in parse_qsl(parsed.query, keep_blank_values=True):
                normalized = _normalized_evidence_key(query_key)
                if any(marker in normalized for marker in SECRET_KEY_MARKERS):
                    raise ValueError(f"credential-bearing URL rejected at {item_path}")

    visit(value, path)


def _validate_safe_evaluation_config(config: Any) -> dict[str, Any]:
    config = _require_exact_keys(
        config, SAFE_EVALUATION_CONFIG_FIELDS, "evaluation_configuration",
    )
    _require_nonempty_string(config["DEEPSEEK_MODEL"], "evaluation_configuration.DEEPSEEK_MODEL")
    _validate_sha256_field(
        config["DEEPSEEK_BASE_URL_SHA256"], "evaluation_configuration.DEEPSEEK_BASE_URL_SHA256",
    )
    _validate_sha256_field(
        config["QDRANT_URL_SHA256"], "evaluation_configuration.QDRANT_URL_SHA256",
    )
    _require_nonempty_string(config["QDRANT_COLLECTION"], "evaluation_configuration.QDRANT_COLLECTION")
    _require_true_int(config["QDRANT_EMBEDDING_DIM"], "evaluation_configuration.QDRANT_EMBEDDING_DIM", positive=True)
    _require_nonnegative_int(config["KB_N_RESULTS"], "evaluation_configuration.KB_N_RESULTS")
    _require_nonnegative_int(config["TAVILY_MAX_RESULTS"], "evaluation_configuration.TAVILY_MAX_RESULTS")
    _require_finite_number(config["TAVILY_MIN_AVG_SCORE"], "evaluation_configuration.TAVILY_MIN_AVG_SCORE")
    _require_nonempty_string(config["PROMPT_VERSION"], "evaluation_configuration.PROMPT_VERSION")
    if not isinstance(config["PROMPT_HASHES"], dict):
        raise ValueError("evaluation_configuration.PROMPT_HASHES must be an object")
    _require_finite_number(config["GROUNDING_FLOOR"], "evaluation_configuration.GROUNDING_FLOOR")
    _require_finite_number(config["UVR_THRESHOLD"], "evaluation_configuration.UVR_THRESHOLD")
    _validate_no_secrets(config, path="evaluation_configuration")
    return config


def _child_environment(evaluation_config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["DEEPSEEK_MODEL"] = evaluation_config["DEEPSEEK_MODEL"]
    env["DEEPSEEK_BASE_URL"] = DEEPSEEK_BASE_URL
    env["QDRANT_URL"] = QDRANT_URL
    env["QDRANT_COLLECTION"] = evaluation_config["QDRANT_COLLECTION"]
    return env


def _select_smoke_topics(
    topics: list[dict[str, Any]],
    *,
    topic_id: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if topic_id is not None and limit is not None:
        _fail_preflight("smoke mode requires exactly one of --id or --limit")
    if topic_id is None and limit is None:
        _fail_preflight("smoke mode requires exactly one of --id or --limit")
    if topic_id is not None:
        if topic_id <= 0:
            _fail_preflight("--id must be a positive integer")
        selected = [topic for topic in topics if topic["id"] == topic_id]
        if not selected:
            _fail_preflight(f"unknown topic id: {topic_id}")
        return selected
    assert limit is not None
    if limit <= 0:
        _fail_preflight("--limit must be a positive integer")
    if limit > 20:
        _fail_preflight("--limit cannot exceed 20")
    selected = topics[:limit]
    if not selected:
        _fail_preflight("smoke selection resolved to zero topics")
    return selected


def _select_release_topics(contract: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_validated_manifest(contract)


def _run_topic(
    topic: dict[str, Any],
    *,
    child_env: dict[str, str],
) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        ["uv", "run", "python", "main.py", "run",
         "--topic", topic["topic"], "--card-id", topic["card_id"],
         "--series", topic["series"], "--auto"],
        capture_output=True,
        text=True,
        timeout=600,
        env=child_env,
    )
    elapsed = time.time() - started
    status = "success" if proc.returncode == 0 else "failed"
    telemetry = None
    validation_error = None
    run_id = extract_run_id(proc.stdout)

    if status == "success":
        if run_id is None:
            status = "failed"
            validation_error = "successful CLI output did not emit RUN_ID"
        else:
            try:
                telemetry = load_exact_telemetry(run_id, topic["topic"])
            except TelemetryValidationError as error:
                status = "failed"
                validation_error = str(error)
            else:
                validation_error = telemetry_validation_error(telemetry)
                if validation_error:
                    status = "failed"

    if telemetry is None:
        outcome = {
            "verification_status": "unknown",
            "evaluation_status": "telemetry_unavailable",
            "uvr": None,
            "validation_error": validation_error or f"CLI exited with status {proc.returncode}",
        }
    else:
        try:
            outcome = verification_outcome(telemetry, topic)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            status = "failed"
            outcome = {
                "verification_status": telemetry.get("verification_status", "unknown"),
                "evaluation_status": "telemetry_invalid",
                "uvr": None,
                "validation_error": f"telemetry arithmetic invalid: {type(error).__name__}",
            }
    if outcome["validation_error"] and validation_error is None:
        validation_error = outcome["validation_error"]

    return {
        "id": topic["id"],
        "topic": topic["topic"],
        "status": status,
        "run_id": run_id,
        "wall_time_s": round(elapsed, 1),
        "telemetry": telemetry,
        "verification_status": outcome["verification_status"],
        "evaluation_status": outcome["evaluation_status"],
        "uvr": outcome["uvr"],
        "validation_error": validation_error,
        "stderr": proc.stderr[-500:] if proc.returncode != 0 else None,
        "subprocess_exit_code": proc.returncode,
    }


def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        result for result in results
        if result.get("status") == "success" and not result.get("validation_error")
        and isinstance(result.get("telemetry"), dict)
    ]
    scorable = [result for result in valid if _is_numeric(result.get("uvr"))]

    def numeric(values: list[Any]) -> list[float | int]:
        return [value for value in values if _is_numeric(value)]

    return {
        "total_runs": len(results),
        "successful": len(valid),
        "failed": len(results) - len(valid),
        "unscorable": len([result for result in valid if result["uvr"] is None]),
        "mean_cost_usd": _mean(numeric([result["telemetry"].get("total_cost_usd") for result in valid]), 5),
        "mean_grounding": _mean(numeric([result["telemetry"].get("grounding_score", 0) for result in valid]), 3),
        "mean_reflection": _mean(numeric([result["telemetry"].get("reflection_score", 0) for result in valid]), 1),
        "mean_wall_time_s": _mean(numeric([result.get("wall_time_s") for result in results]), 1),
        "mean_html_errors": _mean([len(result["telemetry"].get("error_log", [])) for result in valid], 1),
        "mean_unverified_rate": _mean([result["uvr"] for result in scorable], 3),
        "runs_below_grounding_floor": sum(
            1 for result in valid if result["telemetry"].get("grounding_score", 1.0) < GROUNDING_FLOOR
        ),
    }


def _validate_release_telemetry_types(telemetry: dict[str, Any], label: str) -> None:
    for field in ("claims_verified", "claims_weak", "claims_unverified"):
        _require_nonnegative_int(telemetry.get(field), f"{label} telemetry.{field}")
    if _claim_total(telemetry) <= 0:
        raise ValueError(f"{label}: unscorable zero-verdict telemetry claim total")
    for field in ("web_sources_count", "kb_results_count", "total_tokens"):
        _require_nonnegative_int(telemetry.get(field), f"{label} telemetry.{field}")
    _require_finite_number(telemetry.get("total_cost_usd"), f"{label} telemetry.total_cost_usd", minimum=0)
    _require_finite_number(
        telemetry.get("grounding_score"), f"{label} telemetry.grounding_score", minimum=0, maximum=1,
    )
    reflection = _require_true_int(telemetry.get("reflection_score"), f"{label} telemetry.reflection_score")
    if not 1 <= reflection <= 10:
        raise ValueError(f"{label} telemetry.reflection_score must be in 1..10")
    latency = telemetry.get("latency_ms")
    if not isinstance(latency, dict):
        raise ValueError(f"{label} telemetry.latency_ms must be an object")
    for name, value in latency.items():
        _require_finite_number(value, f"{label} telemetry.latency_ms.{name}", minimum=0)


def _validate_release_units(
    results: list[dict[str, Any]],
    *,
    expected_topic_ids: list[int],
    evaluation_config: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if len(results) != len(expected_topic_ids):
        failures.append(
            f"expected {len(expected_topic_ids)} release units, got {len(results)}",
        )
        return failures

    result_ids = [result.get("id") if isinstance(result, dict) else None for result in results]
    if result_ids != expected_topic_ids:
        failures.append("release unit topic IDs are missing, extra, duplicate, or out of order")

    run_ids = [result.get("run_id") for result in results if isinstance(result, dict)]
    if len(run_ids) != len(results) or any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        failures.append("every release unit must have a nonempty run ID")
    elif len(set(run_ids)) != len(run_ids):
        failures.append("release run IDs must be unique")

    try:
        _validate_safe_evaluation_config(evaluation_config)
    except ValueError as error:
        return [str(error)]
    expected_prompt_version = evaluation_config["PROMPT_VERSION"]
    expected_prompt_hashes = evaluation_config["PROMPT_HASHES"]
    try:
        v1_topics = _frozen_v1_topic_names_by_id()
    except ValueError as error:
        failures.append(str(error))
        return failures

    for index, result in enumerate(results):
        label = f"topic {expected_topic_ids[index]:02d}"
        if not isinstance(result, dict):
            failures.append(f"{label}: result must be an object")
            continue
        try:
            _require_exact_keys(result, RELEASE_RESULT_FIELDS, f"{label} result")
            result_id = _require_true_int(result.get("id"), f"{label} result.id", positive=True)
            _require_true_int(result.get("subprocess_exit_code"), f"{label} subprocess_exit_code")
            _require_finite_number(result.get("wall_time_s"), f"{label} wall_time_s", minimum=0)
        except ValueError as error:
            failures.append(str(error))
            continue
        expected_topic = v1_topics.get(result_id)
        if expected_topic is not None and result.get("topic") != expected_topic:
            failures.append(f"{label}: topic does not match frozen V1 manifest")
        if result["subprocess_exit_code"] != 0:
            failures.append(f"{label}: subprocess exit code {result['subprocess_exit_code']}")
            continue
        if result["status"] != "success" or result["validation_error"]:
            failures.append(f"{label}: {result['validation_error'] or 'CLI failed'}")
            continue
        telemetry = result["telemetry"]
        if not isinstance(telemetry, dict):
            failures.append(f"{label}: telemetry unavailable")
            continue
        # Preserve the existing validator as the first telemetry gate.  This is
        # deterministic schema validation, not cryptographic artifact proof.
        try:
            telemetry_error = telemetry_validation_error(telemetry)
        except Exception as error:  # malformed values must never fail open
            failures.append(f"{label}: telemetry validator exception: {type(error).__name__}")
            continue
        if telemetry_error:
            failures.append(f"{label}: telemetry validator: {telemetry_error}")
            continue
        try:
            _validate_release_telemetry_types(telemetry, label)
        except ValueError as error:
            failures.append(str(error))
            continue
        if telemetry.get("run_id") != result["run_id"]:
            failures.append(f"{label}: telemetry run_id mismatch")
        if telemetry.get("topic") != result["topic"]:
            failures.append(f"{label}: telemetry topic mismatch")
        if telemetry.get("verification_status") != result["verification_status"]:
            failures.append(f"{label}: telemetry verification_status mismatch")
        if result["verification_status"] != "completed":
            failures.append(f"{label}: verification_status={result['verification_status']}")
        if telemetry.get("verification_status") != "completed":
            failures.append(
                f"{label}: telemetry verification_status={telemetry.get('verification_status')}",
            )
        outcome = verification_outcome(telemetry, {})
        if result["evaluation_status"] != "scorable":
            failures.append(f"{label}: evaluation_status={result['evaluation_status']}")
        if outcome["evaluation_status"] != "scorable":
            failures.append(f"{label}: telemetry is {outcome['evaluation_status']}")
        if result["evaluation_status"] != outcome["evaluation_status"]:
            failures.append(f"{label}: evaluation_status contradicts telemetry")
        total = _claim_total(telemetry)
        uvr = result["uvr"]
        try:
            _require_finite_number(uvr, f"{label} UVR", minimum=0, maximum=1)
        except ValueError as error:
            failures.append(str(error))
        else:
            if uvr > UVR_THRESHOLD:
                failures.append(f"{label}: UVR {uvr:.2f} > {UVR_THRESHOLD:.2f}")
        if total == 0:
            failures.append(f"{label}: unscorable or zero-verdict unit")
        elif _is_numeric(uvr):
            computed_uvr = telemetry.get("claims_unverified", 0) / total
            if uvr != computed_uvr:
                failures.append(f"{label}: UVR contradicts claim counts")
        prompt_version = telemetry.get("prompt_version")
        prompt_hashes = telemetry.get("prompt_hashes")
        if prompt_version != expected_prompt_version:
            failures.append(f"{label}: prompt_version mismatch")
        if prompt_hashes != expected_prompt_hashes:
            failures.append(f"{label}: prompt_hashes mismatch")

    return failures


def _validate_smoke_gate(results: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for result in results:
        if result["status"] != "success" or result["validation_error"]:
            failures.append(f"topic {result['id']:02d}: {result['validation_error'] or 'CLI failed'}")
    scorable = [result for result in results if result["uvr"] is not None]
    for result in scorable:
        if result["uvr"] > UVR_THRESHOLD:
            failures.append(
                f"topic {result['id']:02d} UVR {result['uvr']:.2f} > {UVR_THRESHOLD:.2f}",
            )
    return failures


def _compute_evidence_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_git_sha_field(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not GIT_SHA_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a 40-char lowercase hex git SHA")


def _validate_sha256_field(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a 64-char lowercase hex SHA-256 digest")


def _validate_code_identity_object(value: Any, field_name: str) -> None:
    value = _require_exact_keys(value, CODE_IDENTITY_FIELDS, field_name)
    _validate_git_sha_field(value["head_sha"], f"{field_name}.head_sha")
    for flag in ("staged_clean", "unstaged_clean"):
        if not isinstance(value.get(flag), bool):
            raise ValueError(f"{field_name}.{flag} must be a boolean")


def _require_nonempty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")


def _validate_release_pass_github_identity(
    identity: dict[str, Any],
    expected_code_sha: str,
) -> None:
    _require_exact_keys(identity, GITHUB_IDENTITY_FIELDS, "github_actions_identity")
    if identity.get("github_actions") != "true":
        raise ValueError("release PASS requires github_actions == 'true'")
    if identity.get("github_ref") != "refs/heads/main":
        raise ValueError("release PASS requires github_ref == 'refs/heads/main'")
    if identity.get("github_sha") != expected_code_sha:
        raise ValueError("release PASS requires github_sha to equal expected_code_sha")
    for field in ("github_run_id", "github_run_attempt", "github_workflow_ref"):
        _require_nonempty_string(identity.get(field), f"release PASS {field}")


def _validate_release_pass_code_identity(
    preflight: dict[str, Any],
    final: dict[str, Any],
    expected_code_sha: str,
) -> None:
    if preflight["head_sha"] != expected_code_sha:
        raise ValueError("release PASS requires preflight HEAD to equal expected_code_sha")
    if final["head_sha"] != expected_code_sha:
        raise ValueError("release PASS requires final HEAD to equal expected_code_sha")
    if preflight["head_sha"] != final["head_sha"]:
        raise ValueError("release PASS requires stable code identity across execution")
    for identity, name in ((preflight, "preflight"), (final, "final")):
        if identity["staged_clean"] is not True:
            raise ValueError(f"release PASS requires {name} staged_clean")
        if identity["unstaged_clean"] is not True:
            raise ValueError(f"release PASS requires {name} unstaged_clean")


def _validate_release_pass_contract_identity(contract_identity: dict[str, Any]) -> None:
    _validate_v1_contract_types(contract_identity, "release_contract_identity")
    for field, expected in V1_RELEASE_CONTRACT.items():
        actual = contract_identity.get(field)
        if actual != expected:
            raise ValueError(
                f"release PASS release_contract_identity.{field} must match frozen V1",
            )


def _validate_release_pass_selected_manifest(selected_identity: dict[str, Any]) -> None:
    selected_identity = _require_exact_keys(
        selected_identity, RELEASE_SELECTED_MANIFEST_FIELDS, "selected_manifest_identity",
    )
    _require_nonempty_string(selected_identity["manifest_id"], "selected_manifest_identity.manifest_id")
    _require_nonempty_string(selected_identity["manifest_path"], "selected_manifest_identity.manifest_path")
    _validate_sha256_field(selected_identity["manifest_sha256"], "selected_manifest_identity.manifest_sha256")
    _require_true_int(
        selected_identity["expected_topic_count"], "selected_manifest_identity.expected_topic_count", positive=True,
    )
    _require_true_int(
        selected_identity["actual_topic_count"], "selected_manifest_identity.actual_topic_count", positive=True,
    )
    for field in ("ordered_topic_ids", "actual_topic_ids"):
        values = selected_identity[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"selected_manifest_identity.{field} must be a nonempty list")
        for index, topic_id in enumerate(values):
            _require_true_int(topic_id, f"selected_manifest_identity.{field}[{index}]", positive=True)
    if selected_identity.get("manifest_id") != V1_MANIFEST_ID:
        raise ValueError("release PASS requires V1 manifest identity")
    if selected_identity.get("manifest_path") != V1_RELEASE_CONTRACT["manifest_path"]:
        raise ValueError("release PASS requires frozen V1 manifest path")
    if selected_identity.get("manifest_sha256") != V1_RELEASE_CONTRACT["manifest_sha256"]:
        raise ValueError("release PASS requires frozen V1 manifest SHA")
    expected_count = V1_RELEASE_CONTRACT["expected_topic_count"]
    expected_ids = V1_RELEASE_CONTRACT["ordered_topic_ids"]
    if selected_identity.get("actual_topic_count") != expected_count:
        raise ValueError("release PASS requires actual_topic_count == 20")
    if selected_identity.get("expected_topic_count") != expected_count:
        raise ValueError("release PASS requires expected_topic_count == 20")
    if selected_identity.get("actual_topic_ids") != expected_ids:
        raise ValueError("release PASS requires actual_topic_ids == 1..20")
    if selected_identity.get("ordered_topic_ids") != expected_ids:
        raise ValueError("release PASS requires ordered_topic_ids == 1..20")


def _validate_release_pass_results(
    results: list[Any],
    evaluation_config: dict[str, Any],
) -> None:
    expected_ids = V1_RELEASE_CONTRACT["ordered_topic_ids"]
    if len(results) != len(expected_ids):
        raise ValueError("release PASS requires exactly 20 ordered unit results")
    run_ids: list[str] = []
    typed_results: list[dict[str, Any]] = []
    for index, expected_id in enumerate(expected_ids):
        result = results[index]
        if not isinstance(result, dict):
            raise ValueError(f"release PASS result at index {index} must be an object")
        _require_exact_keys(result, RELEASE_RESULT_FIELDS, f"release PASS result {expected_id}")
        if _require_true_int(result.get("id"), f"release PASS result {expected_id}.id", positive=True) != expected_id:
            raise ValueError("release PASS result IDs must be exactly 1..20 in order")
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("release PASS requires nonempty unique run IDs")
        run_ids.append(run_id)
        typed_results.append(result)
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("release PASS run IDs must be unique")
    failures = _validate_release_units(
        typed_results,
        expected_topic_ids=expected_ids,
        evaluation_config=evaluation_config,
    )
    if failures:
        raise ValueError("; ".join(failures))


def _validate_release_pass_aggregates(payload: dict[str, Any]) -> None:
    results = payload["ordered_unit_results"]
    aggregate = payload["aggregate_metrics"]
    if not isinstance(aggregate, dict):
        raise ValueError("aggregate_metrics must be an object")
    expected = _aggregate_metrics(results)
    missing = sorted(set(expected) - set(aggregate))
    extra = sorted(set(aggregate) - set(expected))
    if missing:
        raise ValueError(f"release PASS aggregate_metrics missing fields: {missing}")
    if extra:
        raise ValueError(f"release PASS aggregate_metrics has unexpected fields: {extra}")
    for field in ("total_runs", "successful", "failed", "unscorable", "runs_below_grounding_floor"):
        _require_nonnegative_int(aggregate[field], f"aggregate_metrics.{field}")
    for field in (
        "mean_cost_usd", "mean_grounding", "mean_reflection", "mean_wall_time_s",
        "mean_html_errors", "mean_unverified_rate",
    ):
        if aggregate[field] is not None:
            _require_finite_number(aggregate[field], f"aggregate_metrics.{field}")
    if aggregate != expected:
        mismatches = sorted(
            field for field in expected if aggregate[field] != expected[field]
        )
        raise ValueError(
            "release PASS aggregate_metrics contradicts ordered units: "
            f"{mismatches}",
        )


def _validate_release_pass_evidence(
    payload: dict[str, Any],
    *,
    github_identity: dict[str, Any],
    expected_code_sha: str,
    preflight: dict[str, Any],
    final: dict[str, Any],
    contract_identity: dict[str, Any],
    selected_identity: dict[str, Any],
) -> None:
    _validate_release_pass_github_identity(github_identity, expected_code_sha)
    _validate_release_pass_code_identity(preflight, final, expected_code_sha)
    _validate_release_pass_contract_identity(contract_identity)
    _validate_release_pass_selected_manifest(selected_identity)
    if payload["gate_requested"] is not True:
        raise ValueError("release PASS requires gate_requested")
    if payload["gate_failures"] != []:
        raise ValueError("release PASS requires empty gate_failures")
    _validate_release_pass_results(
        payload["ordered_unit_results"],
        payload["evaluation_configuration"],
    )
    _validate_release_pass_aggregates(payload)


def _validate_evidence_structure(payload: dict[str, Any]) -> None:
    _require_exact_keys(payload, EVIDENCE_REQUIRED_FIELDS, "evidence")

    if payload["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("invalid evidence schema_version")
    if not isinstance(payload["timestamp_utc"], str) or not payload["timestamp_utc"]:
        raise ValueError("timestamp_utc must be a nonempty string")
    if payload["mode"] not in VALID_MODES:
        raise ValueError("mode must be smoke or release")
    if payload["release_qualification"] not in VALID_RELEASE_QUALIFICATIONS:
        raise ValueError("release_qualification must be PASS, FAIL, or NON_RELEASE")
    if payload["mode"] == "smoke" and payload["release_qualification"] != "NON_RELEASE":
        raise ValueError("smoke evidence requires release_qualification=NON_RELEASE")
    if payload["mode"] == "release" and payload["release_qualification"] not in {"PASS", "FAIL"}:
        raise ValueError("release evidence requires release_qualification=PASS or FAIL")
    if not isinstance(payload["gate_requested"], bool):
        raise ValueError("gate_requested must be a boolean")

    github_identity = _require_exact_keys(
        payload["github_actions_identity"], GITHUB_IDENTITY_FIELDS, "github_actions_identity",
    )

    expected_code_sha = payload["expected_code_sha"]
    if payload["mode"] == "release":
        _validate_git_sha_field(expected_code_sha, "expected_code_sha")
    elif expected_code_sha is not None:
        raise ValueError("expected_code_sha must be null for smoke mode")

    preflight = payload["preflight_code_identity"]
    final = payload["final_code_identity"]
    if payload["mode"] == "release":
        _validate_code_identity_object(preflight, "preflight_code_identity")
        _validate_code_identity_object(final, "final_code_identity")
    else:
        if preflight is not None:
            raise ValueError("preflight_code_identity must be null for smoke mode")
        if final is not None:
            raise ValueError("final_code_identity must be null for smoke mode")

    contract_identity = _validate_v1_contract_types(
        payload["release_contract_identity"], "release_contract_identity",
    )

    selected_identity = payload["selected_manifest_identity"]
    if payload["mode"] == "release":
        _validate_release_pass_selected_manifest(selected_identity)
    else:
        _require_exact_keys(
            selected_identity,
            frozenset({"manifest_id", "manifest_path", "manifest_sha256", "selected_topic_count", "selected_topic_ids"}),
            "selected_manifest_identity",
        )

    evaluation_config = _validate_safe_evaluation_config(payload["evaluation_configuration"])
    _validate_sha256_field(payload["evaluation_config_sha256"], "evaluation_config_sha256")
    expected_config_sha = hashlib.sha256(
        _canonical_json(evaluation_config).encode("utf-8"),
    ).hexdigest()
    if payload["evaluation_config_sha256"] != expected_config_sha:
        raise ValueError("evaluation_config_sha256 mismatch")

    if not isinstance(payload["ordered_unit_results"], list):
        raise ValueError("ordered_unit_results must be a list")
    if not isinstance(payload["aggregate_metrics"], dict):
        raise ValueError("aggregate_metrics must be an object")
    if not isinstance(payload["gate_failures"], list):
        raise ValueError("gate_failures must be a list")

    if payload["mode"] == "release" and payload["release_qualification"] == "PASS":
        _validate_release_pass_evidence(
            payload,
            github_identity=github_identity,
            expected_code_sha=expected_code_sha,
            preflight=preflight,
            final=final,
            contract_identity=contract_identity,
            selected_identity=selected_identity,
        )


def _validate_evidence_payload(payload: dict[str, Any]) -> None:
    _validate_no_secrets(payload)
    _validate_evidence_structure(payload)
    digest = payload.get("evidence_sha256")
    if not digest:
        raise ValueError("missing evidence_sha256")
    _validate_sha256_field(digest, "evidence_sha256")
    body = dict(payload)
    body.pop("evidence_sha256", None)
    expected = _compute_evidence_digest(body)
    if digest != expected:
        raise ValueError("evidence_sha256 mismatch")


def _build_trusted_release_context(
    *,
    expected_code_sha: str,
    preflight_github_identity: dict[str, Any],
    preflight_code_identity: dict[str, Any],
    preflight_contract: dict[str, Any],
    preflight_manifest_identity: dict[str, Any],
    preflight_evaluation_config: dict[str, Any],
    results: list[dict[str, Any]],
    gate_failures: list[str],
) -> TrustedReleaseFinalizationContext:
    """Resolve every trust root again; a payload cannot authenticate itself."""
    final_github_identity = _github_actions_identity()
    final_code_identity = _resolve_code_identity()
    final_config, final_config_sha256 = resolve_evaluation_config()
    final_contract = load_release_contract()
    _, final_manifest_identity = load_validated_manifest(final_contract)
    if final_github_identity != preflight_github_identity:
        raise ValueError("trusted-context comparison: GitHub identity drifted")
    if final_code_identity != preflight_code_identity:
        raise ValueError("trusted-context comparison: code identity drifted")
    if final_config != preflight_evaluation_config:
        raise ValueError("trusted-context comparison: evaluation configuration drifted")
    if final_contract != preflight_contract:
        raise ValueError("trusted-context comparison: release contract drifted")
    if final_manifest_identity != preflight_manifest_identity:
        raise ValueError("trusted-context comparison: manifest identity drifted")
    frozen_contract = deepcopy(V1_RELEASE_CONTRACT)
    if final_contract != frozen_contract:
        raise ValueError("trusted-context comparison: immutable V1 contract drifted")
    trusted_results = deepcopy(results)
    trusted_failures = deepcopy(gate_failures)
    if trusted_failures:
        raise ValueError("trusted finalization requires no prior gate failures")
    unit_failures = _validate_release_units(
        trusted_results,
        expected_topic_ids=deepcopy(frozen_contract["ordered_topic_ids"]),
        evaluation_config=deepcopy(final_config),
    )
    if unit_failures:
        raise ValueError("trusted finalization unit validation: " + "; ".join(unit_failures))
    return TrustedReleaseFinalizationContext(
        expected_code_sha=expected_code_sha,
        preflight_github_identity=deepcopy(preflight_github_identity),
        final_github_identity=deepcopy(final_github_identity),
        preflight_code_identity=deepcopy(preflight_code_identity),
        final_code_identity=deepcopy(final_code_identity),
        release_contract_identity=frozen_contract,
        selected_manifest_identity=deepcopy(final_manifest_identity),
        evaluation_configuration=deepcopy(final_config),
        evaluation_config_sha256=final_config_sha256,
        ordered_unit_results=trusted_results,
        aggregate_metrics=_aggregate_metrics(trusted_results),
        gate_failures=trusted_failures,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        release_qualification="PASS",
    )


def _trusted_release_payload(context: TrustedReleaseFinalizationContext) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "timestamp_utc": context.timestamp_utc,
        "mode": "release",
        "release_qualification": context.release_qualification,
        "gate_requested": True,
        "github_actions_identity": deepcopy(context.final_github_identity),
        "expected_code_sha": context.expected_code_sha,
        "preflight_code_identity": deepcopy(context.preflight_code_identity),
        "final_code_identity": deepcopy(context.final_code_identity),
        "release_contract_identity": deepcopy(context.release_contract_identity),
        "selected_manifest_identity": deepcopy(context.selected_manifest_identity),
        "evaluation_configuration": deepcopy(context.evaluation_configuration),
        "evaluation_config_sha256": context.evaluation_config_sha256,
        "ordered_unit_results": deepcopy(context.ordered_unit_results),
        "aggregate_metrics": deepcopy(context.aggregate_metrics),
        "gate_failures": deepcopy(context.gate_failures),
    }


def _sanitized_release_fail_evidence(
    *, expected_code_sha: str, github_identity: dict[str, Any],
    preflight_identity: dict[str, Any], final_identity: dict[str, Any],
    manifest_identity: dict[str, Any], gate_failures: list[str],
) -> dict[str, Any]:
    config, config_sha256 = resolve_evaluation_config()
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "release",
        "release_qualification": "FAIL",
        "gate_requested": True,
        "github_actions_identity": deepcopy(github_identity),
        "expected_code_sha": expected_code_sha,
        "preflight_code_identity": deepcopy(preflight_identity),
        "final_code_identity": deepcopy(final_identity),
        "release_contract_identity": deepcopy(V1_RELEASE_CONTRACT),
        "selected_manifest_identity": deepcopy(manifest_identity),
        "evaluation_configuration": config,
        "evaluation_config_sha256": config_sha256,
        "ordered_unit_results": [],
        "aggregate_metrics": _aggregate_metrics([]),
        "gate_failures": deepcopy(gate_failures),
    }


def _validate_trusted_release_payload(
    payload: dict[str, Any], context: TrustedReleaseFinalizationContext | None,
) -> None:
    if context is None:
        raise ValueError("release PASS requires external trusted finalization context")
    expected = _trusted_release_payload(context)
    expected["evidence_sha256"] = _compute_evidence_digest(expected)
    if payload != expected:
        raise ValueError("trusted-context comparison: persisted PASS differs from intended report")


def _write_evidence_report(
    payload: dict[str, Any], out_path: Path,
    *, trusted_context: TrustedReleaseFinalizationContext | None = None,
) -> dict[str, Any]:
    body = deepcopy(payload)
    if body.get("release_qualification") == "PASS":
        _validate_trusted_release_payload(
            {**body, "evidence_sha256": _compute_evidence_digest(body)}, trusted_context,
        )
    _validate_no_secrets(body)
    body["evidence_sha256"] = _compute_evidence_digest(body)
    _validate_evidence_payload(body)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    is_pass = body.get("release_qualification") == "PASS"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, out_path)
        reread = json.loads(out_path.read_text(encoding="utf-8"))
        _validate_evidence_payload(reread)
        _validate_no_secrets(reread)
        if is_pass:
            _validate_trusted_release_payload(reread, trusted_context)
        return reread
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        if is_pass:
            out_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)
        raise


def _print_result_summary(result: dict[str, Any]) -> None:
    telemetry = result["telemetry"]
    if telemetry is None:
        print(f"  {result['status']}: {result['validation_error']} | {result['wall_time_s']:.0f}s")
        return
    try:
        verified = telemetry.get("claims_verified", 0)
        weak = telemetry.get("claims_weak", 0)
        unverified = telemetry.get("claims_unverified", 0)
        total = verified + weak + unverified
        uvr_label = f"{result['uvr']:.2f}" if result["uvr"] is not None else "N/A"
        print(f"  cost=${telemetry['total_cost_usd']:.4f} | "
              f"grounding={telemetry.get('grounding_score', 0):.2f} | "
              f"reflection={telemetry.get('reflection_score', 0)} | "
              f"claims={total} (v={verified} w={weak} u={unverified}) | uvr={uvr_label} | "
              f"verification={result['verification_status']} | "
              f"evaluation={result['evaluation_status']} | {result['wall_time_s']:.0f}s")
    except (KeyError, TypeError, ValueError):
        print("  telemetry formatting deferred to fail-closed release validation")
    if result["validation_error"]:
        print(f"    ↳ validation: {result['validation_error']}")


@click.command()
@click.option("--mode", required=True, type=click.Choice(["smoke", "release"]))
@click.option("--limit", default=None, type=int, help="Smoke mode: run first N manifest topics")
@click.option("--id", "topic_id", default=None, type=int, help="Smoke mode: run single topic by id")
@click.option("--gate", is_flag=True, default=False, help="Enable gate checks for selected mode")
@click.option(
    "--expected-code-sha",
    default=None,
    type=str,
    help="Release mode: exact 40-char lowercase git SHA for qualification",
)
def run_benchmark(mode, limit, topic_id, gate, expected_code_sha):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gate_failures: list[str] = []
    release_qualification = "NON_RELEASE"
    contract: dict[str, Any] | None = None
    manifest_identity: dict[str, Any] | None = None
    preflight_identity: dict[str, Any] | None = None
    final_identity: dict[str, Any] | None = None
    evaluation_config, evaluation_config_sha256 = resolve_evaluation_config()
    child_env = _child_environment(evaluation_config)
    github_identity = _github_actions_identity()

    if mode == "release":
        if topic_id is not None or limit is not None:
            _fail_preflight("release mode forbids --id and --limit")
        if not gate:
            _fail_preflight("release mode requires --gate")
        if expected_code_sha is None:
            _fail_preflight("release mode requires --expected-code-sha")
        gate_failures.extend(_validate_github_release_identity(expected_code_sha))
        preflight_identity = _resolve_code_identity()
        gate_failures.extend(_validate_code_identity(expected_code_sha, preflight_identity))
        contract = load_release_contract()
        topics, manifest_identity = _select_release_topics(contract)
    else:
        if expected_code_sha is not None:
            _fail_preflight("smoke mode forbids --expected-code-sha")
        contract = load_release_contract()
        manifest_topics, _ = load_validated_manifest(contract)
        topics = _select_smoke_topics(manifest_topics, topic_id=topic_id, limit=limit)
        manifest_identity = {
            "manifest_id": contract["manifest_id"],
            "manifest_path": contract["manifest_path"],
            "manifest_sha256": contract["manifest_sha256"],
            "selected_topic_count": len(topics),
            "selected_topic_ids": [topic["id"] for topic in topics],
        }

    if gate_failures:
        for failure in gate_failures:
            click.echo(f"PREFLIGHT FAIL: {failure}", err=True)
        raise SystemExit(2)

    print(f"Benchmark — {len(topics)} topics — {timestamp} — mode={mode}")
    print(f"{'─' * 60}")

    results: list[dict[str, Any]] = []
    for topic in topics:
        print(f"\n[{topic['id']:02d}/{len(topics)}] {topic['topic']}")
        result = _run_topic(topic, child_env=child_env)
        results.append(result)
        _print_result_summary(result)

    aggregate = _aggregate_metrics(results)
    valid_count = aggregate["successful"]
    scorable_count = len([result for result in results if result["uvr"] is not None])

    if mode == "release":
        final_identity = _resolve_code_identity()
        gate_failures.extend(_validate_code_identity(expected_code_sha, final_identity))
        if final_identity["head_sha"] != preflight_identity["head_sha"]:
            gate_failures.append("final HEAD drifted from preflight HEAD")
        gate_failures.extend(
            _validate_release_units(
                results,
                expected_topic_ids=contract["ordered_topic_ids"],
                evaluation_config=evaluation_config,
            ),
        )
        release_qualification = "PASS" if not gate_failures else "FAIL"
    elif gate:
        gate_failures.extend(_validate_smoke_gate(results))
        release_qualification = "NON_RELEASE"

    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "release_qualification": release_qualification,
        "gate_requested": gate,
        "github_actions_identity": github_identity,
        "expected_code_sha": expected_code_sha,
        "preflight_code_identity": preflight_identity,
        "final_code_identity": final_identity,
        "release_contract_identity": {
            "manifest_id": contract["manifest_id"],
            "schema_version": contract["schema_version"],
            "manifest_path": contract["manifest_path"],
            "manifest_sha256": contract["manifest_sha256"],
            "expected_topic_count": contract["expected_topic_count"],
            "ordered_topic_ids": contract["ordered_topic_ids"],
        },
        "selected_manifest_identity": manifest_identity,
        "evaluation_configuration": evaluation_config,
        "evaluation_config_sha256": evaluation_config_sha256,
        "ordered_unit_results": results,
        "aggregate_metrics": aggregate,
        "gate_failures": gate_failures,
    }

    out_path = Path(f"outputs/benchmark_results/benchmark_{timestamp}.json")
    written_evidence: dict[str, Any]
    if mode == "release" and release_qualification == "PASS":
        try:
            context = _build_trusted_release_context(
                expected_code_sha=expected_code_sha,
                preflight_github_identity=deepcopy(github_identity),
                preflight_code_identity=deepcopy(preflight_identity),
                preflight_contract=deepcopy(contract),
                preflight_manifest_identity=deepcopy(manifest_identity),
                preflight_evaluation_config=deepcopy(evaluation_config),
                results=results,
                gate_failures=gate_failures,
            )
            evidence = _trusted_release_payload(context)
            written_evidence = _write_evidence_report(
                evidence, out_path, trusted_context=context,
            )
        except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
            # A failed finalization must never leave an apparently valid PASS
            # artifact.  Do not retain untrusted result/telemetry content here.
            out_path.unlink(missing_ok=True)
            gate_failures.append("trusted finalization failed")
            release_qualification = "FAIL"
            final_identity = _resolve_code_identity()
            evidence = _sanitized_release_fail_evidence(
                expected_code_sha=expected_code_sha,
                github_identity=_github_actions_identity(),
                preflight_identity=preflight_identity,
                final_identity=final_identity,
                manifest_identity=manifest_identity,
                gate_failures=gate_failures,
            )
            try:
                written_evidence = _write_evidence_report(evidence, out_path)
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                out_path.unlink(missing_ok=True)
                raise SystemExit(1) from error
    else:
        try:
            written_evidence = _write_evidence_report(evidence, out_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            if mode != "release":
                raise
            out_path.unlink(missing_ok=True)
            gate_failures.append("release failure evidence sanitized")
            release_qualification = "FAIL"
            written_evidence = _write_evidence_report(
                _sanitized_release_fail_evidence(
                    expected_code_sha=expected_code_sha,
                    github_identity=_github_actions_identity(),
                    preflight_identity=preflight_identity,
                    final_identity=_resolve_code_identity(),
                    manifest_identity=manifest_identity,
                    gate_failures=gate_failures,
                ),
                out_path,
            )

    print(f"\n{'═' * 60}")
    print("Benchmark Complete")
    print(f"  Valid : {valid_count}/{aggregate['total_runs']}")
    print(f"  Scorable UVR runs: {scorable_count}/{valid_count}")
    print(f"  Mean unverified rate: {aggregate['mean_unverified_rate'] if scorable_count else 'N/A'}")
    print(f"  Release qualification: {release_qualification}")
    print(f"  Report: {out_path}")

    if gate:
        if gate_failures:
            print("\nCI GATE: FAIL")
            for failure in gate_failures:
                print(f"  - {failure}")
            raise SystemExit(1)
        if mode == "smoke":
            print("\nSMOKE GATE: PASS — NON-RELEASE")
        else:
            print("\nRELEASE GATE: PASS")
            print(f"  evidence_sha256={written_evidence['evidence_sha256']}")


if __name__ == "__main__":
    run_benchmark()
