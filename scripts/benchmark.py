"""Run benchmark topics and fail closed on unverifiable evaluation evidence."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
TOPIC_REQUIRED_FIELDS = ("id", "topic", "card_id", "series", "slug")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _fail_preflight(message: str) -> None:
    click.echo(f"Error: {message}", err=True)
    raise SystemExit(2)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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


def load_release_contract(path: Path = RELEASE_CONTRACT_PATH) -> dict[str, Any]:
    if not path.is_file():
        _fail_preflight(f"release contract missing: {path}")
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        _fail_preflight(f"release contract malformed: {error}")
    if not isinstance(contract, dict):
        _fail_preflight("release contract must be a JSON object")
    extra = set(contract) - RELEASE_CONTRACT_REQUIRED_FIELDS
    missing = RELEASE_CONTRACT_REQUIRED_FIELDS - set(contract)
    if extra:
        _fail_preflight(f"release contract has unexpected fields: {sorted(extra)}")
    if missing:
        _fail_preflight(f"release contract missing fields: {sorted(missing)}")
    if contract["schema_version"] != 1:
        _fail_preflight("release contract schema_version must be 1")
    ordered_ids = contract["ordered_topic_ids"]
    if not isinstance(ordered_ids, list) or not ordered_ids:
        _fail_preflight("ordered_topic_ids must be a nonempty list")
    if len(ordered_ids) != contract["expected_topic_count"]:
        _fail_preflight("ordered_topic_ids length must equal expected_topic_count")
    return contract


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
        topic_name = topic.get("topic")
        slug = topic.get("slug")
        if isinstance(topic_id, int):
            if topic_id in seen_ids:
                failures.append(f"duplicate manifest topic id: {topic_id}")
            seen_ids.add(topic_id)
        if isinstance(topic_name, str):
            if topic_name in seen_topics:
                failures.append(f"duplicate manifest topic name: {topic_name}")
            seen_topics.add(topic_name)
        if isinstance(slug, str):
            if slug in seen_slugs:
                failures.append(f"duplicate manifest slug: {slug}")
            seen_slugs.add(slug)
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
    if not isinstance(topics, list) or not topics:
        _fail_preflight("manifest must be a nonempty JSON array")
    manifest_failures = _validate_manifest_topics(topics)
    if manifest_failures:
        _fail_preflight("; ".join(manifest_failures))
    topic_ids = [topic["id"] for topic in topics]
    if len(topics) != contract["expected_topic_count"]:
        _fail_preflight(
            f"manifest topic count mismatch: expected {contract['expected_topic_count']}, got {len(topics)}",
        )
    if topic_ids != contract["ordered_topic_ids"]:
        _fail_preflight("manifest topic IDs do not match release contract order")
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

    outcome = (
        verification_outcome(telemetry, topic)
        if telemetry is not None else {
            "verification_status": "unknown",
            "evaluation_status": "telemetry_unavailable",
            "uvr": None,
            "validation_error": validation_error or f"CLI exited with status {proc.returncode}",
        }
    )
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
        if result["status"] == "success" and not result["validation_error"]
    ]
    scorable = [result for result in valid if result["uvr"] is not None]
    return {
        "total_runs": len(results),
        "successful": len(valid),
        "failed": len(results) - len(valid),
        "unscorable": len([result for result in valid if result["uvr"] is None]),
        "mean_cost_usd": _mean([result["telemetry"]["total_cost_usd"] for result in valid], 5),
        "mean_grounding": _mean([result["telemetry"].get("grounding_score", 0) for result in valid], 3),
        "mean_reflection": _mean([result["telemetry"].get("reflection_score", 0) for result in valid], 1),
        "mean_wall_time_s": _mean([result["wall_time_s"] for result in results], 1),
        "mean_html_errors": _mean([len(result["telemetry"].get("error_log", [])) for result in valid], 1),
        "mean_unverified_rate": _mean([result["uvr"] for result in scorable], 3),
        "runs_below_grounding_floor": sum(
            1 for result in valid if result["telemetry"].get("grounding_score", 1.0) < GROUNDING_FLOOR
        ),
    }


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

    result_ids = [result["id"] for result in results]
    if result_ids != expected_topic_ids:
        failures.append("release unit topic IDs are missing, extra, duplicate, or out of order")

    run_ids = [result.get("run_id") for result in results]
    if any(not run_id for run_id in run_ids):
        failures.append("every release unit must have a nonempty run ID")
    if len(set(run_ids)) != len(run_ids):
        failures.append("release run IDs must be unique")

    expected_prompt_version = evaluation_config["PROMPT_VERSION"]
    expected_prompt_hashes = evaluation_config["PROMPT_HASHES"]

    for result in results:
        label = f"topic {result['id']:02d}"
        if result["subprocess_exit_code"] != 0:
            failures.append(f"{label}: subprocess exit code {result['subprocess_exit_code']}")
            continue
        if result["status"] != "success" or result["validation_error"]:
            failures.append(f"{label}: {result['validation_error'] or 'CLI failed'}")
            continue
        telemetry = result["telemetry"]
        if telemetry is None:
            failures.append(f"{label}: telemetry unavailable")
            continue
        if telemetry.get("run_id") != result["run_id"]:
            failures.append(f"{label}: telemetry run_id mismatch")
        if telemetry.get("topic") != result["topic"]:
            failures.append(f"{label}: telemetry topic mismatch")
        if result["verification_status"] != "completed":
            failures.append(f"{label}: verification_status={result['verification_status']}")
        if result["evaluation_status"] != "scorable":
            failures.append(f"{label}: evaluation_status={result['evaluation_status']}")
        if result["uvr"] is None or _claim_total(telemetry) == 0:
            failures.append(f"{label}: unscorable or zero-verdict unit")
        if result["uvr"] is not None and result["uvr"] > UVR_THRESHOLD:
            failures.append(f"{label}: UVR {result['uvr']:.2f} > {UVR_THRESHOLD:.2f}")
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


def _validate_evidence_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("invalid evidence schema_version")
    digest = payload.get("evidence_sha256")
    if not digest:
        raise ValueError("missing evidence_sha256")
    body = dict(payload)
    body.pop("evidence_sha256", None)
    expected = _compute_evidence_digest(body)
    if digest != expected:
        raise ValueError("evidence_sha256 mismatch")


def _write_evidence_report(payload: dict[str, Any], out_path: Path) -> None:
    body = dict(payload)
    body["evidence_sha256"] = _compute_evidence_digest(body)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)
    reread = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_evidence_payload(reread)


def _print_result_summary(result: dict[str, Any]) -> None:
    telemetry = result["telemetry"]
    if telemetry is None:
        print(f"  {result['status']}: {result['validation_error']} | {result['wall_time_s']:.0f}s")
        return
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
    _write_evidence_report(evidence, out_path)

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
            print(f"  evidence_sha256={json.loads(out_path.read_text())['evidence_sha256']}")


if __name__ == "__main__":
    run_benchmark()
