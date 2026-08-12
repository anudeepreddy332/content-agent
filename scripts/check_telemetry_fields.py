"""Validate fields in telemetry emitted by one exact successful CLI run."""
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_FIELDS = [
    "run_id", "topic", "slug", "timestamp",
    "prompt_version", "prompt_hashes",
    "iteration_metrics", "experiment_flags",
    "claims_verified", "claims_weak", "claims_unverified",
    "grounding_score", "grounding_breakdown", "grounding_report",
    "reflection_score", "reflection_notes",
    "web_sources_count", "kb_results_count",
    "web_sources", "kb_results", "attribution",
    "total_cost_usd", "total_tokens",
    "latency_ms", "error_log",
    "hitl_status", "git_status", "verification_status",
]


class TelemetryValidationError(ValueError):
    """The CLI result cannot be bound to valid telemetry for this invocation."""


def extract_run_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            return run_id or None
    return None


def load_exact_telemetry(run_id: str, topic: str, runs_dir: Path = Path("outputs/runs")) -> dict:
    telemetry_path = runs_dir / f"{run_id}.json"
    if not telemetry_path.is_file():
        raise TelemetryValidationError(f"telemetry missing for emitted RUN_ID={run_id}")
    try:
        data = json.loads(telemetry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TelemetryValidationError(f"telemetry unreadable for emitted RUN_ID={run_id}: {error}") from error
    if data.get("run_id") != run_id:
        raise TelemetryValidationError("telemetry run_id does not match emitted RUN_ID")
    if data.get("topic") != topic:
        raise TelemetryValidationError("telemetry topic does not match requested topic")
    return data


def telemetry_validation_error(data: dict) -> str | None:
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return f"missing telemetry fields: {missing}"
    if not isinstance(data.get("web_sources_count"), int) or data["web_sources_count"] < 0:
        return f"web_sources_count is invalid: {data.get('web_sources_count')}"

    attr = data.get("attribution", {})
    report = data.get("grounding_report", [])
    if report and sum(attr.get(kind, 0) for kind in ("web", "kb", "none", "unresolved")) != len(report):
        return f"attribution counts {attr} do not sum to claim count {len(report)}"
    if report and any("source_kind" not in verdict for verdict in report):
        return "grounding_report entries missing source_kind"
    if any("chunk_index" not in chunk for chunk in data.get("kb_results", [])):
        return "kb_results entries missing chunk_index"
    return None


def run_check(topic: str = "Gradient Descent", runs_dir: Path = Path("outputs/runs")) -> int:
    proc = subprocess.run(
        ["uv", "run", "python", "main.py", "run",
         "--topic", topic, "--card-id", "standalone", "--series", "Test", "--auto"],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        print(f"FAIL: CLI exited with status {proc.returncode}")
        return 1

    run_id = extract_run_id(proc.stdout)
    if run_id is None:
        print("FAIL: successful CLI output did not emit RUN_ID")
        return 1
    try:
        data = load_exact_telemetry(run_id, topic, runs_dir)
    except TelemetryValidationError as error:
        print(f"FAIL: {error}")
        return 1

    error = telemetry_validation_error(data)
    if error:
        print(f"FAIL: {error}")
        return 1
    print("PASS: exact-run telemetry fields and identity validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_check())
