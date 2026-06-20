"""Assert every required telemetry field is present in a fresh run JSON."""
import sys
import json
import subprocess
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
    "hitl_status", "git_status",
]

topic = "Gradient Descent"
proc = subprocess.run(
    ["uv", "run", "python", "main.py", "run",
     "--topic", topic, "--card-id", "standalone", "--series", "Test", "--auto"],
    capture_output=True, text=True, timeout=300,
)

runs = sorted(Path("outputs/runs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
data = json.loads(runs[0].read_text())

missing = [f for f in REQUIRED_FIELDS if f not in data]
if missing:
    print(f"FAIL: missing telemetry fields: {missing}")
    sys.exit(1)

# Check web_sources_count is a positive integer
if not isinstance(data.get("web_sources_count"), int) or data["web_sources_count"] < 0:
    print(f"FAIL: web_sources_count is invalid: {data.get('web_sources_count')}")
    sys.exit(1)

# M5 correctness checks — presence alone passed the broken retrieve_kb field.
attr = data.get("attribution", {})
report = data.get("grounding_report", [])
if report and sum(attr.get(k, 0) for k in ("web", "kb", "none", "unresolved")) != len(report):
    print(f"FAIL: attribution counts {attr} do not sum to claim count {len(report)}")
    sys.exit(1)
if report and any("source_kind" not in r for r in report):
    print("FAIL: grounding_report entries missing source_kind")
    sys.exit(1)
if any("chunk_index" not in k for k in data.get("kb_results", [])):
    print("FAIL: kb_results entries missing chunk_index")
    sys.exit(1)


print("PASS: all required telemetry fields present")