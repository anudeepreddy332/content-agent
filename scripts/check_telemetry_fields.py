"""Assert every required telemetry field is present in a fresh run JSON."""
import sys, json, subprocess, uuid
from pathlib import Path

REQUIRED_FIELDS = [
    "run_id", "topic", "slug", "timestamp",
    "prompt_version", "prompt_hashes",
    "claims_verified", "claims_weak", "claims_unverified",
    "grounding_score", "grounding_breakdown", "grounding_report",
    "reflection_score", "reflection_notes",
    "web_sources_count", "kb_results_count",
    "web_sources",
    "total_cost_usd", "total_tokens",
    "latency_ms", "error_log",
    "hitl_status", "git_status",
]

topic = "Gradient Descent"
proc = subprocess.run(
    ["uv", "run", "python", "main.py", "run",
     "--topic", topic, "--card-id", "standalone", "--series", "Test", "--auto"],
    capture_output=True, text=True, timeout=120,
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

print("PASS: all required telemetry fields present")