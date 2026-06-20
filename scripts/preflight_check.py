"""Assert retrieval health before trusting any A/B experiment."""
import sys
import json
import subprocess
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Module identity guard (M4 post-mortem): imports must resolve to this source tree,
# not a stale installed copy (.venv shadowing voided a full M4 round).
import inspect
import tools.web_search as _ws
import agent.nodes as _an
_root = Path(__file__).resolve().parent.parent
for _m in (_ws, _an):
    _f = Path(inspect.getsourcefile(_m)).resolve()
    if _root not in _f.parents:
        print(f"FAIL: {_m.__name__} resolves to {_f} — stale installed copy shadows the source tree")
        sys.exit(1)
print("module identity OK")


# Run one quick smoke topic with auto-approve
run_id = str(uuid.uuid4())[:8]
topic = "Gradient Descent"
proc = subprocess.run(
    ["uv", "run", "python", "main.py", "run",
     "--topic", topic, "--card-id", "standalone", "--series", "Test", "--auto"],
    capture_output=True, text=True, timeout=120,
)
if proc.returncode != 0:
    print("FAIL: pipeline exited with error")
    sys.exit(1)

# Find the telemetry file for this run
runs = sorted(Path("outputs/runs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    print("FAIL: no telemetry found")
    sys.exit(1)

data = json.loads(runs[0].read_text())
web_count = data.get("web_sources_count", 0)
kb_count = data.get("kb_results_count", 0)
retrieve_ms = data.get("latency_ms", {}).get("retrieve", 0)

print(f"web_sources: {web_count}, kb_chunks: {kb_count}, retrieve_ms: {retrieve_ms}")

if web_count == 0:
    print("FAIL: web_sources_count is 0 — retrieval is broken")
    sys.exit(1)

print("PASS: retrieval healthy, ready for experiments")