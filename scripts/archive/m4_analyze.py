"""M4 analysis: per-cell means of iteration-1 vs iteration-2 verify metrics.

Usage:
    uv run python scripts/archive/m4_analyze.py [min_iso_timestamp]

Selects outputs/runs/*.json with experiment_flags.m4_force_revise == "1" and
exactly 2 iteration_metrics entries, optionally at/after min_iso_timestamp.
Prints per (topic, arm): n, mean SV1/SV2/dSV, mean UVR1/UVR2/dUVR, mean
feedback-claims on iteration 2, and validity-check summaries.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

min_ts = sys.argv[1] if len(sys.argv) > 1 else ""
cells = defaultdict(list)
prompt_versions, retrieve_ms_max = set(), 0

for f in Path("outputs/runs").glob("*.json"):
    d = json.loads(f.read_text())
    if d.get("timestamp", "") < min_ts:
        continue
    flags = d.get("experiment_flags") or {}
    im = d.get("iteration_metrics") or []
    if flags.get("m4_force_revise") != "1" or len(im) != 2:
        continue
    if any(it.get("N", 0) == 0 for it in im):
        print(f"WARNING: dropping run {d.get('run_id')} — verify parse dropout (N=0 iteration)")
        continue

    arm = "treatment" if flags.get("m4_grounding_feedback") == "1" else "control"
    cells[(d["topic"], arm)].append(im)
    prompt_versions.add(d.get("prompt_version"))
    web_ms = d.get("latency_ms", {}).get("retrieve_web")
    if web_ms is None:
        web_ms = d.get("latency_ms", {}).get("retrieve", 0)  # legacy runs: total only
    retrieve_ms_max = max(retrieve_ms_max, web_ms)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

print(f"{'topic':38} {'arm':10} {'n':>2} {'SV1':>6} {'SV2':>6} {'dSV':>6} "
      f"{'UVR1':>6} {'UVR2':>6} {'dUVR':>7} {'fb2':>5}")
for (topic, arm), runs in sorted(cells.items()):
    sv1 = mean([r[0]["SV"] for r in runs]); sv2 = mean([r[1]["SV"] for r in runs])
    u1  = mean([r[0]["uvr"] for r in runs]); u2  = mean([r[1]["uvr"] for r in runs])
    fb  = mean([r[1].get("m4_feedback_claims", 0) for r in runs])
    print(f"{topic:38.38} {arm:10} {len(runs):>2} {sv1:>6.1f} {sv2:>6.1f} "
          f"{sv2 - sv1:>+6.1f} {u1:>6.3f} {u2:>6.3f} {u2 - u1:>+7.3f} {fb:>5.1f}")

print(f"\nvalidity: prompt_versions={sorted(prompt_versions)} "
      f"(must be exactly one, equal to the M6a baseline)")
print(f"validity: max web-retrieve latency = {retrieve_ms_max} ms "
      f"(< 1000 under frozen cache; legacy runs report total retrieve incl. KB)")