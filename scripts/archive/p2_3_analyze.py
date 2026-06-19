"""P2.3 Phase 1 — MAX_ITERATIONS 2 vs 3 on final SV/UVR. Groups by (topic, arm) via
experiment_flags.p2_3_max_iter. Reports final SV/UVR, mean iterations, and the fraction
of treatment runs that actually executed a 3rd pass (lever engagement). Read-only, $0."""
import json, sys
from pathlib import Path
from collections import defaultdict

min_ts = sys.argv[1] if len(sys.argv) > 1 else ""
cells, pv = defaultdict(list), set()
for f in Path("outputs/runs").glob("*.json"):
    d = json.loads(f.read_text())
    if d.get("timestamp", "") < min_ts:
        continue
    arm = (d.get("experiment_flags") or {}).get("p2_3_max_iter")
    im = d.get("iteration_metrics") or []
    if arm not in ("2", "3") or not im:
        continue
    if any(it.get("N", 0) == 0 for it in im):
        print(f"WARNING drop {d.get('run_id')} (N=0 parse dropout)"); continue
    cells[(d["topic"], arm)].append(im)
    pv.add(d.get("prompt_version"))

def mean(xs): return sum(xs) / len(xs) if xs else 0.0
print(f"{'topic':32} {'arm':>3} {'n':>2} {'SVf':>6} {'UVRf':>6} {'iters':>5} {'used3':>6}")
for (topic, arm), runs in sorted(cells.items()):
    print(f"{topic:32.32} {arm:>3} {len(runs):>2} "
          f"{mean([r[-1]['SV'] for r in runs]):>6.1f} {mean([r[-1]['uvr'] for r in runs]):>6.3f} "
          f"{mean([len(r) for r in runs]):>5.1f} {mean([1 if len(r) >= 3 else 0 for r in runs]):>6.0%}")
print(f"\nvalidity: prompt_versions={sorted(pv)} (must be exactly one == baseline sha-6687240c8cd8)")
print("PASS (adopt MAX=3): SV up >+7 vs control on >=3 topics, UVR not worse by >0.05, used3>=50% on those.")
print("REJECT: SV gain <=+7 on a majority; OR SV up but UVR worse >0.05 (vagueness); OR used3<50% (lever idle).")