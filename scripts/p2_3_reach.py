"""P2.3 Phase 0 — does raising MAX_ITERATIONS have any population to act on?
Counts runs that revised to iteration 2 and were STILL over-claiming (last-iteration
UVR > 0.15) — the only runs a 3rd pass could touch. Read-only, $0."""
import json
from pathlib import Path
from collections import defaultdict

two_iter = lever_eligible = 0
topics = defaultdict(int)
for f in Path("outputs/runs").glob("*.json"):
    d = json.loads(f.read_text())
    im = d.get("iteration_metrics") or []
    if len(im) < 2:
        continue
    two_iter += 1
    if (im[-1].get("uvr", 0) > 0.15):          # still over-claiming after the 2nd verify
        lever_eligible += 1
        topics[d["topic"]] += 1

print(f"runs that revised to iteration 2         : {two_iter}")
print(f"  of those, still UVR>0.15 (lever-eligible): {lever_eligible}")
for t, n in sorted(topics.items(), key=lambda x: -x[1]):
    print(f"    {n:>2}  {t}")
print("\nPRE-REGISTERED: if lever-eligible < 3, the 3rd iteration has ~no population under the")
print("current gate -> CONCLUDE no-adopt, SKIP Phase 1. Otherwise run Phase 1 on these topics.")