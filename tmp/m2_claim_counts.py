"""tmp/m2_claim_counts.py — decompose UVR into claim-count movements (no new runs)."""
import json
from pathlib import Path
from statistics import mean
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
RUNS = Path("outputs/runs")
B = Path("outputs/m2/baseline_ids.txt")
T = Path("outputs/m2/treatment_ids.txt")

def ids(p):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

def load(idlist, arm):
    out = []
    for rid in idlist:
        f = RUNS / f"{rid}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        v = d.get("claims_verified", 0); w = d.get("claims_weak", 0); u = d.get("claims_unverified", 0)
        tot = v + w + u
        out.append({"arm": arm, "topic": d.get("topic", "?"),
                    "v": v, "w": w, "u": u, "total": tot,
                    "uvr": (u / tot) if tot else 0})
    return out

def agg(rows, topic, arm):
    s = [r for r in rows if r["topic"] == topic and r["arm"] == arm]
    if not s:
        return None
    return {k: mean([r[k] for r in s]) for k in ("v", "w", "u", "total", "uvr")}

rows = load(ids(B), "baseline") + load(ids(T), "treatment")
topics = sorted({r["topic"] for r in rows})
print(f"{'topic':<32}{'arm':<10}{'total':>6}{'ver':>5}{'weak':>5}{'unv':>5}{'uvr':>6}")
print("-" * 72)
for t in topics:
    b = agg(rows, t, "baseline"); tr = agg(rows, t, "treatment")
    for arm, a in (("baseline", b), ("treatment", tr)):
        if a:
            print(f"{t:<32}{arm:<10}{a['total']:>6.1f}{a['v']:>5.1f}{a['w']:>5.1f}{a['u']:>5.1f}{a['uvr']:>6.2f}")
    if b and tr:
        print(f"  -> delta: total {tr['total']-b['total']:+.1f}, verified {tr['v']-b['v']:+.1f}, "
              f"unverified {tr['u']-b['u']:+.1f}")
    print("-" * 72)
print("\nREAD:")
print("  total UP + unverified UP        -> denser drafts; marginal specific claims unsupported (over-claim via volume)")
print("  verified DOWN + unverified UP   -> source-awareness replaced groundable generic claims with ungroundable specific ones")
print("  verified FLAT + unverified UP   -> pure addition of unsupported claims (over-claiming)")
