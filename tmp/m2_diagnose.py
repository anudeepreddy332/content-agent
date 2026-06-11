"""tmp/m2_diagnose.py — mechanism decomposition for the M2 verifier hypothesis."""
import json
from pathlib import Path
from statistics import mean

RUNS = Path("outputs/runs")
BASELINE = Path("outputs/m2/baseline_ids.txt")
TREATMENT = Path("outputs/m2/treatment_ids.txt")

def load_ids(p):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

def rows(ids, arm):
    out = []
    for rid in ids:
        f = RUNS / f"{rid}.json"
        if not f.exists():
            print(f"  [warn] missing {rid}"); continue
        t = json.loads(f.read_text())
        v = t.get("claims_verified", 0); w = t.get("claims_weak", 0); u = t.get("claims_unverified", 0)
        total = v + w + u
        bd = t.get("grounding_breakdown", {})
        out.append({
            "arm": arm, "topic": t.get("topic", "?"),
            "total_claims": total,
            "verified": v, "weak": w, "unverified": u,
            "uvr": (u/total) if total else 0,
            "unv_no_source": bd.get("unverified_no_source", 0),
            "unv_has_source": bd.get("unverified_has_source", 0),
            "mean_conf_unv": bd.get("mean_confidence_unverified", 0),
            "mean_conf_ver": bd.get("mean_confidence_verified", 0),
        })
    return out

def agg(rs, topic, arm):
    s = [r for r in rs if r["topic"] == topic and r["arm"] == arm]
    if not s: return None
    return {k: mean([r[k] for r in s]) for k in
            ("total_claims","verified","weak","unverified","uvr",
             "unv_no_source","unv_has_source","mean_conf_unv","mean_conf_ver")}

def main():
    rs = rows(load_ids(BASELINE), "baseline") + rows(load_ids(TREATMENT), "treatment")
    topics = sorted({r["topic"] for r in rs})
    print(f"\n{'topic':<34}{'arm':<10}{'claims':>7}{'uvr':>6}{'no_src':>7}{'has_src':>8}{'conf_unv':>9}")
    print("-"*81)
    for topic in topics:
        for arm in ("baseline","treatment"):
            a = agg(rs, topic, arm)
            if a:
                print(f"{topic:<34}{arm:<10}{a['total_claims']:>7.1f}{a['uvr']:>6.2f}"
                      f"{a['unv_no_source']:>7.1f}{a['unv_has_source']:>8.1f}{a['mean_conf_unv']:>9.2f}")
        print("-"*81)
    print("\nREAD:")
    print("  - If treatment unv_HAS_src rose  -> verifier found the source but scored low")
    print("    => strictness / paraphrase mismatch (verifier problem, prompt-fixable).")
    print("  - If treatment unv_NO_src rose   -> verifier could not attach a source it HAD")
    print("    => extraction/matching failure (verifier problem).")
    print("  - If treatment total_claims rose -> denser drafts, more surface to fail (density).")
    print("  - If mean_conf_unv is mid (0.3-0.6), the verifier is hedging, not cleanly rejecting.")

if __name__ == "__main__":
    main()