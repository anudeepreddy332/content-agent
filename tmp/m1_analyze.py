"""
tmp/m1_analyze.py
-----------------
M1 freshness-baseline analysis. Read-only over outputs/runs/<id>.json.

Reads two id lists:
    outputs/m1/control_ids.txt    (one run_id per line, control arm)
    outputs/m1/treatment_ids.txt  (one run_id per line, treatment arm)

Primary metric: claim-level unverified-rate = claims_unverified / (verified + weak + unverified).
Topic and arm are derived from telemetry + which file the id came from.

Outputs a per-(topic, arm) table, per-topic control-vs-treatment delta,
healthy-control movement, plus validity columns:
    retrieve_ms  — low (~<300ms) means cache hit; high means live retrieval
    gate_fired   — True if error_log shows the score-gate forced a refresh
    n_claims     — total claims (draft-side; should be similar across arms)

Run:
    uv run python tmp/m1_analyze.py
"""

import json
from pathlib import Path
from statistics import mean, pstdev

RUNS = Path("outputs/runs")
CTRL = Path("outputs/m1/control_ids.txt")
TREAT = Path("outputs/m1/treatment_ids.txt")

FAILURE_TOPICS = {"CatBoost", "ReAct Agent Pattern",
                  "Embedding Models & Vector Search",
                  "Multi-Agent Systems — When and Why"}


def _load_ids(p: Path) -> list[str]:
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _row(run_id: str, arm: str) -> dict | None:
    f = RUNS / f"{run_id}.json"
    if not f.exists():
        print(f"  [warn] missing telemetry: {run_id}")
        return None
    t = json.loads(f.read_text())
    v = t.get("claims_verified", 0)
    w = t.get("claims_weak", 0)
    u = t.get("claims_unverified", 0)
    total = v + w + u
    bd = t.get("grounding_breakdown", {})
    err = t.get("error_log", []) or []
    return {
        "run_id": run_id,
        "arm": arm,
        "topic": t.get("topic", "?"),
        "n_claims": total,
        "unverified_rate": (u / total) if total else None,
        "unverified_no_source": bd.get("unverified_no_source", 0),
        "unverified_has_source": bd.get("unverified_has_source", 0),
        "grounding_score": t.get("grounding_score"),
        "retrieve_ms": (t.get("latency_ms", {}) or {}).get("retrieve"),
        "gate_fired": any("forced refresh" in str(e).lower() for e in err),
        "iterations": t.get("iterations"),
    }


def _agg(rows: list[dict], topic: str, arm: str):
    rs = [r for r in rows if r["topic"] == topic and r["arm"] == arm
          and r["unverified_rate"] is not None]
    if not rs:
        return None
    vals = [r["unverified_rate"] for r in rs]
    return {
        "n": len(rs),
        "mean_uvr": mean(vals),
        "std_uvr": pstdev(vals) if len(vals) > 1 else 0.0,
        "vals": [round(x, 3) for x in vals],
        "mean_retrieve_ms": mean([r["retrieve_ms"] for r in rs if r["retrieve_ms"] is not None]) if any(r["retrieve_ms"] is not None for r in rs) else None,
        "any_gate_fired": any(r["gate_fired"] for r in rs),
        "mean_claims": mean([r["n_claims"] for r in rs]),
    }


def main():
    rows = []
    for rid in _load_ids(CTRL):
        r = _row(rid, "control")
        if r:
            rows.append(r)
    for rid in _load_ids(TREAT):
        r = _row(rid, "treatment")
        if r:
            rows.append(r)

    if not rows:
        print("No telemetry found. Did the runs complete and were ids written?")
        return

    topics = sorted({r["topic"] for r in rows})

    print("\nM1 — claim-level unverified-rate (lower is better)\n" + "=" * 78)
    print(f"{'topic':<38}{'arm':<10}{'n':>2} {'mean':>6} {'std':>6}  {'retr_ms':>8} {'gate':>5}")
    print("-" * 78)

    deltas = {}
    for topic in topics:
        is_failure = topic in FAILURE_TOPICS
        c = _agg(rows, topic, "control")
        t = _agg(rows, topic, "treatment")
        for arm, a in (("control", c), ("treatment", t)):
            if a:
                gate = "YES" if a["any_gate_fired"] else "-"
                rms = f"{a['mean_retrieve_ms']:.0f}" if a["mean_retrieve_ms"] is not None else "?"
                print(f"{topic:<38}{arm:<10}{a['n']:>2} {a['mean_uvr']:>6.3f} {a['std_uvr']:>6.3f}  {rms:>8} {gate:>5}")
        if c and t:
            d = c["mean_uvr"] - t["mean_uvr"]   # positive = treatment improved (lower uvr)
            deltas[topic] = (d, is_failure, c, t)
        print("-" * 78)

    print("\nDelta (control_uvr - treatment_uvr); positive = freshness reduced unverified-rate\n" + "=" * 78)
    fail_deltas = []
    ctrl_deltas = []
    for topic, (d, is_failure, c, t) in deltas.items():
        tag = "FAILURE" if is_failure else "control-topic"
        print(f"  {topic:<40} delta={d:+.3f}   [{tag}]   "
              f"claims c/t={c['mean_claims']:.0f}/{t['mean_claims']:.0f}")
        (fail_deltas if is_failure else ctrl_deltas).append(d)

    print("\nVERDICT INPUTS\n" + "-" * 40)
    if fail_deltas:
        improved = sum(1 for d in fail_deltas if d >= 0.15)
        print(f"  failure topics with delta >= 0.15 : {improved}/{len(fail_deltas)}")
        print(f"  failure-topic deltas              : {[round(d,3) for d in fail_deltas]}")
    if ctrl_deltas:
        max_ctrl = max(abs(d) for d in ctrl_deltas)
        print(f"  max |delta| on control topics     : {max_ctrl:.3f}  (want < 0.05)")

    # Interpretation guard
    gate_in_control = any(
        _agg(rows, topic, "control") and _agg(rows, topic, "control")["any_gate_fired"]
        for topic in FAILURE_TOPICS if any(r["topic"] == topic for r in rows)
    )
    if gate_in_control:
        print("\n  [interpretation] Score-gate fired in at least one CONTROL failure-topic run.")
        print("  For those runs control already applied freshness, so a small delta does NOT")
        print("  mean freshness is useless — it means the existing gate already handles them.")
        print("  Treat that as: residual unverified is over-claiming -> go to M2.")

    print()


if __name__ == "__main__":
    main()