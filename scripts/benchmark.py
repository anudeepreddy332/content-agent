"""
Runs all 20 topics from evals/topics.json through the full pipeline.
Writes per-run JSON to outputs/runs/ and aggregate to outputs/benchmark_<timestamp>.json.

Usage:
    uv run python scripts/benchmark.py               # all 20 topics
    uv run python scripts/benchmark.py --limit 5     # first 5 only
    uv run python scripts/benchmark.py --id 3        # single topic by id

"""

import sys, json, time, click, subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROUNDING_FLOOR

@click.command()
@click.option("--limit", default=None, type=int, help="Run first N topics only")
@click.option("--id", "topic_id", default=None, type=int, help="Run single topic by id")
@click.option("--gate", is_flag=True, default=False,
              help="CI mode: exit 1 if any run fails or any run's UVR > 0.15")
def run_benchmark(limit, topic_id, gate):
    topics = json.loads(Path("evals/topics.json").read_text())

    if topic_id:
        topics = [t for t in topics if t["id"] == topic_id]
    elif limit:
        topics = topics[:limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"Benchmark — {len(topics)} topics — {timestamp}")
    print(f"{'─' * 60}")

    for t in topics:
        print(f"\n[{t['id']:02d}/{len(topics)}] {t['topic']}")
        t_start = time.time()

        # Run each topic via CLI subprocess so state is fully isolated
        proc = subprocess.run(
            [
                "uv", "run", "python", "main.py", "run",
                "--topic", t["topic"],
                "--card-id", t["card_id"],
                "--series", t["series"],
                "--auto" # auto-approve for benchmark
            ],
            capture_output=True, text=True, timeout=600,
        )

        elapsed = time.time() - t_start
        status = "success" if proc.returncode == 0 else "failed"

        # Parse run_id from subprocess stdout - deterministic, no timestamp guessing
        run_id = None
        for line in proc.stdout.splitlines():
            if line.startswith("RUN_ID="):
                run_id = line.split("=", 1)[1].strip()
                break

        telemetry = None
        if run_id:
            telemetry_path = Path(f"outputs/runs/{run_id}.json")
            if telemetry_path.exists():
                telemetry = json.loads(telemetry_path.read_text())

        elif proc.returncode == 0:
            # Fallback to mtime scan only if RUN_ID was missing from stdout
            run_files = sorted(Path("outputs/runs").glob("*.json"),
                               key=lambda x: x.stat().st_mtime, reverse=True)

            for rf in run_files[:3]:
                t_data = json.loads(rf.read_text())
                if t_data.get("topic") == t["topic"]:
                    telemetry = t_data
                    break

        result = {
            "id": t["id"],
            "topic": t["topic"],
            "status": status,
            "wall_time_s": round(elapsed, 1),
            "telemetry": telemetry,
            "stderr": proc.stderr[-500:] if proc.returncode != 0 else None,
        }
        results.append(result)

        if telemetry:
            v = telemetry.get("claims_verified", 0)
            w = telemetry.get("claims_weak", 0)
            u = telemetry.get("claims_unverified", 0)
            total = v + w + u
            uvr = u / total if total else 0
            print(f"  cost=${telemetry['total_cost_usd']:.4f} | "
                  f"grounding={telemetry.get('grounding_score', 0):.2f} | "
                  f"reflection={telemetry.get('reflection_score', 0)} | "
                  f"claims={total} (v={v} w={w} u={u}) | uvr={uvr:.2f} | "
                  f"hitl={telemetry.get('hitl_status')} | "
                  f"git={telemetry.get('git_status')} | "
                  f"{elapsed:.0f}s")
            bd = telemetry.get("grounding_breakdown", {})
            if bd and telemetry.get("grounding_score", 1.0) < 0.75:
                print(f"    ↳ unverified: {telemetry.get('claims_unverified', 0)} claims  "
                      f"({bd.get('unverified_no_source', 0)} no-source / "
                      f"{bd.get('unverified_has_source', 0)} has-source) | "
                      f"conf_unverified={bd.get('mean_confidence_unverified', 0):.2f}")
        else:
            print(f"  {status} (no telemetry found) | {elapsed:.0f}s")

    # Aggregate report
    successful = [r for r in results if r["status"] == "success" and r["telemetry"]]
    aggregate = {
        "timestamp": timestamp,
        "total_runs": len(results),
        "successful": successful,
        "failed": len(results) - len(successful),
        "mean_cost_usd": round(sum(r["telemetry"]["total_cost_usd"] for r in successful) / max(len(successful), 1), 5),
        "mean_grounding": round(sum(r["telemetry"].get("grounding_score", 0)
                                    for r in successful) / max(len(successful), 1), 3),
        "mean_reflection": round(sum(r["telemetry"].get("reflection_score", 0)
                                     for r in successful) / max(len(successful), 1), 1),
        "mean_wall_time_s": round(sum(r["wall_time_s"] for r in results) / len(results), 1),
        "mean_html_errors": round(
            sum(len(r["telemetry"].get("error_log", [])) for r in successful)
            / max(len(successful), 1),
            1
        ),
        "mean_unverified_rate": round(
            sum(
                r["telemetry"].get("claims_unverified", 0) /
                max(
                    r["telemetry"].get("claims_verified", 0) +
                    r["telemetry"].get("claims_weak", 0) +
                    r["telemetry"].get("claims_unverified", 0),
                    1,
                )
                for r in successful
            ) / max(len(successful), 1),
            3,
        ),
        "runs_below_grounding_floor": sum(
            1 for r in successful
            if r["telemetry"].get("grounding_score", 1.0) < GROUNDING_FLOOR
        ),
        "runs": results,
    }

    out_path = Path(f"outputs/benchmark_results/benchmark_{timestamp}.json")
    out_path.write_text(json.dumps(aggregate, indent=2))

    print(f"\n{'═' * 60}")
    print(f"Benchmark Complete")
    print(f"  Successful : {len(successful)}/{aggregate['total_runs']}")
    print(f"  Mean cost  : ${aggregate['mean_cost_usd']:.5f}")
    print(f"  Mean grounding: {aggregate['mean_grounding']:.2f}")
    print(f"  Mean unverified rate: {aggregate['mean_unverified_rate']:.1%}")
    print(f"  Runs below floor ({GROUNDING_FLOOR:.0%}): {aggregate['runs_below_grounding_floor']}/{len(successful)}")
    print(f"  Mean reflection: {aggregate['mean_reflection']:.1f}")
    print(f"  Mean HTML errors/run: {aggregate['mean_html_errors']:.1f}")
    print(f"  Report: {out_path}")

    # B3 CI gate. Gates: zero outright failures + per-run UVR <= 0.15 (the locked
    # grounding gate). SV is deliberately NOT gated here — at n=3 the +/-6-7 noise
    # band (DECISIONS) would make an SV threshold flake; it is reported only.
    if gate:
        gate_failures = []
        if aggregate["failed"] > 0:
            gate_failures.append(f"{aggregate['failed']} run(s) failed outright")
        for r in successful:
            t = r["telemetry"]
            tot = (t.get("claims_verified", 0) + t.get("claims_weak", 0)
                   + t.get("claims_unverified", 0))
            uvr = t.get("claims_unverified", 0) / tot if tot else 0.0
            if uvr > 0.15:
                gate_failures.append(f"topic {r['id']:02d} UVR {uvr:.2f} > 0.15")
        if gate_failures:
            print("\nCI GATE: FAIL")
            for gf in gate_failures:
                print(f"  - {gf}")
            sys.exit(1)
        print("\nCI GATE: PASS — all runs succeeded, all UVR <= 0.15")

    # Gate report trigger
    if len(successful) >= 10:
        print(f"\nGate criteria met (10+ runs). Write phase4a_gate_report.md.")


if __name__ == "__main__":
    run_benchmark()
