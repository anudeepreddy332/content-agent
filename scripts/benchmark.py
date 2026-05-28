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

@click.command()
@click.option("--limit", default=None, type=int, help="Run first N topics only")
@click.option("--id", "topic_id", default=None, type=int, help="Run single topic by id")
def run_benchmark(limit, topic_id):
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
            capture_output=True, text=True, timeout=300,
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
            print(f"  cost=${telemetry['total_cost_usd']:.4f} | "
                  f"grounding={telemetry.get('grounding_score', 0):.2f} | "
                  f"reflection={telemetry.get('reflection_score', 0)} | "
                  f"hitl={telemetry.get('hitl_status')} | "
                  f"git={telemetry.get('git_status')} | "
                  f"{elapsed:.0f}s")
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
        "runs": results,
    }

    out_path = Path(f"outputs/benchmark_{timestamp}.json")
    out_path.write_text(json.dumps(aggregate, indent=2))

    print(f"\n{'═' * 60}")
    print(f"Benchmark Complete")
    print(f"  Successful : {aggregate['successful']}/{aggregate['total_runs']}")
    print(f"  Mean cost  : ${aggregate['mean_cost_usd']:.5f}")
    print(f"  Mean grounding: {aggregate['mean_grounding']:.2f}")
    print(f"  Mean reflection: {aggregate['mean_reflection']:.1f}")
    print(f"  Mean HTML errors/run: {aggregate['mean_html_errors']:.1f}")
    print(f"  Report: {out_path}")

    # Gate report trigger
    if len(successful) >= 10:
        print(f"\nGate criteria met (10+ runs). Write phase4a_gate_report.md.")


if __name__ == "__main__":
    run_benchmark()
