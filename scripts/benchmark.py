"""Run benchmark topics and fail closed on unverifiable evaluation evidence."""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROUNDING_FLOOR
from scripts.check_telemetry_fields import (
    TelemetryValidationError,
    extract_run_id,
    load_exact_telemetry,
    telemetry_validation_error,
)


UVR_THRESHOLD = 0.15


def _claim_total(telemetry: dict) -> int:
    return sum(telemetry.get(field, 0) for field in (
        "claims_verified", "claims_weak", "claims_unverified",
    ))


def verification_outcome(telemetry: dict, topic: dict) -> dict:
    """Separate verifier completion from whether UVR is meaningful for this topic."""
    status = telemetry.get("verification_status", "unknown")
    total = _claim_total(telemetry)
    if status != "completed":
        return {
            "verification_status": status,
            "evaluation_status": "verification_incomplete",
            "uvr": None,
            "validation_error": f"verification_status={status}",
        }
    if total == 0:
        if topic.get("allow_zero_claims", False):
            return {
                "verification_status": status,
                "evaluation_status": "allowed_zero_claims",
                "uvr": None,
                "validation_error": None,
            }
        return {
            "verification_status": status,
            "evaluation_status": "unscorable_incomplete",
            "uvr": None,
            "validation_error": "zero verdicts without allow_zero_claims",
        }
    return {
        "verification_status": status,
        "evaluation_status": "scorable",
        "uvr": telemetry.get("claims_unverified", 0) / total,
        "validation_error": None,
    }


def _mean(values: list[float], digits: int) -> float | None:
    return round(sum(values) / len(values), digits) if values else None


@click.command()
@click.option("--limit", default=None, type=int, help="Run first N topics only")
@click.option("--id", "topic_id", default=None, type=int, help="Run single topic by id")
@click.option("--gate", is_flag=True, default=False,
              help="CI mode: fail on invalid verification evidence or UVR > 0.15")
def run_benchmark(limit, topic_id, gate):
    topics = json.loads(Path("evals/topics.json").read_text())
    if topic_id:
        topics = [topic for topic in topics if topic["id"] == topic_id]
    elif limit:
        topics = topics[:limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    print(f"Benchmark — {len(topics)} topics — {timestamp}")
    print(f"{'─' * 60}")

    for topic in topics:
        print(f"\n[{topic['id']:02d}/{len(topics)}] {topic['topic']}")
        started = time.time()
        proc = subprocess.run(
            ["uv", "run", "python", "main.py", "run",
             "--topic", topic["topic"], "--card-id", topic["card_id"],
             "--series", topic["series"], "--auto"],
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - started
        status = "success" if proc.returncode == 0 else "failed"
        telemetry = None
        validation_error = None
        run_id = extract_run_id(proc.stdout)

        if status == "success":
            if run_id is None:
                status = "failed"
                validation_error = "successful CLI output did not emit RUN_ID"
            else:
                try:
                    telemetry = load_exact_telemetry(run_id, topic["topic"])
                except TelemetryValidationError as error:
                    status = "failed"
                    validation_error = str(error)
                else:
                    validation_error = telemetry_validation_error(telemetry)
                    if validation_error:
                        status = "failed"

        outcome = (
            verification_outcome(telemetry, topic)
            if telemetry is not None else {
                "verification_status": "unknown",
                "evaluation_status": "telemetry_unavailable",
                "uvr": None,
                "validation_error": validation_error or f"CLI exited with status {proc.returncode}",
            }
        )
        if outcome["validation_error"] and validation_error is None:
            validation_error = outcome["validation_error"]

        result = {
            "id": topic["id"],
            "topic": topic["topic"],
            "status": status,
            "run_id": run_id,
            "wall_time_s": round(elapsed, 1),
            "telemetry": telemetry,
            "verification_status": outcome["verification_status"],
            "evaluation_status": outcome["evaluation_status"],
            "uvr": outcome["uvr"],
            "validation_error": validation_error,
            "stderr": proc.stderr[-500:] if proc.returncode != 0 else None,
        }
        results.append(result)

        if telemetry is None:
            print(f"  {status}: {validation_error} | {elapsed:.0f}s")
            continue
        verified = telemetry.get("claims_verified", 0)
        weak = telemetry.get("claims_weak", 0)
        unverified = telemetry.get("claims_unverified", 0)
        total = verified + weak + unverified
        uvr_label = f"{outcome['uvr']:.2f}" if outcome["uvr"] is not None else "N/A"
        print(f"  cost=${telemetry['total_cost_usd']:.4f} | "
              f"grounding={telemetry.get('grounding_score', 0):.2f} | "
              f"reflection={telemetry.get('reflection_score', 0)} | "
              f"claims={total} (v={verified} w={weak} u={unverified}) | uvr={uvr_label} | "
              f"verification={outcome['verification_status']} | "
              f"evaluation={outcome['evaluation_status']} | {elapsed:.0f}s")
        if validation_error:
            print(f"    ↳ validation: {validation_error}")

    valid = [result for result in results if result["status"] == "success" and not result["validation_error"]]
    scorable = [result for result in valid if result["uvr"] is not None]
    aggregate = {
        "timestamp": timestamp,
        "total_runs": len(results),
        "successful": valid,
        "failed": len(results) - len(valid),
        "unscorable": [result for result in valid if result["uvr"] is None],
        "mean_cost_usd": _mean([result["telemetry"]["total_cost_usd"] for result in valid], 5),
        "mean_grounding": _mean([result["telemetry"].get("grounding_score", 0) for result in valid], 3),
        "mean_reflection": _mean([result["telemetry"].get("reflection_score", 0) for result in valid], 1),
        "mean_wall_time_s": _mean([result["wall_time_s"] for result in results], 1),
        "mean_html_errors": _mean([len(result["telemetry"].get("error_log", [])) for result in valid], 1),
        "mean_unverified_rate": _mean([result["uvr"] for result in scorable], 3),
        "runs_below_grounding_floor": sum(
            1 for result in valid if result["telemetry"].get("grounding_score", 1.0) < GROUNDING_FLOOR
        ),
        "runs": results,
    }
    out_path = Path(f"outputs/benchmark_results/benchmark_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(f"\n{'═' * 60}")
    print("Benchmark Complete")
    print(f"  Valid : {len(valid)}/{aggregate['total_runs']}")
    print(f"  Scorable UVR runs: {len(scorable)}/{len(valid)}")
    print(f"  Mean unverified rate: {aggregate['mean_unverified_rate'] if scorable else 'N/A'}")
    print(f"  Report: {out_path}")

    if gate:
        gate_failures = []
        for result in results:
            if result["status"] != "success" or result["validation_error"]:
                gate_failures.append(f"topic {result['id']:02d}: {result['validation_error'] or 'CLI failed'}")
        for result in scorable:
            if result["uvr"] > UVR_THRESHOLD:
                gate_failures.append(f"topic {result['id']:02d} UVR {result['uvr']:.2f} > {UVR_THRESHOLD:.2f}")
        if gate_failures:
            print("\nCI GATE: FAIL")
            for failure in gate_failures:
                print(f"  - {failure}")
            raise SystemExit(1)
        print("\nCI GATE: PASS — all runs valid; all scorable UVR <= 0.15")


if __name__ == "__main__":
    run_benchmark()
