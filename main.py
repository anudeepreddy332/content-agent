"""
Content agent CLI entry point

Usage:

# Minimal — drops straight into Learning Log
    uv run python main.py run --topic "Gradient Descent"

# With series (for supervised-learning-models.html cards)
    uv run python main.py run --topic "Linear & Logistic Regression" \
      --card-id "01-A" \
      --series "Family 01 — Linear Models · supervised-learning-models.html"

# Benchmark mode
uv run python main.py run --topic "Gradient Descent" --auto

"""

import os
import click
import uuid
import json
from pathlib import Path
from agent.graph import build_graph
from config import PROMPT_VERSION


def _write_telemetry(state: dict):
    """Write telemetry results to outputs/runs/<run_id>.json."""
    run_id = state.get("run_id", "unknown")
    out_path = Path(f"outputs/runs/{run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "topic": state.get("topic"),
        "slug": state.get("slug"),
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "prompt_version": state.get("prompt_version", "unknown"),
        "iterations": state.get("iterations", 0),
        "reflection_score": state.get("reflection_score"),
        "grounding_score": state.get("grounding_score"),
        "hitl_status": state.get("hitl_status"),
        "git_status": state.get("git_status"),
        "total_tokens": state.get("total_tokens", 0),
        "total_cost_usd": round(state.get("total_cost_usd", 0), 5),
        "latency_ms": state.get("latency_ms", {}),
        "error_log": state.get("error_log", []),
        "claims_verified": sum(1 for r in state.get("grounding_report", [])
                               if r.get("status") == "verified"),
        "claims_weak": sum(1 for r in state.get("grounding_report", [])
                           if r.get("status") == "weak"),
        "claims_unverified": sum(1 for r in state.get("grounding_report", [])
                                 if r.get("status") == "unverified"),
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path

@click.group()
def cli():
    pass

@cli.command()
@click.option("--topic", required=True, help="Article topic")
@click.option("--card-id", default="standalone",
              show_default=True,
              help="Card ID e.g. 01-A. Defaults to 'standalone' for Learning Log articles.")
@click.option("--series", default="Learning Log",
              show_default=True,
              help="Series context string. Defaults to 'Learning Log'.")

@click.option("--auto", is_flag=True, default=False,
              help="Auto-approve HITL (benchmark mode only)")

def run(topic, card_id, series, auto):
    # Validate required credentials before doing any work.
    # Fail here instead of mid-pipeline with a cryptic API error.
    missing = [v for v in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY") if not os.getenv(v)]
    if missing:
        raise click.UsageError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in the missing values."
        )

    os.environ["HITL_AUTO_APPROVE"] = "1" if auto else "0"

    slug = (
        topic.lower()
        .replace(" ", "-")
        .replace("&", "and")
        .replace(",", "")
        .replace("—", "")
        .strip("-")
    )
    run_id = str(uuid.uuid4())

    initial_state = {
        "topic": topic,
        "slug": slug,
        "card_id": card_id,
        "series_context": series,
        "draft_sections": {},
        "draft_markdown": "",
        "web_sources": [],
        "kb_results": [],
        "grounding_report": [],
        "grounding_score": 0.0,
        "reflection_score": 0,
        "reflection_notes": "",
        "iterations": 0,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "html_output": None,
        "html_filename": None,
        "branch_name": None,
        "git_status": None,
        "run_id": run_id,
        "prompt_version": PROMPT_VERSION,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "latency_ms": {},
        "error_log": [],
    }

    graph = build_graph()
    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        initial_state["error_log"] = [f"pipeline crash: {e}"]
        _write_telemetry(initial_state)
        raise

    telemetry_path = _write_telemetry(result)
    click.echo(f"\nRun complete. Telemetry: {telemetry_path}")
    click.echo(
        f"Cost: ${result['total_cost_usd']:.4f} | "
        f"Grounding: {result['grounding_score']:.2f} | "
        f"Git: {result.get('git_status', 'n/a')}"
    )
    click.echo(f"RUN_ID={result['run_id']}")


if __name__ == "__main__":
    cli()