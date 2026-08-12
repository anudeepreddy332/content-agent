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
from config import PROMPT_VERSION, PROMPT_HASHES
from observability.logger import get_logger
from observability.tracing import setup_langsmith_tracing
import re
import datetime


def _write_telemetry(state: dict):
    """Write telemetry results to outputs/runs/<run_id>.json."""
    run_id = state.get("run_id", "unknown")
    out_path = Path(f"outputs/runs/{run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-compute slices once — used for both aggregate counts and breakdown
    report = state.get("grounding_report", [])
    unverified = [r for r in report if r.get("status") == "unverified"]
    verified = [r for r in report if r.get("status") == "verified"]


    record = {
        "run_id": run_id,
        "topic": state.get("topic"),
        "slug": state.get("slug"),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompt_version": state.get("prompt_version", "unknown"),

        # Per-file prompt hashes (M6a). Read from config, not state: the hashes
        # are a property of the checkout at write time, and this guarantees the
        # crash-path telemetry carries them too. Comparability rule: grounding/SV
        # metrics are comparable across runs iff verify_system hashes match.
        "prompt_hashes": PROMPT_HASHES,

        "iterations": state.get("iterations", 0),
        # M4: per-iteration verify metrics + experiment toggles. The independent
        # variable must be visible in every run record (preflight rule).
        "iteration_metrics": state.get("iteration_metrics", []),
        "m4_feedback_claims": state.get("m4_feedback_claims", 0),
        "experiment_flags": {
            "m4_force_revise": os.environ.get("M4_FORCE_REVISE", "0"),
            "m4_grounding_feedback": os.environ.get("M4_GROUNDING_FEEDBACK", "0"),
            "m4_freeze_cache": os.environ.get("M4_FREEZE_CACHE", "0"),
        },

        "reflection_score": state.get("reflection_score"),
        "reflection_notes": state.get("reflection_notes", ""),
        "grounding_score": state.get("grounding_score"),
        # Unknown is deliberate for historical or partial states. Benchmark code
        # must not infer a completed verification from a missing status.
        "verification_status": state.get("verification_status", "unknown"),
        "hitl_status": state.get("hitl_status"),
        "html_review_status": state.get("html_review_status"),
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

        # Categorised breakdown: directly answers "why did grounding fail?"
        #   unverified_no_source  → retrieval gap (Tavily never found a relevant source)
        #   unverified_has_source → precision mismatch or hallucination
        #   mean_confidence_*     → verify model calibration signal
        "grounding_breakdown": {
            "unverified_no_source": sum(1 for r in unverified if not r.get("source_url")),
            "unverified_has_source": sum(1 for r in unverified if r.get("source_url")),
            "weak_count": sum(1 for r in report if r.get("status") == "weak"),
            "mean_confidence_verified": round(
                sum(r.get("confidence", 0) for r in verified) / max(len(verified), 1), 3,
            ),
            "mean_confidence_unverified": round(
                sum(r.get("confidence", 0) for r in unverified) / max(len(unverified), 1), 3,
            ),
        },
        # Grounded-depth metric (M3): rewards substantive claims that are verified.
        #   SV = substantive AND verified (the objective to maximize)
        #   unverified_fraction = UVR (the grounding gate; must stay <= 0.15)
        "grounded_depth": {
            "SV": sum(1 for r in report
                      if r.get("specificity") == "substantive" and r.get("status") == "verified"),
            "S": sum(1 for r in report if r.get("specificity") == "substantive"),
            "V": len(verified),
            "N": len(report),
            "verified_fraction": round(len(verified) / len(report), 3) if report else None,
            "unverified_fraction": round(len(unverified) / len(report), 3) if report else None,
        },
        # Full claim-level evidence
        "grounding_report": report,
        "web_sources_count": len(state.get("web_sources", []) or []),
        "kb_results_count": len(state.get("kb_results", []) or []),
        "kb_context_stats": state.get("kb_context_stats", {}),

        "web_sources": [
            {"url": s.get("url"), "score": s.get("score"),
             "content": (s.get("content") or "")[:2000]}
            for s in (state.get("web_sources", []) or [])
        ],
        # M5: persist the retrieved KB set with chunk identity — without this,
        # KB-verified claims were not reconstructable (only a count was stored).
        "kb_results": [
            {"source": k.get("source"), "chunk_index": k.get("chunk_index", 0),
             "chunk_indices": k.get("chunk_indices", [k.get("chunk_index", 0)]),
             "distance": k.get("distance"), "rrf_score": k.get("rrf_score"),
             "text": (k.get("text") or "")[:2000]}
            for k in (state.get("kb_results", []) or [])
        ],

        # M5: attribution resolution summary. unresolved > 0 means the verifier
        # cited a source not in the retrieved set — surface it, don't hide it.
        "attribution": {
            "web": sum(1 for r in report if r.get("source_kind") == "web"),
            "kb": sum(1 for r in report if r.get("source_kind") == "kb"),
            "none": sum(1 for r in report if r.get("source_kind") == "none"),
            "unresolved": sum(1 for r in report if r.get("source_kind") == "unresolved"),
        },
    }

    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path


def _make_slug(topic: str) -> str:
    """B1: allowlist sanitization. The slug reaches the filesystem
        (../themachinist-website/<slug>.html) and git branch/tag names, so it must
        be incapable of path traversal or ref injection BY CONSTRUCTION:
        only [a-z0-9-] survives, length-capped. Caller rejects empty results."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower().replace("&", "and"))
    return re.sub(r"-{2,}", "-", slug).strip("-")[:80]




def _build_initial_state(topic, slug, card_id, series, run_id, category="concept-exploration"):
    """Canonical AgentState seed. Used by the CLI and the API server so the
    state shape is defined in exactly one place (LangGraph merges silently on
    key drift — see graph.py docstring)."""
    return {
        "topic": topic,
        "slug": slug,
        "card_id": card_id,
        "category": category,
        "series_context": series,
        "draft_sections": {},
        "draft_markdown": "",
        "web_sources": [],
        "kb_results": [],
        "kb_context_stats": {},
        "grounding_report": [],
        "grounding_score": 0.0,
        "verification_status": "not_started",
        "reflection_score": 0,
        "reflection_notes": "",
        "iterations": 0,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "html_review_status": None,
        "html_feedback": None,
        "html_output": None,
        "html_filename": None,
        "branch_name": None,
        "git_status": None,
        "iteration_metrics": [],
        "m4_feedback_claims": 0,
        "run_id": run_id,
        "prompt_version": PROMPT_VERSION,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "latency_ms": {},
        "error_log": [],
    }


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

    # Auto-on, OFF by default — see observability/tracing.py. Enables when
    # LANGCHAIN_API_KEY + LANGCHAIN_PROJECT are set; LANGSMITH_TRACING=0 forces off.
    setup_langsmith_tracing()

    slug = _make_slug(topic)
    if not slug:
        raise click.UsageError(f"Topic {topic!r} produces an empty slug.")
    run_id = str(uuid.uuid4())

    initial_state = _build_initial_state(topic, slug, card_id, series, run_id)

    # KB warmup (M5): pay the one-time encoder load + BM25 build BEFORE the graph
    # runs, so latency_ms.retrieve_kb measures steady-state query cost only.
    from tools.query_kb import warmup as kb_warmup
    warmup_times = kb_warmup()
    get_logger("main").info("kb.warmup", run_id=run_id, **warmup_times)


    graph = build_graph()
    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        initial_state["error_log"] = [f"pipeline crash: {e}"]
        initial_state["verification_status"] = "upstream_failed"
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


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
def serve(host, port):
    """Launch the HITL API server (FastAPI). Requires API_BEARER_TOKEN in env."""
    import uvicorn
    uvicorn.run("api.server:app", host=host, port=port, log_level="info")



if __name__ == "__main__":
    cli()
