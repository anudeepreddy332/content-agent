import sys
import json
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import build_graph

def smoke_test():
    initial_state = {
        "topic": "Gradient Descent",
        "slug": "gradient-descent-smoke",
        "card_id": "SMOKE-01",
        "series_context": "Concept Exploration",
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
        "run_id": f"smoke-{uuid.uuid4().hex[:8]}",
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "latency_ms": {},
        "error_log": [],
    }

    graph = build_graph()
    result = graph.invoke(initial_state)

    failures = []

    if not result.get("draft_markdown"):
        failures.append("draft_markdown is empty")

    grounding = result.get("grounding_score", -1)

    if grounding < 0:
        failures.append("grounding_score missing from result")
    elif grounding == 0 and not result.get("error_log"):
        failures.append(f"grounding_score is 0 with no error_log — silent failure")

    if result.get("total_cost_usd", 0) >= 0.05:
        failures.append(f"cost exceeded: ${result['total_cost_usd']:.4f}")

    run_id = result["run_id"]
    telemetry_path = Path(f"outputs/runs/{run_id}.json")
    if not telemetry_path.exists():
        failures.append(f"telemetry not written to {telemetry_path}")

    if failures:
        print("SMOKE FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print(f"SMOKE PASS | run_id={run_id} | "
          f"cost=${result['total_cost_usd']:.4f} | "
          f"grounding={result['grounding_score']:.2f} | "
          f"reflection={result['reflection_score']} | "
          f"tokens={result['total_tokens']}")
    sys.exit(0)

if __name__ == "__main__":
    smoke_test()









