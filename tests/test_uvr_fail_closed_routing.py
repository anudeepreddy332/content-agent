"""P0-2b slice 1: UVR-aware fail-closed routing. Deterministic, $0, no providers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.graph import END

import agent.graph as graph_mod
import agent.nodes as nodes
from config import (
    COST_GATE_USD,
    GROUNDING_FLOOR,
    MAX_ITERATIONS,
    REFLECTION_THRESHOLD,
    UVR_THRESHOLD,
)
from tests.conftest import FakeLLMClient, fake_response

# Frozen topic-10 baseline (GHA 32480353168, Feedforward Neural Networks).
TOPIC10_UNVERIFIED = [
    "Linear models like logistic regression can only draw straight-line boundaries, which fail on data with nonlinear structure.",
    "Given enough capacity and training data, a feedforward network can approximate any continuous function.",
    "Understanding feedforward networks is the prerequisite for everything else in neural networks, from CNNs to transformers.",
    "ReLU helps but can cause dead neurons if weights push activations into the negative region where the gradient is zero.",
    "Another failure mode is overfitting: a network with many parameters can memorize the training set, so you need regularization like dropout or weight decay, or more data.",
]


def _row(claim: str, status: str, confidence: float) -> dict:
    return {
        "claim": claim,
        "source_url": None,
        "confidence": confidence,
        "status": status,
        "specificity": "generic",
    }


def _report(*, verified: int, weak: int, unverified: int | list[str],
            unverified_confidence: float = 0.0) -> list[dict]:
    claims = unverified if isinstance(unverified, list) else [
        f"unverified-{i}" for i in range(unverified)
    ]
    rows = [_row(c, "unverified", unverified_confidence) for c in claims]
    rows.extend(_row(f"verified-{i}", "verified", 0.9) for i in range(verified))
    rows.extend(_row(f"weak-{i}", "weak", 0.5) for i in range(weak))
    return rows


def topic10_report() -> list[dict]:
    return _report(verified=14, weak=2, unverified=TOPIC10_UNVERIFIED)


def topic10_state(base_state: dict) -> dict:
    report = topic10_report()
    assert len(report) == 21
    state = dict(base_state)
    state.update(
        topic="Feedforward Neural Networks",
        slug="feedforward-neural-networks",
        iterations=1,
        verification_status="completed",
        grounding_report=report,
        grounding_score=0.669,
        reflection_score=7,
        total_cost_usd=0.01,
    )
    return state


def test_thresholds_and_prompt_files_unchanged():
    assert UVR_THRESHOLD == 0.15
    assert GROUNDING_FLOOR == 0.60
    assert REFLECTION_THRESHOLD == 7
    assert MAX_ITERATIONS == 2
    assert COST_GATE_USD == 0.10
    bench_src = Path("scripts/benchmark.py").read_text()
    assert "UVR_THRESHOLD = 0.15" in bench_src
    assert nodes.UVR_THRESHOLD == 0.15
    assert nodes.CLAIM_COMPLETENESS == "unknown"


def test_topic10_preserved_state_routes_to_draft(base_state):
    state = topic10_state(base_state)
    assert nodes.unverified_rate(state["grounding_report"]) == 5 / 21
    assert 5 / 21 > UVR_THRESHOLD
    assert state["grounding_score"] > GROUNDING_FLOOR
    assert state["reflection_score"] == REFLECTION_THRESHOLD
    assert nodes.route_after_reflect(state) == "draft"


def test_uvr_exactly_threshold_is_acceptable(base_state):
    # 3/20 == 0.15 exactly.
    report = _report(verified=17, weak=0, unverified=3)
    assert nodes.unverified_rate(report) == 0.15
    base_state.update(
        iterations=1,
        verification_status="completed",
        grounding_report=report,
        grounding_score=0.80,
        reflection_score=8,
    )
    assert nodes.semantic_verification_accepted(base_state) is True
    assert nodes.route_after_reflect(base_state) == "hitl"


def test_uvr_above_threshold_routes_to_revision_when_capacity_remains(base_state):
    report = _report(verified=16, weak=0, unverified=4)  # 4/20 = 0.20
    base_state.update(
        iterations=1,
        verification_status="completed",
        grounding_report=report,
        grounding_score=0.80,
        reflection_score=8,
    )
    assert nodes.unverified_rate(report) > UVR_THRESHOLD
    assert nodes.route_after_reflect(base_state) == "draft"


def test_grounding_and_reflection_cannot_override_uvr_failure(base_state):
    state = topic10_state(base_state)
    state.update(grounding_score=0.99, reflection_score=10)
    assert nodes.route_after_reflect(state) == "draft"


def test_completed_empty_verdict_set_fails_closed(base_state):
    base_state.update(
        iterations=1,
        verification_status="completed",
        grounding_report=[],
        grounding_score=0.99,
        reflection_score=10,
    )
    assert nodes.semantic_verification_accepted(base_state) is False
    assert nodes.unverified_rate([]) is None
    assert nodes.route_after_reflect(base_state) == "draft"


@pytest.mark.parametrize(
    "status",
    ["parse_failed", "skipped_cost_gate", "upstream_failed", "not_started", "unknown", None],
)
def test_incomplete_verification_status_fails_closed(base_state, status):
    report = _report(verified=20, weak=0, unverified=0)
    base_state.update(
        iterations=1,
        verification_status=status,
        grounding_report=report,
        grounding_score=0.99,
        reflection_score=10,
        total_cost_usd=0.0,
    )
    assert nodes.semantic_verification_accepted(base_state) is False
    assert nodes.route_after_reflect(base_state) == "draft"


def test_high_confidence_unverified_cannot_pass_via_confidence(base_state):
    report = _report(verified=0, weak=0, unverified=["unsupported claim"],
                     unverified_confidence=0.99)
    base_state.update(
        iterations=1,
        verification_status="completed",
        grounding_report=report,
        grounding_score=0.99,
        reflection_score=10,
    )
    assert nodes.semantic_verification_accepted(base_state) is False
    assert nodes.route_after_reflect(base_state) == "draft"


def test_topic10_revision_feedback_contains_exact_five_claims(base_state, monkeypatch):
    state = topic10_state(base_state)
    captured = {}
    client = FakeLLMClient(response=fake_response(json.dumps({
        "problem_framing": "p",
        "technical_dive": "t",
        "code_snippets": "c",
        "takeaways": "k",
    })))
    orig_create = client.chat.completions.create

    def create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return orig_create(**kwargs)

    client.chat.completions.create = create
    monkeypatch.setattr(nodes, "_get_client", lambda: client)

    result = nodes.draft_node(state)
    assert result["m4_feedback_claims"] == 5
    user = next(m["content"] for m in captured["messages"] if m["role"] == "user")
    for claim in TOPIC10_UNVERIFIED:
        assert claim in user
    assert nodes.unverified_claims(state["grounding_report"]) == TOPIC10_UNVERIFIED


def test_revision_is_followed_by_reverification(base_state, monkeypatch):
    verify_calls = []
    draft_calls = []
    report = topic10_report()

    def fake_retrieve(state):
        return {}

    def fake_draft(state):
        draft_calls.append({
            "iterations_in": state.get("iterations", 0),
            "claims": nodes.unverified_claims(state.get("grounding_report")),
            "m4": len(nodes.unverified_claims(state.get("grounding_report")))
            if state.get("iterations", 0) >= 1 else 0,
        })
        return {
            "draft_markdown": f"draft-{len(draft_calls)}",
            "m4_feedback_claims": draft_calls[-1]["m4"],
            "iterations": state.get("iterations", 0) + 1,
        }

    def fake_verify(state):
        verify_calls.append(state.get("iterations"))
        if len(verify_calls) == 1:
            return {
                "grounding_report": report,
                "grounding_score": 0.669,
                "verification_status": "completed",
            }
        ok = _report(verified=10, weak=0, unverified=0)
        return {
            "grounding_report": ok,
            "grounding_score": 0.85,
            "verification_status": "completed",
        }

    def fake_reflect(state):
        return {"reflection_score": 7, "reflection_notes": "ok"}

    def fake_hitl(state):
        return {"hitl_status": "rejected", "hitl_feedback": None}

    monkeypatch.setattr(graph_mod, "retrieve_node", fake_retrieve)
    monkeypatch.setattr(graph_mod, "draft_node", fake_draft)
    monkeypatch.setattr(graph_mod, "verify_node", fake_verify)
    monkeypatch.setattr(graph_mod, "reflect_node", fake_reflect)
    monkeypatch.setattr(graph_mod, "hitl_node", fake_hitl)

    graph = graph_mod.build_graph()
    init = topic10_state(base_state)
    init["iterations"] = 0
    init["grounding_report"] = []
    init["verification_status"] = "not_started"
    graph.invoke(init)

    assert verify_calls == [1, 2], "revision must execute verify again"
    assert len(draft_calls) == 2
    assert draft_calls[1]["claims"] == TOPIC10_UNVERIFIED
    assert draft_calls[1]["m4"] == 5


def test_acceptable_verified_state_proceeds_to_hitl(base_state):
    report = _report(verified=10, weak=1, unverified=0)
    base_state.update(
        iterations=1,
        verification_status="completed",
        grounding_report=report,
        grounding_score=0.80,
        reflection_score=8,
    )
    assert nodes.semantic_verification_accepted(base_state) is True
    assert nodes.route_after_reflect(base_state) == "hitl"


def test_iteration_ceiling_does_not_auto_approve_semantic_failure(base_state, monkeypatch):
    state = topic10_state(base_state)
    state["iterations"] = MAX_ITERATIONS
    assert nodes.route_after_reflect(state) == "hitl"
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    merged = {**state, **result}
    assert nodes.route_after_hitl(merged) == END


def test_cost_ceiling_does_not_auto_approve_semantic_failure(base_state, monkeypatch):
    state = topic10_state(base_state)
    state["total_cost_usd"] = COST_GATE_USD
    assert nodes.route_after_reflect(state) == "hitl"
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_auto_approve_still_works_when_semantic_verification_passes(base_state, monkeypatch):
    report = _report(verified=10, weak=0, unverified=0)
    base_state.update(
        iterations=1,
        verification_status="completed",
        grounding_report=report,
        grounding_score=0.80,
        reflection_score=8,
    )
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(base_state)
    assert result["hitl_status"] == "approved"
    assert nodes.route_after_hitl({**base_state, **result}) == "html_gen"
