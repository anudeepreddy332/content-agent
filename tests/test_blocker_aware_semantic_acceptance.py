"""Blocker-aware semantic acceptance: WEAK + contradiction/limitation fail closed."""
from __future__ import annotations

import pytest
from langgraph.graph import END

import agent.nodes as nodes
from config import UVR_THRESHOLD
from tests.test_verify_node_semantic_cutover import (
    _fixture_state,
    _load_case,
    _run_fixture_case,
)


def _verified_row(claim: str = "verified claim") -> dict:
    return {"claim": claim, "status": "verified", "blockers": []}


def _weak_row(claim: str, *, blocker_kind: str | None = None) -> dict:
    blockers = []
    if blocker_kind is not None:
        blockers = [{"kind": blocker_kind, "explanation": "test", "evidence_spans": []}]
    return {"claim": claim, "status": "weak", "blockers": blockers}


def _unverified_row(claim: str = "unverified claim") -> dict:
    return {"claim": claim, "status": "unverified", "blockers": []}


def _state(base_state: dict, report: list[dict], **extra) -> dict:
    state = dict(base_state)
    defaults = {
        "iterations": 1,
        "verification_status": "completed",
        "grounding_report": report,
        "grounding_score": 0.75,
        "reflection_score": 8,
        "total_cost_usd": 0.01,
    }
    defaults.update(extra)
    state.update(defaults)
    return state


@pytest.mark.parametrize(
    "report,expected",
    [
        pytest.param([_weak_row(f"c-{i}", blocker_kind="contradiction") for i in range(5)], False, id="A"),
        pytest.param([_weak_row(f"l-{i}", blocker_kind="limitation") for i in range(5)], False, id="B"),
        pytest.param(
            [_weak_row("blocked", blocker_kind="contradiction")]
            + [_verified_row(f"v-{i}") for i in range(4)],
            False,
            id="C",
        ),
        pytest.param(
            [_weak_row("blocked", blocker_kind="limitation")]
            + [_verified_row(f"v-{i}") for i in range(9)],
            False,
            id="D",
        ),
        pytest.param([_verified_row(f"v-{i}") for i in range(10)], True, id="E"),
        pytest.param(
            [_unverified_row("u-1")] + [_verified_row(f"v-{i}") for i in range(9)],
            True,
            id="F",
        ),
        pytest.param(
            [_unverified_row(f"u-{i}") for i in range(4)]
            + [_verified_row(f"v-{i}") for i in range(16)],
            False,
            id="G",
        ),
        pytest.param(
            [_weak_row("ordinary weak")] + [_verified_row(f"v-{i}") for i in range(9)],
            True,
            id="H",
        ),
    ],
)
def test_routing_matrix_semantic_acceptance(base_state, report, expected):
    state = _state(base_state, report)
    assert nodes.semantic_verification_accepted(state) is expected


def test_matrix_f_uvr_below_threshold_without_blockers(base_state):
    report = [_unverified_row("u-1")] + [_verified_row(f"v-{i}") for i in range(9)]
    assert nodes.unverified_rate(report) == 0.1
    assert nodes.unverified_rate(report) <= UVR_THRESHOLD
    assert nodes.has_blocking_semantic_weakness(report) is False
    assert nodes.semantic_verification_accepted(_state(base_state, report)) is True


def test_matrix_g_uvr_above_threshold(base_state):
    report = [_unverified_row(f"u-{i}") for i in range(4)] + [_verified_row(f"v-{i}") for i in range(16)]
    assert nodes.unverified_rate(report) == 0.2
    assert nodes.semantic_verification_accepted(_state(base_state, report)) is False


def test_p6_contradiction_weak_routes_to_revision(base_state, monkeypatch):
    result = _run_fixture_case(monkeypatch, "P6", legacy_status="verified")
    state = {**_fixture_state(_load_case("P6")), **result}
    assert state["grounding_report"][0]["status"] == "weak"
    assert nodes.unverified_rate(state["grounding_report"]) == 0.0
    assert state["grounding_score"] == 0.75
    assert nodes.semantic_verification_accepted(state) is False
    assert nodes.route_after_reflect(state) == "draft"


def test_p7_limitation_weak_routes_to_revision(base_state, monkeypatch):
    result = _run_fixture_case(monkeypatch, "P7")
    state = {**_fixture_state(_load_case("P7")), **result}
    assert state["grounding_report"][0]["status"] == "weak"
    assert nodes.semantic_verification_accepted(state) is False
    assert nodes.route_after_reflect(state) == "draft"


def test_p1_verified_clean_accepted(base_state, monkeypatch):
    result = _run_fixture_case(monkeypatch, "P1")
    state = {**_fixture_state(_load_case("P1")), **result}
    assert state["grounding_report"][0]["status"] == "verified"
    assert nodes.semantic_verification_accepted(state) is True
    assert nodes.route_after_reflect(state) == "hitl"


def test_auto_approve_rejects_contradiction_weak(base_state, monkeypatch):
    report = [_weak_row("blocked", blocker_kind="contradiction")]
    state = _state(base_state, report)
    assert nodes.unverified_rate(report) == 0.0
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_auto_approve_rejects_limitation_weak(base_state, monkeypatch):
    report = [_weak_row("blocked", blocker_kind="limitation")]
    state = _state(base_state, report)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_auto_approve_still_passes_blocker_free_weak(base_state, monkeypatch):
    report = [_weak_row("ordinary")] + [_verified_row(f"v-{i}") for i in range(9)]
    state = _state(base_state, report, grounding_score=0.85)
    assert nodes.semantic_verification_accepted(state) is True
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "approved"
    assert nodes.route_after_hitl({**state, **result}) == "html_gen"


def test_has_blocking_semantic_weakness_ignores_non_weak_rows():
    report = [
        {"claim": "x", "status": "verified", "blockers": [{"kind": "contradiction"}]},
        {"claim": "y", "status": "unverified", "blockers": [{"kind": "limitation"}]},
    ]
    assert nodes.has_blocking_semantic_weakness(report) is False


def test_has_blocking_semantic_weakness_ignores_other_blocker_kinds():
    report = [_weak_row("x", blocker_kind="unsupported_kind")]
    assert nodes.has_blocking_semantic_weakness(report) is False
    assert nodes.semantic_verification_accepted(
        {"verification_status": "completed", "grounding_report": report}
    ) is True
