"""Phase 4 Slice 1: acceptance/remediation foundation. Deterministic, $0, no providers.

Covers the required causal matrix:
  A/B  WEAK + contradiction/limitation       -> block
  C/D  UNVERIFIED + contradiction/limitation -> block (even when UVR alone passes)
  E    blocker-free UNVERIFIED below UVR 0.15 -> preserved UVR behavior (accept)
  F    blocker-free UNVERIFIED above UVR 0.15 -> reject
  G    blocker-free WEAK                      -> preserved behavior (accept)
  H    all VERIFIED                           -> pass semantic gate
  I    INVALID evaluation state               -> fail closed
  J    legacy LLM status/confidence           -> cannot alter any decision
Plus: targeted P6/P7 revision feedback, grounding_score routing removal,
auto-approval blocker-bypass prevention, and a mocked end-to-end graph run.
"""
from __future__ import annotations

import copy
import json

import pytest
from langgraph.graph import END

import agent.graph as graph_mod
import agent.nodes as nodes
from config import MAX_ITERATIONS, REFLECTION_THRESHOLD, UVR_THRESHOLD
from tests.conftest import FakeLLMClient, fake_response
from tests.test_verify_node_semantic_cutover import (
    _fixture_state,
    _load_case,
    _run_fixture_case,
)

ADVERSE_QUOTE = "Writes are disabled whenever the CM-OFF condition holds."
SUPPORT_QUOTE = "Cache tier T3 participates in the write path."


def _row(
    claim: str,
    status: str,
    *,
    blocker_kind: str | None = None,
    explanation: str = "structured blocker explanation",
    quote: str | None = None,
    evidence_id: str = "WEB-001",
    source_url: str | None = "https://src.test/evidence",
    confidence: float | None = None,
) -> dict:
    blockers = []
    if blocker_kind is not None:
        blockers = [
            {
                "kind": blocker_kind,
                "explanation": explanation,
                "evidence_spans": (
                    [{"evidence_id": evidence_id, "start": 0, "end": len(quote), "text": quote}]
                    if quote
                    else []
                ),
            }
        ]
    row = {
        "claim_id": "claim-001",
        "claim": claim,
        "status": status,
        "blockers": blockers,
        "support_spans": (
            [{"evidence_id": evidence_id, "start": 0, "end": len(SUPPORT_QUOTE), "text": SUPPORT_QUOTE}]
            if status == "weak"
            else []
        ),
        "reason_codes": ["MEANINGFUL_SUPPORT_PRESENT"] if status == "weak" else ["NO_MEANINGFUL_SUPPORT"],
        "source_url": source_url if status != "unverified" else None,
        "source_ref": source_url if status != "unverified" else None,
    }
    if confidence is not None:
        row["confidence"] = confidence
    return row


def _verified_rows(n: int) -> list[dict]:
    return [
        {"claim_id": f"claim-v{i:03d}", "claim": f"verified-{i}", "status": "verified",
         "blockers": [], "support_spans": [], "reason_codes": []}
        for i in range(n)
    ]


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


# ---------------------------------------------------------------------------
# Causal cases A-J (spec §10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status,kind,case",
    [
        ("weak", "contradiction", "A"),
        ("weak", "limitation", "B"),
        ("unverified", "contradiction", "C"),
        ("unverified", "limitation", "D"),
    ],
)
def test_cases_abcd_blocker_blocks_acceptance_and_revision_routing(base_state, status, kind, case):
    # 9 verified + 1 blocker row: UVR is 0.0 (weak) or 0.10 (unverified) — both
    # <= 0.15, so the blocker alone must be what blocks acceptance.
    report = [_row(f"blocked-{case}", status, blocker_kind=kind)] + _verified_rows(9)
    state = _state(base_state, report)
    assert nodes.unverified_rate(report) <= UVR_THRESHOLD
    assert nodes.has_blocking_semantic_blockers(report) is True
    assert nodes.semantic_verification_accepted(state) is False
    assert nodes.route_after_reflect(state) == "draft"
    # Status is NOT changed by the blocker: acceptance is what flips.
    assert state["grounding_report"][0]["status"] == status


def test_case_e_blocker_free_unverified_below_threshold_accepted(base_state):
    report = [
        {"claim": "u-1", "status": "unverified", "blockers": []},
        *_verified_rows(9),
    ]
    state = _state(base_state, report)
    assert nodes.unverified_rate(report) == 0.1
    assert nodes.has_blocking_semantic_blockers(report) is False
    assert nodes.semantic_verification_accepted(state) is True
    assert nodes.route_after_reflect(state) == "hitl"


def test_case_f_blocker_free_unverified_above_threshold_rejected(base_state):
    report = [
        *[{"claim": f"u-{i}", "status": "unverified", "blockers": []} for i in range(4)],
        *_verified_rows(16),
    ]
    state = _state(base_state, report)
    assert nodes.unverified_rate(report) == 0.2
    assert nodes.semantic_verification_accepted(state) is False
    assert nodes.route_after_reflect(state) == "draft"


def test_case_g_blocker_free_weak_accepted(base_state):
    report = [
        {"claim": "ordinary weak", "status": "weak", "blockers": []},
        *_verified_rows(9),
    ]
    state = _state(base_state, report)
    assert nodes.semantic_verification_accepted(state) is True
    assert nodes.route_after_reflect(state) == "hitl"


def test_case_h_all_verified_passes(base_state):
    state = _state(base_state, _verified_rows(10))
    assert nodes.unresolved_semantic_obligations(state["grounding_report"]) == []
    assert nodes.semantic_verification_accepted(state) is True
    assert nodes.route_after_reflect(state) == "hitl"


@pytest.mark.parametrize(
    "status",
    ["parse_failed", "verification_error", "skipped_cost_gate", "upstream_failed", "not_started", None],
)
def test_case_i_invalid_evaluation_state_fails_closed(base_state, status):
    state = _state(
        base_state,
        _verified_rows(10),
        verification_status=status,
        grounding_score=0.99,
        reflection_score=10,
    )
    assert nodes.semantic_verification_accepted(state) is False
    assert nodes.route_after_reflect(state) == "draft"


def test_case_j_legacy_confidence_cannot_launder_blocker(base_state):
    """A legacy confidence field on a blocker row changes nothing."""
    report = [
        _row("blocked", "unverified", blocker_kind="contradiction", confidence=0.99),
        *_verified_rows(9),
    ]
    state = _state(base_state, report, grounding_score=0.99, reflection_score=10)
    assert nodes.semantic_verification_accepted(state) is False
    assert nodes.route_after_reflect(state) == "draft"


def test_case_j_legacy_status_cannot_override_engine_p6(base_state, monkeypatch):
    """P6: engine says weak+contradiction; legacy Call A verdict is irrelevant."""
    for legacy_status in ("verified", "weak", "unverified"):
        result = _run_fixture_case(monkeypatch, "P6", legacy_status=legacy_status)
        state = {**_fixture_state(_load_case("P6")), **result}
        assert state["grounding_report"][0]["status"] == "weak"
        assert state["grounding_report"][0]["blockers"][0]["kind"] == "contradiction"
        assert nodes.semantic_verification_accepted(state) is False
        assert nodes.route_after_reflect(state) == "draft"


def test_case_j_legacy_unverified_cannot_sink_engine_verified_p1(base_state, monkeypatch):
    result = _run_fixture_case(monkeypatch, "P1", legacy_status="unverified")
    state = {**_fixture_state(_load_case("P1")), **result, "reflection_score": 8}
    assert state["grounding_report"][0]["status"] == "verified"
    assert nodes.semantic_verification_accepted(state) is True
    assert nodes.route_after_reflect(state) == "hitl"


# ---------------------------------------------------------------------------
# Explicit semantic obligations (spec §7)
# ---------------------------------------------------------------------------

def test_obligations_expose_blocker_provenance_and_resolution():
    report = [
        _row(
            "Cache tier T3 admits writes when isolation level IL-2 holds.",
            "weak",
            blocker_kind="limitation",
            explanation="Source qualifies writes with CM-OFF disable condition absent from the draft claim.",
            quote=ADVERSE_QUOTE,
        ),
        *_verified_rows(3),
    ]
    obligations = nodes.unresolved_semantic_obligations(report)
    assert len(obligations) == 1
    ob = obligations[0]
    assert ob["claim_id"] == "claim-001"
    assert ob["status"] == "weak"
    assert ob["blocker_kinds"] == ["limitation"]
    assert ob["blockers"][0]["explanation"].startswith("Source qualifies writes")
    assert ob["blockers"][0]["evidence_spans"][0]["text"] == ADVERSE_QUOTE
    assert ob["support_spans"][0]["text"] == SUPPORT_QUOTE
    assert ob["reason_codes"] == ["MEANINGFUL_SUPPORT_PRESENT"]
    assert ob["source_ref"] == "https://src.test/evidence"
    assert ob["blocking"] is True
    assert ob["resolution"] == "unresolved"


def test_obligations_include_blocker_free_unverified_as_non_blocking():
    report = [
        {"claim_id": "claim-001", "claim": "u", "status": "unverified", "blockers": []},
        *_verified_rows(9),
    ]
    obligations = nodes.unresolved_semantic_obligations(report)
    assert len(obligations) == 1
    assert obligations[0]["blocking"] is False
    assert obligations[0]["resolution"] == "unresolved"


# ---------------------------------------------------------------------------
# Targeted revision feedback (spec §3, §4, §11)
# ---------------------------------------------------------------------------

def _capture_draft_user_message(monkeypatch, state: dict) -> str:
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
    nodes.draft_node(state)
    return next(m["content"] for m in captured["messages"] if m["role"] == "user")


def test_p6_contradiction_revision_feedback_is_targeted(base_state, monkeypatch):
    result = _run_fixture_case(monkeypatch, "P6", legacy_status="verified")
    state = {**_fixture_state(_load_case("P6")), **result}
    row = state["grounding_report"][0]
    assert row["status"] == "weak"
    assert row["blockers"][0]["kind"] == "contradiction"

    user = _capture_draft_user_message(monkeypatch, state)

    # affected claim, blocker kind, explanation, exact adverse evidence, ids
    assert row["claim"] in user
    assert row["claim_id"] in user
    assert "contradiction" in user
    assert row["blockers"][0]["explanation"] in user
    adverse_quote = row["blockers"][0]["evidence_spans"][0]["text"]
    assert adverse_quote and adverse_quote in user
    assert row["blockers"][0]["evidence_spans"][0]["evidence_id"] in user
    # supporting evidence quote is also carried
    support_quote = row["support_spans"][0]["text"]
    assert support_quote and support_quote in user
    # required-content guardrail, not a bare "improve grounding"
    assert "required substantive content" in user
    assert "grounding_score" not in user


def test_p7_limitation_revision_feedback_is_targeted(base_state, monkeypatch):
    result = _run_fixture_case(monkeypatch, "P7")
    state = {**_fixture_state(_load_case("P7")), **result}
    row = state["grounding_report"][0]
    assert row["status"] == "weak"
    assert row["blockers"][0]["kind"] == "limitation"

    user = _capture_draft_user_message(monkeypatch, state)

    assert row["claim"] in user
    assert row["claim_id"] in user
    assert "limitation" in user
    # omitted qualifier is named via the blocker explanation + repair guidance
    assert row["blockers"][0]["explanation"] in user
    assert "qualifier" in user
    adverse_quote = row["blockers"][0]["evidence_spans"][0]["text"]
    assert adverse_quote and adverse_quote in user
    assert "required substantive content" in user


def test_blocker_feedback_reaches_trace_revision_linkage(base_state, monkeypatch):
    result = _run_fixture_case(monkeypatch, "P6", legacy_status="verified")
    state = {**_fixture_state(_load_case("P6")), **result}
    client = FakeLLMClient(response=fake_response(json.dumps({
        "problem_framing": "p", "technical_dive": "t", "code_snippets": "c", "takeaways": "k",
    })))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    out = nodes.draft_node(state)
    linkage = out["semantic_trace"]["iterations"][-1]["revision_linkage"]
    obligations = linkage["semantic_obligations"]
    assert len(obligations) == 1
    assert obligations[0]["blocker_kinds"] == ["contradiction"]
    assert obligations[0]["blocking"] is True
    assert "TARGETED SEMANTIC BLOCKER FEEDBACK" in linkage["grounding_feedback_block"]


def test_unverified_only_revision_keeps_m4_block_without_blocker_block(base_state, monkeypatch):
    report = [{"claim": "unsupported claim", "status": "unverified", "blockers": []}] + _verified_rows(9)
    state = _state(base_state, report)
    user = _capture_draft_user_message(monkeypatch, state)
    assert "unsupported claim" in user
    assert "GROUNDING REPORT FEEDBACK" in user
    assert "TARGETED SEMANTIC BLOCKER FEEDBACK" not in user


# ---------------------------------------------------------------------------
# grounding_score routing removal (spec §5, §12)
# ---------------------------------------------------------------------------

GROUNDING_SCORE_SWEEP = [0.0, 0.3, 0.59, 0.60, 0.61, 0.74, 0.75, 0.9, 1.0]


def _routes_over_score_sweep(state: dict) -> set:
    return {
        nodes.route_after_reflect({**state, "grounding_score": score})
        for score in GROUNDING_SCORE_SWEEP
    }


def test_grounding_score_cannot_alter_routing_when_accepted(base_state):
    """Pre-slice, score < 0.60 (hard floor) would have forced 'draft' on this
    exact accepted state. After Slice 1 the scalar has no routing authority."""
    state = _state(base_state, _verified_rows(10), reflection_score=8)
    assert _routes_over_score_sweep(state) == {"hitl"}
    acceptances = {
        nodes.semantic_verification_accepted({**state, "grounding_score": s})
        for s in GROUNDING_SCORE_SWEEP
    }
    assert acceptances == {True}


def test_grounding_score_cannot_alter_routing_when_blocker_present(base_state):
    report = [_row("blocked", "unverified", blocker_kind="contradiction")] + _verified_rows(9)
    state = _state(base_state, report, reflection_score=10)
    assert _routes_over_score_sweep(state) == {"draft"}


def test_grounding_score_cannot_alter_routing_when_uvr_fails(base_state):
    report = [
        *[{"claim": f"u-{i}", "status": "unverified", "blockers": []} for i in range(4)],
        *_verified_rows(16),
    ]
    state = _state(base_state, report, reflection_score=10)
    assert _routes_over_score_sweep(state) == {"draft"}


def test_reflection_gate_still_routes_without_grounding(base_state):
    """The reflection quality gate is unchanged and is not a grounding_score
    composite: low reflection revises at ANY grounding_score."""
    state = _state(base_state, _verified_rows(10), reflection_score=REFLECTION_THRESHOLD - 1)
    assert _routes_over_score_sweep(state) == {"draft"}
    ok = _state(base_state, _verified_rows(10), reflection_score=REFLECTION_THRESHOLD)
    assert nodes.route_after_reflect({**ok, "grounding_score": 0.0}) == "hitl"


# ---------------------------------------------------------------------------
# Auto-approval / publication safety (spec §9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status,kind",
    [
        ("weak", "contradiction"),
        ("weak", "limitation"),
        ("unverified", "contradiction"),
        ("unverified", "limitation"),
    ],
)
def test_auto_approve_cannot_bypass_blocker_at_iteration_exhaustion(
    base_state, monkeypatch, status, kind
):
    report = [_row("blocked", status, blocker_kind=kind)] + _verified_rows(9)
    state = _state(base_state, report, iterations=MAX_ITERATIONS)
    # Exhaustion routes to the human gate, never silently onward.
    assert nodes.route_after_reflect(state) == "hitl"
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


@pytest.mark.parametrize("kind", ["contradiction", "limitation"])
def test_api_approve_cannot_bypass_unverified_blocker(base_state, monkeypatch, kind):
    report = [_row("blocked", "unverified", blocker_kind=kind)] + _verified_rows(9)
    state = _state(base_state, report, iterations=MAX_ITERATIONS)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: {"action": "approve"})
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_hitl_api_payload_exposes_semantic_obligations(base_state, monkeypatch):
    report = [
        _row("blocked", "weak", blocker_kind="contradiction", quote=ADVERSE_QUOTE),
        *_verified_rows(9),
    ]
    state = _state(base_state, report, iterations=MAX_ITERATIONS)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("HITL_MODE", "api")
    captured = {}

    def fake_interrupt(payload):
        captured["payload"] = payload
        return {"action": "reject"}

    monkeypatch.setattr("langgraph.types.interrupt", fake_interrupt)
    nodes.hitl_node(state)
    obligations = captured["payload"]["semantic_obligations"]
    assert len(obligations) == 1
    assert obligations[0]["blocking"] is True
    assert obligations[0]["blocker_kinds"] == ["contradiction"]
    assert captured["payload"]["claim_completeness"] == "unknown"


# ---------------------------------------------------------------------------
# UVR non-regression (spec §6)
# ---------------------------------------------------------------------------

def test_uvr_v1_definition_unchanged():
    report = [
        *[{"claim": f"u-{i}", "status": "unverified", "blockers": []} for i in range(2)],
        {"claim": "w", "status": "weak", "blockers": []},
        *_verified_rows(7),
    ]
    # UVR_v1 = UNVERIFIED / (VERIFIED + WEAK + UNVERIFIED) = 2/10
    assert nodes.unverified_rate(report) == 0.2
    assert UVR_THRESHOLD == 0.15
    assert nodes.unverified_rate([]) is None


def test_uvr_cannot_override_blocker_or_invalid_state(base_state):
    report = [_row("blocked", "unverified", blocker_kind="limitation")] + _verified_rows(19)
    assert nodes.unverified_rate(report) == 0.05  # well below threshold
    assert nodes.semantic_verification_accepted(_state(base_state, report)) is False
    invalid = _state(base_state, _verified_rows(20), verification_status="parse_failed")
    assert nodes.semantic_verification_accepted(invalid) is False


# ---------------------------------------------------------------------------
# Mocked end-to-end (spec §16.12): blocker-bearing UNVERIFIED revises with
# targeted feedback, exhausts iterations, then holds at HITL — never publishes.
# ---------------------------------------------------------------------------

def test_mocked_e2e_unverified_blocker_revises_then_holds(base_state, monkeypatch):
    draft_json = json.dumps({
        "problem_framing": "p", "technical_dive": "t", "code_snippets": "c", "takeaways": "k",
    })
    client = FakeLLMClient(response=fake_response(draft_json))
    captured: list[list[dict]] = []
    orig_create = client.chat.completions.create

    def create(**kwargs):
        captured.append(kwargs["messages"])
        return orig_create(**kwargs)

    client.chat.completions.create = create
    monkeypatch.setattr(nodes, "_get_client", lambda: client)

    report = [
        _row(
            "Cache tier T3 admits writes unconditionally.",
            "unverified",
            blocker_kind="limitation",
            explanation="Source states writes are disabled under CM-OFF.",
            quote=ADVERSE_QUOTE,
        ),
        *_verified_rows(9),
    ]

    monkeypatch.setattr(graph_mod, "retrieve_node", lambda state: {})
    monkeypatch.setattr(graph_mod, "verify_node", lambda state: {
        "grounding_report": copy.deepcopy(report),
        "grounding_score": 0.9,
        "verification_status": "completed",
    })
    monkeypatch.setattr(graph_mod, "reflect_node", lambda state: {
        "reflection_score": 8, "reflection_notes": "ok",
    })
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")

    graph = graph_mod.build_graph()
    init = dict(base_state)
    init.update(iterations=0, grounding_report=[], verification_status="not_started")
    result = graph.invoke(init)

    # Two drafting passes (initial + one revision), then exhaustion -> HITL hold.
    assert result["iterations"] == MAX_ITERATIONS
    assert len(captured) == MAX_ITERATIONS
    assert result["hitl_status"] == "rejected"
    assert result.get("html_output") is None

    revision_user = next(m["content"] for m in captured[1] if m["role"] == "user")
    assert "TARGETED SEMANTIC BLOCKER FEEDBACK" in revision_user
    assert "Cache tier T3 admits writes unconditionally." in revision_user
    assert "limitation" in revision_user
    assert ADVERSE_QUOTE in revision_user
    assert "Source states writes are disabled under CM-OFF." in revision_user
