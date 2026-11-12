"""Narrow ambiguity terminal routing; portable doubles, no private article fixture."""
from contextlib import closing
from copy import deepcopy
import json
import sqlite3
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END

from agent import nodes, workflow
import agent.graph as graph_mod
from agent.claim_inventory import (
    ANCHOR_AMBIGUOUS, ANCHOR_BOUND, ANCHOR_BOUND_MULTIPLE, ANCHOR_FAILED,
    build_claim_inventory,
)
from tests.conftest import FakeLLMClient, fake_response
from tests.test_claim_inventory import _row
from tests.test_call_b_durable_retry import _state, _inventory, _observation, CLAIM


ANCHOR = "and accelerates convergence"
PROPOSITION = "Momentum accelerates convergence."
DRAFT = f"A velocity term damps oscillation {ANCHOR}.\n\nA takeaway repeats {ANCHOR}."
SECTIONS = {key: DRAFT for key in (
    "problem_framing", "technical_dive", "code_snippets", "takeaways")}


def ambiguous_state(base_state):
    state = {**base_state, "quality_episode": 1, "quality_attempts": 1,
             "iterations": 1, "draft_markdown": DRAFT, "draft_status": "valid",
             "draft_sections": dict(SECTIONS), "verification_status": "inventory_failed",
             "total_tokens": 12347, "total_cost_usd": 0.00823235,
             "provider_usage_status": "known", "policy_diagnostics": [{"reason": "retained"}]}
    state["claim_inventory"] = build_claim_inventory(
        run_id=state["run_id"], iteration=1, draft_markdown=DRAFT,
        raw_claims=[_row(PROPOSITION, ANCHOR)], brief_requirements=[])
    state["verification_identity"] = workflow.identity(state)
    return state


def test_current_ambiguity_stops_without_mutating_evidence(base_state):
    state = ambiguous_state(base_state)
    before = deepcopy(state)
    result = workflow.verification_failure(state)
    assert result == {
        "terminal_status": "execution_failed",
        "terminal_message": "Claim inventory has an ambiguous draft anchor; content redraft blocked.",
    }
    assert state == before
    current = {**state, **result}
    assert nodes.route_after_verify(current) == END
    assert current["quality_attempts"] == 1
    assert len(current["claim_inventory"]["claims"][0]["occurrences"]) == 2
    assert not workflow.verification_eligible(current)
    assert not workflow.gate1_eligible(current)
    assert workflow.verification_failure(current) == {}


@pytest.mark.parametrize("field,value", [
    ("run_id", "other-run"), ("iteration", 2), ("draft_sha256", "stale"),
])
def test_stale_inventory_does_not_gain_new_subtype_authority(base_state, field, value):
    state = ambiguous_state(base_state)
    state["claim_inventory"][field] = value
    assert not workflow.inventory_current(state)
    # Existing recoverable-output handling is unchanged, with no acceptance authority.
    assert workflow.verification_failure(state) == {}
    assert not workflow.verification_eligible(state) and not workflow.gate1_eligible(state)
    # Completed verification cannot use the stale inventory to advance.
    state["verification_status"] = "completed"
    assert workflow.verification_failure(state)["terminal_status"] == "execution_failed"
    assert nodes.route_after_verify(state) == END


@pytest.mark.parametrize("field", ["run_id", "draft_sha256", "episode", "attempt", "iteration"])
def test_wrong_verification_identity_remains_terminal(base_state, field):
    state = ambiguous_state(base_state)
    state["verification_identity"][field] = "wrong"
    result = workflow.verification_failure(state)
    assert result["terminal_status"] == "execution_failed"
    assert "ambiguous" not in result["terminal_message"]
    assert nodes.route_after_verify(state) == END


@pytest.mark.parametrize("inventory", [None, [], "malformed", {}, {"claims": None}])
def test_malformed_inventory_retains_existing_output_failure_handling(base_state, inventory):
    state = ambiguous_state(base_state)
    state["claim_inventory"] = inventory
    assert workflow.verification_failure(state) == {}
    assert not workflow.verification_eligible(state)


@pytest.mark.parametrize("claims", [None, {}, "bad", [None], ["bad"]])
def test_malformed_claims_cannot_enter_ambiguity_subtype(base_state, claims):
    state = ambiguous_state(base_state)
    state["claim_inventory"]["claims"] = claims
    assert workflow.verification_failure(state) == {}
    assert not workflow.verification_eligible(state)


@pytest.mark.parametrize("status", ["parse_failed", "verification_error", "skipped_cost_gate"])
def test_unrelated_statuses_keep_existing_routes(base_state, status):
    state = ambiguous_state(base_state)
    state["verification_status"] = status
    result = workflow.verification_failure(state)
    if status == "parse_failed":
        assert result == {}
    else:
        assert result["terminal_status"] == "execution_failed"


@pytest.mark.parametrize("draft,rows,validity,occurrences", [
    ("A improves speed.", [_row("A improves speed.", "A improves speed.")], ANCHOR_BOUND, 1),
    ("A improves speed. A improves speed.", [_row("A improves speed.", "A improves speed.")], ANCHOR_BOUND_MULTIPLE, 2),
    ("A improves speed and storage.", [_row("A improves speed.", "A improves speed and storage.")], ANCHOR_BOUND, 1),
    ("A improves speed.", [_row("A improves speed.", "missing")], ANCHOR_FAILED, 0),
])
def test_binder_controls_and_nonambiguous_inventory_failures_unchanged(base_state, draft, rows, validity, occurrences):
    state = ambiguous_state(base_state)
    state["draft_markdown"] = draft
    state["claim_inventory"] = build_claim_inventory(
        run_id=state["run_id"], iteration=1, draft_markdown=draft, raw_claims=rows)
    state["verification_identity"] = workflow.identity(state)
    claim = state["claim_inventory"]["claims"][0]
    assert claim["anchor_validity"] == validity and len(claim["occurrences"]) == occurrences
    assert workflow.verification_failure(state) == {}
    assert nodes.route_after_verify(state) == "draft"


@pytest.mark.parametrize("reverse", [False, True])
def test_unique_duplicate_or_metadata_cannot_hide_ambiguous_claim(base_state, reverse):
    state = ambiguous_state(base_state)
    state["draft_markdown"] += "\nA unique momentum sentence."
    rows = [_row(PROPOSITION, ANCHOR), _row(PROPOSITION, "A unique momentum sentence.")]
    if reverse:
        rows.reverse()
    state["claim_inventory"] = build_claim_inventory(
        run_id=state["run_id"], iteration=1, draft_markdown=state["draft_markdown"], raw_claims=rows)
    state["verification_identity"] = workflow.identity(state)
    claim = state["claim_inventory"]["claims"][0]
    assert claim["anchor_validity"] == ANCHOR_AMBIGUOUS and len(claim["occurrences"]) == 3
    # All inventory claims require resolution; an eligibility flag cannot hide ambiguity.
    claim["call_b_eligible"] = False
    assert workflow.verification_failure(state)["terminal_status"] == "execution_failed"


@pytest.mark.parametrize("terminal,cost,expected", [
    ("policy_blocked", 0.01, {}), (None, 0.10, {"terminal_status": "resource_exhausted", "terminal_message": "resource exhausted"}),
])
def test_existing_terminal_and_cost_precedence(base_state, terminal, cost, expected):
    state = ambiguous_state(base_state)
    state.update(terminal_status=terminal, total_cost_usd=cost)
    assert workflow.verification_failure(state) == expected


def test_real_verify_graph_stops_and_sqlite_resume_dispatches_nothing(base_state, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    draft = Mock(return_value={"draft_status": "valid", "draft_sections": SECTIONS, "draft_markdown": DRAFT})
    monkeypatch.setattr(graph_mod, "retrieve_node", lambda _state: {})
    monkeypatch.setattr(graph_mod, "draft_node", draft)
    client = FakeLLMClient(responses=[fake_response(json.dumps([_row(PROPOSITION, ANCHOR)]))])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    path = tmp_path / "checkpoint.sqlite"
    config = {"configurable": {"thread_id": base_state["run_id"]}}
    with closing(sqlite3.connect(path, check_same_thread=False)) as conn:
        graph = graph_mod.build_graph(checkpointer=SqliteSaver(conn))
        result = graph.invoke(base_state, config)
        assert result["terminal_status"] == "execution_failed" and result["quality_attempts"] == 1
        assert result["verification_status"] == "inventory_failed"
        assert result["total_tokens"] == 100 and len(result["claim_inventory"]["claims"]) == 1
        assert not graph.get_state(config).next and "__interrupt__" not in result
        assert not result.get("html_output") and not result.get("git_status")
    assert draft.call_count == 1 and client.calls == 1  # only initial Call A, no Call B
    monkeypatch.setattr(nodes, "_get_client", Mock(side_effect=AssertionError("resume dispatched provider")))
    draft.reset_mock(side_effect=True)
    draft.side_effect = AssertionError("resume dispatched another draft")
    with closing(sqlite3.connect(path, check_same_thread=False)) as conn:
        graph = graph_mod.build_graph(checkpointer=SqliteSaver(conn))
        for _ in range(2):
            replay = graph.invoke(None, config)
            assert replay["terminal_status"] == "execution_failed" and replay["quality_attempts"] == 1
            assert replay["claim_inventory"] == result["claim_inventory"]
            assert replay["total_tokens"] == 100 and not graph.get_state(config).next
    draft.assert_not_called()
    nodes._get_client.assert_not_called()
    assert not (tmp_path / "outputs/articles").exists()


def test_valid_material_failure_still_routes_targeted_feedback(base_state, monkeypatch):
    state = _state(base_state)
    observation = json.loads(_observation(entailed=False))
    observation["observations"][0]["blockers"] = [{
        "kind": "limitation", "explanation": "Only suitably small steps are supported.",
        "evidence_quotes": [{"evidence_id": "WEB-001", "quote": "A small step follows the negative gradient."}],
    }]
    client = FakeLLMClient(responses=[fake_response(_inventory()), fake_response(json.dumps(observation))])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    update = workflow.verify_step(state, run=nodes.verify_node)
    current = {**state, **update}
    assert update["verification_status"] == "completed" and not update.get("terminal_status")
    assert nodes.route_after_verify(current) == "draft" and client.calls == 2
    assert nodes.material_policy_result(current).material_safety_state == "repairable"
    assert not workflow.gate1_eligible(current)
    feedback, obligations = nodes._format_semantic_blocker_feedback(current["grounding_report"])
    assert len(obligations) == 1 and CLAIM in feedback and "WEB-001" in feedback
    assert "Only suitably small steps" in feedback and "Do NOT resolve a blocker by deleting" in feedback
    # The real next request must receive the obligation, not merely a helper string.
    captured = []
    class StopBeforeTransport(BaseException):
        pass
    def capture(_client, **kwargs):
        captured.append(kwargs)
        raise StopBeforeTransport()
    monkeypatch.setattr(nodes, "_llm_call", capture)
    with pytest.raises(StopBeforeTransport):
        nodes.draft_node(current)
    user_message = captured[0]["messages"][1]["content"]
    assert feedback in user_message and current["grounding_report"][0]["claim_id"] in user_message
    exhausted = {**current, "quality_attempts": 2}
    exhausted["verification_identity"] = workflow.identity(exhausted)
    assert workflow.verification_failure(exhausted)["terminal_status"] == "terminal_quality_exhausted"
