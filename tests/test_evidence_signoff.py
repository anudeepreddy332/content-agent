"""Offline adversarial Gate-1 evidence signoff; no providers or publication."""
from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock
import queue
import sqlite3

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from agent import nodes, workflow
from agent.evidence_signoff import EvidenceSignoffError, material_review_bundle, receipt_valid
from agent.reviewer_identity import reviewer_context
from tests.test_citation_policy import _claim_row, _verified
from tests.test_real_hitl_api import _gate_graph

REVIEW_ID = "server-issued-review-0123456789"
REVIEWER = "authenticated-reviewer"


@pytest.fixture
def article(base_state):
    draft = "Gradient descent converges with a suitably chosen learning rate."
    passage = "Convergence requires a suitably chosen learning rate for this setting."
    state, inventory = _verified(base_state, draft, [_claim_row(draft)], [["WEB-001"]])
    state["web_sources"] = [{"title": "Learning-rate analysis", "url": "https://example.com/analysis",
                             "content": passage, "score": 0.9}]
    state["grounding_report"][0].update(
        claim=draft,
        support_spans=[{"evidence_id": "WEB-001", "start": 0, "end": len(passage), "text": passage}],
        full_entailment=True,
    )
    state["draft_sections"] = {"problem_framing": draft, "technical_dive": draft,
                               "code_snippets": "print(1)", "takeaways": draft}
    assert inventory["claims"][0]["material"] is True
    assert workflow.gate1_eligible(state)
    return state


def submission(state, decision="SUPPORTED"):
    bundle = material_review_bundle(state)
    return {
        "evidence_binding_sha256": bundle["evidence_binding_sha256"],
        "decisions": [{"claim_id": claim_id, "decision": decision,
                       "rationale": "The source passage supports the stated condition."}
                      for claim_id in bundle["material_claim_ids"]],
        "coverage_attestation": {"no_omitted_material_claims": True,
                                 "materiality_classification_correct": True},
    }


def approve(state, body):
    with reviewer_context(REVIEWER, state["run_id"], review_id=REVIEW_ID):
        return nodes.hitl_approve_update(state, body, submitted_review_id=REVIEW_ID)


def test_supported_claim_gets_server_bound_receipt_and_can_render(article):
    bundle = material_review_bundle(article)
    claim = bundle["claims"][0]
    assert claim["claim_text"] == article["draft_markdown"]
    assert claim["draft_contexts"][0]["quote"] == claim["claim_text"]
    assert claim["support_spans"][0]["quote"] in claim["support_spans"][0]["source_context"]["text"]
    assert claim["support_spans"][0]["citation_identity"] == "WEB-001"
    assert "status" not in claim and "full_entailment" not in claim

    accepted = approve(article, submission(article))
    assert accepted["hitl_status"] == "approved"
    receipt = accepted["gate1_evidence_signoff"]
    assert receipt["review_id"] == REVIEW_ID
    assert receipt["reviewer_id"] == REVIEWER
    assert receipt["claim_evidence_bindings"][0]["spans"][0]["evidence_id"] == "WEB-001"
    approved_state = {**article, **accepted}
    assert receipt_valid(approved_state)
    rendered = workflow.html_step(approved_state, run=lambda _state: {"html_output": "<p>rendered</p>"})
    assert rendered["html_output"] == "<p>rendered</p>"


@pytest.mark.parametrize("decision", ["UNSUPPORTED", "CANNOT_DETERMINE", ""])
def test_favorable_call_b_does_not_override_human_negative(article, decision):
    assert article["grounding_report"][0]["status"] == "verified"
    outcome = approve(article, submission(article, decision))
    assert outcome["hitl_status"] == "rejected"
    assert outcome["terminal_status"] == "execution_failed"
    assert "gate1_evidence_signoff" not in outcome


def test_missing_duplicate_and_invalid_coverage_block(article):
    valid = submission(article)
    for changed in (
        {**valid, "decisions": []},
        {**valid, "decisions": valid["decisions"] * 2},
        {**valid, "coverage_attestation": {"no_omitted_material_claims": False,
                                           "materiality_classification_correct": True}},
        {**valid, "coverage_attestation": {"no_omitted_material_claims": True,
                                           "materiality_classification_correct": False}},
    ):
        assert approve(article, changed)["hitl_status"] == "rejected"


def test_changed_draft_inventory_source_and_quote_invalidate_receipt(article):
    accepted = approve(article, submission(article))
    for change in ("draft", "inventory", "source", "span"):
        stale = deepcopy({**article, **accepted})
        if change == "draft":
            stale["draft_markdown"] += " Extra unsupported claim."
        elif change == "inventory":
            stale["claim_inventory"]["claims"][0]["materiality_rationale"] = "changed"
        elif change == "source":
            stale["web_sources"][0]["content"] += " Extra caveat."
        else:
            stale["grounding_report"][0]["support_spans"][0]["end"] -= 1
        assert not receipt_valid(stale), change
        assert "html_output" not in workflow.html_step(stale, run=lambda _: {"html_output": "unsafe"})


def test_receipt_cannot_override_changed_automated_gate_or_write_archive(article, monkeypatch):
    accepted = approve(article, submission(article))
    stale = {**article, **accepted, "reflection_score": 0,
             "gate2_reviewer_id": REVIEWER, "gate2_review_origin": "authenticated",
             "html_output": "<p>unsafe</p>", "html_filename": "unsafe.html"}
    assert not receipt_valid(stale)
    write = Mock()
    monkeypatch.setattr(nodes, "_atomic_write_utf8", write)
    assert nodes.git_node(stale)["git_status"] == "failed"
    write.assert_not_called()


def test_synthetic_approval_cannot_publish_after_runtime_switch(article, monkeypatch):
    monkeypatch.setenv("HITL_TEST_MODE", "1")
    monkeypatch.setenv("HITL_TEST_REVIEWER_ID", "test:signoff")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    monkeypatch.setenv("GIT_PUSH_ENABLED", "false")
    monkeypatch.delenv("PUBLISH_TARGET", raising=False)
    result = nodes.hitl_node(article)
    assert result["hitl_status"] == "approved"
    state = {**article, **result}
    assert receipt_valid(state) and not receipt_valid(state, publishing=True)
    monkeypatch.setenv("HITL_TEST_MODE", "0")
    assert not receipt_valid(state)


def test_reported_omission_uses_existing_feedback_route(article, monkeypatch):
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setattr("langgraph.types.interrupt", lambda _payload: {
        "action": "approve", "signoff": {"coverage_attestation": {
            "no_omitted_material_claims": False,
            "materiality_classification_correct": True,
        }, "feedback": "A material condition was omitted from the draft."},
    })
    result = nodes.hitl_node(article)
    assert result["hitl_status"] == "feedback"
    assert "omitted" in result["hitl_feedback"]


def test_foreign_review_id_and_unbound_source_are_rejected(article):
    with reviewer_context(REVIEWER, article["run_id"], review_id=REVIEW_ID):
        stale = nodes.hitl_approve_update(article, submission(article),
                                          submitted_review_id="other-review")
    assert stale["hitl_status"] == "rejected"
    article["grounding_report"][0]["support_spans"][0]["text"] = "fabricated quote"
    with pytest.raises(EvidenceSignoffError):
        material_review_bundle(article)


def test_receipt_cannot_cross_runs_with_same_draft_and_evidence(article):
    accepted = approve(article, submission(article))
    other = deepcopy({**article, **accepted})
    other["run_id"] = "different-run"
    other["claim_inventory"]["run_id"] = "different-run"
    other["verification_identity"] = workflow.identity(other)
    other["reflection_identity"] = workflow.identity(other)
    other["approved_draft_identity"] = workflow.identity(other)
    assert workflow.gate1_eligible(other)
    assert not receipt_valid(other)


def test_invalid_receipt_has_zero_publish_side_effects(article, monkeypatch):
    import api.server as server

    accepted = approve(article, submission(article))
    invalid = deepcopy({**article, **accepted})
    invalid["web_sources"][0]["content"] += " Changed after review."
    invalid.update(git_status="merged", slug="gradient-descent",
                   gate2_reviewer_id=REVIEWER, gate2_review_origin="authenticated",
                   git_commit_sha="a" * 40, publish_expected_remote_sha="a" * 40)
    assert not receipt_valid(invalid, publishing=True)
    monkeypatch.setenv("API_BEARER_TOKEN", "signoff-test-bearer")
    monkeypatch.setattr(server, "REGISTRY", {article["run_id"]: {
        "status": "complete", "initial_state": {}, "interrupt_payload": None,
        "result": invalid, "error": None,
    }})
    run = Mock()
    monkeypatch.setattr(server.subprocess, "run", run)
    client = TestClient(server.app)
    try:
        response = client.post(f"/ui/runs/{article['run_id']}/publish",
                               headers={"Authorization": "Bearer signoff-test-bearer"})
        assert response.status_code == 409
        run.assert_not_called()
    finally:
        client.close()


@pytest.mark.parametrize("surface", ["runs", "ui/runs"])
def test_api_and_ui_require_exact_signoff_and_reject_replay(article, monkeypatch, surface):
    import api.server as server

    monkeypatch.setenv("API_BEARER_TOKEN", "signoff-test-bearer")
    monkeypatch.setenv("HITL_REVIEWER_TOKEN", "r" * 32)
    monkeypatch.setenv("HITL_REVIEWER_ID", REVIEWER)
    monkeypatch.setenv("API_SYNC", "1")
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    graph, visited = _gate_graph("article")
    run_id = article["run_id"]
    monkeypatch.setattr(server, "GRAPH", graph)
    monkeypatch.setattr(server, "_write_telemetry", lambda _state: None)
    monkeypatch.setattr(server, "REGISTRY", {run_id: {
        "status": "queued", "initial_state": article, "result": None,
        "interrupt_payload": None, "error": None, "events": queue.Queue(),
    }})
    advance = server._advance_streaming if surface == "ui/runs" else server._advance
    advance(run_id, article, False)
    review = server.REGISTRY[run_id]["interrupt_payload"]
    assert review["material_review"]["claims"][0]["support_spans"][0]["quote"]
    assert "review_claims" not in review and "reflection_score" not in review
    headers = {"Authorization": "Bearer signoff-test-bearer", "X-Reviewer-Token": "r" * 32,
               "X-Review-Id": review["review_id"]}
    client = TestClient(server.app)
    try:
        assert client.post(f"/{surface}/{run_id}/approve", headers={**headers,
                           "X-Review-Id": "foreign"}, json=submission(article)).status_code == 409
        assert client.post(f"/{surface}/{run_id}/approve", headers=headers,
                           json=submission(article)).status_code == 200
        assert client.post(f"/{surface}/{run_id}/approve", headers=headers,
                           json=submission(article)).status_code == 409
    finally:
        client.close()
    assert visited == ["html_gen"]
    assert server.REGISTRY[run_id]["result"]["gate1_evidence_signoff"]["review_id"] == review["review_id"]


def test_restart_reissues_id_and_preserves_signoff_binding(article, monkeypatch, tmp_path):
    import api.server as server

    monkeypatch.setenv("API_BEARER_TOKEN", "signoff-test-bearer")
    monkeypatch.setenv("HITL_REVIEWER_TOKEN", "r" * 32)
    monkeypatch.setenv("HITL_REVIEWER_ID", REVIEWER)
    monkeypatch.setenv("API_SYNC", "1")
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    db = tmp_path / "signoff.sqlite"
    conn = sqlite3.connect(db, check_same_thread=False)
    first, _ = _gate_graph("article", SqliteSaver(conn))
    run_id = article["run_id"]
    monkeypatch.setattr(server, "GRAPH", first)
    monkeypatch.setattr(server, "_write_telemetry", lambda _state: None)
    monkeypatch.setattr(server, "REGISTRY", {run_id: {
        "status": "queued", "initial_state": article, "result": None,
        "interrupt_payload": None, "error": None, "events": queue.Queue(),
    }})
    server._advance(run_id, article, False)
    old_id = server.REGISTRY[run_id]["interrupt_payload"]["review_id"]
    conn.close()
    conn = sqlite3.connect(db, check_same_thread=False)
    restarted, visited = _gate_graph("article", SqliteSaver(conn))
    monkeypatch.setattr(server, "GRAPH", restarted)
    server.REGISTRY.clear()
    client = TestClient(server.app)
    headers = {"Authorization": "Bearer signoff-test-bearer", "X-Reviewer-Token": "r" * 32}
    try:
        review = client.get(f"/runs/{run_id}", headers=headers).json()["review"]
        assert review["review_id"] != old_id
        assert client.post(f"/runs/{run_id}/approve", headers={**headers, "X-Review-Id": old_id},
                           json=submission(article)).status_code == 409
        assert client.post(f"/runs/{run_id}/approve", headers={**headers,
                           "X-Review-Id": review["review_id"]}, json=submission(article)).status_code == 200
        assert visited == ["html_gen"]
        checkpoint = restarted.get_state({"configurable": {"thread_id": run_id}}).values
        assert receipt_valid(checkpoint)
    finally:
        client.close()
        conn.close()
