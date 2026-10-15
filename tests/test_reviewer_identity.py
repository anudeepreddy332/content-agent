"""Reviewer credential and resume boundaries; all external dependencies are offline."""
from __future__ import annotations

import builtins
import queue
import sqlite3
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent import nodes
from agent.reviewer_identity import (
    InvalidReviewerCredential,
    ReviewerConfigurationError,
    authenticate_reviewer,
    current_reviewer_id,
    reviewer_context,
    test_auto_reviewer_id as auto_reviewer_id,
)
from tests.test_citation_policy import E1, _claim_row, _verified
from tests.test_real_hitl_api import _gate_graph
from tests.workflow_fixtures import current_quality

API_TOKEN = "general-test-token"
REVIEWER_TOKEN = "reviewer-secret-" + "x" * 32
REVIEWER_ID = "test:pilot-reviewer"


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv("API_BEARER_TOKEN", API_TOKEN)
    monkeypatch.setenv("HITL_REVIEWER_TOKEN", REVIEWER_TOKEN)
    monkeypatch.setenv("HITL_REVIEWER_ID", REVIEWER_ID)
    monkeypatch.setenv("GIT_PUSH_ENABLED", "false")
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("API_SYNC", "1")


@pytest.fixture
def article(base_state):
    claim = "Dropout randomly zeroes activations during training."
    state, _ = _verified(base_state, claim, [_claim_row(claim)], [[E1]])
    return current_quality(state)


def test_reviewer_mapping_is_distinct_and_server_owned(credentials, monkeypatch):
    assert authenticate_reviewer(REVIEWER_TOKEN) == REVIEWER_ID
    with pytest.raises(InvalidReviewerCredential):
        authenticate_reviewer(API_TOKEN)
    with pytest.raises(InvalidReviewerCredential):
        authenticate_reviewer("")
    monkeypatch.setenv("HITL_REVIEWER_TOKEN", API_TOKEN)
    with pytest.raises(ReviewerConfigurationError):
        authenticate_reviewer(API_TOKEN)
    monkeypatch.delenv("HITL_REVIEWER_ID")
    with pytest.raises(ReviewerConfigurationError):
        authenticate_reviewer(REVIEWER_TOKEN)


def test_context_is_run_bound_and_untrusted_resume_fields_do_not_approve(credentials, article):
    assert current_reviewer_id(article["run_id"]) is None
    without = nodes.hitl_approve_update({**article, "reviewer_id": REVIEWER_ID})
    assert without["hitl_status"] == "rejected"
    with reviewer_context(REVIEWER_ID, "different-run"):
        assert nodes.hitl_approve_update(article)["hitl_status"] == "rejected"
    with reviewer_context(REVIEWER_ID, article["run_id"]):
        accepted = nodes.hitl_approve_update(article)
        assert accepted["gate1_reviewer_id"] == REVIEWER_ID
    assert current_reviewer_id(article["run_id"]) is None


@pytest.mark.parametrize("surface", ["runs", "ui/runs"])
def test_api_approval_needs_reviewer_secret_and_ignores_impersonation(
    credentials, monkeypatch, article, surface
):
    import api.server as server

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
    assert server.REGISTRY[run_id]["status"] == "awaiting_review"
    general = {"Authorization": f"Bearer {API_TOKEN}"}
    client = TestClient(server.app)
    try:
        url = f"/{surface}/{run_id}/approve"
        assert client.post(url, headers=general).status_code == 401
        assert client.post(url, headers={**general, "X-Reviewer-Token": API_TOKEN}).status_code == 401
        assert visited == []
        response = client.post(url, headers={**general, "X-Reviewer-Token": REVIEWER_TOKEN},
                               json={"reviewer_id": "attacker"})
        assert response.status_code == 200
    finally:
        client.close()
    assert server.REGISTRY[run_id]["result"]["gate1_reviewer_id"] == REVIEWER_ID
    assert visited == ["html_gen"]
    assert REVIEWER_TOKEN not in repr(graph.get_state({"configurable": {"thread_id": run_id}}).values)


def test_gate2_and_legacy_publish_require_both_identities(credentials, monkeypatch, base_state):
    import api.server as server

    html = "<html><body>Approved layout</body></html>"
    from hashlib import sha256
    state = {**base_state, "html_output": html, "html_sha256": sha256(html.encode()).hexdigest(),
             "gate1_reviewer_id": REVIEWER_ID}
    graph, visited = _gate_graph("html")
    run_id = state["run_id"]
    monkeypatch.setattr(server, "GRAPH", graph)
    monkeypatch.setattr(server, "_write_telemetry", lambda _state: None)
    monkeypatch.setattr(server, "REGISTRY", {run_id: {
        "status": "queued", "initial_state": state, "result": None,
        "interrupt_payload": None, "error": None, "events": queue.Queue(),
    }})
    server._advance(run_id, state, False)
    client = TestClient(server.app)
    try:
        general = {"Authorization": f"Bearer {API_TOKEN}"}
        assert client.post(f"/runs/{run_id}/approve", headers=general).status_code == 401
        ok = client.post(f"/runs/{run_id}/approve", headers={**general,
                                                              "X-Reviewer-Token": REVIEWER_TOKEN})
        assert ok.status_code == 200
        assert server.REGISTRY[run_id]["result"]["gate2_reviewer_id"] == REVIEWER_ID
        assert visited == ["git"]
        server.REGISTRY[run_id]["result"] = {"git_status": "merged", "run_id": run_id,
                                               "git_commit_sha": "a" * 40,
                                               "publish_expected_remote_sha": "b" * 40}
        assert client.post(f"/ui/runs/{run_id}/publish", headers=general).status_code == 409
    finally:
        client.close()


def test_sqlite_restart_requires_fresh_auth_context_and_stores_no_secret(
    credentials, article, tmp_path
):
    db = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": article["run_id"]}}
    first_connection = sqlite3.connect(db, check_same_thread=False)
    first, _ = _gate_graph("article", SqliteSaver(first_connection))
    assert first.invoke(article, config)["__interrupt__"][0].value["type"] == "hitl_review"
    first_connection.close()

    second_connection = sqlite3.connect(db, check_same_thread=False)
    restarted, _ = _gate_graph("article", SqliteSaver(second_connection))
    try:
        with reviewer_context(REVIEWER_ID, article["run_id"]):
            accepted = restarted.invoke(Command(resume={"action": "approve", "reviewer_id": "attacker"}), config)
        assert accepted["gate1_reviewer_id"] == REVIEWER_ID
    finally:
        second_connection.close()
    assert REVIEWER_TOKEN.encode() not in db.read_bytes()


def test_cli_and_auto_paths_fail_closed_without_identity(credentials, monkeypatch, article):
    monkeypatch.setenv("HITL_MODE", "cli")
    monkeypatch.setattr(builtins, "input", lambda _prompt: "a")
    import getpass
    monkeypatch.setattr(getpass, "getpass", lambda _prompt: API_TOKEN)
    assert nodes.hitl_node(article)["terminal_status"] == "execution_failed"
    monkeypatch.setattr(getpass, "getpass", lambda _prompt: REVIEWER_TOKEN)
    assert nodes.hitl_node(article)["gate1_reviewer_id"] == REVIEWER_ID

    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    assert nodes.hitl_node(article)["terminal_status"] == "execution_failed"
    monkeypatch.setenv("HITL_TEST_MODE", "1")
    monkeypatch.setenv("HITL_TEST_REVIEWER_ID", "test:e2e")
    assert auto_reviewer_id() == "test:e2e"
    assert nodes.hitl_node(article)["gate1_reviewer_id"] == "test:e2e"
    monkeypatch.setenv("GIT_PUSH_ENABLED", "true")
    assert auto_reviewer_id() is None
    assert nodes.hitl_node(article)["terminal_status"] == "execution_failed"


def test_invalid_identity_cannot_reach_git_archive(credentials, monkeypatch, base_state):
    html = "<html><body>content</body></html>"
    state = {**base_state, "html_output": html, "html_filename": "test.html",
             "html_sha256": nodes.sha256_utf8(html), "approved_html_sha256": nodes.sha256_utf8(html)}
    write = Mock(side_effect=AssertionError("Git archive write attempted"))
    monkeypatch.setattr(nodes, "_atomic_write_utf8", write)
    out = nodes.git_node(state)
    assert out["git_status"] == "failed"
    write.assert_not_called()
