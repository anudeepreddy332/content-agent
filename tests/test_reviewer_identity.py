"""Reviewer credential and resume boundaries; all external dependencies are offline."""
from __future__ import annotations

import builtins
import queue
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent import nodes
from agent import workflow
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
REVIEWER_ID = "pilot-reviewer"


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
    review_id = server.REGISTRY[run_id]["interrupt_payload"]["review_id"]
    general = {"Authorization": f"Bearer {API_TOKEN}"}
    client = TestClient(server.app)
    try:
        url = f"/{surface}/{run_id}/approve"
        assert client.post(url, headers=general).status_code == 401
        assert client.post(url, headers={**general, "X-Reviewer-Token": API_TOKEN}).status_code == 401
        assert visited == []
        response = client.post(url, headers={**general, "X-Reviewer-Token": REVIEWER_TOKEN,
                                             "X-Review-Id": review_id},
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
             "gate1_reviewer_id": REVIEWER_ID, "gate1_review_origin": "authenticated"}
    graph, visited = _gate_graph("html")
    run_id = state["run_id"]
    monkeypatch.setattr(server, "GRAPH", graph)
    monkeypatch.setattr(server, "_write_telemetry", lambda _state: None)
    monkeypatch.setattr(server, "REGISTRY", {run_id: {
        "status": "queued", "initial_state": state, "result": None,
        "interrupt_payload": None, "error": None, "events": queue.Queue(),
    }})
    server._advance(run_id, state, False)
    review_id = server.REGISTRY[run_id]["interrupt_payload"]["review_id"]
    client = TestClient(server.app)
    try:
        general = {"Authorization": f"Bearer {API_TOKEN}"}
        assert client.post(f"/runs/{run_id}/approve", headers=general).status_code == 401
        ok = client.post(f"/runs/{run_id}/approve", headers={**general,
                                                              "X-Reviewer-Token": REVIEWER_TOKEN,
                                                              "X-Review-Id": review_id})
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


def test_synthetic_approval_cannot_cross_into_production(credentials, monkeypatch, article):
    monkeypatch.setenv("HITL_TEST_MODE", "1")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    monkeypatch.setenv("HITL_TEST_REVIEWER_ID", "test:synthetic")
    synthetic = nodes.hitl_node(article)
    assert synthetic["gate1_review_origin"] == "synthetic"
    monkeypatch.setenv("HITL_TEST_MODE", "0")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("PUBLISH_TARGET", "demo")
    resumed = {**article, **synthetic}
    assert workflow.html_step(resumed, run=lambda _: {"html_output": "unsafe"})["terminal_status"] == "execution_failed"
    assert nodes.html_gen_node(resumed)["terminal_status"] == "execution_failed"
    assert nodes.hitl_html_node({**resumed, "html_output": "<p>unsafe</p>"})["terminal_status"] == "execution_failed"
    writes = Mock(side_effect=AssertionError("archive write attempted"))
    monkeypatch.setattr(nodes, "_atomic_write_utf8", writes)
    assert nodes.git_node({**resumed, "gate2_reviewer_id": "test:synthetic",
                           "gate2_review_origin": "synthetic"})["git_status"] == "failed"
    writes.assert_not_called()
    import api.server as server
    monkeypatch.setattr(server, "REGISTRY", {article["run_id"]: {
        "status": "complete", "result": {**resumed, "gate2_reviewer_id": "test:synthetic",
                                          "gate2_review_origin": "synthetic", "git_status": "merged"},
    }})
    push = Mock(side_effect=AssertionError("publication attempted"))
    monkeypatch.setattr(server.subprocess, "run", push)
    client = TestClient(server.app)
    try:
        assert client.post(f"/ui/runs/{article['run_id']}/publish",
                           headers={"Authorization": f"Bearer {API_TOKEN}"}).status_code == 409
    finally:
        client.close()
    push.assert_not_called()
    monkeypatch.setenv("HITL_REVIEWER_ID", "test:synthetic")
    with pytest.raises(ReviewerConfigurationError):
        authenticate_reviewer(REVIEWER_TOKEN)


def test_synthetic_provenance_survives_checkpoint_restart(credentials, monkeypatch, article, tmp_path):
    monkeypatch.setenv("HITL_TEST_MODE", "1")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    monkeypatch.setenv("HITL_TEST_REVIEWER_ID", "test:checkpoint")
    db = tmp_path / "synthetic.sqlite"
    config = {"configurable": {"thread_id": article["run_id"]}}
    connection = sqlite3.connect(db, check_same_thread=False)
    graph, _ = _gate_graph("article", SqliteSaver(connection))
    result = graph.invoke(article, config)
    assert result["gate1_review_origin"] == "synthetic"
    connection.close()
    monkeypatch.setenv("HITL_TEST_MODE", "0")
    monkeypatch.setenv("PUBLISH_TARGET", "demo")
    restarted_connection = sqlite3.connect(db, check_same_thread=False)
    try:
        restarted, _ = _gate_graph("article", SqliteSaver(restarted_connection))
        checkpointed = restarted.get_state(config).values
        assert checkpointed["gate1_review_origin"] == "synthetic"
        assert workflow.html_step(checkpointed, run=lambda _: {"html_output": "unsafe"})["terminal_status"] == "execution_failed"
    finally:
        restarted_connection.close()


def test_concurrent_mixed_surface_approvals_consume_once(credentials, monkeypatch, article):
    import api.server as server

    run_id = article["run_id"]
    submitted = []
    monkeypatch.setattr(server, "_submit", lambda *a, **kw: submitted.append((a, kw)))
    monkeypatch.setattr(server, "_submit_stream", lambda *a, **kw: submitted.append((a, kw)))
    monkeypatch.setattr(server, "REGISTRY", {run_id: {
        "status": "queued", "initial_state": article, "result": None,
        "interrupt_payload": None, "error": None, "events": queue.Queue(),
    }})
    with server.LOCK:
        review = server._issue_review(run_id, {"type": "hitl_review", "run_id": run_id,
                                               "draft_sha256": workflow.identity(article)["draft_sha256"]})
    headers = {"Authorization": f"Bearer {API_TOKEN}", "X-Reviewer-Token": REVIEWER_TOKEN,
               "X-Review-Id": review["review_id"]}

    def approve(surface):
        client = TestClient(server.app)
        try:
            return client.post(f"/{surface}/{run_id}/approve", headers=headers).status_code
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(approve, ("runs", "ui/runs")))
    assert sorted(statuses) == [200, 409]
    assert len(submitted) == 1
    assert server.REGISTRY[run_id]["status"] == "resuming"


def test_checkpoint_restart_issues_fresh_review_id(credentials, monkeypatch, article, tmp_path):
    import api.server as server

    db = tmp_path / "review-checkpoints.sqlite"
    connection = sqlite3.connect(db, check_same_thread=False)
    graph, visited = _gate_graph("article", SqliteSaver(connection))
    run_id = article["run_id"]
    monkeypatch.setattr(server, "GRAPH", graph)
    monkeypatch.setattr(server, "_write_telemetry", lambda _state: None)
    monkeypatch.setattr(server, "REGISTRY", {run_id: {
        "status": "queued", "initial_state": article, "result": None,
        "interrupt_payload": None, "error": None, "events": queue.Queue(),
    }})
    server._advance(run_id, article, False)
    old_id = server.REGISTRY[run_id]["interrupt_payload"]["review_id"]
    connection.close()
    restarted_connection = sqlite3.connect(db, check_same_thread=False)
    restarted, _ = _gate_graph("article", SqliteSaver(restarted_connection))
    monkeypatch.setattr(server, "GRAPH", restarted)
    server.REGISTRY.clear()
    headers = {"Authorization": f"Bearer {API_TOKEN}", "X-Reviewer-Token": REVIEWER_TOKEN}
    try:
        client = TestClient(server.app)
        try:
            review = client.get(f"/runs/{run_id}", headers=headers).json()["review"]
            assert review["review_id"] != old_id
            assert client.post(f"/runs/{run_id}/approve", headers={**headers, "X-Review-Id": old_id}).status_code == 409
            assert client.post(f"/runs/{run_id}/approve", headers={**headers, "X-Review-Id": review["review_id"]}).status_code == 200
        finally:
            client.close()
        assert server.REGISTRY[run_id]["result"]["gate1_review_origin"] == "authenticated"
    finally:
        restarted_connection.close()


def test_changed_checkpoint_artifact_blocks_reserved_resume(credentials, monkeypatch, article):
    import api.server as server

    run_id = article["run_id"]
    original = workflow.identity(article)["draft_sha256"]
    payload = {"type": "hitl_review", "run_id": run_id, "draft_sha256": original}
    changed = {**payload, "draft_sha256": "f" * 64 if original != "f" * 64 else "e" * 64}
    interrupt = SimpleNamespace(value=changed)
    snapshot = SimpleNamespace(config={"configurable": {"thread_id": run_id}},
                               tasks=(SimpleNamespace(interrupts=(interrupt,)),))
    invoke = Mock(side_effect=AssertionError("stale approval resumed graph"))
    monkeypatch.setattr(server, "GRAPH", SimpleNamespace(get_state=lambda _config: snapshot,
                                                          invoke=invoke))
    monkeypatch.setattr(server, "REGISTRY", {run_id: {
        "status": "queued", "initial_state": article, "result": None,
        "interrupt_payload": None, "error": None, "events": queue.Queue(),
    }})
    with server.LOCK:
        issued = server._issue_review(run_id, payload)
        server._reserve_review(run_id, issued["review_id"])
    server._advance(run_id, Command(resume={"action": "approve", "review_id": issued["review_id"]}),
                    False, reviewer_id=REVIEWER_ID)
    assert server.REGISTRY[run_id]["status"] == "execution_failed"
    invoke.assert_not_called()
