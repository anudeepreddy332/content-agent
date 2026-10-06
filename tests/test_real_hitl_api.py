"""Real HITL interrupt/resume and HTTP contracts; synthetic state, no providers."""

import builtins
import hashlib
import queue
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

import agent.nodes as nodes
from agent.state import AgentState
from tests.test_citation_policy import E1, _claim_row, _verified
from tests.workflow_fixtures import current_quality


@pytest.fixture
def api_gate_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.delenv("HITL_AUTO_APPROVE", raising=False)
    monkeypatch.setenv("GIT_PUSH_ENABLED", "false")
    cli_input = Mock(side_effect=AssertionError("API gate must never call CLI input"))
    monkeypatch.setattr(builtins, "input", cli_input)
    monkeypatch.setattr(
        nodes, "_get_client", Mock(side_effect=AssertionError("No provider calls"))
    )
    monkeypatch.setattr(
        nodes, "git_node", Mock(side_effect=AssertionError("No real publication"))
    )
    monkeypatch.chdir(tmp_path)
    return cli_input


@pytest.fixture
def html_state(base_state):
    html = "<html><body><p>Safe synthetic gate payload.</p></body></html>"
    return {
        **base_state,
        "html_output": html,
        "html_filename": "gate-test.html",
        "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
        "html_review_status": None,
        "approved_html_sha256": None,
    }


@pytest.fixture
def article_state(base_state):
    draft = "Dropout randomly zeroes activations during training."
    state, _ = _verified(base_state, draft, [_claim_row(draft)], [[E1]])
    return current_quality(state)


def _gate_graph(gate):
    """Keep production node/router real; downstream markers have no side effects."""
    visited = []
    builder = StateGraph(AgentState)
    node = nodes.hitl_html_node if gate == "html" else nodes.hitl_node
    router = nodes.route_after_html_review if gate == "html" else nodes.route_after_hitl
    destinations = ("git", "html_revise") if gate == "html" else ("html_gen", "draft")
    builder.add_node("review", node)
    builder.set_entry_point("review")
    for destination in destinations:
        def marker(state, name=destination):
            visited.append(name)
            return {}

        builder.add_node(destination, marker)
        builder.add_edge(destination, END)
    builder.add_conditional_edges(
        "review", router, {END: END, **{name: name for name in destinations}}
    )
    return builder.compile(checkpointer=InMemorySaver()), visited


@pytest.mark.parametrize(
    "decision,status,route",
    [
        ({"action": "reject"}, "rejected", END),
        ({"action": "unknown"}, "rejected", END),
        ({"unrelated": "missing action"}, "rejected", END),
        ({"action": None}, "rejected", END),
        ({"action": "feedback", "feedback": "  "}, "rejected", END),
        ({"action": "approve"}, "approved", "git"),
        ({"action": "feedback", "feedback": "  Move the sources.  "}, "changes", "html_revise"),
        ({"action": "request_changes", "feedback": "Move the sources."}, "changes", "html_revise"),
    ],
)
def test_real_html_interrupt_resume(api_gate_environment, html_state, decision, status, route):
    graph, visited = _gate_graph("html")
    config = {"configurable": {"thread_id": "html-gate"}}
    initial = graph.invoke(html_state, config)
    assert initial["__interrupt__"][0].value["type"] == "hitl_html_review"
    assert visited == []
    assert graph.get_state(config).values.get("approved_html_sha256") is None

    result = graph.invoke(Command(resume=decision), config)

    assert "__interrupt__" not in result
    assert result["html_review_status"] == status
    assert nodes.route_after_html_review(result) == route
    assert visited == ([] if route == END else [route])
    api_gate_environment.assert_not_called()
    if status == "approved":
        assert result["approved_html_sha256"] == html_state["html_sha256"]
    elif status == "changes":
        assert result["html_feedback"] == "Move the sources."
        assert result["approved_html_sha256"] is None
    else:
        assert "git" not in visited
        assert result["approved_html_sha256"] is None


@pytest.mark.parametrize(
    "decision,status,route",
    [
        ({"action": "approve"}, "approved", "html_gen"),
        ({"action": "reject"}, "rejected", END),
        ({"action": "feedback", "feedback": "Simplify the example."}, "feedback", "draft"),
    ],
)
def test_real_article_api_contract(api_gate_environment, article_state, decision, status, route):
    graph, visited = _gate_graph("article")
    config = {"configurable": {"thread_id": "article-gate"}}
    initial = graph.invoke(article_state, config)
    assert initial["__interrupt__"][0].value["type"] == "hitl_review"
    assert visited == []

    result = graph.invoke(Command(resume=decision), config)

    assert result["hitl_status"] == status
    assert nodes.route_after_hitl(result) == route
    assert visited == ([] if route == END else [route])
    api_gate_environment.assert_not_called()
    if status == "feedback":
        assert result["hitl_feedback"] == decision["feedback"]


@pytest.mark.parametrize("surface", ["runs", "ui/runs"])
@pytest.mark.parametrize("gate", ["html", "article"])
@pytest.mark.parametrize("action", ["approve", "reject", "feedback"])
def test_http_review_resumes_real_gate(
    monkeypatch, api_gate_environment, html_state, article_state, surface, gate, action
):
    import api.server as server

    graph, visited = _gate_graph(gate)
    state = html_state if gate == "html" else article_state
    run_id = "real-http-gate"
    monkeypatch.setenv("API_BEARER_TOKEN", "real-gate-test-token")
    monkeypatch.setenv("API_SYNC", "1")
    monkeypatch.setattr(server, "GRAPH", graph)
    monkeypatch.setattr(server, "_write_telemetry", lambda state: None)
    monkeypatch.setattr(server, "REGISTRY", {
        run_id: {
            "status": "queued", "initial_state": state, "result": None,
            "interrupt_payload": None, "error": None, "events": queue.Queue(),
        }
    })
    advance = server._advance_streaming if surface == "ui/runs" else server._advance
    advance(run_id, state, False)
    assert server.REGISTRY[run_id]["status"] == "awaiting_review"
    assert visited == []
    headers = {"Authorization": "Bearer real-gate-test-token"}
    body = {"feedback": "Simplify the layout."} if action == "feedback" else None
    client = TestClient(server.app)
    try:
        response = client.post(f"/{surface}/{run_id}/{action}", headers=headers, json=body)
    finally:
        client.close()
    assert response.status_code == 200
    result = server.REGISTRY[run_id]["result"]
    status_key = "html_review_status" if gate == "html" else "hitl_status"
    expected_status = "changes" if action == "feedback" and gate == "html" else {
        "approve": "approved", "reject": "rejected", "feedback": "feedback"
    }[action]
    assert result[status_key] == expected_status
    assert server.REGISTRY[run_id]["status"] == ("terminal_rejected" if action == "reject" else "complete")
    expected_route = {
        "html": {"approve": "git", "reject": END, "feedback": "html_revise"},
        "article": {"approve": "html_gen", "reject": END, "feedback": "draft"},
    }[gate][action]
    assert visited == ([] if expected_route == END else [expected_route])
    api_gate_environment.assert_not_called()
    if action == "reject":
        assert "git" not in visited
