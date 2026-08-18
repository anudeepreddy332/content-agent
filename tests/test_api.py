"""B4: API state machine + auth, with the graph fully mocked. $0, zero network.
API_SYNC=1 makes run advancement inline so assertions are deterministic."""
import json
import os
from pathlib import Path

import pytest
from tests.conftest import fake_response

os.environ["API_BEARER_TOKEN"] = "test-token"
os.environ["API_SYNC"] = "1"

from langgraph.types import Command
from types import SimpleNamespace


@pytest.fixture
def client(monkeypatch):
    import api.server as srv

    class FakeGraph:
        """Models both gates per thread_id. draft gate -> approve -> html gate;
        html-gate feedback (request_changes) loops back to the draft gate."""

        def __init__(self):
            self.phase = {}  # thread_id -> "draft" | "html"

        def invoke(self, arg, config):
            tid = config["configurable"]["thread_id"]
            if not isinstance(arg, Command):
                self.phase[tid] = "draft"
                return {"__interrupt__": [SimpleNamespace(value={"type": "hitl_review"})]}
            action = arg.resume.get("action")
            phase = self.phase.get(tid, "draft")
            if phase == "draft":
                if action == "approve":
                    self.phase[tid] = "html"
                    return {"__interrupt__": [SimpleNamespace(value={"type": "hitl_html_review"})]}
                if action == "reject":
                    return {"hitl_status": "rejected", "run_id": "r"}
                # draft-gate feedback -> back to draft gate
                return {"__interrupt__": [SimpleNamespace(value={"type": "hitl_review"})]}
            # phase == "html"
            if action == "approve":
                return {"hitl_status": "approved", "html_review_status": "approved",
                        "git_status": "merged", "grounding_score": 0.9,
                        "total_cost_usd": 0.01, "html_filename": "x.html", "run_id": "r"}
            if action == "reject":
                return {"hitl_status": "approved", "html_review_status": "rejected",
                        "grounding_score": 0.9, "total_cost_usd": 0.01, "run_id": "r"}
            # request_changes -> html_revise -> back to the HTML gate
            self.phase[tid] = "draft"
            return {"__interrupt__": [SimpleNamespace(value={"type": "hitl_html_review"})]}

        def get_state(self, config):
            return SimpleNamespace(tasks=())

    monkeypatch.setattr(srv, "GRAPH", FakeGraph())
    monkeypatch.setattr(srv, "_write_telemetry", lambda state: "outputs/runs/r.json")
    from fastapi.testclient import TestClient
    return TestClient(srv.app)


H = {"Authorization": "Bearer test-token"}


def test_auth_required(client):
    assert client.post("/runs", json={"topic": "x"}).status_code == 401

def test_health_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}

def test_empty_slug_rejected(client):
    assert client.post("/runs", json={"topic": "———"}, headers=H).status_code == 422

def test_two_gate_approve_cycle(client):
    rid = client.post("/runs", json={"topic": "Gradient Descent"}, headers=H).json()["run_id"]
    assert client.get(f"/runs/{rid}", headers=H).json()["review"]["type"] == "hitl_review"
    client.post(f"/runs/{rid}/approve", headers=H)                     # draft gate -> html gate
    g = client.get(f"/runs/{rid}", headers=H).json()
    assert g["status"] == "awaiting_review"
    assert g["review"]["type"] == "hitl_html_review"
    client.post(f"/runs/{rid}/approve", headers=H)                     # html gate -> publish
    body = client.get(f"/runs/{rid}", headers=H).json()
    assert body["status"] == "complete"
    assert body["summary"]["html_review_status"] == "approved"

def test_html_gate_request_changes_loops_to_html_gate(client):
    rid = client.post("/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    client.post(f"/runs/{rid}/approve", headers=H)                      # -> html gate
    client.post(f"/runs/{rid}/feedback", json={"feedback": "move sources below takeaways"}, headers=H)
    g = client.get(f"/runs/{rid}", headers=H).json()
    assert g["status"] == "awaiting_review"
    assert g["review"]["type"] == "hitl_html_review"                    # re-verify the REVISED render

def test_html_gate_reject_does_not_publish(client):
    rid = client.post("/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    client.post(f"/runs/{rid}/approve", headers=H)                     # -> html gate
    client.post(f"/runs/{rid}/reject", headers=H)                      # reject the render
    body = client.get(f"/runs/{rid}", headers=H).json()
    assert body["status"] == "rejected"
    assert body["summary"]["html_review_status"] == "rejected"
    assert body["summary"]["git_status"] is None                      # git never ran

def test_reject_cycle(client):
    rid = client.post("/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    client.post(f"/runs/{rid}/reject", headers=H)
    assert client.get(f"/runs/{rid}", headers=H).json()["status"] == "rejected"

def test_feedback_loops_back_to_review(client):
    rid = client.post("/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    client.post(f"/runs/{rid}/feedback", json={"feedback": "tighten section 2"}, headers=H)
    assert client.get(f"/runs/{rid}", headers=H).json()["status"] == "awaiting_review"

def test_feedback_requires_nonempty(client):
    rid = client.post("/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    assert client.post(f"/runs/{rid}/feedback", json={"feedback": ""}, headers=H).status_code == 422

def test_approve_unknown_run_404(client):
    assert client.post("/runs/nope/approve", headers=H).status_code == 404

def test_approve_after_terminal_409(client):
    rid = client.post("/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    client.post(f"/runs/{rid}/approve", headers=H)                     # draft -> html
    client.post(f"/runs/{rid}/approve", headers=H)                     # html -> complete
    assert client.post(f"/runs/{rid}/approve", headers=H).status_code == 409


def test_poll_graph_crash_writes_upstream_failed_telemetry(tmp_path, monkeypatch):
    """The poll endpoint must classify graph crashes before persistence."""
    import api.server as srv
    from fastapi.testclient import TestClient

    class CrashingGraph:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("injected poll failure")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(srv, "GRAPH", CrashingGraph())
    client = TestClient(srv.app)

    created = client.post("/runs", json={"topic": "Gradient Descent"}, headers=H)
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    status = client.get(f"/runs/{run_id}", headers=H).json()
    assert status["status"] == "error"
    assert status["error"] == "injected poll failure"
    telemetry = json.loads((Path("outputs/runs") / f"{run_id}.json").read_text())
    assert telemetry["verification_status"] == "upstream_failed"
    assert telemetry["error_log"] == ["pipeline crash: injected poll failure"]


def test_html_revise_discards_content_change(base_state, monkeypatch):
    import agent.nodes as nodes
    body = "<p>Gradient descent minimizes loss.</p>"
    base_state["article_body_html"] = body
    base_state["html_output"] = body
    base_state["html_sha256"] = "orig"
    base_state["html_feedback"] = "make the paragraph bold"
    bad = "<p><strong>Gradient descent minimizes loss. It is the best.</strong></p>"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(bad))
    out = nodes.html_revise_node(base_state)
    assert out["html_output"] == base_state["html_output"]
    assert any("DISCARDED" in e for e in out["error_log"])

def test_html_revise_applies_layout_only(base_state, monkeypatch):
    import agent.nodes as nodes
    body = "<p>Gradient descent minimizes loss.</p>"
    base_state["article_body_html"] = body
    base_state["html_output"] = body
    base_state["html_feedback"] = "make the paragraph bold"
    good = "<p><strong>Gradient descent minimizes loss.</strong></p>"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(good))
    out = nodes.html_revise_node(base_state)
    assert "Gradient descent minimizes loss" in out["html_output"]
    assert "<strong>" in out["html_output"]
    assert out["html_output"] != body
    assert out["approved_html_sha256"] is None
    assert "<script>" not in out["html_output"].lower()


def _revise(base_state, monkeypatch, original, revised):
    import agent.nodes as nodes
    base_state["article_body_html"] = original
    base_state["html_output"] = original
    base_state["html_sha256"] = "orig"
    base_state["html_feedback"] = "layout only"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(revised))
    return nodes.html_revise_node(base_state)


@pytest.mark.parametrize("original,revised,label", [
    ("<p>Gradient descent minimizes loss.</p>",
     "<p>Gradient descent minimizes error.</p>", "changed-word"),
    ("<p>Gradient descent minimizes loss.</p>",
     "<p>Gradient descent loss.</p>", "removed-word"),
    ("<p>Gradient descent minimizes loss.</p>",
     "<p>Gradient descent minimizes loss quickly.</p>", "added-word"),
    ("<p>Gradient descent minimizes loss.</p>",
     "<p>Loss minimizes gradient descent.</p>", "reordered-words"),
    ("<p>see</p><pre><code>print(1)\nprint(2)</code></pre>",
     "<p>see</p><pre><code>print(9)\nprint(2)</code></pre>", "changed-code-token"),
    ("<p>see</p><pre><code>print(1)\nprint(2)</code></pre>",
     "<p>see</p><pre><code>print(2)\nprint(1)</code></pre>", "reordered-code-lines"),
])
def test_html_revise_discards_any_visible_or_code_change(base_state, monkeypatch, original, revised, label):
    out = _revise(base_state, monkeypatch, original, revised)
    assert out["html_output"] == original, label
    assert any("DISCARDED" in e for e in out["error_log"]), label


def test_html_revise_accepts_layout_only_with_identical_code(base_state, monkeypatch):
    original = (
        "<p>Gradient descent minimizes loss.</p>"
        "<pre><code>print(1)\nprint(2)</code></pre>"
        "<p>Use <code>lr</code> carefully.</p>"
    )
    revised = (
        "<div class=\"callout callout-info\"><p>Gradient descent minimizes loss.</p></div>"
        "<div class=\"sl-code-block\"><pre><code class=\"language-python\">print(1)\nprint(2)</code></pre></div>"
        "<p>Use <em><code>lr</code></em> carefully.</p>"
    )
    out = _revise(base_state, monkeypatch, original, revised)
    assert not any("DISCARDED" in e for e in out["error_log"])
    assert out["html_output"] != original
    assert "print(1)" in out["html_output"]
    assert "print(2)" in out["html_output"]
    assert "Gradient descent minimizes loss" in out["html_output"]
    import agent.nodes as nodes
    body = "<p>Gradient descent minimizes loss.</p>"
    base_state["article_body_html"] = body
    base_state["html_output"] = body
    base_state["html_sha256"] = "orig"
    base_state["html_feedback"] = "make the paragraph bold"
    bad = "<p><strong>Gradient descent minimizes loss. It is the best.</strong></p>"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(bad))
    out = nodes.html_revise_node(base_state)
    assert out["html_output"] == base_state["html_output"]
    assert any("DISCARDED" in e for e in out["error_log"])

def test_html_revise_applies_layout_only(base_state, monkeypatch):
    import agent.nodes as nodes
    body = "<p>Gradient descent minimizes loss.</p>"
    base_state["article_body_html"] = body
    base_state["html_output"] = body
    base_state["html_feedback"] = "make the paragraph bold"
    good = "<p><strong>Gradient descent minimizes loss.</strong></p>"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(good))
    out = nodes.html_revise_node(base_state)
    assert "Gradient descent minimizes loss" in out["html_output"]
    assert "<strong>" in out["html_output"]
    assert out["html_output"] != body
    assert out["approved_html_sha256"] is None
    assert "<script>" not in out["html_output"].lower()
