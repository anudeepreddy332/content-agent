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
    base_state["html_output"] = "<!DOCTYPE html><html><body><p>Gradient descent minimizes loss.</p></body></html>"
    base_state["html_feedback"] = "make the paragraph bold"
    bad = "<!DOCTYPE html><html><body><p><b>Gradient descent minimizes loss. It is the best.</b></p></body></html>"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(bad))
    out = nodes.html_revise_node(base_state)
    assert out["html_output"] == base_state["html_output"]          # discarded -> original kept
    assert any("DISCARDED" in e for e in out["error_log"])

def test_html_revise_applies_layout_only(base_state, monkeypatch):
    import agent.nodes as nodes
    base_state["html_output"] = "<!DOCTYPE html><html><body><p>Gradient descent minimizes loss.</p></body></html>"
    base_state["html_feedback"] = "make the paragraph bold"
    good = "<!DOCTYPE html><html><body><p><b>Gradient descent minimizes loss.</b></p></body></html>"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(good))
    out = nodes.html_revise_node(base_state)
    assert out["html_output"] == good                # same words, new markup -> applied
