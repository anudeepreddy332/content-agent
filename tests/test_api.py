"""B4: API state machine + auth, with the graph fully mocked. $0, zero network.
API_SYNC=1 makes run advancement inline so assertions are deterministic."""
import os
import pytest

os.environ["API_BEARER_TOKEN"] = "test-token"
os.environ["API_SYNC"] = "1"

from langgraph.types import Command
from types import SimpleNamespace


@pytest.fixture
def client(monkeypatch):
    import api.server as srv

    class FakeGraph:
        """First invoke -> interrupt at HITL. Resume with approve -> complete,
        reject -> rejected, feedback -> interrupt again (revise loop)."""
        def invoke(self, arg, config):
            if isinstance(arg, Command):
                action = arg.resume.get("action")
                if action == "approve":
                    return {"hitl_status": "approved", "git_status": "dry-run",
                            "grounding_score": 0.8, "total_cost_usd": 0.01,
                            "html_filename": "x.html", "run_id": "r"}
                if action == "reject":
                    return {"hitl_status": "rejected", "grounding_score": 0.8,
                            "total_cost_usd": 0.01, "run_id": "r"}
                return {"__interrupt__": [SimpleNamespace(value={"type": "hitl_review"})]}
            return {"__interrupt__": [SimpleNamespace(value={"type": "hitl_review",
                                                             "topic": "t"})]}
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

def test_full_approve_cycle(client):
    rid = client.post("/runs", json={"topic": "Gradient Descent"}, headers=H).json()["run_id"]
    assert client.get(f"/runs/{rid}", headers=H).json()["status"] == "awaiting_review"
    assert client.post(f"/runs/{rid}/approve", headers=H).status_code == 200
    body = client.get(f"/runs/{rid}", headers=H).json()
    assert body["status"] == "complete"
    assert body["summary"]["git_status"] == "dry-run"

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

def test_approve_wrong_state_409(client):
    rid = client.post("/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    client.post(f"/runs/{rid}/approve", headers=H)          # -> complete
    assert client.post(f"/runs/{rid}/approve", headers=H).status_code == 409