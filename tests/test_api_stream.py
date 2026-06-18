"""Streaming UI surface (/ui/runs + SSE). Graph fully mocked with a .stream()
that yields ("updates"|"values", chunk) tuples and pauses via get_state, exactly
like LangGraph's stream_mode=["updates","values"]. $0, zero network.
API_SYNC=1 runs _advance_streaming inline so the per-run queue is fully populated
(node events + gate/done + sentinel) by the time the SSE endpoint drains it."""
import os
import json
import pytest

os.environ["API_BEARER_TOKEN"] = "test-token"
os.environ["API_SYNC"] = "1"

from langgraph.types import Command
from types import SimpleNamespace

TOK = "test-token"
H = {"Authorization": "Bearer " + TOK}


def _interrupt(value):
    return SimpleNamespace(tasks=(SimpleNamespace(interrupts=(SimpleNamespace(value=value),)),),
                           values={})


class FakeStreamGraph:
    """draft gate -> approve -> html gate -> approve -> publish (terminal)."""

    def __init__(self):
        self.phase = {}
        self.values = {}

    def stream(self, arg, config, stream_mode=None):
        tid = config["configurable"]["thread_id"]
        if not isinstance(arg, Command):
            self.phase[tid] = "draft_gate"
            st = {"run_id": tid, "slug": "s", "grounding_score": 0.8}
            self.values[tid] = st
            yield ("updates", {"retrieve": {"web_sources": [1, 2, 3], "kb_results": [1, 2]}})
            yield ("values", dict(st))
            yield ("updates", {"verify": {"grounding_score": 0.8,
                                          "grounding_report": [{"claim": "c"}]}})
            yield ("values", dict(st))
            return  # pauses at hitl (draft gate)
        action = arg.resume.get("action")
        phase = self.phase.get(tid)
        if phase == "draft_gate" and action == "approve":
            self.phase[tid] = "html_gate"
            st = {"run_id": tid, "slug": "s", "html_output": "<html/>",
                  "html_filename": "s.html", "grounding_score": 0.8}
            self.values[tid] = st
            yield ("updates", {"html_gen": {"html_filename": "s.html"}})
            yield ("values", dict(st))
            return  # pauses at hitl_html (layout gate)
        if phase == "html_gate" and action == "approve":
            self.phase[tid] = "done"
            st = {"run_id": tid, "slug": "s", "hitl_status": "approved",
                  "html_review_status": "approved", "git_status": "merged",
                  "html_filename": "s.html", "grounding_score": 0.8, "total_cost_usd": 0.01}
            self.values[tid] = st
            yield ("updates", {"git": {"git_status": "merged"}})
            yield ("values", dict(st))
            return  # terminal

    def get_state(self, config):
        tid = config["configurable"]["thread_id"]
        phase = self.phase.get(tid)
        if phase == "draft_gate":
            return _interrupt({"type": "hitl_review", "draft_markdown": "# draft",
                               "grounding_report": [], "grounding_score": 0.8,
                               "reflection_score": 8, "reflection_notes": "ok"})
        if phase == "html_gate":
            return _interrupt({"type": "hitl_html_review", "html_output": "<html/>",
                               "html_filename": "s.html", "validation_warnings": [],
                               "grounding_score": 0.8})
        return SimpleNamespace(tasks=(), values=self.values.get(tid, {}))


@pytest.fixture
def client(monkeypatch):
    import api.server as srv
    monkeypatch.setattr(srv, "GRAPH", FakeStreamGraph())
    monkeypatch.setattr(srv, "_write_telemetry", lambda state: "outputs/runs/r.json")
    from fastapi.testclient import TestClient
    return TestClient(srv.app)


def _drain(client, rid):
    """Read one SSE segment to its segment_end sentinel; return parsed events."""
    out = []
    with client.stream("GET", f"/ui/runs/{rid}/events?token={TOK}") as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            out.append(json.loads(line[5:].strip()))
            if out[-1].get("event") == "segment_end":
                break
    return out


def test_sse_requires_token(client):
    rid = client.post("/ui/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    assert client.get(f"/ui/runs/{rid}/events?token=wrong").status_code == 401


def test_stream_full_two_gate_publish(client):
    r = client.post("/ui/runs", json={"topic": "Gradient Descent"}, headers=H)
    assert r.status_code == 202
    rid = r.json()["run_id"]

    seg1 = _drain(client, rid)
    nodes = [e["node"] for e in seg1 if e["event"] == "node"]
    assert nodes == ["retrieve", "verify"]
    assert seg1[0]["headline"]["web_sources"] == 3
    gate1 = [e for e in seg1 if e["event"] == "gate"][0]
    assert gate1["review"]["type"] == "hitl_review"

    assert client.post(f"/ui/runs/{rid}/approve", headers=H).status_code == 200
    seg2 = _drain(client, rid)
    assert [e["node"] for e in seg2 if e["event"] == "node"] == ["html_gen"]
    assert [e for e in seg2 if e["event"] == "gate"][0]["review"]["type"] == "hitl_html_review"

    assert client.post(f"/ui/runs/{rid}/approve", headers=H).status_code == 200
    seg3 = _drain(client, rid)
    done = [e for e in seg3 if e["event"] == "done"][0]
    assert done["status"] == "complete"
    assert done["summary"]["git_status"] == "merged"
    assert done["summary"]["slug"] == "s"
    # poll endpoint stays consistent for streamed runs
    assert client.get(f"/runs/{rid}", headers=H).json()["status"] == "complete"


def test_stream_resume_requires_awaiting_review(client):
    rid = client.post("/ui/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    _drain(client, rid)                       # advance to the draft gate
    client.post(f"/ui/runs/{rid}/approve", headers=H)
    _drain(client, rid)                       # advance to the html gate
    client.post(f"/ui/runs/{rid}/approve", headers=H)
    _drain(client, rid)                       # terminal
    assert client.post(f"/ui/runs/{rid}/approve", headers=H).status_code == 409
