"""API/HITL security-boundary tests. Graph mocked. $0."""
import os

os.environ["API_BEARER_TOKEN"] = "test-token"
os.environ["API_SYNC"] = "1"

from fastapi.testclient import TestClient

from tests.test_api_stream import FakeStreamGraph, H, TOK, _drain

from agent.html_policy import sanitize_fragment
from tests.conftest import fake_response


def test_preview_endpoint_removed(client=None):
    import api.server as srv
    from fastapi.testclient import TestClient
    c = TestClient(srv.app)
    rid = "run-x"
    srv.REGISTRY[rid] = {"status": "complete", "initial_state": {}, "interrupt_payload": None,
                         "result": {"html_output": "<p>x</p>"}, "error": None}
    assert c.get(f"/ui/runs/{rid}/preview?token={TOK}").status_code == 404
    assert c.get(f"/ui/runs/{rid}/preview", headers=H).status_code == 404


def test_sse_rejects_query_token_even_if_correct():
    import api.server as srv
    srv.GRAPH = FakeStreamGraph()
    c = TestClient(srv.app)
    rid = c.post("/ui/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    assert c.get(f"/ui/runs/{rid}/events?token={TOK}").status_code == 401
    assert c.get(f"/ui/runs/{rid}/events").status_code == 401


def test_sse_header_auth_and_no_token_in_path():
    import api.server as srv
    srv.GRAPH = FakeStreamGraph()
    c = TestClient(srv.app)
    rid = c.post("/ui/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    events = _drain(c, rid)
    assert events[-1]["event"] == "segment_end"


def test_reviewer_security_headers_present():
    import api.server as srv
    c = TestClient(srv.app)
    res = c.get("/health")
    assert res.headers["content-security-policy"].startswith("default-src 'none'")
    assert "script-src 'self'" in res.headers["content-security-policy"]
    assert res.headers["referrer-policy"] == "no-referrer"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["x-frame-options"] == "DENY"
    assert "geolocation=()" in res.headers["permissions-policy"]
    assert res.headers["cross-origin-opener-policy"] == "same-origin"
    assert res.headers["cache-control"] == "no-store"


def test_sse_has_accel_buffering_header():
    import api.server as srv
    srv.GRAPH = FakeStreamGraph()
    c = TestClient(srv.app)
    rid = c.post("/ui/runs", json={"topic": "X"}, headers=H).json()["run_id"]
    with c.stream("GET", f"/ui/runs/{rid}/events", headers=H) as r:
        assert r.status_code == 200
        assert r.headers.get("x-accel-buffering") == "no"


def test_index_has_no_marked_or_innerhtml_source():
    html = open("static/index.html", encoding="utf-8").read()
    js = open("static/app.js", encoding="utf-8").read()
    assert "marked" not in html
    assert "cdn.jsdelivr.net" not in html
    assert "EventSource" not in js
    assert "innerHTML" not in js
    assert "preview?token" not in js
    assert 'sandbox=""' in html
    assert "allow-scripts" not in html
    assert "/static/app.js" in html
    assert "/static/app.css" in html


def test_malicious_reviewer_feedback_is_sanitized_before_persist(base_state, monkeypatch):
    import agent.nodes as nodes
    base_state["article_body_html"] = "<p>Gradient descent minimizes loss.</p>"
    base_state["html_output"] = "<p>Gradient descent minimizes loss.</p>"
    base_state["html_feedback"] = "<script>alert(1)</script> make it bold"
    unsafe = '<script>alert(1)</script><p onclick="alert(1)">Gradient descent minimizes loss.</p>'
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda c, **kw: fake_response(unsafe))
    out = nodes.html_revise_node(base_state)
    html = out["html_output"]
    assert "<script>" not in html.lower()
    assert "onclick" not in html.lower()
    assert out.get("approved_html_sha256") is None
    assert sanitize_fragment(out["article_body_html"]).html == out["article_body_html"]
