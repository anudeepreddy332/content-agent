"""Cloud publish + preview endpoints (/ui/runs/{id}/publish, /preview).
These never touch GRAPH — they only read REGISTRY state (seeded directly) and
shell out to `git push`, which is mocked. $0, zero network, git_node untouched."""
import os
import pytest

os.environ["API_BEARER_TOKEN"] = "test-token"
os.environ["API_SYNC"] = "1"

TOK = "test-token"
H = {"Authorization": "Bearer " + TOK}


@pytest.fixture
def client(monkeypatch):
    import api.server as srv
    monkeypatch.setattr(srv, "_write_telemetry", lambda state: "outputs/runs/r.json")
    from fastapi.testclient import TestClient
    return TestClient(srv.app)


def _seed_run(git_status="merged", slug="s", html_output="<html/>", awaiting_html=False):
    import api.server as srv
    run_id = "run-" + git_status + "-" + slug
    interrupt_payload = None
    status = "complete"
    if awaiting_html:
        status = "awaiting_review"
        interrupt_payload = {"type": "hitl_html_review", "html_output": html_output}
    srv.REGISTRY[run_id] = {
        "status": status, "initial_state": {}, "interrupt_payload": interrupt_payload,
        "result": None if awaiting_html else
                  {"slug": slug, "git_status": git_status, "html_output": html_output},
        "error": None,
    }
    return run_id


# ---- publish ----

def test_publish_requires_auth(client):
    rid = _seed_run()
    assert client.post(f"/ui/runs/{rid}/publish").status_code == 401


def test_publish_404_unknown_run(client):
    assert client.post("/ui/runs/nope/publish", headers=H).status_code == 404


def test_publish_409_when_not_merged(client):
    rid = _seed_run(git_status="dry_run")
    assert client.post(f"/ui/runs/{rid}/publish", headers=H).status_code == 409


def test_publish_409_when_run_still_in_progress(client):
    rid = _seed_run(awaiting_html=True)
    assert client.post(f"/ui/runs/{rid}/publish", headers=H).status_code == 409


def test_publish_success_merged(client, monkeypatch):
    import api.server as srv
    import subprocess
    rid = _seed_run(git_status="merged", slug="my-article")
    fake = subprocess.CompletedProcess(args=["git", "push"], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(srv.subprocess, "run", lambda *a, **kw: fake)
    monkeypatch.setenv("NETLIFY_BASE_URL", "https://tmw-demo-site.netlify.app")
    monkeypatch.setenv("PUBLISH_REMOTE", "origin")
    res = client.post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 200
    assert res.json() == {"live_url": "https://tmw-demo-site.netlify.app/my-article"}


def test_publish_success_tagged_and_merged(client, monkeypatch):
    import api.server as srv
    import subprocess
    rid = _seed_run(git_status="tagged_and_merged", slug="republished")
    fake = subprocess.CompletedProcess(args=["git", "push"], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(srv.subprocess, "run", lambda *a, **kw: fake)
    res = client.post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 200
    assert res.json()["live_url"].endswith("/republished")


def test_publish_failure_returns_500_with_git_error(client, monkeypatch):
    import api.server as srv
    import subprocess
    rid = _seed_run(git_status="merged", slug="fails")
    fake = subprocess.CompletedProcess(args=["git", "push"], returncode=1,
                                        stdout="", stderr="fatal: authentication failed")
    monkeypatch.setattr(srv.subprocess, "run", lambda *a, **kw: fake)
    res = client.post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 500
    assert "authentication failed" in res.text


# ---- preview ----

def test_preview_requires_token(client):
    rid = _seed_run()
    assert client.get(f"/ui/runs/{rid}/preview?token=wrong").status_code == 401


def test_preview_404_unknown_run(client):
    assert client.get(f"/ui/runs/nope/preview?token={TOK}").status_code == 404


def test_preview_returns_html_from_result(client):
    rid = _seed_run(html_output="<p>hi from result</p>")
    res = client.get(f"/ui/runs/{rid}/preview?token={TOK}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "<p>hi from result</p>" in res.text


def test_preview_returns_html_from_pending_gate2(client):
    rid = _seed_run(awaiting_html=True, html_output="<p>hi from gate 2</p>")
    res = client.get(f"/ui/runs/{rid}/preview?token={TOK}")
    assert res.status_code == 200
    assert "<p>hi from gate 2</p>" in res.text
