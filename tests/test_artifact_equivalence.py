"""Archive / Git / hash equivalence and publication ref binding. $0."""
import hashlib
import os
import subprocess
from pathlib import Path

os.environ["API_BEARER_TOKEN"] = "test-token"
os.environ["API_SYNC"] = "1"

from fastapi.testclient import TestClient

from agent.html_policy import sanitize_fragment, assemble_trusted_article, sha256_utf8
import agent.nodes as nodes

SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
H = {"Authorization": "Bearer test-token"}


def _trusted():
    return assemble_trusted_article(
        topic="Gradient Descent",
        slug="gradient-descent-test",
        meta_description="meta",
        series_label="Test",
        breadcrumb_section="Test",
        read_time="5",
        problem_framing=sanitize_fragment("<p>framing</p>"),
        technical_dive=sanitize_fragment("<p>dive</p>"),
        code_snippets=sanitize_fragment("<pre><code>print(1)</code></pre>"),
        takeaways=sanitize_fragment("<ul><li>one</li></ul>"),
        citations_html="<li>Sources retrieved via Tavily web search.</li>",
    )


def test_git_node_archive_and_repo_hash_equivalence(tmp_path, monkeypatch, base_state):
    article = _trusted()
    repo = tmp_path / "site"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True, capture_output=True)
    (repo / "README").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_PUSH_ENABLED", "true")
    monkeypatch.setattr(nodes, "THEMACHINIST_REPO_PATH", str(repo), raising=False)
    import config
    monkeypatch.setattr(config, "THEMACHINIST_REPO_PATH", str(repo))

    state = {
        **base_state,
        "html_output": article.html,
        "html_filename": "gradient-descent-test.html",
        "html_sha256": article.sha256,
        "approved_html_sha256": article.sha256,
        "article_body_html": article.body_html,
        "slug": "gradient-descent-test",
        "topic": "Gradient Descent",
        "category": "concept-exploration",
        "draft_sections": {"problem_framing": "framing"},
    }
    out = nodes.git_node(state)
    assert out["git_status"] in ("merged", "tagged_and_merged")
    archive = (tmp_path / "outputs" / "articles" / "gradient-descent-test.html").read_bytes()
    git_bytes = subprocess.check_output(
        ["git", "show", f"{out['git_commit_sha']}:gradient-descent-test.html"], cwd=repo
    )
    assert hashlib.sha256(archive).hexdigest() == article.sha256
    assert hashlib.sha256(git_bytes).hexdigest() == article.sha256
    assert hashlib.sha256(state["html_output"].encode("utf-8")).hexdigest() == article.sha256
    assert out["git_commit_sha"]
    assert archive == git_bytes == article.html.encode("utf-8")


def test_git_node_fails_closed_without_approval(tmp_path, monkeypatch, base_state):
    article = _trusted()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_PUSH_ENABLED", "true")
    state = {**base_state, "html_output": article.html, "html_filename": "x.html",
             "html_sha256": article.sha256, "approved_html_sha256": None}
    out = nodes.git_node(state)
    assert out["git_status"] == "failed"
    assert not list((tmp_path / "outputs" / "articles").glob("*.html")) if (tmp_path / "outputs" / "articles").exists() else True


def test_publish_binds_commit_ref_not_ambient_main(monkeypatch):
    import api.server as srv
    rid = "run-bind"
    srv.REGISTRY[rid] = {
        "status": "complete", "initial_state": {}, "interrupt_payload": None, "error": None,
        "result": {"slug": "my-article", "git_status": "merged",
                   "git_commit_sha": SHA, "publish_expected_remote_sha": SHA},
    }
    calls = []

    def fake_run(args, **kw):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=f"{SHA}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    monkeypatch.setenv("NETLIFY_BASE_URL", "https://tmw-demo-site.netlify.app")
    monkeypatch.setenv("PUBLISH_REMOTE", "origin")
    c = TestClient(srv.app)
    res = c.post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 200
    push = [a for a in calls if a[:2] == ["git", "push"]][0]
    assert f"{SHA}:refs/heads/main" in push
    assert push[3] != "main" or ":" in push[3]


def test_publish_conflicts_if_remote_moved(monkeypatch):
    import api.server as srv
    rid = "run-conflict"
    srv.REGISTRY[rid] = {
        "status": "complete", "initial_state": {}, "interrupt_payload": None, "error": None,
        "result": {"slug": "my-article", "git_status": "merged",
                   "git_commit_sha": SHA, "publish_expected_remote_sha": SHA},
    }
    other = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    def fake_run(args, **kw):
        return subprocess.CompletedProcess(args, 0, stdout=f"{other}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    c = TestClient(srv.app)
    res = c.post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 409
    assert "remote main changed" in res.text
