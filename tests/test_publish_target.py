"""Fail-closed publish-target policy. $0, no network, no real git push."""
import os
import subprocess
from pathlib import Path

import pytest

os.environ["API_BEARER_TOKEN"] = "test-token"
os.environ["API_SYNC"] = "1"

from fastapi.testclient import TestClient

from agent.publish_target import (
    DEMO_GITHUB,
    PRODUCTION_GITHUB,
    canonicalize_https_base,
    evaluate_publish_target,
    github_owner_repo,
)
import agent.nodes as nodes
from agent.html_policy import assemble_trusted_article, sanitize_fragment

DEMO_FORK_URL = "git@github.com:anudeepreddy332/themachinist-website-fork.git"
PROD_REPO_URL = "git@github.com:anudeepreddy332/themachinist-website.git"
DEMO_SITE = "https://tmw-demo-site.netlify.app"
SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
H = {"Authorization": "Bearer test-token"}


def make_git_repo(path: Path, remotes: dict[str, str], *, commit: bool = False,
                  pushurls: dict[str, str | list[str]] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.test"], cwd=path, check=True,
                   capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True, capture_output=True)
    for name, url in remotes.items():
        subprocess.run(["git", "remote", "add", name, url], cwd=path, check=True,
                       capture_output=True)
    for name, urls in (pushurls or {}).items():
        first, *rest = [urls] if isinstance(urls, str) else list(urls)
        subprocess.run(["git", "remote", "set-url", "--push", name, first], cwd=path,
                       check=True, capture_output=True)
        for extra in rest:
            subprocess.run(["git", "remote", "set-url", "--add", "--push", name, extra],
                           cwd=path, check=True, capture_output=True)
    if commit:
        (path / "README").write_text("x", encoding="utf-8")
        subprocess.run(["git", "add", "README"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)
    return path


def enable_demo_publish(monkeypatch, repo: Path, *, remote: str = "origin",
                        git_push: str = "true"):
    monkeypatch.setenv("PUBLISH_TARGET", "demo")
    monkeypatch.setenv("GIT_PUSH_ENABLED", git_push)
    monkeypatch.setenv("THEMACHINIST_REPO_PATH", str(repo))
    monkeypatch.setenv("PUBLISH_REMOTE", remote)
    monkeypatch.setenv("NETLIFY_BASE_URL", DEMO_SITE)
    import config
    monkeypatch.setattr(config, "THEMACHINIST_REPO_PATH", str(repo))


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


def _git_state(base_state, article):
    return {
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


def _seed_publishable(monkeypatch, repo: Path):
    import api.server as srv
    enable_demo_publish(monkeypatch, repo)
    rid = "run-policy"
    srv.REGISTRY[rid] = {
        "status": "complete", "initial_state": {}, "interrupt_payload": None, "error": None,
        "result": {"slug": "my-article", "git_status": "merged",
                   "git_commit_sha": SHA, "publish_expected_remote_sha": SHA},
    }
    return srv, rid


@pytest.mark.parametrize("url", [
    DEMO_FORK_URL,
    "https://github.com/anudeepreddy332/themachinist-website-fork.git",
    "https://github.com/anudeepreddy332/themachinist-website-fork",
    "ssh://git@github.com/anudeepreddy332/themachinist-website-fork.git",
])
def test_github_identity_accepts_fork_url_forms(url):
    assert github_owner_repo(url) == DEMO_GITHUB


def test_github_identity_distinguishes_production_repo():
    assert github_owner_repo(PROD_REPO_URL) == PRODUCTION_GITHUB
    assert github_owner_repo(PROD_REPO_URL) != DEMO_GITHUB


def test_demo_site_canonicalization():
    assert canonicalize_https_base(DEMO_SITE) == ("https", "tmw-demo-site.netlify.app", "/")
    assert canonicalize_https_base(DEMO_SITE + "/") == ("https", "tmw-demo-site.netlify.app", "/")
    assert canonicalize_https_base("https://themachinist.org") != (
        "https", "tmw-demo-site.netlify.app", "/")


def test_unset_publish_target_denies(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL})
    monkeypatch.delenv("PUBLISH_TARGET", raising=False)
    monkeypatch.setenv("GIT_PUSH_ENABLED", "true")
    monkeypatch.setenv("NETLIFY_BASE_URL", DEMO_SITE)
    monkeypatch.setenv("PUBLISH_REMOTE", "origin")
    d = evaluate_publish_target(repo_path=str(repo))
    assert d.allowed is False
    assert "unset" in d.reason


def test_demo_allowlist_allows_fork(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL, "demo": DEMO_FORK_URL})
    enable_demo_publish(monkeypatch, repo, remote="demo")
    d = evaluate_publish_target()
    assert d.allowed is True
    assert d.target == "demo"


def test_demo_target_production_repository_denied(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "prod", {"origin": PROD_REPO_URL})
    enable_demo_publish(monkeypatch, repo)
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "production" in d.reason.lower()


def test_demo_target_production_remote_denied(tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "mixed",
        {"origin": DEMO_FORK_URL, "prod": PROD_REPO_URL},
    )
    enable_demo_publish(monkeypatch, repo, remote="prod")
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "production" in d.reason.lower()


def test_demo_target_rejects_production_remote_even_if_publish_remote_is_fork(
        tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "mixed",
        {"origin": PROD_REPO_URL, "demo": DEMO_FORK_URL},
    )
    enable_demo_publish(monkeypatch, repo, remote="demo")
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "production" in d.reason.lower()


def test_demo_target_production_site_denied(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL})
    enable_demo_publish(monkeypatch, repo)
    monkeypatch.setenv("NETLIFY_BASE_URL", "https://themachinist.org")
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "demo site" in d.reason


def test_git_push_disabled_prevents_remote_publish(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL})
    enable_demo_publish(monkeypatch, repo, git_push="false")
    d = evaluate_publish_target(require_push_enabled=True)
    assert d.allowed is False
    assert "GIT_PUSH_ENABLED" in d.reason

    srv, rid = _seed_publishable(monkeypatch, repo)
    monkeypatch.setenv("GIT_PUSH_ENABLED", "false")
    pushed = []

    def fake_run(args, **kw):
        pushed.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=f"{SHA}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    res = TestClient(srv.app).post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 409
    assert "GIT_PUSH_ENABLED" in res.text
    assert pushed == []


def test_production_target_without_confirmation_denied(monkeypatch):
    monkeypatch.setenv("PUBLISH_TARGET", "production")
    monkeypatch.delenv("CONFIRM_PRODUCTION_PUBLISH", raising=False)
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "CONFIRM_PRODUCTION_PUBLISH" in d.reason


def test_production_confirmation_does_not_skip_remote_parent_check(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "any", {"origin": PROD_REPO_URL})
    monkeypatch.setenv("PUBLISH_TARGET", "production")
    monkeypatch.setenv("CONFIRM_PRODUCTION_PUBLISH", "I_UNDERSTAND")
    monkeypatch.setenv("GIT_PUSH_ENABLED", "true")
    monkeypatch.setenv("THEMACHINIST_REPO_PATH", str(repo))
    monkeypatch.setenv("PUBLISH_REMOTE", "origin")
    import config
    monkeypatch.setattr(config, "THEMACHINIST_REPO_PATH", str(repo))

    import api.server as srv
    rid = "run-prod-confirm"
    other = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    srv.REGISTRY[rid] = {
        "status": "complete", "initial_state": {}, "interrupt_payload": None, "error": None,
        "result": {"slug": "my-article", "git_status": "merged",
                   "git_commit_sha": SHA, "publish_expected_remote_sha": SHA},
    }
    calls = []

    def fake_run(args, **kw):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args, 0, stdout=f"{other}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    res = TestClient(srv.app).post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 409
    assert "remote main changed" in res.text
    assert not any(a[:2] == ["git", "push"] for a in calls)


def test_production_confirmation_still_pushes_exact_sha_ref(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "any", {"origin": PROD_REPO_URL})
    monkeypatch.setenv("PUBLISH_TARGET", "production")
    monkeypatch.setenv("CONFIRM_PRODUCTION_PUBLISH", "I_UNDERSTAND")
    monkeypatch.setenv("GIT_PUSH_ENABLED", "true")
    monkeypatch.setenv("THEMACHINIST_REPO_PATH", str(repo))
    monkeypatch.setenv("PUBLISH_REMOTE", "origin")
    import config
    monkeypatch.setattr(config, "THEMACHINIST_REPO_PATH", str(repo))

    import api.server as srv
    rid = "run-prod-sha"
    srv.REGISTRY[rid] = {
        "status": "complete", "initial_state": {}, "interrupt_payload": None, "error": None,
        "result": {"slug": "my-article", "git_status": "merged",
                   "git_commit_sha": SHA, "publish_expected_remote_sha": SHA},
    }
    calls = []

    def fake_run(args, **kw):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args, 0, stdout=f"{SHA}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    res = TestClient(srv.app).post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 200
    push = [a for a in calls if a[:2] == ["git", "push"]][0]
    assert f"{SHA}:refs/heads/main" in push
    assert push[3] != "main"


def test_git_node_denies_merge_when_target_unset(tmp_path, monkeypatch, base_state):
    article = _trusted()
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL}, commit=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_PUSH_ENABLED", "true")
    monkeypatch.delenv("PUBLISH_TARGET", raising=False)
    import config
    monkeypatch.setattr(config, "THEMACHINIST_REPO_PATH", str(repo))
    out = nodes.git_node(_git_state(base_state, article))
    assert out["git_status"] == "failed"
    assert "publish target denied" in " ".join(out["error_log"])
    assert not (repo / "gradient-descent-test.html").exists()


def test_git_node_merges_when_demo_allowlist_matches(tmp_path, monkeypatch, base_state):
    article = _trusted()
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL}, commit=True)
    monkeypatch.chdir(tmp_path)
    enable_demo_publish(monkeypatch, repo)
    parent = subprocess.check_output(["git", "rev-parse", "main"], cwd=repo, text=True).strip()
    state = _git_state(base_state, article)
    state["publish_expected_remote_sha"] = parent
    monkeypatch.setattr(nodes, "_capture_remote_main_sha", lambda: parent)
    out = nodes.git_node(state)
    assert out["git_status"] in ("merged", "tagged_and_merged")
    assert out["git_commit_sha"]


def test_git_node_dry_run_when_push_disabled_even_with_demo_target(
        tmp_path, monkeypatch, base_state):
    article = _trusted()
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL}, commit=True)
    monkeypatch.chdir(tmp_path)
    enable_demo_publish(monkeypatch, repo, git_push="false")
    out = nodes.git_node(_git_state(base_state, article))
    assert out["git_status"] == "dry_run"
    assert not (repo / "gradient-descent-test.html").exists()


def test_api_publish_denied_when_target_unset(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL})
    srv, rid = _seed_publishable(monkeypatch, repo)
    monkeypatch.delenv("PUBLISH_TARGET", raising=False)
    pushed = []

    def fake_run(args, **kw):
        pushed.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=f"{SHA}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    res = TestClient(srv.app).post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 409
    assert "publish target denied" in res.text
    assert pushed == []


def test_api_publish_allowed_when_demo_allowlist_matches(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL})
    srv, rid = _seed_publishable(monkeypatch, repo)

    def fake_run(args, **kw):
        return subprocess.CompletedProcess(args, 0, stdout=f"{SHA}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    res = TestClient(srv.app).post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 200
    assert res.json()["live_url"] == f"{DEMO_SITE}/my-article"


def test_api_publish_denied_for_production_site(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL})
    srv, rid = _seed_publishable(monkeypatch, repo)
    monkeypatch.setenv("NETLIFY_BASE_URL", "https://themachinist.org")
    pushed = []

    def fake_run(args, **kw):
        pushed.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=f"{SHA}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    res = TestClient(srv.app).post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 409
    assert pushed == []


def test_demo_allows_fork_with_no_explicit_pushurl(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL})
    enable_demo_publish(monkeypatch, repo)
    d = evaluate_publish_target()
    assert d.allowed is True


def test_demo_allows_explicit_fork_pushurl(tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "fork", {"origin": DEMO_FORK_URL},
        pushurls={"origin": DEMO_FORK_URL},
    )
    enable_demo_publish(monkeypatch, repo)
    d = evaluate_publish_target()
    assert d.allowed is True


def test_demo_allows_multiple_pushurls_all_fork(tmp_path, monkeypatch):
    https_fork = "https://github.com/anudeepreddy332/themachinist-website-fork.git"
    repo = make_git_repo(
        tmp_path / "fork", {"origin": DEMO_FORK_URL},
        pushurls={"origin": [DEMO_FORK_URL, https_fork]},
    )
    enable_demo_publish(monkeypatch, repo)
    d = evaluate_publish_target()
    assert d.allowed is True


def test_demo_denies_fetch_fork_pushurl_production(tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "split", {"origin": DEMO_FORK_URL},
        pushurls={"origin": PROD_REPO_URL},
    )
    enable_demo_publish(monkeypatch, repo)
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "production" in d.reason.lower()


def test_demo_denies_fetch_production_pushurl_fork(tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "split", {"origin": PROD_REPO_URL},
        pushurls={"origin": DEMO_FORK_URL},
    )
    enable_demo_publish(monkeypatch, repo)
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "production" in d.reason.lower()


def test_demo_denies_other_remote_production_pushurl(tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "mixed",
        {"origin": DEMO_FORK_URL, "extra": DEMO_FORK_URL},
        pushurls={"extra": PROD_REPO_URL},
    )
    enable_demo_publish(monkeypatch, repo, remote="origin")
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "production" in d.reason.lower()


def test_demo_denies_multiple_pushurls_including_production(tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "split", {"origin": DEMO_FORK_URL},
        pushurls={"origin": [DEMO_FORK_URL, PROD_REPO_URL]},
    )
    enable_demo_publish(monkeypatch, repo)
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "production" in d.reason.lower()


def test_demo_denies_malformed_selected_pushurl(tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "fork", {"origin": DEMO_FORK_URL},
        pushurls={"origin": "not-a-git-remote"},
    )
    enable_demo_publish(monkeypatch, repo)
    d = evaluate_publish_target()
    assert d.allowed is False
    assert "does not resolve" in d.reason


def test_git_node_denies_fetch_fork_pushurl_production(
        tmp_path, monkeypatch, base_state):
    article = _trusted()
    repo = make_git_repo(
        tmp_path / "split", {"origin": DEMO_FORK_URL},
        pushurls={"origin": PROD_REPO_URL}, commit=True,
    )
    monkeypatch.chdir(tmp_path)
    enable_demo_publish(monkeypatch, repo)
    out = nodes.git_node(_git_state(base_state, article))
    assert out["git_status"] == "failed"
    assert "publish target denied" in " ".join(out["error_log"])
    assert not (repo / "gradient-descent-test.html").exists()


def test_api_publish_denies_fetch_fork_pushurl_production(tmp_path, monkeypatch):
    repo = make_git_repo(
        tmp_path / "split", {"origin": DEMO_FORK_URL},
        pushurls={"origin": PROD_REPO_URL},
    )
    srv, rid = _seed_publishable(monkeypatch, repo)
    pushed = []

    def fake_run(args, **kw):
        pushed.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=f"{SHA}\trefs/heads/main\n", stderr="")

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    res = TestClient(srv.app).post(f"/ui/runs/{rid}/publish", headers=H)
    assert res.status_code == 409
    assert "publish target denied" in res.text
    assert pushed == []


def test_get_root_returns_spa():
    import api.server as srv
    res = TestClient(srv.app).get("/")
    assert res.status_code == 200
    body = res.text
    assert "content-agent" in body
    assert "/static/app.js" in body
    assert "Bearer token" in body


def test_static_app_has_no_browser_token_persistence():
    html = Path("static/index.html").read_text(encoding="utf-8")
    js = Path("static/app.js").read_text(encoding="utf-8")
    blob = html + "\n" + js
    assert "localStorage" not in blob
    assert "sessionStorage" not in blob
    assert "document.cookie" not in blob
    assert "API_BEARER_TOKEN=" not in js
    assert "TOKEN = field.value || TOKEN" in js
