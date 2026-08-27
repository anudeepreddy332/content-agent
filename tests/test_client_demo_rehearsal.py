"""Focused regressions from the supervised client-demo rehearsal. $0."""

from __future__ import annotations

import hashlib
import json
import subprocess

import agent.nodes as nodes
from main import _build_initial_state, _write_telemetry
from agent.html_policy import (
    assemble_trusted_article,
    render_markdown_review_html,
    sanitize_fragment,
)
from tests.conftest import FakeLLMClient, fake_response
from tests.test_publish_target import DEMO_FORK_URL, enable_demo_publish, make_git_repo


REMOTE_PARENT = "b" * 40


def _trusted_article():
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


def test_gate1_renders_canonical_markdown_without_mutating_it():
    """The assembler must not turn headings into an indented code block."""
    markdown = nodes._assemble_markdown(
        "A reviewable topic",
        {
            "problem_framing": "A **bold** and *italic* [source](https://example.com/a).",
            "technical_dive": (
                "> A quoted point.\n\n"
                "Use `inline_code`.\n\n"
                "| Metric | Value |\n| --- | ---: |\n| loss | 0.1 |\n\n"
                "Inline $E = mc^2$.\n\n$$\n\\nabla f(x) = 0\n$$"
            ),
            "code_snippets": "```python\nprint('safe')\n```",
            "takeaways": "- First\n- Second\n\n<script>alert('x')</script>",
        },
    )

    rendered = render_markdown_review_html(markdown)

    assert "<h1>A reviewable topic</h1>" in rendered
    assert "<h2>Problem Framing</h2>" in rendered
    assert "<h2>Technical Deep-Dive</h2>" in rendered
    assert "<strong>bold</strong>" in rendered
    assert "<em>italic</em>" in rendered
    assert '<a href="https://example.com/a"' in rendered
    assert "<blockquote>" in rendered
    assert "<code>inline_code</code>" in rendered
    assert '<code class="language-python">' in rendered
    assert "<table>" in rendered
    assert 'class="math-inline"' in rendered
    assert 'class="math-block"' in rendered
    assert "<ul>" in rendered
    assert "<script" not in rendered.lower()
    assert "<pre><code>        ## Problem Framing" not in rendered


def test_reflection_marks_a_real_judge_score_as_real(base_state, monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_get_client",
        lambda: FakeLLMClient(response=fake_response('{"score": 7, "notes": "specific judge critique"}')),
    )

    out = nodes.reflect_node(base_state)

    assert out["reflection_score"] == 7
    assert out["reflection_provenance"] == {
        "origin": "judge",
        "reason": "json_score",
        "provider_called": True,
        "parse_status": "ok",
    }


def test_reflection_marks_parse_fallback_as_fallback(base_state, monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_get_client",
        lambda: FakeLLMClient(response=fake_response("not JSON")),
    )

    out = nodes.reflect_node(base_state)

    assert out["reflection_score"] == 7
    assert out["reflection_provenance"] == {
        "origin": "fallback",
        "reason": "parse_failed",
        "provider_called": True,
        "parse_status": "failed",
    }


def test_gate2_approval_captures_the_fresh_parent_at_approval(base_state, monkeypatch):
    html = "<html><body>trusted</body></html>"
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    base_state.update(html_output=html, html_sha256=digest)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    monkeypatch.setenv("GIT_PUSH_ENABLED", "true")
    monkeypatch.setattr(nodes, "_capture_remote_main_sha", lambda: REMOTE_PARENT)

    out = nodes.hitl_html_node(base_state)

    assert out["html_review_status"] == "approved"
    assert out["approved_html_sha256"] == digest
    assert out["publish_expected_remote_sha"] == REMOTE_PARENT


def test_parent_capture_reads_selected_remote_instead_of_a_local_tracking_ref(tmp_path, monkeypatch):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL}, commit=True)
    enable_demo_publish(monkeypatch, repo)
    fresh_parent = "c" * 40
    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs.get("cwd")))
        return subprocess.CompletedProcess(
            args, 0, stdout=f"{fresh_parent}\trefs/heads/main\n", stderr=""
        )

    monkeypatch.setattr(nodes.subprocess, "run", fake_run)

    assert nodes._capture_remote_main_sha() == fresh_parent
    assert calls == [(["git", "ls-remote", "origin", "refs/heads/main"], repo.resolve())]


def test_git_node_refuses_a_stale_local_main_before_mutating_the_site(
    tmp_path, monkeypatch, base_state
):
    """A locally stale main must never become the base of an approved merge."""
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL}, commit=True)
    article = _trusted_article()
    monkeypatch.chdir(tmp_path)
    enable_demo_publish(monkeypatch, repo)
    monkeypatch.setattr(nodes, "_capture_remote_main_sha", lambda: REMOTE_PARENT)

    state = {
        **base_state,
        "html_output": article.html,
        "html_filename": "gradient-descent-test.html",
        "html_sha256": article.sha256,
        "approved_html_sha256": article.sha256,
        "article_body_html": article.body_html,
        "publish_expected_remote_sha": REMOTE_PARENT,
        "slug": "gradient-descent-test",
        "topic": "Gradient Descent",
        "category": "concept-exploration",
        "draft_sections": {"problem_framing": "framing"},
    }

    out = nodes.git_node(state)

    assert out["git_status"] == "failed"
    assert "approved remote parent" in " ".join(out["error_log"])
    assert not (repo / "gradient-descent-test.html").exists()
    assert subprocess.check_output(["git", "rev-parse", "main"], cwd=repo).decode().strip() != REMOTE_PARENT


def test_git_node_preserves_the_remote_race_check_before_mutating_the_site(
    tmp_path, monkeypatch, base_state
):
    repo = make_git_repo(tmp_path / "fork", {"origin": DEMO_FORK_URL}, commit=True)
    article = _trusted_article()
    parent = subprocess.check_output(["git", "rev-parse", "main"], cwd=repo, text=True).strip()
    monkeypatch.chdir(tmp_path)
    enable_demo_publish(monkeypatch, repo)
    monkeypatch.setattr(nodes, "_capture_remote_main_sha", lambda: "d" * 40)

    state = {
        **base_state,
        "html_output": article.html,
        "html_filename": "gradient-descent-test.html",
        "html_sha256": article.sha256,
        "approved_html_sha256": article.sha256,
        "article_body_html": article.body_html,
        "publish_expected_remote_sha": parent,
        "slug": "gradient-descent-test",
        "topic": "Gradient Descent",
        "category": "concept-exploration",
        "draft_sections": {"problem_framing": "framing"},
    }

    out = nodes.git_node(state)

    assert out["git_status"] == "failed"
    assert "remote main changed after Gate 2 approval" in " ".join(out["error_log"])
    assert not (repo / "gradient-descent-test.html").exists()


def test_html_generation_records_safe_placeholder_diagnostics(base_state, monkeypatch):
    base_state.update(
        draft_sections={
            "problem_framing": "framing",
            "technical_dive": "dive",
            "code_snippets": "```python\nprint(1)\n```",
            "takeaways": "- one",
        },
        draft_markdown="# Test\n\n## Technical Deep-Dive\ndive",
    )
    monkeypatch.setattr(
        nodes,
        "_get_client",
        lambda: FakeLLMClient(response=fake_response("<p>{{TOPIC}}</p>")),
    )

    out = nodes.html_gen_node(base_state)

    assert out["html_output"] is None
    assert out["policy_diagnostics"] == [{
        "stage": "html_gen",
        "rule_id": "template_delimiter_outside_code_v1",
        "match_count": 1,
        "locations": [{
            "range": out["policy_diagnostics"][0]["locations"][0]["range"],
            "element": "p",
            "token_class": "template_delimiter",
        }],
    }]
    assert "TOPIC" not in repr(out["policy_diagnostics"])


def test_telemetry_persists_policy_diagnostics_without_raw_placeholder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = _build_initial_state("Placeholder review", "placeholder-review", "test", "Test", "policy-run")
    state["policy_diagnostics"] = [{
        "stage": "html_gen",
        "rule_id": "template_delimiter_outside_code_v1",
        "match_count": 1,
        "locations": [{"range": [12, 14], "element": "p", "token_class": "template_delimiter"}],
    }]

    path = _write_telemetry(state)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["policy_diagnostics"] == state["policy_diagnostics"]
    assert "TOPIC" not in path.read_text(encoding="utf-8")
