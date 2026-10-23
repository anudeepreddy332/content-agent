"""Offline initial-render and downstream content-binding regressions."""

from copy import deepcopy
import sqlite3

import pytest

from agent import nodes
from agent.html_policy import PolicyError, reassemble_from_body, sha256_utf8
from agent.initial_html_equivalence import receipt_valid, visible_key


@pytest.fixture
def draft_state():
    sections = {
        "problem_framing": "Gradient descent does not always converge.",
        "technical_dive": "It takes 10 steps only when the rate is suitable.\n\nThe result is approximate.",
        "code_snippets": "```python\nif ready:\n    step()\n```",
        "takeaways": "- Keep the condition.",
    }
    return {
        "run_id": "offline-render", "topic": "Gradient Descent", "slug": "gradient-descent",
        "draft_sections": sections,
        "draft_markdown": nodes._assemble_markdown("Gradient Descent", sections),
        "series_context": "Learning Log", "total_cost_usd": 0.0,
        "latency_ms": {}, "error_log": [],
    }


def render(state, technical_html, monkeypatch):
    monkeypatch.setattr(nodes, "approval_origin_valid", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(nodes, "receipt_valid", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(nodes, "_render_technical_dive_via_llm",
                        lambda **_kwargs: (technical_html, 0, 0.0))
    return nodes.html_gen_node(state)


def changed_rendered_body(state, before, after):
    """Replace displayed code while retaining the original Gate 1 receipt."""
    body = state["article_body_html"].replace(before, after, 1)
    assert body != state["article_body_html"]
    article = reassemble_from_body(**nodes._article_shell_fields(state), body_html=body,
                                   citations_html=nodes._citations_html_for_state(state))
    return {**state, "article_body_html": article.body_html,
            "html_output": article.html, "html_sha256": article.sha256,
            "approved_html_sha256": article.sha256}


GOOD = "<p>It takes 10 steps only when the rate is suitable.</p><p>The result is approximate.</p>"


@pytest.mark.parametrize("altered", [
    "<p>It takes 10 steps even when the rate is unsuitable.</p><p>The result is approximate.</p>",
    "<p>It takes 100 steps only when the rate is suitable.</p><p>The result is approximate.</p>",
    "<p>It takes 10 steps only when the rate is suitable.</p>",
    "<p>The result is approximate.</p><p>It takes 10 steps only when the rate is suitable.</p>",
    "<p>It takes 10 steps only when the rate is suitable.</p><p>The result is approximate and guaranteed.</p>",
    "<p>It takes 10 steps only when the rate is suitable.<p>The result is approximate.</p>",
])
def test_altered_or_malformed_initial_render_fails_closed(draft_state, altered, monkeypatch):
    result = render(draft_state, altered, monkeypatch)
    assert result["html_output"] is None
    assert result["initial_html_equivalence"] is None
    assert result["approved_html_sha256"] is None


def test_code_indentation_change_fails_closed(draft_state, monkeypatch):
    monkeypatch.setattr(nodes, "_render_code_snippets",
                        lambda _raw: "<pre><code>if ready:\nstep()\n</code></pre>")
    assert render(draft_state, GOOD, monkeypatch)["html_output"] is None


def test_removed_negation_fails_closed(draft_state, monkeypatch):
    state = deepcopy(draft_state)
    state["draft_sections"]["technical_dive"] = "This method does not guarantee convergence."
    state["draft_markdown"] = nodes._assemble_markdown(state["topic"], state["draft_sections"])
    result = render(state, "<p>This method guarantees convergence.</p>", monkeypatch)
    assert result["html_output"] is None


def test_equivalent_markup_and_preserved_code_pass(draft_state, monkeypatch):
    equivalent = ("<div><p>It takes <strong>10</strong> steps only when the rate is suitable.</p>"
                  "<p>The result is <em>approximate</em>.</p></div>")
    result = render(draft_state, equivalent, monkeypatch)
    assert result["html_output"]
    assert "if ready:\n    step()\n" in result["html_output"]
    assert receipt_valid({**draft_state, **result})


def test_server_added_citation_marker_keeps_bound_content(draft_state, monkeypatch):
    result = render(draft_state, GOOD, monkeypatch)
    state = {**draft_state, **result}
    body = state["article_body_html"].replace("The result is approximate.",
                                              "The result is approximate.<sup>1</sup>")
    article = reassemble_from_body(**nodes._article_shell_fields(state), body_html=body,
                                   citations_html=nodes._citations_html_for_state(state))
    state.update(article_body_html=article.body_html, html_output=article.html,
                 html_sha256=article.sha256)
    # The initial render records the trusted post-citation body; a bare model
    # addition never reaches this receipt minting point.
    state["initial_html_equivalence"] = nodes.initial_equivalence_receipt(state, article.body_html)
    assert receipt_valid(state)


def test_stale_draft_body_shell_and_receipt_block_gate2_and_git(draft_state, monkeypatch, tmp_path):
    result = render(draft_state, GOOD, monkeypatch)
    state = {**draft_state, **result}
    assert receipt_valid(state)
    changes = []
    changed_draft = deepcopy(state)
    changed_draft["draft_sections"]["technical_dive"] = "A changed approved section."
    changes.append(changed_draft)
    changed_body = dict(state, article_body_html=state["article_body_html"].replace("10 steps", "100 steps"))
    changes.append(changed_body)
    changed_shell = dict(state, html_output=state["html_output"].replace("Intermediate", "Guaranteed"))
    changed_shell["html_sha256"] = sha256_utf8(changed_shell["html_output"])
    changes.append(changed_shell)
    changes.append(dict(state, initial_html_equivalence=None))
    for unsafe_body in (state["article_body_html"] + "<script></script>",
                        state["article_body_html"].replace("<p>", '<p onclick="steal()">', 1)):
        altered_html = state["html_output"].replace(state["article_body_html"], unsafe_body, 1)
        changes.append(dict(state, article_body_html=unsafe_body,
                            html_output=altered_html, html_sha256=sha256_utf8(altered_html)))
    for stale in changes:
        assert not receipt_valid(stale)
        gate = nodes.hitl_html_node(stale)
        assert gate["terminal_status"] == "rendering_failed"
        monkeypatch.chdir(tmp_path)
        git_result = nodes.git_node({**stale, "html_filename": "gradient-descent.html"})
        assert git_result["git_status"] == "failed"
        assert not (tmp_path / "outputs" / "articles" / "gradient-descent.html").exists()


def test_visible_key_keeps_code_whitespace_significant():
    assert visible_key("<pre><code>if ready:\n    step()\n</code></pre>") != visible_key(
        "<pre><code>if ready:\nstep()\n</code></pre>")
    assert visible_key("<p>A<sup>1</sup>B</p>") != visible_key("<p>AB<sup>1</sup></p>")


def test_code_line_break_markup_and_unicode_literal_fail_closed(draft_state, monkeypatch):
    with pytest.raises(PolicyError):
        visible_key("<pre><code>if ready:<br>    step()</code></pre>")
    with pytest.raises(PolicyError):
        visible_key("<pre><code>if ready:<span>\n</span>    step()</code></pre>")
    assert visible_key('<pre><code>print("é")</code></pre>') != visible_key(
        '<pre><code>print("e\u0301")</code></pre>')
    assert visible_key('<pre><code>print(&quot;é&quot;)</code></pre>') == visible_key(
        '<pre><code>print("é")</code></pre>')
    assert render(draft_state, GOOD, monkeypatch)["html_output"]
    monkeypatch.setattr(nodes, "_render_code_snippets",
                        lambda _raw: "<pre><code>if ready:<br>    step()\n</code></pre>")
    assert render(draft_state, GOOD, monkeypatch)["html_output"] is None


@pytest.mark.parametrize("raw,expected", [
    ("A **gradient descent** learning note.", "<strong>gradient descent</strong>"),
    ("- Use `fit()` on training data.", "<code>fit()</code>"),
])
def test_trusted_markdown_section_rendering(raw, expected, draft_state, monkeypatch):
    section = "problem_framing" if raw.startswith("A ") else "takeaways"
    state = deepcopy(draft_state)
    state["draft_sections"][section] = raw
    state["draft_markdown"] = nodes._assemble_markdown(state["topic"], state["draft_sections"])
    result = render(state, GOOD, monkeypatch)
    assert result["html_output"]
    assert expected in result["article_body_html"]
    assert receipt_valid({**state, **result})


def test_markdown_lists_and_actual_content_changes(draft_state, monkeypatch):
    state = deepcopy(draft_state)
    state["draft_sections"]["takeaways"] = "1. Use `fit()` on training data.\n2. Check the loss."
    state["draft_markdown"] = nodes._assemble_markdown(state["topic"], state["draft_sections"])
    result = render(state, GOOD, monkeypatch)
    assert "<ol>" in result["article_body_html"]
    assert "<code>fit()</code>" in result["article_body_html"]
    assert receipt_valid({**state, **result})
    monkeypatch.setattr(nodes, "_render_takeaways",
                        lambda _raw: "<ol><li>Use <code>fit()</code> on all data.</li><li>Check the loss.</li></ol>")
    assert render(state, GOOD, monkeypatch)["html_output"] is None


def test_renderable_approved_fixture_is_a_valid_positive_control(base_state, monkeypatch):
    from tests.signoff_fixtures import render_approved_offline, renderable_approved_state

    state = renderable_approved_state(base_state)
    rendered = render_approved_offline(state, monkeypatch)
    assert receipt_valid(rendered)


def test_layout_only_change_keeps_initial_binding(draft_state, monkeypatch):
    result = render(draft_state, GOOD, monkeypatch)
    state = {**draft_state, **result}
    body = state["article_body_html"].replace("The result is approximate.",
                                              "The result is <strong>approximate</strong>.")
    revised = reassemble_from_body(**nodes._article_shell_fields(state), body_html=body,
                                   citations_html=nodes._citations_html_for_state(state))
    state.update(article_body_html=revised.body_html, html_output=revised.html,
                 html_sha256=revised.sha256)
    assert receipt_valid(state)


def test_checkpoint_restart_preserves_gate2_binding(base_state, monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command
    from agent.reviewer_identity import reviewer_context
    from tests.signoff_fixtures import renderable_approved_state, render_approved_offline
    from tests.test_real_hitl_api import _gate_graph

    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("GIT_PUSH_ENABLED", "false")
    state = render_approved_offline(
        renderable_approved_state(base_state, reviewer_id="restart-reviewer"), monkeypatch)
    path = tmp_path / "render-checkpoint.sqlite"
    config = {"configurable": {"thread_id": state["run_id"]}}
    first_connection = sqlite3.connect(path, check_same_thread=False)
    graph, _ = _gate_graph("html", SqliteSaver(first_connection))
    assert graph.invoke(state, config)["__interrupt__"][0].value["type"] == "hitl_html_review"
    first_connection.close()

    second_connection = sqlite3.connect(path, check_same_thread=False)
    resumed, visited = _gate_graph("html", SqliteSaver(second_connection))
    with reviewer_context("restart-reviewer", state["run_id"]):
        result = resumed.invoke(Command(resume={"action": "approve"}), config)
    second_connection.close()
    assert result["html_review_status"] == "approved"
    assert visited == ["git"]


def test_code_tampering_invalidates_receipt_gate2_and_git(draft_state, monkeypatch, tmp_path):
    state = {**draft_state, **render(draft_state, GOOD, monkeypatch)}
    assert receipt_valid(state)
    changed = changed_rendered_body(state, "if ready:\n", "if ready:<br>\n")
    assert not receipt_valid(changed)
    assert nodes.hitl_html_node(changed)["terminal_status"] == "rendering_failed"
    monkeypatch.chdir(tmp_path)
    assert nodes.git_node({**changed, "html_filename": "gradient-descent.html"})["git_status"] == "failed"
    assert not (tmp_path / "outputs" / "articles" / "gradient-descent.html").exists()

    unicode_state = deepcopy(draft_state)
    unicode_state["draft_sections"]["code_snippets"] = 'print("é")'
    unicode_state["draft_markdown"] = nodes._assemble_markdown(
        unicode_state["topic"], unicode_state["draft_sections"])
    unicode_state.update(render(unicode_state, GOOD, monkeypatch))
    assert receipt_valid(unicode_state)
    changed_unicode = changed_rendered_body(unicode_state, 'print("é")', 'print("e\u0301")')
    assert not receipt_valid(changed_unicode)
    assert nodes.hitl_html_node(changed_unicode)["terminal_status"] == "rendering_failed"


def test_checkpoint_restart_rejects_inserted_code_break(base_state, monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command
    from agent.reviewer_identity import reviewer_context
    from tests.signoff_fixtures import renderable_approved_state, render_approved_offline
    from tests.test_real_hitl_api import _gate_graph

    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    state = render_approved_offline(renderable_approved_state(base_state), monkeypatch)
    path = tmp_path / "tampered-code.sqlite"
    config = {"configurable": {"thread_id": state["run_id"]}}
    first_connection = sqlite3.connect(path, check_same_thread=False)
    graph, _ = _gate_graph("html", SqliteSaver(first_connection))
    assert graph.invoke(state, config)["__interrupt__"][0].value["type"] == "hitl_html_review"
    changed = changed_rendered_body(state, "print(1)", "print(<br>1)")
    graph.update_state(config, {
        "article_body_html": changed["article_body_html"],
        "html_output": changed["html_output"],
        "html_sha256": changed["html_sha256"],
    })
    first_connection.close()

    second_connection = sqlite3.connect(path, check_same_thread=False)
    resumed, visited = _gate_graph("html", SqliteSaver(second_connection))
    with reviewer_context("fixture-reviewer", state["run_id"]):
        result = resumed.invoke(Command(resume={"action": "approve"}), config)
    second_connection.close()
    assert result["terminal_status"] == "rendering_failed"
    assert visited == []


def test_stale_equivalence_blocks_publish_before_git_network(monkeypatch):
    from fastapi.testclient import TestClient
    import api.server as server
    from tests.signoff_fixtures import publishable_result

    run_id = "stale-html-publish"
    state = publishable_result(run_id, "stale-html", monkeypatch)
    state.update(git_status="merged", git_commit_sha="a" * 40,
                 publish_expected_remote_sha="b" * 40)
    state["initial_html_equivalence"] = {**state["initial_html_equivalence"],
                                          "body_key_sha256": "0" * 64}
    monkeypatch.setenv("API_BEARER_TOKEN", "offline-publish-token")
    monkeypatch.setattr(server, "REGISTRY", {run_id: {"result": state, "status": "complete"}})
    monkeypatch.setattr(server.subprocess, "run",
                        lambda *_args, **_kwargs: pytest.fail("invalid render reached git network"))
    with TestClient(server.app) as client:
        response = client.post(f"/ui/runs/{run_id}/publish",
                               headers={"Authorization": "Bearer offline-publish-token"})
    assert response.status_code == 409
    assert "initial HTML" in response.text


def test_inserted_code_break_blocks_publication_before_git_network(monkeypatch):
    from fastapi.testclient import TestClient
    import api.server as server
    from tests.signoff_fixtures import publishable_result

    run_id = "code-break-publish"
    state = publishable_result(run_id, "code-break", monkeypatch)
    state = changed_rendered_body(state, "print(1)", "print(<br>1)")
    assert not receipt_valid(state)
    state.update(git_status="merged", git_commit_sha="a" * 40,
                 publish_expected_remote_sha="b" * 40)
    monkeypatch.setenv("API_BEARER_TOKEN", "offline-publish-token")
    monkeypatch.setattr(server, "REGISTRY", {run_id: {"result": state, "status": "complete"}})
    monkeypatch.setattr(server.subprocess, "run",
                        lambda *_args, **_kwargs: pytest.fail("tampered code reached git network"))
    with TestClient(server.app) as client:
        response = client.post(f"/ui/runs/{run_id}/publish",
                               headers={"Authorization": "Bearer offline-publish-token"})
    assert response.status_code == 409
