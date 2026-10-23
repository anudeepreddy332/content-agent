"""Independent structural and citation acceptance cases for article publication."""

from copy import deepcopy
import sqlite3
from types import SimpleNamespace

import pytest

from agent import nodes
from agent.citation_policy import (evidence_numbering, occurrence_marked_markdown,
                                   replace_rendered_occurrence_markers)
from agent.html_policy import PolicyError, render_publication_markdown
from agent.initial_html_equivalence import (assert_section_equivalent, receipt_valid,
                                            source_key, visible_key)
from tests.conftest import fake_response
from tests.signoff_fixtures import renderable_approved_state


@pytest.mark.parametrize("markdown,required", [
    ("## Method\n\nFirst paragraph.\n\nSecond paragraph.",
     ("<h2>Method</h2>", "<p>First paragraph.</p>", "<p>Second paragraph.</p>")),
    ("3. Third\n4. Fourth\n   - Nested", ("<ol start=\"3\">", "<li>Third</li>", "<ul>")),
    ("A  \nB", ("<p>A<br>", "B</p>")),
    ("```python\nif x:\n    y()\n```", ("<pre><code class=\"language-python\">", "if x:\n    y()\n")),
    ("| A | B |\n|---|---|\n| x | y |", ("<table>", "<th>A</th>", "<td>y</td>")),
    ("$a+b$\n\n$$\na+b\n$$", ('class="math-inline"', 'class="math-block"')),
])
def test_independent_markdown_structure_expectations(markdown, required):
    html = render_publication_markdown(markdown).html
    for literal in required:
        assert literal in html
    assert_section_equivalent("technical_dive", markdown, html)


@pytest.mark.parametrize("markdown", [
    "# Article-level heading", "##### Too deep", "10000. Out of range",
    "| A | B |\n|:---|---:|\n| x | y |",
    "[source](https://example.com)",
    "![diagram](https://example.com/diagram.png)",
])
def test_unsupported_publication_markdown_fails_before_sanitization(markdown):
    with pytest.raises(PolicyError):
        render_publication_markdown(markdown)
    with pytest.raises(PolicyError):
        source_key("technical_dive", markdown)


def test_structure_distinguishes_paragraphs_lists_numbering_and_math():
    assert visible_key("<p>A.</p><p>B.</p>") != visible_key("<p>A. B.</p>")
    assert visible_key("<ul><li>A</li><li>B</li></ul>") != visible_key("<p>A B</p>")
    assert visible_key('<ol start="3"><li>Third</li></ol>') != visible_key(
        "<ol><li>Third</li></ol>")
    assert visible_key('<span class="math-inline">x+y</span>') != visible_key(
        '<div class="math-block">x+y</div>')
    assert visible_key("<p>A<br>B</p>") != visible_key("<p>A B</p>")


def test_citation_anchor_survives_emphasis_and_duplicate_quotes():
    first = "Alpha."
    second = "A **gradient descent** note. Alpha."
    draft = first + "\n\n" + second
    start = len(first) + 2
    phrase = "gradient descent"
    anchor = start + second.index(phrase)
    groups = [{"anchor_start": anchor, "anchor_end": anchor + len(phrase),
               "attached_evidence_ids": ["WEB-002", "WEB-001"]}]
    numbering = evidence_numbering(["WEB-002", "WEB-001"])
    unchanged, none = occurrence_marked_markdown(first, section_start=0,
                                                  groups=groups, numbering=numbering)
    assert unchanged == first and none == {}
    annotated, markers = occurrence_marked_markdown(second, section_start=start,
                                                    groups=groups, numbering=numbering)
    html = render_publication_markdown(annotated).html
    marked = replace_rendered_occurrence_markers(html, markers)
    assert marked.count("<sup>2,1</sup>") == 1
    assert "<strong>gradient descent<sup>2,1</sup></strong>" in marked
    assert draft[anchor:anchor + len(phrase)] == phrase
    with pytest.raises(PolicyError):
        replace_rendered_occurrence_markers("<p>lost marker</p>", markers)
    marker = next(iter(markers))
    with pytest.raises(PolicyError):
        replace_rendered_occurrence_markers(f"<p>{marker}{marker}</p>", markers)
    with pytest.raises(PolicyError):
        occurrence_marked_markdown(second, section_start=start, groups=[{
            "anchor_start": start - 1, "anchor_end": start + 2,
            "attached_evidence_ids": ["WEB-001"],
        }], numbering=numbering)


def test_bibliography_order_uses_the_same_numbering_as_markers(monkeypatch):
    from agent.citation_policy import bibliography_items
    registry = {
        "WEB-001": {"kind": "web", "title": "First source", "url": "https://example.com/first"},
        "WEB-002": {"kind": "web", "title": "Second source", "url": "https://example.com/second"},
    }
    attached = ["WEB-002", "WEB-001"]
    assert evidence_numbering(attached) == {"WEB-001": 1, "WEB-002": 2}
    monkeypatch.setattr(nodes, "evaluate_citation_policy", lambda **_kw: SimpleNamespace(
        attached_evidence_ids=attached, required_evidence_ids=attached))
    monkeypatch.setattr(nodes, "build_evidence_registry", lambda _state: registry)
    bibliography = nodes._citations_html_for_state({"claim_inventory": {}})
    assert bibliography.index("First source") < bibliography.index("Second source")
    assert len(bibliography_items(sorted(attached), registry)) == 2


def test_initial_render_is_provider_free_and_receipt_versioned(base_state, monkeypatch):
    state = renderable_approved_state(base_state)
    monkeypatch.setattr(nodes, "_get_client", lambda: pytest.fail("initial HTML called provider"))
    monkeypatch.setattr(nodes, "_render_technical_dive_via_llm",
                        lambda **_kw: pytest.fail("initial HTML called LLM formatter"), raising=False)
    update = nodes.html_gen_node(state)
    assert update["html_output"]
    assert update["total_tokens"] == state.get("total_tokens", 0)
    current = {**state, **update}
    assert receipt_valid(current)
    stale = deepcopy(current)
    stale["initial_html_equivalence"]["contract_version"] = "initial-html-content-v1"
    assert not receipt_valid(stale)


def test_current_gate2_layout_revision_remains_approvable(base_state, monkeypatch):
    state = renderable_approved_state(base_state)
    state.update(nodes.html_gen_node(state))
    assert receipt_valid(state)
    old = "<p>A gradient descent learning note.</p>"
    new = '<div class="callout callout-info"><p>A gradient descent learning note.</p></div>'
    revised = state["article_body_html"].replace(old, new, 1)
    assert revised != state["article_body_html"]
    state["html_feedback"] = "Add an information callout"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda _client, **_kw: fake_response(revised))
    update = nodes.html_revise_node(state)
    assert update["html_output"] != state["html_output"]
    assert new in update["article_body_html"]
    assert receipt_valid({**state, **update})


def test_stale_contract_checkpoint_cannot_resume_gate2(base_state, monkeypatch, tmp_path):
    from agent.reviewer_identity import reviewer_context
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command
    from tests.test_real_hitl_api import _gate_graph

    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    state = renderable_approved_state(base_state)
    state.update(nodes.html_gen_node(state))
    path = tmp_path / "stale-structure.sqlite"
    config = {"configurable": {"thread_id": state["run_id"]}}
    first_connection = sqlite3.connect(path, check_same_thread=False)
    graph, _ = _gate_graph("html", SqliteSaver(first_connection))
    assert graph.invoke(state, config)["__interrupt__"][0].value["type"] == "hitl_html_review"
    stale = {**state["initial_html_equivalence"], "contract_version": "initial-html-content-v1"}
    graph.update_state(config, {"initial_html_equivalence": stale})
    first_connection.close()
    second_connection = sqlite3.connect(path, check_same_thread=False)
    resumed, visited = _gate_graph("html", SqliteSaver(second_connection))
    with reviewer_context("fixture-reviewer", state["run_id"]):
        result = resumed.invoke(Command(resume={"action": "approve"}), config)
    second_connection.close()
    assert result["terminal_status"] == "rendering_failed"
    assert visited == []


@pytest.mark.parametrize("before,after", [
    ("<p>A gradient descent learning note.</p>", "<div>A gradient descent learning note.</div>"),
    ("<ul>", "<ol>"),
    ("<pre><code>print(1)</code></pre>", "<code>print(1)</code>"),
])
def test_gate2_rejects_structure_change_and_keeps_prior_render(
    base_state, monkeypatch, before, after,
):
    state = renderable_approved_state(base_state)
    state.update(nodes.html_gen_node(state))
    assert receipt_valid(state)
    changed = state["article_body_html"].replace(before, after, 1)
    assert changed != state["article_body_html"]
    state["html_feedback"] = "Improve layout"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda _client, **_kw: fake_response(changed))
    update = nodes.html_revise_node(state)
    assert update["html_output"] == state["html_output"]
    assert update["article_body_html"] == state["article_body_html"]
    assert update["html_sha256"] == state["html_sha256"]
    assert any("DISCARDED" in entry for entry in update["error_log"])
    assert receipt_valid({**state, **update})
