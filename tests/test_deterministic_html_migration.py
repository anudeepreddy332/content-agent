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
                                            content_key, source_key, visible_key)
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


@pytest.mark.parametrize("prose,expected", [
    ("A claim.<sup>1</sup> Another sentence.", "A claim. Another sentence."),
    ("A <sup>1</sup>claim.", "A claim."),
    ("A claim<sup>1</sup> with a qualifier.", "A claim with a qualifier."),
    ("A <strong>claim<sup>1</sup></strong> with care.", "A <strong>claim</strong> with care."),
    ("First<sup>1</sup> and second<sup>2</sup>.", "First and second."),
])
def test_approved_citations_are_removed_before_prose_aggregation(prose, expected):
    assert content_key(f"<p>{prose}</p>") == visible_key(f"<p>{expected}</p>")
    assert [value for kind, value in visible_key(f"<p>{prose}</p>") if kind == "citation"] == [
        value for value in ("1", "2") if f"<sup>{value}</sup>" in prose]


@pytest.mark.parametrize("altered", [
    "<p>A changed claim.<sup>1</sup> Another sentence.</p>",
    "<p>Another sentence. A claim.<sup>1</sup></p>",
    "<p>A claim. Another sentence.<sup>1</sup></p>",
])
def test_citation_normalization_does_not_accept_changed_prose_or_shifted_marker(altered):
    approved = "<p>A claim.<sup>1</sup> Another sentence.</p>"
    if "shifted" not in altered:
        assert (content_key(altered) != content_key(approved)
                or visible_key(altered) != visible_key(approved))


def test_approved_mid_paragraph_citation_renders_and_receipt_is_current(base_state):
    state = renderable_approved_state(base_state, sections={
        "problem_framing": "A learning note.",
        "technical_dive": "Gradient descent minimizes a loss function. The rate matters.",
        "code_snippets": "print(1)",
        "takeaways": "- Use care.",
    })
    update = nodes.html_gen_node(state)
    assert update["html_output"]
    assert "minimizes a loss function.<sup>1</sup> The rate matters." in update["article_body_html"]
    assert receipt_valid({**state, **update})


@pytest.mark.parametrize("markdown", [
    "| Claim | Scope |\n|---|---|\n| works | here | not in deployment |",
    "| A | B |\n|---|---|\n| x |",
])
def test_table_rows_that_would_lose_or_invent_cells_fail_closed(markdown):
    with pytest.raises(PolicyError, match="table row"):
        render_publication_markdown(markdown)
    with pytest.raises(PolicyError, match="table row"):
        source_key("technical_dive", markdown)


@pytest.mark.parametrize("markdown,expected", [
    ("| A | B |\n|---|---|\n| x | y |", ("<td>x</td>", "<td>y</td>")),
    ("| A | B | C |\n|---|---|---|\n| x | y | z |", ("<td>x</td>", "<td>z</td>")),
    ("| A | B |\n|---|---|\n| x | |", ("<td>x</td>", "<td></td>")),
    (r"| A | B |" + "\n|---|---|\n" + r"| a\|b | c |", ("<td>a|b</td>", "<td>c</td>")),
])
def test_supported_tables_preserve_expected_source_cells(markdown, expected):
    html = render_publication_markdown(markdown).html
    for cell in expected:
        assert cell in html
    assert_section_equivalent("technical_dive", markdown, html)


def test_table_like_text_in_fenced_code_remains_literal():
    markdown = "```text\n| A | B |\n|---|---|\n| x | y | lost |\n```"
    html = render_publication_markdown(markdown).html
    assert "| x | y | lost |" in html
    assert "<table>" not in html


def test_subscript_is_not_equivalent_to_plain_numeric_content():
    assert visible_key("<p>The example multiplier is 10.</p>")
    for changed in ("1<sub>0</sub>", "1<sup>0</sup>", "10<sub>2</sub>"):
        with pytest.raises(PolicyError):
            visible_key(f"<p>The example multiplier is {changed}.</p>")
    assert visible_key('<p>Term <span class="math-inline">x_0</span>.</p>')


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


def test_gate2_rejects_numeric_subscript_and_keeps_valid_render(base_state, monkeypatch):
    state = renderable_approved_state(base_state, sections={
        "problem_framing": "The example multiplier is 10.",
        "technical_dive": "Gradient descent minimizes a loss function.",
        "code_snippets": "print(1)",
        "takeaways": "- Use care.",
    })
    state.update(nodes.html_gen_node(state))
    assert receipt_valid(state)
    changed = state["article_body_html"].replace("multiplier is 10.", "multiplier is 1<sub>0</sub>.")
    assert changed != state["article_body_html"]
    state["html_feedback"] = "Improve layout"
    monkeypatch.setattr(nodes, "_get_client", lambda: object())
    monkeypatch.setattr(nodes, "_llm_call", lambda _client, **_kw: fake_response(changed))
    update = nodes.html_revise_node(state)
    assert update["article_body_html"] == state["article_body_html"]
    assert update["html_output"] == state["html_output"]
    assert any("DISCARDED" in entry for entry in update["error_log"])
    assert receipt_valid({**state, **update})
