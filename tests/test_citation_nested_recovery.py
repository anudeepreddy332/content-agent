"""Offline citation regression from the November 3 blocked E2E draft."""

import pytest

from agent import nodes
from agent.citation_policy import (
    build_citation_plan,
    citation_policy_passed,
    evaluate_citation_policy,
    evidence_numbering,
    occurrence_marked_markdown,
    replace_rendered_occurrence_markers,
)
from agent.html_policy import PolicyError, render_publication_markdown
from agent.initial_html_equivalence import content_key, visible_key
from tests.test_citation_policy import _claim_row, _verified


OUTER = (
    "so in practice you use a mini-batch of size B (typically 32–256) "
    "and average gradients over that batch"
)
INNER = "(typically 32–256)"
OUTER_CLAIM = "In practice a mini-batch of size B is used and gradients are averaged over that batch."
INNER_CLAIM = "Mini-batch sizes are typically 32–256."
PARENT_SUPPORT = "KB-006"
CHILD_SUPPORT = "KB-002"


def _nested(base_state, draft=OUTER, *, child_support=(CHILD_SUPPORT,),
            outer_anchor=OUTER):
    rows = [
        _claim_row(OUTER_CLAIM, outer_anchor),
        _claim_row(INNER_CLAIM, INNER),
    ]
    state, inventory = _verified(
        base_state, draft, rows, [[PARENT_SUPPORT], list(child_support)],
    )
    state["evidence_registry"] = {
        PARENT_SUPPORT: {"evidence_id": PARENT_SUPPORT, "kind": "kb",
                         "source_ref": "kb:mini-batch-reference"},
        CHILD_SUPPORT: {"evidence_id": CHILD_SUPPORT, "kind": "kb",
                        "source_ref": "kb:batch-size-reference"},
        "KB-003": {"evidence_id": "KB-003", "kind": "kb",
                   "source_ref": "kb:other-reference"},
    }
    return state, inventory


def _groups_by_claim(result):
    return {claim_id: group for group in result.anchor_groups
            for claim_id in group["claim_ids"]}


def test_real_strict_nesting_keeps_independent_claim_evidence(base_state):
    state, inventory = _nested(base_state)
    result = evaluate_citation_policy(state=state)
    assert citation_policy_passed(result)
    assert len(result.anchor_groups) == 2
    outer, inner = [claim["claim_id"] for claim in inventory["claims"]]
    groups = _groups_by_claim(result)
    assert groups[outer]["required_evidence_ids"] == [PARENT_SUPPORT]
    assert groups[outer]["attached_evidence_ids"] == [PARENT_SUPPORT]
    assert groups[inner]["required_evidence_ids"] == [CHILD_SUPPORT]
    assert groups[inner]["attached_evidence_ids"] == [CHILD_SUPPORT]
    assert (groups[outer]["anchor_start"] < groups[inner]["anchor_start"]
            < groups[inner]["anchor_end"] < groups[outer]["anchor_end"])


def test_nested_markers_render_at_separate_ends_without_changing_prose(base_state):
    state, _ = _nested(base_state)
    result = evaluate_citation_policy(state=state)
    assert citation_policy_passed(result)
    numbering = evidence_numbering(result.attached_evidence_ids)
    annotated, markers = occurrence_marked_markdown(
        OUTER, section_start=0, groups=result.anchor_groups, numbering=numbering,
    )
    html = replace_rendered_occurrence_markers(
        render_publication_markdown(annotated).html, markers,
    )
    assert f"{INNER}<sup>{numbering[CHILD_SUPPORT]}</sup>" in html
    assert f"average gradients over that batch<sup>{numbering[PARENT_SUPPORT]}</sup>" in html
    assert content_key(html) == visible_key(render_publication_markdown(OUTER).html)
    assert [value for kind, value in visible_key(html) if kind == "citation"] == [
        str(numbering[CHILD_SUPPORT]), str(numbering[PARENT_SUPPORT]),
    ]


def test_nested_full_article_citations_preserve_section_equivalence(base_state):
    sections = {
        "problem_framing": "A learning note.",
        "technical_dive": f"In practice, {OUTER}.",
        "code_snippets": "```python\nprint(1)\n```",
        "takeaways": "- Use a suitable batch size.",
    }
    draft = nodes._assemble_markdown(base_state["topic"], sections)
    state, _ = _nested(base_state, draft)
    state["draft_sections"] = sections
    result = evaluate_citation_policy(state=state)
    assert citation_policy_passed(result)
    fragments = {name: render_publication_markdown(raw)
                 for name, raw in sections.items()}
    rendered = nodes._render_occurrence_bound_citations(state, fragments, result)
    assert set(rendered) == set(sections)
    for name in sections:
        assert content_key(rendered[name].html) == visible_key(fragments[name].html)
    technical = rendered["technical_dive"].html
    numbering = evidence_numbering(result.attached_evidence_ids)
    assert f"{INNER}<sup>{numbering[CHILD_SUPPORT]}</sup>" in technical
    assert f"average gradients over that batch<sup>{numbering[PARENT_SUPPORT]}</sup>" in technical


def test_nested_missing_or_swapped_child_support_fails_closed(base_state):
    state, _ = _nested(base_state, child_support=())
    missing = evaluate_citation_policy(state=state)
    assert not citation_policy_passed(missing)
    state, _ = _nested(base_state)
    plan = build_citation_plan(state)
    state["citation_render_overlay"] = [
        {"anchor_start": g["anchor_start"], "anchor_end": g["anchor_end"],
         "evidence_ids": [CHILD_SUPPORT if PARENT_SUPPORT in g["required_evidence_ids"]
                          else PARENT_SUPPORT]}
        for g in plan["anchor_groups"]
    ]
    swapped = evaluate_citation_policy(state=state)
    assert not citation_policy_passed(swapped)
    assert swapped.invalid_citation_count == 2
    assert swapped.missing_required_citation_count == 2


def test_nested_supplied_plan_cannot_launder_changed_support(base_state):
    state, inventory = _nested(base_state)
    stale = build_citation_plan(state)
    state["citation_plan"] = stale
    child_id = inventory["claims"][1]["claim_id"]
    for row in state["grounding_report"]:
        if row["claim_id"] == child_id:
            row["support_spans"] = [{"evidence_id": "KB-003", "start": 0,
                                     "end": 6, "text": "KB-003"}]
    result = evaluate_citation_policy(state=state)
    assert not citation_policy_passed(result)
    assert "stale_citation_plan_support" in result.reason_codes


def test_nested_model_citation_with_ambiguous_owner_fails_closed(base_state):
    marked = OUTER.replace(INNER, INNER + " [KB-002]")
    state, _ = _nested(base_state, marked, outer_anchor=marked)
    # The child hint is inside the outer location span. Do not assign it to
    # the outer group simply because that group is scanned first.
    result = evaluate_citation_policy(state=state)
    assert not citation_policy_passed(result)
    assert "ambiguous_nested_model_citation" in result.reason_codes


def test_partial_overlap_remains_blocked(base_state):
    draft = "Alpha beta gamma delta."
    state, _ = _verified(base_state, draft, [
        _claim_row("Alpha beta gamma", "Alpha beta gamma"),
        _claim_row("beta gamma delta", "beta gamma delta"),
    ], [[PARENT_SUPPORT], [CHILD_SUPPORT]])
    result = evaluate_citation_policy(state=state)
    assert not citation_policy_passed(result)
    assert "overlapping_nonidentical_anchors" in result.reason_codes


def test_shared_start_remains_blocked(base_state):
    draft = "Alpha beta gamma delta."
    state, _ = _verified(base_state, draft, [
        _claim_row("Alpha beta gamma delta", "Alpha beta gamma delta"),
        _claim_row("Alpha beta", "Alpha beta"),
    ], [[PARENT_SUPPORT], [CHILD_SUPPORT]])
    result = evaluate_citation_policy(state=state)
    assert not citation_policy_passed(result)
    assert "overlapping_nonidentical_anchors" in result.reason_codes


def test_three_nested_atomic_claims_keep_three_support_bases(base_state):
    middle = "use a mini-batch of size B (typically 32–256) and average gradients"
    state, inventory = _verified(base_state, OUTER, [
        _claim_row(OUTER_CLAIM, OUTER),
        _claim_row("A mini-batch of size B is used to average gradients.", middle),
        _claim_row(INNER_CLAIM, INNER),
    ], [[PARENT_SUPPORT], ["KB-003"], [CHILD_SUPPORT]])
    state["evidence_registry"] = {
        eid: {"evidence_id": eid, "kind": "kb", "source_ref": f"kb:{eid}"}
        for eid in (PARENT_SUPPORT, "KB-003", CHILD_SUPPORT)
    }
    result = evaluate_citation_policy(state=state)
    assert citation_policy_passed(result)
    assert len(result.anchor_groups) == 3
    groups = _groups_by_claim(result)
    for claim, support in zip(inventory["claims"],
                              (PARENT_SUPPORT, "KB-003", CHILD_SUPPORT)):
        assert groups[claim["claim_id"]]["required_evidence_ids"] == [support]
    numbering = evidence_numbering(result.attached_evidence_ids)
    annotated, markers = occurrence_marked_markdown(
        OUTER, section_start=0, groups=result.anchor_groups, numbering=numbering,
    )
    html = replace_rendered_occurrence_markers(
        render_publication_markdown(annotated).html, markers,
    )
    assert [value for kind, value in visible_key(html) if kind == "citation"] == [
        str(numbering[CHILD_SUPPORT]), str(numbering["KB-003"]),
        str(numbering[PARENT_SUPPORT]),
    ]


def test_nested_unknown_attachment_fails_closed(base_state):
    state, _ = _nested(base_state)
    plan = build_citation_plan(state)
    child = next(g for g in plan["anchor_groups"]
                 if CHILD_SUPPORT in g["required_evidence_ids"])
    state["citation_render_overlay"] = [{
        "anchor_start": child["anchor_start"], "anchor_end": child["anchor_end"],
        "evidence_ids": [CHILD_SUPPORT, "KB-999"],
    }]
    result = evaluate_citation_policy(state=state)
    assert not citation_policy_passed(result)
    assert result.invalid_citation_count >= 1
    assert "KB-999" in result.dangling_citations


def test_repeated_nested_occurrences_keep_each_claim_and_marker(base_state):
    draft = f"{OUTER}. Again: {OUTER}."
    state, inventory = _verified(base_state, draft, [
        _claim_row(OUTER, OUTER), _claim_row(INNER, INNER),
    ], [[PARENT_SUPPORT], [CHILD_SUPPORT]])
    result = evaluate_citation_policy(state=state)
    assert citation_policy_passed(result)
    assert len(result.anchor_groups) == 4
    assert len(inventory["claims"][0]["occurrences"]) == 2
    assert len(inventory["claims"][1]["occurrences"]) == 2
    numbering = evidence_numbering(result.attached_evidence_ids)
    annotated, markers = occurrence_marked_markdown(
        draft, section_start=0, groups=result.anchor_groups, numbering=numbering,
    )
    html = replace_rendered_occurrence_markers(
        render_publication_markdown(annotated).html, markers,
    )
    assert html.count(f"<sup>{numbering[CHILD_SUPPORT]}</sup>") == 2
    assert html.count(f"<sup>{numbering[PARENT_SUPPORT]}</sup>") == 2
    assert content_key(html) == visible_key(render_publication_markdown(draft).html)


def test_cross_section_anchor_still_fails_rendering():
    with pytest.raises(PolicyError, match="crosses or leaves approved section"):
        occurrence_marked_markdown(
            "Alpha text", section_start=20,
            groups=[{"anchor_start": 18, "anchor_end": 24,
                     "attached_evidence_ids": [PARENT_SUPPORT]}],
            numbering={PARENT_SUPPORT: 1},
        )
