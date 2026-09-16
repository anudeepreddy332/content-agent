"""Phase 4 Slice 2C: citation correctness, completeness, placement.

Deterministic, $0, no providers. Covers spec matrix A-U plus rendering,
shared-anchor, HITL/API-approval bypass, and publication-safety conjunction.
"""
from __future__ import annotations

from langgraph.graph import END

import agent.nodes as nodes
from agent.semantic_analyzer.status_engine import sha256_utf8
from agent.claim_inventory import build_claim_inventory
from agent.citation_policy import (
    CITATION_HITL,
    CITATION_PASS,
    bibliography_items,
    build_citation_plan,
    build_evidence_registry,
    citation_policy_passed,
    evaluate_citation_policy,
    insert_citation_markers,
)
from agent.html_policy import render_citations_html
from agent.material_policy import MATERIAL_PASS, evaluate_material_policy
from config import MAX_ITERATIONS

E1 = "WEB-001"
E2 = "WEB-002"
E99 = "E99"


def _claim_row(
    claim_text, anchor_quote=None, *, claim_type="factual", material=True,
    satisfies_req_ids=None,
):
    return {
        "claim_text": claim_text,
        "anchor_quote": anchor_quote or claim_text,
        "section": "technical_dive",
        "claim_type": claim_type,
        "material": material,
        "materiality_reason_code": None,
        "materiality_rationale": None,
        "satisfies_req_ids": satisfies_req_ids or [],
        "specificity": "substantive",
        "requires_citation": None,
    }


def _inventory(draft, rows, reqs=None):
    return build_claim_inventory(
        run_id="t", iteration=1, draft_markdown=draft,
        raw_claims=rows, brief_requirements=reqs or [],
    )


def _span(*eids):
    return [{"evidence_id": e, "start": 0, "end": 3, "text": e} for e in eids]


def _row(claim_id, status, *, support=None, analysis_validity="VALID"):
    return {
        "claim_id": claim_id,
        "claim": claim_id,
        "status": status,
        "blockers": [],
        "support_spans": _span(*(support or [])),
        "reason_codes": [],
        "analysis_validity": analysis_validity,
    }


def _registry():
    return {
        E1: {"evidence_id": E1, "kind": "web",
             "url": "https://example.com/e1", "title": "Source One"},
        E2: {"evidence_id": E2, "kind": "web",
             "url": "https://example.com/e2", "title": "Source Two"},
        "KB-001": {"evidence_id": "KB-001", "kind": "kb",
                   "url": None, "title": "notes.md", "source_ref": "kb:notes.md"},
    }


def _state_with(base_state, draft, inv, report, *, iterations=1, **extra):
    state = dict(base_state)
    state.update({
        "draft_markdown": draft,
        "claim_inventory": inv,
        "grounding_report": report,
        "verification_status": "completed",
        "grounding_score": 0.9,
        "reflection_score": 8,
        "iterations": iterations,
        "total_cost_usd": 0.01,
        "evidence_registry": _registry(),
        "web_sources": [
            {"title": "Source One", "url": "https://example.com/e1",
             "content": "e1 body", "score": 0.9},
            {"title": "Source Two", "url": "https://example.com/e2",
             "content": "e2 body", "score": 0.8},
        ],
    })
    state.update(extra)
    return state


def _verified(base_state, draft, rows, supports, **extra):
    inv = _inventory(draft, rows)
    report = [
        _row(c["claim_id"], "verified", support=s)
        for c, s in zip(inv["claims"], supports)
    ]
    return _state_with(base_state, draft, inv, report, **extra), inv


# ---------------------------------------------------------------------------
# A-G core citation gates
# ---------------------------------------------------------------------------

def test_A_complete_valid_support_citation_passes(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1, E2]])
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_PASS
    assert citation_policy_passed(res) is True
    assert set(res.required_evidence_ids) == {E1, E2}
    assert res.missing_required_citation_count == 0
    assert res.invalid_citation_count == 0
    assert res.citation_placement_failure_count == 0


def test_B_undercitation_subset_fails_completeness(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1, E2]])
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1],
    }]
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert res.missing_required_citation_count >= 1
    assert E2 in res.missing_citation_obligations[0]["missing_evidence_ids"]
    assert citation_policy_passed(res) is False


def test_C_unsupported_extra_citation_fails_correctness(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]])
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1, E2],
    }]
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert res.invalid_citation_count >= 1
    extras = res.invalid_citation_obligations[0]["invalid_evidence_ids"]
    assert E2 in extras


def test_D_citation_elsewhere_fails_placement(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]])
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["end"] + 1, "anchor_end": occ["end"] + 8,
        "evidence_ids": [E1],
    }]
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert res.citation_placement_failure_count >= 1
    assert any(p["reason"] == "citation_not_at_occurrence" for p in res.placement_failures)


def test_E_dangling_evidence_id_fails_integrity(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]])
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1, E99],
    }]
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert E99 in res.dangling_citations
    assert res.dangling_citation_count >= 1


def test_F_stale_citation_plan_draft_sha_fails(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]])
    plan = build_citation_plan(state)
    plan["draft_sha256"] = "0" * 64
    state["citation_plan"] = plan
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert "stale_citation_plan_draft_sha" in res.reason_codes
    assert res.stale_plan_failures


def test_G_stale_claim_roster_fails(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]])
    plan = build_citation_plan(state)
    plan["claim_roster_ids"] = ["clm-not-current"]
    state["citation_plan"] = plan
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert "stale_citation_plan_claim_roster" in res.reason_codes


# ---------------------------------------------------------------------------
# H-M grouping, non-material, claim_type, model citations
# ---------------------------------------------------------------------------

def test_H_shared_exact_anchor_is_one_group_union_support(base_state):
    draft = "System A improves latency and reduces storage cost."
    rows = [
        _claim_row("System A improves latency.", draft),
        _claim_row("System A reduces storage cost.", draft),
    ]
    state, inv = _verified(base_state, draft, rows, [[E1], [E2]])
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_PASS
    assert res.citation_anchor_group_count == 1
    group = res.anchor_groups[0]
    assert set(group["claim_ids"]) == {c["claim_id"] for c in inv["claims"]}
    assert set(group["required_evidence_ids"]) == {E1, E2}
    marked = insert_citation_markers(draft, res.anchor_groups)
    assert marked.count("⟦cite:") == 1
    assert E1 in marked and E2 in marked
    assert draft in state["draft_markdown"]
    assert "⟦cite:" not in state["draft_markdown"]


def test_I_overlapping_nonidentical_anchors_fail_closed(base_state):
    draft = "System A improves latency and reduces storage cost."
    rows = [
        _claim_row("System A improves latency and reduces storage cost.", draft),
        _claim_row("reduces storage cost.", "reduces storage cost."),
    ]
    state, inv = _verified(base_state, draft, rows, [[E1], [E2]])
    spans = [(o["start"], o["end"]) for c in inv["claims"] for o in c["occurrences"]]
    assert len(spans) == 2
    assert spans[0] != spans[1]
    assert spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert "overlapping_nonidentical_anchors" in res.reason_codes
    assert nodes.route_after_reflect(state) == "hitl"


def test_J_nonmaterial_without_citation_does_not_hard_fail(base_state):
    draft = "We find this design elegant.\n\nDropout randomly zeroes activations."
    rows = [
        _claim_row("We find this design elegant.", material=False, claim_type="editorial"),
        _claim_row("Dropout randomly zeroes activations"),
    ]
    state, inv = _verified(base_state, draft, rows, [[], [E1]])
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_PASS
    editorial_id = inv["claims"][0]["claim_id"]
    assert editorial_id not in res.applicable_claim_ids
    assert res.missing_required_citation_count == 0


def test_K_nonmaterial_unsupported_rendered_citation_fails_correctness(base_state):
    draft = "We find this design elegant."
    rows = [_claim_row(draft, material=False, claim_type="editorial")]
    state, inv = _verified(base_state, draft, rows, [[]])
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1],
    }]
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert res.invalid_citation_count >= 1


def test_L_claim_type_cannot_exempt_material_verified_claim(base_state):
    draft = "A transformer is a neural architecture."
    rows = [_claim_row(draft, claim_type="editorial", material=True)]
    state, inv = _verified(base_state, draft, rows, [[E1]])
    res = evaluate_citation_policy(state=state)
    assert inv["claims"][0]["claim_id"] in res.applicable_claim_ids
    state["citation_render_overlay"] = [{
        "anchor_start": inv["claims"][0]["occurrences"][0]["start"],
        "anchor_end": inv["claims"][0]["occurrences"][0]["end"],
        "evidence_ids": [],
    }]
    res_miss = evaluate_citation_policy(state=state)
    assert res_miss.decision == CITATION_HITL
    assert res_miss.missing_required_citation_count >= 1


def test_M_model_written_citation_cannot_satisfy_disagreeing_support(base_state):
    draft = "Dropout randomly zeroes activations during training. [E99]"
    rows = [_claim_row("Dropout randomly zeroes activations during training.")]
    state, inv = _verified(base_state, draft, rows, [[E1]])
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_HITL
    assert res.invalid_citation_count >= 1 or res.dangling_citation_count >= 1


def test_M2_model_citation_cannot_satisfy_completeness_alone(base_state):
    """Model [WEB-001] at the claim cannot stand in for a missing plan attach
    of the complete {E1,E2} basis when overlay renders only nothing extra
    from the plan. Production plan attaches the full set; overlay strips it."""
    draft = "Dropout randomly zeroes activations during training. [WEB-001]"
    rows = [_claim_row("Dropout randomly zeroes activations during training.")]
    state, inv = _verified(base_state, draft, rows, [[E1, E2]])
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [],
    }]
    res = evaluate_citation_policy(state=state)
    # Model may add E1 at the occurrence; E2 still missing. Model cannot
    # satisfy the complete support set by itself.
    assert res.decision == CITATION_HITL
    assert E2 in (res.missing_citation_obligations[0]["missing_evidence_ids"] if res.missing_citation_obligations else [E2])


# ---------------------------------------------------------------------------
# N-Q scores cannot override; approval cannot bypass
# ---------------------------------------------------------------------------

def test_N_coverage_rate_cannot_override_raw_failure(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1, E2]])
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1],
    }]
    res = evaluate_citation_policy(state=state)
    assert res.citation_coverage_rate == 0.5
    assert res.decision != CITATION_PASS
    assert citation_policy_passed(res) is False


def test_O_high_reflection_cannot_override_citation_failure(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1, E2]],
                           iterations=MAX_ITERATIONS, reflection_score=10)
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1],
    }]
    state["reflection_score"] = 10
    assert nodes.route_after_reflect(state) == "hitl"
    assert nodes.publication_safety_accepted(state) is False


def test_P_hitl_auto_approve_cannot_bypass_missing_citation(base_state, monkeypatch):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1, E2]],
                           iterations=MAX_ITERATIONS)
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1],
    }]
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_Pb_hitl_auto_approve_cannot_bypass_unsupported_citation(base_state, monkeypatch):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]],
                           iterations=MAX_ITERATIONS)
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1, E2],
    }]
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_Pc_hitl_auto_approve_cannot_bypass_placement_failure(base_state, monkeypatch):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]],
                           iterations=MAX_ITERATIONS)
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["end"] + 1, "anchor_end": occ["end"] + 8,
        "evidence_ids": [E1],
    }]
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_Q_api_approve_cannot_bypass_citation_failure(base_state, monkeypatch):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1, E2]],
                           iterations=MAX_ITERATIONS)
    occ = inv["claims"][0]["occurrences"][0]
    state["citation_render_overlay"] = [{
        "anchor_start": occ["start"], "anchor_end": occ["end"],
        "evidence_ids": [E1],
    }]
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: {"action": "approve"})
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


# ---------------------------------------------------------------------------
# R-U rendering identity, stale N+1, conjunction, no autonomous publish
# ---------------------------------------------------------------------------

def test_R_citation_rendering_leaves_canonical_draft_sha_unchanged(base_state, monkeypatch):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]])
    state["draft_sections"] = {
        "problem_framing": draft,
        "technical_dive": draft,
        "code_snippets": "```python\nprint(1)\n```",
        "takeaways": "- keep going",
    }
    sha_before = sha256_utf8(state["draft_markdown"])
    monkeypatch.setattr(
        nodes, "_render_technical_dive_via_llm",
        lambda **kw: (f"<p>{draft}</p>", 0, 0.0),
    )
    out = nodes.html_gen_node(state)
    assert sha256_utf8(state["draft_markdown"]) == sha_before
    assert inv["draft_sha256"] == sha_before
    assert out.get("html_output")
    assert "⟦cite:" not in state["draft_markdown"]
    html = out["html_output"]
    assert "Source One" in html or "example.com/e1" in html
    assert "example.com/e2" not in html  # uncited retrieved source must not appear


def test_S_plan_for_draft_n_cannot_certify_draft_n1(base_state):
    draft_n = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft_n)]
    state, inv = _verified(base_state, draft_n, rows, [[E1]])
    plan_n = build_citation_plan(state)
    draft_n1 = "Dropout randomly zeroes activations during training. Also, momentum helps."
    rows_n1 = [_claim_row("Dropout randomly zeroes activations during training.")]
    state_n1, _ = _verified(base_state, draft_n1, rows_n1, [[E1]])
    state_n1["citation_plan"] = plan_n
    res = evaluate_citation_policy(state=state_n1)
    assert res.decision == CITATION_HITL
    assert "stale_citation_plan_draft_sha" in res.reason_codes


def test_T_all_hard_conditions_yield_publication_safety_pass(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]],
                           iterations=MAX_ITERATIONS, reflection_score=8)
    assert nodes.semantic_verification_accepted(state) is True
    mat = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert mat.decision == MATERIAL_PASS
    cite = evaluate_citation_policy(state=state)
    assert cite.decision == CITATION_PASS
    assert nodes.publication_safety_accepted(state) is True
    assert nodes.publication_safety_pass(state) is True


def test_U_publication_safety_pass_does_not_autonomously_publish(base_state):
    draft = "Dropout randomly zeroes activations during training."
    rows = [_claim_row(draft)]
    state, inv = _verified(base_state, draft, rows, [[E1]],
                           iterations=MAX_ITERATIONS)
    assert nodes.publication_safety_accepted(state) is True
    state["hitl_status"] = "pending"
    assert nodes.route_after_hitl(state) != "html_gen"
    assert nodes.route_after_hitl(state) == END
    state["hitl_status"] = "approved"
    assert nodes.route_after_hitl(state) == "html_gen"


def test_bibliography_deduplicates_source_used_on_multiple_claims(base_state):
    draft = "Momentum accelerates convergence. Dropout zeroes activations."
    rows = [
        _claim_row("Momentum accelerates convergence."),
        _claim_row("Dropout zeroes activations."),
    ]
    state, inv = _verified(base_state, draft, rows, [[E1], [E1]])
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_PASS
    assert res.citation_anchor_group_count == 2
    items = bibliography_items(res.attached_evidence_ids, build_evidence_registry(state))
    assert len(items) == 1
    html = render_citations_html(items)
    assert html.count("example.com/e1") == 1
    marked = insert_citation_markers(draft, res.anchor_groups)
    assert marked.count("⟦cite:") == 2


def test_pre2a_absent_inventory_preserves_slice1_pass(base_state):
    state = dict(base_state)
    state.update({
        "claim_inventory": None,
        "grounding_report": [
            {"claim": "x", "status": "verified", "claim_id": "c1"}
            for _ in range(10)
        ],
        "verification_status": "completed",
        "iterations": MAX_ITERATIONS,
        "reflection_score": 8,
        "hitl_status": "approved",
    })
    res = evaluate_citation_policy(state=state)
    assert res.decision == CITATION_PASS
    assert nodes.citation_policy_accepted(state) is True
    assert nodes.route_after_hitl(state) == "html_gen"
