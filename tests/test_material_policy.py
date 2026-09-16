"""Phase 4 Slice 2B: material-claim + required-content acceptance policy.

Deterministic, $0, no providers. Covers the spec §19 causal matrix A-Q plus
denominator-gaming, version integrity, routing, HITL/auto-approval bypass
prevention, API-approval bypass prevention, and the UNKNOWN-materiality
no-stochastic-retry invariant.
"""
from __future__ import annotations

import json
from langgraph.graph import END

import agent.nodes as nodes
from agent.claim_inventory import build_claim_inventory
from agent.material_policy import (
    MATERIAL_HITL,
    MATERIAL_PASS,
    MATERIAL_REVISION,
    REQ_MISSING,
    REQ_SATISFIED,
    REQ_UNRESOLVED,
    evaluate_material_policy,
    format_required_content_feedback,
    material_policy_passed,
)
from config import MAX_ITERATIONS

DRAFT = "System A improves latency and reduces storage cost."


def _claim_row(
    claim_text, anchor_quote=None, *, claim_type="factual", material=True,
    satisfies_req_ids=None, specificity="substantive",
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
        "specificity": specificity,
        "requires_citation": None,
    }


def _inventory(draft, rows, reqs=None):
    return build_claim_inventory(
        run_id="t", iteration=1, draft_markdown=draft,
        raw_claims=rows, brief_requirements=reqs or [],
    )


def _row(claim_id, status, *, blockers=None, analysis_validity="VALID"):
    return {
        "claim_id": claim_id,
        "claim": claim_id,
        "status": status,
        "blockers": blockers or [],
        "support_spans": [],
        "reason_codes": [],
        "analysis_validity": analysis_validity,
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
    })
    state.update(extra)
    return state


def _verified_state(base_state, draft, rows, statuses, reqs=None, iterations=1):
    inv = _inventory(draft, rows, reqs)
    report = [_row(c["claim_id"], s) for c, s in zip(inv["claims"], statuses)]
    return _state_with(base_state, draft, inv, report, iterations=iterations), inv


# ---------------------------------------------------------------------------
# §19 causal matrix A-G (material claim resolution)
# ---------------------------------------------------------------------------

def test_A_material_verified_resolved(base_state):
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]
    state, inv = _verified_state(base_state, draft, rows, ["verified"])
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_PASS
    assert res.unresolved_material_claim_count == 0
    assert res.resolved_material_claim_count == 1
    assert res.material_verified_rate == 1.0


def test_B_material_weak_revision_required(base_state):
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]
    state, inv = _verified_state(base_state, draft, rows, ["weak"], iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_REVISION
    assert res.unresolved_material_claim_ids == [inv["claims"][0]["claim_id"]]
    assert nodes.route_after_reflect(state) == "draft"


def test_C_material_unverified_revision_required(base_state):
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]
    state, inv = _verified_state(base_state, draft, rows, ["unverified"], iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_REVISION
    assert nodes.route_after_reflect(state) == "draft"


def test_D_material_contradiction_revision_required(base_state):
    draft = "System A improves latency."
    inv = _inventory(draft, [_claim_row("System A improves latency.", material=True)])
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "weak", blockers=[
        {"kind": "contradiction", "explanation": "x", "evidence_spans": []}])]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_REVISION
    assert cid in res.unresolved_material_claim_ids
    # Weak status is unresolved for a material claim; the contradiction is
    # already captured by the semantic blocker gate, so material resolution
    # reports the non-verified status.
    assert res.material_claim_resolution[cid].startswith("semantic_status_")


def test_E_material_limitation_revision_required(base_state):
    draft = "System A improves latency."
    inv = _inventory(draft, [_claim_row("System A improves latency.", material=True)])
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "weak", blockers=[
        {"kind": "limitation", "explanation": "x", "evidence_spans": []}])]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_REVISION


def test_F_material_invalid_fail_closed(base_state):
    draft = "A is invalid. B is verified. C is verified. D is verified. E is verified. F is verified. G is verified. H is verified. I is verified. J is verified."
    rows = [_claim_row(t, material=True) for t in [
        "A is invalid.", "B is verified.", "C is verified.", "D is verified.",
        "E is verified.", "F is verified.", "G is verified.", "H is verified.",
        "I is verified.", "J is verified.",
    ]]
    inv = _inventory(draft, rows)
    ids = [cl["claim_id"] for cl in inv["claims"]]
    # Claim A is a material INVALID row; the other nine are verified so UVR
    # is 0.1 (semantic gate passes) but the material INVALID claim fails closed.
    report = [_row(ids[0], "unverified", analysis_validity="INVALID")]
    report += [_row(i, "verified") for i in ids[1:]]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    assert nodes.semantic_verification_accepted(state) is True  # UVR gate passes
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    # INVALID is fail-closed: HITL, never revision (not repairable).
    assert res.decision == MATERIAL_HITL
    assert res.invalid_material_claim_ids == [ids[0]]
    assert res.unresolved_material_claim_ids == [ids[0]]
    assert "invalid_material_claim" in res.reason_codes
    assert nodes.route_after_reflect(state) == "hitl"


def test_G_materiality_unknown_hitl_not_stochastic_retry(base_state):
    draft = "System A improves latency."
    inv = _inventory(draft, [_claim_row("System A improves latency.", material="unknown")])
    cid = inv["claims"][0]["claim_id"]
    # Even a VERIFIED semantic row cannot auto-pass UNKNOWN materiality.
    report = [_row(cid, "verified")]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_HITL
    assert res.unknown_materiality_claim_ids == [cid]
    # Unknown materiality does NOT route to revision (no retry-to-green),
    # even with full revision budget remaining.
    assert nodes.route_after_reflect(state) == "hitl"


def test_Gb_unknown_materiality_live_cases(base_state):
    """Deterministic equivalents of the qualified provider run's real UNKNOWN
    examples (cache/dropout/incidental dashboard-color)."""
    for label, claim_text in [
        ("cache", "The cache tier persists writes across restarts."),
        ("dropout", "Dropout with rate 0.5 is applied at inference."),
        ("dashboard", "The dashboard uses a blue color scheme."),
    ]:
        inv = _inventory(claim_text, [_claim_row(claim_text, material="unknown")])
        cid = inv["claims"][0]["claim_id"]
        report = [_row(cid, "verified")]
        state = _state_with(base_state, claim_text, inv, report, iterations=1)
        res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
        assert res.decision == MATERIAL_HITL, label
        assert res.unknown_materiality_claim_ids == [cid], label


# ---------------------------------------------------------------------------
# §19 H-Q (required content, denominator-gaming, version, routing, approval)
# ---------------------------------------------------------------------------

def test_H_required_claim_deleted_requirement_becomes_missing(base_state):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    # Draft N had claim A linked to REQ-1; revision deletes A and replaces it
    # with unrelated content. The current inventory no longer links REQ-1.
    draft = "Some unrelated content."
    rows = [_claim_row("Some unrelated content.", material=False)]
    inv = _inventory(draft, rows, reqs)
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "verified")]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.missing_requirement_ids == ["REQ-1"]
    assert res.mandatory_requirement_states["REQ-1"] == REQ_MISSING
    assert res.decision == MATERIAL_REVISION  # repairable: add the content


def test_I_mandatory_requirement_supported_only_by_weak_unresolved(base_state):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True,
                        satisfies_req_ids=["REQ-1"])]
    inv = _inventory(draft, rows, reqs)
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "weak")]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.mandatory_requirement_states["REQ-1"] == REQ_UNRESOLVED
    assert res.unresolved_requirement_ids == ["REQ-1"]
    assert res.decision == MATERIAL_REVISION


def test_J_mandatory_requirement_fulfilled_by_verified_material_claim(base_state):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True,
                        satisfies_req_ids=["REQ-1"])]
    state, inv = _verified_state(base_state, draft, rows, ["verified"], reqs=reqs)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.mandatory_requirement_states["REQ-1"] == REQ_SATISFIED
    assert res.decision == MATERIAL_PASS


def test_K_optional_requirement_missing_no_hard_block(base_state):
    reqs = [{"req_id": "REQ-OPT", "kind": "required_content", "mandatory": False,
             "description": "optional aside"}]
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]
    state, inv = _verified_state(base_state, draft, rows, ["verified"], reqs=reqs)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert "REQ-OPT" not in res.mandatory_requirement_states
    assert res.decision == MATERIAL_PASS


def test_L_stale_inventory_fail_closed(base_state):
    draft = "System A improves latency."
    inv = _inventory(draft, [_claim_row("System A improves latency.", material=True)])
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "verified")]
    state = _state_with(base_state, "System A improves p99 latency.", inv, report,
                         iterations=1)  # different draft -> stale
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.material_safety_state == "integrity_failure"
    assert "stale_inventory_draft_sha" in res.reason_codes
    assert res.decision == MATERIAL_HITL
    # Fail closed: the stale inventory never auto-passes. The semantic gate
    # already rejects a stale inventory, so routing is fail-closed (re-verify
    # via draft, or HITL on exhaustion) — never onward to publish.
    assert nodes.semantic_verification_accepted(state) is False
    assert nodes.route_after_reflect(state) in ("draft", "hitl")
    assert nodes.route_after_reflect({**state, "iterations": MAX_ITERATIONS}) == "hitl"


def test_M_stale_grounding_roster_fail_closed(base_state):
    draft = "System A improves latency. System A reduces storage cost."
    rows = [
        _claim_row("System A improves latency."),
        _claim_row("System A reduces storage cost."),
    ]
    inv = _inventory(draft, rows)
    # Roster has only one of the two inventory claims -> mismatch.
    report = [_row(inv["claims"][0]["claim_id"], "verified")]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.material_safety_state == "integrity_failure"
    assert "grounding_roster_mismatch" in res.reason_codes
    assert res.decision == MATERIAL_HITL


def test_N_all_resolved_and_mandatory_satisfied_pass(base_state):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    draft = "System A improves latency. System A reduces storage cost."
    rows = [
        _claim_row("System A improves latency.", material=True,
                    satisfies_req_ids=["REQ-1"]),
        _claim_row("System A reduces storage cost.", material=True),
    ]
    state, inv = _verified_state(base_state, draft, rows, ["verified", "verified"],
                                 reqs=reqs)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_PASS
    assert res.material_safety_state == "pass"
    assert res.material_verified_rate == 1.0
    assert nodes.route_after_reflect(state) == "hitl"


def test_O_high_reflection_cannot_override_material_failure(base_state):
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]
    state, inv = _verified_state(base_state, draft, rows, ["weak"], iterations=1)
    state["reflection_score"] = 10  # maximum reflection
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_REVISION
    assert nodes.route_after_reflect(state) == "draft"


# ---------------------------------------------------------------------------
# §9 exhaustion, §10 denominator-gaming, §16 HITL/auto-approval, API approval
# ---------------------------------------------------------------------------

def test_exhaustion_routes_unresolved_material_to_hitl(base_state):
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]
    state, inv = _verified_state(base_state, draft, rows, ["weak"],
                                 iterations=MAX_ITERATIONS)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_HITL
    assert res.material_safety_state == "exhausted"
    assert nodes.route_after_reflect(state) == "hitl"


def test_exhaustion_routes_missing_requirement_to_hitl(base_state):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    draft = "Some unrelated content."
    inv = _inventory(draft, [], reqs)
    state = _state_with(base_state, draft, inv, [], iterations=MAX_ITERATIONS)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_HITL
    assert nodes.route_after_reflect(state) == "hitl"


def test_denominator_gaming_deleting_required_claim_does_not_improve_gate(base_state):
    """Spec §10: deleting required claim A must not make the artifact safer."""
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    # Draft N: A present, verified, linked -> pass.
    draft_n = "System A improves latency."
    rows_n = [_claim_row("System A improves latency.", material=True,
                          satisfies_req_ids=["REQ-1"])]
    state_n, inv_n = _verified_state(base_state, draft_n, rows_n, ["verified"],
                                     reqs=reqs)
    res_n = evaluate_material_policy(state=state_n, max_iterations=MAX_ITERATIONS)
    assert res_n.decision == MATERIAL_PASS

    # Revision deletes A: requirement R is now missing -> NOT safer (blocked).
    draft_rev = "Some unrelated content."
    rows_rev = [_claim_row("Some unrelated content.", material=False)]
    inv_rev = _inventory(draft_rev, rows_rev, reqs)
    cid_rev = inv_rev["claims"][0]["claim_id"]
    state_rev = _state_with(base_state, draft_rev, inv_rev,
                             [_row(cid_rev, "verified")], iterations=1)
    res_rev = evaluate_material_policy(state=state_rev, max_iterations=MAX_ITERATIONS)
    assert res_rev.decision != MATERIAL_PASS
    assert res_rev.missing_requirement_ids == ["REQ-1"]
    assert material_policy_passed(res_rev) is False


def test_P_hitl_auto_approve_cannot_bypass_material_failure(base_state, monkeypatch):
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]
    state, inv = _verified_state(base_state, draft, rows, ["weak"],
                                 iterations=MAX_ITERATIONS)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_Pb_hitl_auto_approve_cannot_bypass_unknown_materiality(base_state, monkeypatch):
    draft = "System A improves latency."
    inv = _inventory(draft, [_claim_row("System A improves latency.",
                                          material="unknown")])
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "verified")]
    state = _state_with(base_state, draft, inv, report, iterations=MAX_ITERATIONS)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_Pc_hitl_auto_approve_cannot_bypass_missing_requirement(base_state, monkeypatch):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]  # not linked
    state, inv = _verified_state(base_state, draft, rows, ["verified"],
                                 reqs=reqs, iterations=MAX_ITERATIONS)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_Q_api_approve_cannot_bypass_material_failure(base_state, monkeypatch):
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True)]
    state, inv = _verified_state(base_state, draft, rows, ["weak"],
                                 iterations=MAX_ITERATIONS)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: {"action": "approve"})
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


def test_Qb_api_approve_cannot_bypass_unknown_materiality(base_state, monkeypatch):
    draft = "System A improves latency."
    inv = _inventory(draft, [_claim_row("System A improves latency.",
                                          material="unknown")])
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "verified")]
    state = _state_with(base_state, draft, inv, report, iterations=MAX_ITERATIONS)
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: {"action": "approve"})
    result = nodes.hitl_node(state)
    assert result["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**state, **result}) == END


# ---------------------------------------------------------------------------
# §20 metrics/observability, §8 targeted feedback, §18 false positives,
# §11 new claim after revision, materiality != semantic status
# ---------------------------------------------------------------------------

def test_metrics_exposed_separately_not_combined(base_state):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    draft = "A is verified. B is weak. C is unknown."
    rows = [
        _claim_row("A is verified.", material=True, satisfies_req_ids=["REQ-1"]),
        _claim_row("B is weak.", material=True),
        _claim_row("C is unknown.", material="unknown"),
    ]
    inv = _inventory(draft, rows, reqs)
    a, b, c = [cl["claim_id"] for cl in inv["claims"]]
    report = [_row(a, "verified"), _row(b, "weak"), _row(c, "verified")]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.total_factual_claims == 3
    assert res.material_claim_count == 2  # A and B (C is unknown-materiality)
    assert res.resolved_material_claim_count == 1
    assert res.unresolved_material_claim_count == 1
    assert res.unknown_materiality_count == 1
    assert res.material_verified_rate == round(1 / 2, 6)
    assert res.mandatory_requirements_total == 1
    assert res.mandatory_requirement_states["REQ-1"] == REQ_SATISFIED
    assert res.satisfied_requirement_count == 1
    assert "score" not in res.to_dict()


def test_material_verified_rate_is_not_gate_authority(base_state):
    draft = "A is verified. B is weak."
    rows = [
        _claim_row("A is verified.", material=True),
        _claim_row("B is weak.", material=True),
    ]
    inv = _inventory(draft, rows)
    a, b = [cl["claim_id"] for cl in inv["claims"]]
    report = [_row(a, "verified"), _row(b, "weak")]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.material_verified_rate == 0.5
    assert res.decision == MATERIAL_REVISION
    assert material_policy_passed(res) is False


def test_required_content_feedback_is_targeted_not_generic(base_state):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    draft = "System A improves latency."
    rows = [_claim_row("System A improves latency.", material=True,
                        satisfies_req_ids=["REQ-1"])]
    inv = _inventory(draft, rows, reqs)
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "weak")]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    feedback = format_required_content_feedback(res)
    assert feedback
    assert "REQ-1" in feedback
    assert "must state the mechanism" in feedback
    assert cid in feedback
    assert "improve completeness" not in feedback
    inv_u = _inventory(draft, [_claim_row("System A improves latency.",
                                           material="unknown")], reqs)
    cu = inv_u["claims"][0]["claim_id"]
    res_u = evaluate_material_policy(
        state=_state_with(base_state, draft, inv_u, [_row(cu, "verified")], iterations=1),
        max_iterations=MAX_ITERATIONS,
    )
    assert format_required_content_feedback(res_u) == ""


def test_false_positives_participate_not_deleted(base_state):
    draft = "Real claim. Incidental extra claim."
    rows = [
        _claim_row("Real claim.", material=True),
        _claim_row("Incidental extra claim.", material=True),
    ]
    state, inv = _verified_state(base_state, draft, rows, ["verified", "weak"],
                                 iterations=1)
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.material_claim_count == 2
    assert res.decision == MATERIAL_REVISION


def test_materiality_does_not_change_semantic_status(base_state):
    draft = "System A improves latency."
    inv = _inventory(draft, [_claim_row("System A improves latency.",
                                          material="unknown")])
    cid = inv["claims"][0]["claim_id"]
    report = [_row(cid, "verified")]
    state = _state_with(base_state, draft, inv, report, iterations=1)
    assert nodes.semantic_verification_accepted(state) is True
    res = evaluate_material_policy(state=state, max_iterations=MAX_ITERATIONS)
    assert res.decision == MATERIAL_HITL


def test_new_claim_after_revision_evaluated_from_current_inventory(base_state, monkeypatch):
    import agent.graph as graph_mod
    from agent.claim_inventory import compute_claim_id
    from tests.conftest import verify_llm_client, fake_response
    claim_v1 = "Gradient descent minimizes a loss function."
    claim_new = "Momentum smooths parameter updates."
    draft_v1 = claim_v1
    draft_v2 = f"{claim_v1}\n\n{claim_new}"
    rows_v1 = json.dumps([_claim_row(claim_v1, claim_v1)])
    rows_v2 = json.dumps([_claim_row(claim_v1, claim_v1),
                          _claim_row(claim_new, claim_new, material=True)])
    analyzer_v1 = json.dumps({"observations": [{
        "claim_id": compute_claim_id(claim_v1),
        "support_quotes": [{"evidence_id": "WEB-001", "quote": claim_v1}],
        "full_entailment": True, "blockers": []}]})
    analyzer_v2 = json.dumps({"observations": [
        {"claim_id": compute_claim_id(claim_v1),
         "support_quotes": [{"evidence_id": "WEB-001", "quote": claim_v1}],
         "full_entailment": True, "blockers": []},
        {"claim_id": compute_claim_id(claim_new),
         "support_quotes": [], "full_entailment": False, "blockers": []},
    ]})
    client = verify_llm_client(rows_v1, analyzer_v1)
    client._responses.extend([fake_response(rows_v2), fake_response(analyzer_v2)])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)

    drafts = [draft_v1, draft_v2]

    def fake_draft(state):
        text = drafts[min(state.get("iterations", 0), 1)]
        return {"draft_markdown": text, "iterations": state.get("iterations", 0) + 1}

    monkeypatch.setattr(graph_mod, "retrieve_node", lambda state: {})
    monkeypatch.setattr(graph_mod, "draft_node", fake_draft)
    monkeypatch.setattr(graph_mod, "reflect_node", lambda state: {
        "reflection_score": 8, "reflection_notes": "ok"})
    monkeypatch.setattr(graph_mod, "hitl_node", lambda state: {
        "hitl_status": "rejected", "hitl_feedback": None})

    graph = graph_mod.build_graph()
    init = dict(base_state)
    init.update(iterations=0, grounding_report=[], verification_status="not_started",
                claim_inventory=None)
    result = graph.invoke(init)
    final_inv = result["claim_inventory"]
    assert compute_claim_id(claim_new) in {c["claim_id"] for c in final_inv["claims"]}
    assert result["hitl_status"] == "rejected"
    assert result.get("html_output") is None




