"""Deterministic positive-control review decisions for unrelated offline tests."""

from agent import nodes, workflow
from agent.claim_inventory import build_claim_inventory
from agent.evidence_signoff import material_review_bundle
from agent.reviewer_identity import reviewer_context
from agent.html_policy import render_markdown_fragment
from tests.workflow_fixtures import current_quality


def supported_signoff(state: dict) -> dict:
    bundle = material_review_bundle(state)
    return {
        "evidence_binding_sha256": bundle["evidence_binding_sha256"],
        "decisions": [
            {"claim_id": claim_id, "decision": "SUPPORTED",
             "rationale": "The bound passage supports the exact claim."}
            for claim_id in bundle["material_claim_ids"]
        ],
        "coverage_attestation": {
            "no_omitted_material_claims": True,
            "materiality_classification_correct": True,
        },
    }


def approved_material_state(state: dict, *, reviewer_id: str | None = None) -> dict:
    """Attach one genuinely supported claim and mint a real Gate-1 test receipt.

    Use only for positive controls downstream of Gate 1. Negative tests must
    retain their missing/invalid approvals instead of calling this helper.
    """
    state = dict(state)
    claim_text = "Gradient descent minimizes a loss function."
    draft = state.get("draft_markdown") or claim_text
    if claim_text not in draft:
        draft = f"{draft}\n{claim_text}"
    source = {"title": "Gradient-descent reference", "url": "https://example.com/gd",
              "content": claim_text, "score": 0.9}
    state.update(draft_markdown=draft, web_sources=[source], kb_results=[],
                 iterations=1, verification_status="completed", grounding_score=0.9,
                 reflection_score=8, total_cost_usd=0.01)
    inventory = build_claim_inventory(
        run_id=state["run_id"], iteration=1, draft_markdown=draft,
        brief_requirements=[], raw_claims=[{
            "claim_text": claim_text, "anchor_quote": claim_text,
            "section": "technical_dive", "claim_type": "factual", "material": True,
            "materiality_reason_code": None, "materiality_rationale": "Core factual claim",
            "satisfies_req_ids": [], "specificity": "substantive", "requires_citation": None,
        }],
    )
    state["claim_inventory"] = inventory
    state["grounding_report"] = [{
        "claim_id": inventory["claims"][0]["claim_id"], "claim": claim_text,
        "status": "verified", "analysis_validity": "VALID", "full_entailment": True,
        "blockers": [], "reason_codes": [],
        "support_spans": [{"evidence_id": "WEB-001", "start": 0,
                           "end": len(claim_text), "text": claim_text}],
    }]
    sections = state.get("draft_sections")
    state = current_quality(state)
    if isinstance(sections, dict) and all(isinstance(sections.get(key), str) and sections[key].strip()
                                           for key in ("problem_framing", "technical_dive",
                                                       "code_snippets", "takeaways")):
        state["draft_sections"] = sections
    identity = reviewer_id or state.get("gate1_reviewer_id") or "fixture-reviewer"
    review_id = f"fixture-review-{state['run_id']}"
    with reviewer_context(identity, state["run_id"], review_id=review_id):
        update = nodes.hitl_approve_update(
            state, supported_signoff(state), submitted_review_id=review_id,
        )
    if update.get("hitl_status") != "approved":
        raise AssertionError("positive-control material signoff was rejected")
    result = {**state, **update}
    if not workflow.gate1_eligible(result):
        raise AssertionError("positive-control automated Gate 1 eligibility failed")
    return result


def renderable_approved_state(state: dict, *, reviewer_id: str = "fixture-reviewer",
                              sections: dict | None = None) -> dict:
    """A coherent approved draft for downstream HTML/Git positive controls."""
    sections = dict(sections or {
        "problem_framing": "A gradient descent learning note.",
        "technical_dive": "Gradient descent minimizes a loss function.",
        "code_snippets": "print(1)",
        "takeaways": "- Use a suitable learning rate.",
    })
    claim = "Gradient descent minimizes a loss function."
    if claim not in sections["technical_dive"]:
        sections["technical_dive"] += f"\n\n{claim}"
    state = {**state, "draft_sections": sections,
             "draft_markdown": nodes._assemble_markdown(state["topic"], sections)}
    state.pop("evidence_registry", None)  # production graph derives this from current sources
    return approved_material_state(state, reviewer_id=reviewer_id)


def render_approved_offline(state: dict, monkeypatch) -> dict:
    """Run the production initial renderer with deterministic, provider-free HTML."""
    monkeypatch.setattr(nodes, "_render_technical_dive_via_llm",
                        lambda **kwargs: (render_markdown_fragment(kwargs["technical_dive"]).html,
                                          0, 0.0))
    update = nodes.html_gen_node(state)
    if not update.get("html_output"):
        raise AssertionError(f"valid offline article failed to render: {update.get('error_log')}")
    return {**state, **update}


def publishable_result(run_id: str, slug: str, monkeypatch, *, reviewer_id: str = "fixture-reviewer") -> dict:
    """A complete two-gate positive control for isolated publish-boundary tests."""
    state = renderable_approved_state({
        "run_id": run_id, "topic": "Gradient Descent", "slug": slug,
        "series_context": "Test", "total_cost_usd": 0.0,
        "latency_ms": {}, "error_log": [],
    }, reviewer_id=reviewer_id)
    state = render_approved_offline(state, monkeypatch)
    state.update(gate2_reviewer_id=reviewer_id, gate2_review_origin="authenticated",
                 approved_html_sha256=state["html_sha256"])
    return state
