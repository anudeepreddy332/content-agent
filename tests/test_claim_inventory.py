"""Phase 4 Slice 2A: claim inventory + draft anchoring + materiality.

Deterministic, $0, no providers. Covers spec §24 cases A–O plus the Call-A
parse/contract boundary and Call-A→Call-B verify_node integration.
"""
from __future__ import annotations

import json

import pytest
from langgraph.graph import END

import agent.nodes as nodes
from agent.claim_inventory import (
    ANCHOR_AMBIGUOUS,
    ANCHOR_BOUND,
    ANCHOR_BOUND_MULTIPLE,
    ANCHOR_FAILED,
    ClaimInventoryError,
    bind_anchor_occurrences,
    build_claim_inventory,
    call_b_roster,
    compute_claim_id,
    inventory_critical_failures,
    parse_claim_inventory_rows,
)
from agent.semantic_analyzer.status_engine import sha256_utf8
from tests.conftest import fake_response, verify_llm_client

DRAFT = "System A improves latency and reduces storage cost."


def _row(
    claim_text: str,
    anchor_quote: str,
    *,
    section: str | None = "technical_dive",
    claim_type: str = "factual",
    material=True,
    satisfies_req_ids: list[str] | None = None,
    requires_citation=None,
    specificity: str = "substantive",
) -> dict:
    return {
        "claim_text": claim_text,
        "anchor_quote": anchor_quote,
        "section": section,
        "claim_type": claim_type,
        "material": material,
        "materiality_reason_code": None,
        "materiality_rationale": None,
        "satisfies_req_ids": satisfies_req_ids or [],
        "specificity": specificity,
        "requires_citation": requires_citation,
    }


def _build(draft: str, rows: list[dict], reqs: list[dict] | None = None) -> dict:
    return build_claim_inventory(
        run_id="t", iteration=1, draft_markdown=draft,
        raw_claims=rows, brief_requirements=reqs or [],
    )


# ---------------------------------------------------------------------------
# A: claim_text can differ from verbatim anchor_quote
# ---------------------------------------------------------------------------

def test_a_atomic_claim_text_differs_from_anchor_quote():
    rows = [
        _row("System A improves latency.", DRAFT),
        _row("System A reduces storage cost.", DRAFT),
    ]
    inv = _build(DRAFT, rows)
    assert len(inv["claims"]) == 2
    for claim in inv["claims"]:
        assert claim["anchor_quote"] == DRAFT
        assert claim["claim_text"] != claim["anchor_quote"]
        assert claim["occurrences"] == [{"start": 0, "end": len(DRAFT)}]
        assert claim["anchor_validity"] == ANCHOR_BOUND
    roster = call_b_roster(inv)
    assert {r["claim_text"] for r in roster} == {
        "System A improves latency.", "System A reduces storage cost.",
    }


def test_b_anchor_binds_exact_zero_based_half_open():
    draft = "Intro. " + DRAFT + " Outro."
    inv = _build(draft, [_row("System A improves latency.", DRAFT)])
    occ = inv["claims"][0]["occurrences"][0]
    assert (occ["start"], occ["end"]) == (7, 7 + len(DRAFT))
    assert draft[occ["start"]:occ["end"]] == DRAFT
    assert bind_anchor_occurrences(draft, DRAFT) == [(7, 7 + len(DRAFT))]


def test_c_repeated_claim_deterministic_occurrences():
    sentence = "T3 caches aggressively."
    draft = f"{sentence}\n\n{sentence}"
    inv = _build(draft, [_row(sentence, sentence)])
    assert len(inv["claims"]) == 1  # one logical claim_id, two anchors
    claim = inv["claims"][0]
    assert claim["anchor_validity"] == ANCHOR_BOUND_MULTIPLE
    n = len(sentence)
    expected = [(0, n), (n + 2, 2 * n + 2)]
    assert [(o["start"], o["end"]) for o in claim["occurrences"]] == expected
    again = _build(draft, [_row("T3 caches aggressively.", "T3 caches aggressively.")])
    assert again["claims"][0]["occurrences"] == claim["occurrences"]
    assert again["claims"][0]["claim_id"] == claim["claim_id"]


def test_d_zero_match_anchor_fails_closed_and_is_retained():
    inv = _build(DRAFT, [_row("System A improves latency.", "System A improves p99 latency.")])
    claim = inv["claims"][0]
    assert claim["anchor_validity"] == ANCHOR_FAILED
    assert claim["occurrences"] == []
    # retained, not dropped:
    assert inv["counts"]["total"] == 1
    assert inventory_critical_failures(inv)  # eligible + unresolved => critical


def test_e_ambiguous_anchor_unresolved_fails_closed():
    draft = DRAFT + "\n\n" + DRAFT
    inv = _build(draft, [_row("System A improves latency.", DRAFT)])
    claim = inv["claims"][0]
    assert claim["anchor_validity"] == ANCHOR_AMBIGUOUS
    assert len(claim["occurrences"]) == 2  # preserved, not dropped
    assert inventory_critical_failures(inv)


def test_f_python_assigns_ids_model_ids_and_offsets_rejected():
    with pytest.raises(ClaimInventoryError):
        parse_claim_inventory_rows(json.dumps([
            {**_row("X is Y.", "X is Y."), "claim_id": "claim-001"}
        ]))
    with pytest.raises(ClaimInventoryError):
        parse_claim_inventory_rows(json.dumps([
            {**_row("X is Y.", "X is Y."), "start": 0, "end": 6}
        ]))
    inv = _build(DRAFT, [_row("System A improves latency.", DRAFT)])
    claim = inv["claims"][0]
    assert claim["claim_id"] == compute_claim_id("System A improves latency.")
    assert claim["claim_id"].startswith("clm-")
    # changed proposition => changed identity
    assert compute_claim_id("System A improves latency.") != compute_claim_id("System A reduces latency.")


def test_g_materiality_does_not_alter_roster_identity_or_semantics_path():
    rows_t = [_row("System A improves latency.", DRAFT, material=True)]
    rows_f = [_row("System A improves latency.", DRAFT, material=False)]
    rows_u = [_row("System A improves latency.", DRAFT, material="unknown")]
    rosters = [
        call_b_roster(_build(DRAFT, rows)) for rows in (rows_t, rows_f, rows_u)
    ]
    # Same claim enters Call B identically; materiality is not semantic input.
    assert rosters[0] == rosters[1] == rosters[2]
    materials = [
        _build(DRAFT, rows)["claims"][0]["material"] for rows in (rows_t, rows_f, rows_u)
    ]
    assert materials == [True, False, "unknown"]


def test_h_required_claim_deterministically_becomes_material():
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    # Even when Call A misclassifies as nonmaterial, the override wins.
    inv = _build(DRAFT, [_row("System A improves latency.", DRAFT,
                              material=False, satisfies_req_ids=["REQ-1"])], reqs)
    claim = inv["claims"][0]
    assert claim["material"] is True
    assert claim["materiality_override"] == "required_by_brief"
    assert inv["satisfied_req_ids"] == ["REQ-1"]
    # Non-mandatory requirement does not force materiality.
    reqs_opt = [{**reqs[0], "mandatory": False}]
    inv2 = _build(DRAFT, [_row("System A improves latency.", DRAFT,
                               material=False, satisfies_req_ids=["REQ-1"])], reqs_opt)
    assert inv2["claims"][0]["material"] is False


def test_i_unknown_materiality_remains_unknown():
    inv = _build(DRAFT, [_row("System A improves latency.", DRAFT, material="unknown")])
    claim = inv["claims"][0]
    assert claim["material"] == "unknown"
    assert inv["counts"]["material_unknown"] == 1
    assert claim["call_b_eligible"] is True  # unknown factual claims are still verified


def test_j_material_factual_requires_citation_even_if_model_says_no():
    inv = _build(DRAFT, [_row("System A improves latency.", DRAFT,
                              material=True, requires_citation=False)])
    assert inv["claims"][0]["requires_citation"] is True
    # Nonmaterial factual: model classification preserved (not policy authority).
    inv2 = _build(DRAFT, [_row("System A improves latency.", DRAFT,
                               material=False, requires_citation=True)])
    assert inv2["claims"][0]["requires_citation"] is True
    inv3 = _build(DRAFT, [_row("System A improves latency.", DRAFT,
                               material=False, requires_citation=False)])
    assert inv3["claims"][0]["requires_citation"] is False


def test_k_revision_changes_draft_sha_and_invalidates_stale_inventory(base_state, monkeypatch):
    rows = [_row("Gradient descent minimizes a loss function.", "Gradient descent minimizes a loss function.")]
    client = verify_llm_client(json.dumps(rows))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    r1 = nodes.verify_node({**base_state, "iterations": 1, "run_id": "rev-1"})
    assert r1["verification_status"] == "completed"
    sha1 = r1["claim_inventory"]["draft_sha256"]

    rows2 = [_row("Gradient descent updates parameters.", "Gradient descent updates parameters.")]
    base_state2 = {**base_state, "draft_markdown": "Gradient descent updates parameters."}
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(json.dumps(rows2)))
    r2 = nodes.verify_node({**base_state2, "iterations": 2, "run_id": "rev-1"})
    sha2 = r2["claim_inventory"]["draft_sha256"]
    assert sha1 != sha2 == sha256_utf8("Gradient descent updates parameters.")

    # A stale inventory (draft N) cannot certify the revised draft (N+1).
    stale = {**base_state2, **r2, "claim_inventory": r1["claim_inventory"],
             "verification_status": "completed", "reflection_score": 8,
             "grounding_report": [
                 {"claim": "c", "status": "verified", "blockers": []} for _ in range(10)
             ]}
    assert nodes.semantic_verification_accepted(stale) is False


def test_l_new_claim_after_revision_enters_inventory(base_state, monkeypatch):
    rows = [
        _row("Gradient descent minimizes a loss function.", "Gradient descent minimizes a loss function."),
        _row("Momentum smooths updates.", "Momentum smooths updates."),
    ]
    draft = "Gradient descent minimizes a loss function.\n\nMomentum smooths updates."
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(json.dumps(rows)))
    result = nodes.verify_node({**base_state, "draft_markdown": draft, "iterations": 2, "run_id": "rev-2"})
    assert result["verification_status"] == "completed"
    texts = {c["claim_text"] for c in result["claim_inventory"]["claims"]}
    assert "Momentum smooths updates." in texts
    assert len(result["grounding_report"]) == 2


def test_m_deleted_required_claim_does_not_remove_requirement_identity(base_state, monkeypatch):
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "mechanism must be stated"}]
    rows = [_row("Gradient descent minimizes a loss function.", "Gradient descent minimizes a loss function.")]
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(json.dumps(rows)))
    result = nodes.verify_node({
        **base_state, "iterations": 2, "run_id": "rev-3", "brief_requirements": reqs,
    })
    inv = result["claim_inventory"]
    # Claim carries no req link after revision; the requirement identity persists.
    assert inv["claims"][0]["satisfies_req_ids"] == []
    assert inv["satisfied_req_ids"] == []
    assert [r["req_id"] for r in inv["brief_requirements"]] == ["REQ-1"]


def test_n_call_b_receives_claim_text_not_anchor(base_state, monkeypatch):
    atomic = "System A reduces storage cost."
    rows = [_row(atomic, DRAFT)]
    state = {**base_state, "draft_markdown": DRAFT, "iterations": 1, "run_id": "cb-input"}
    client = verify_llm_client(json.dumps(rows))
    captured = []
    orig = client.chat.completions.create

    def create(**kwargs):
        captured.append(kwargs)
        return orig(**kwargs)

    client.chat.completions.create = create
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node(state)
    assert result["verification_status"] == "completed"
    analyzer_user = json.loads(captured[1]["messages"][1]["content"])
    claim = analyzer_user["claims"][0]
    assert claim["claim_text"] == atomic
    assert claim["claim_text"] != DRAFT  # never the synthetic anchor proposition
    assert claim["claim_span"] == [0, len(DRAFT)]  # first bound occurrence


def test_o_full_support_evidence_set_preserved(base_state, monkeypatch):
    claim = "Gradient descent minimizes a loss function."
    rows = [_row(claim, claim)]
    analyzer = json.dumps({
        "observations": [
            {
                "claim_id": compute_claim_id(claim),
                "support_quotes": [
                    {"evidence_id": "WEB-001", "quote": "gradient descent"},
                    {"evidence_id": "KB-001", "quote": "kb chunk"},
                ],
                "full_entailment": True,
                "blockers": [],
            }
        ]
    })
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(json.dumps(rows), analyzer))
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "support-set"})
    row = result["grounding_report"][0]
    assert row["status"] == "verified"
    # FULL support set retained — not collapsed to the first/primary source.
    assert [s["evidence_id"] for s in row["support_spans"]] == ["WEB-001", "KB-001"]
    # Primary-source compatibility field remains for UI/backward compatibility.
    assert row["source_ref"] == "https://example.com/gd"


# ---------------------------------------------------------------------------
# Call-A contract boundary
# ---------------------------------------------------------------------------

def test_call_a_fences_tolerated():
    raw = "```json\n" + json.dumps([_row("X is Y.", "X is Y.")]) + "\n```"
    assert parse_claim_inventory_rows(raw)[0]["claim_text"] == "X is Y."


@pytest.mark.parametrize("mutation", [
    pytest.param({"claim_type": "opinion"}, id="bad_type"),
    pytest.param({"material": "maybe"}, id="bad_material"),
    pytest.param({"claim_text": ""}, id="empty_claim_text"),
    pytest.param({"anchor_quote": ""}, id="empty_anchor"),
    pytest.param({"section": "appendix"}, id="bad_section"),
    pytest.param({"claim_text": None}, id="null_claim_text"),
])
def test_call_a_schema_violations_fail_closed(mutation):
    row = _row("X is Y.", "X is Y.")
    for key, value in mutation.items():
        row[key] = value
    with pytest.raises(ClaimInventoryError):
        parse_claim_inventory_rows(json.dumps([row]))


def test_call_a_zero_factual_inventory_fails_closed(base_state, monkeypatch):
    rows = [_row("We think this is neat.", "We think this is neat.", claim_type="editorial", material=False)]
    draft = "We think this is neat."
    client = verify_llm_client(json.dumps(rows))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "draft_markdown": draft, "iterations": 1, "run_id": "no-factual"})
    assert result["verification_status"] == "inventory_failed"
    assert client.calls == 1  # Call B never ran
    assert result["claim_inventory"]["counts"]["call_b_eligible"] == 0
    assert nodes.semantic_verification_accepted({**base_state, **result}) is False


def test_anchor_failure_on_factual_claim_skips_call_b(base_state, monkeypatch):
    rows = [_row("Gradient descent minimizes a loss function.", "text not in the draft")]
    client = verify_llm_client(json.dumps(rows))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "iterations": 1, "run_id": "anchor-fail"})
    assert result["verification_status"] == "inventory_failed"
    assert client.calls == 1
    assert result["claim_inventory"]["claims"][0]["anchor_validity"] == ANCHOR_FAILED
    assert nodes.semantic_verification_accepted({**base_state, **result}) is False


def test_ambiguous_anchor_on_factual_claim_skips_call_b(base_state, monkeypatch):
    draft = DRAFT + "\n\n" + DRAFT
    rows = [_row("System A improves latency.", DRAFT)]
    client = verify_llm_client(json.dumps(rows))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "draft_markdown": draft, "iterations": 1, "run_id": "anchor-amb"})
    assert result["verification_status"] == "inventory_failed"
    assert client.calls == 1
    assert nodes.semantic_verification_accepted({**base_state, **result}) is False


def test_non_factual_anchor_failure_does_not_fail_closed(base_state, monkeypatch):
    factual = "Gradient descent minimizes a loss function."
    draft = factual + "\n\nWe like it."
    rows = [
        _row(factual, factual),
        _row("We like it.", "absent editorial anchor", claim_type="editorial", material=False, section="takeaways"),
    ]
    client = verify_llm_client(json.dumps(rows))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "draft_markdown": draft, "iterations": 1, "run_id": "nonfactual-anchor"})
    assert result["verification_status"] == "completed"
    inv = result["claim_inventory"]
    editorial = [c for c in inv["claims"] if c["claim_type"] == "editorial"][0]
    assert editorial["anchor_validity"] == ANCHOR_FAILED  # recorded, non-critical
    assert inventory_critical_failures(inv) == []


def test_present_inventory_with_critical_anchor_failure_blocks_acceptance(base_state):
    """Even if a state somehow reaches 'completed', an anchor-critical inventory
    blocks acceptance (defense in depth)."""
    draft = "Gradient descent minimizes a loss function."
    rows = [_row("Gradient descent minimizes a loss function.", "not in draft")]
    inv = _build(draft, rows)
    state = {
        **base_state,
        "iterations": 1,
        "draft_markdown": draft,
        "verification_status": "completed",
        "claim_inventory": inv,
        "grounding_report": [{"claim": "c", "status": "verified", "blockers": []} for _ in range(10)],
        "grounding_score": 0.9,
        "reflection_score": 8,
    }
    assert nodes.semantic_verification_accepted(state) is False
    assert nodes.route_after_reflect(state) == "draft"
    monkey_state = {**state, "iterations": 2}
    assert nodes.route_after_reflect(monkey_state) == "hitl"


def test_materiality_unknown_preserved_through_grounding_rows(base_state, monkeypatch):
    claim = "Gradient descent minimizes a loss function."
    rows = [_row(claim, claim, material="unknown")]
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(json.dumps(rows)))
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "unk"})
    assert result["grounding_report"][0]["material"] == "unknown"
    assert result["grounding_report"][0]["requires_citation"] is True


def test_definition_with_checkable_content_cannot_evade_call_b(base_state, monkeypatch):
    claim = "Gradient descent minimizes a loss function."
    rows = [_row(claim, claim, claim_type="definition", material=True)]
    client = verify_llm_client(json.dumps(rows))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "defn"})
    assert result["verification_status"] == "completed"
    assert client.calls == 2  # Call B RAN for the material definition
    assert result["grounding_report"][0]["claim_type"] == "definition"
    # A deterministically nonmaterial definition stays out of Call B.
    rows_nm = [_row(claim, claim, claim_type="definition", material=False)]
    client2 = verify_llm_client(json.dumps(rows_nm))
    monkeypatch.setattr(nodes, "_get_client", lambda: client2)
    result2 = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "defn-nm"})
    assert result2["verification_status"] == "inventory_failed"  # no eligible claims
    assert client2.calls == 1


def test_inventory_recorded_in_trace(base_state, monkeypatch):
    claim = "Gradient descent minimizes a loss function."
    rows = [_row(claim, claim)]
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(json.dumps(rows)))
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "trace-inv"})
    slot = result["semantic_trace"]["iterations"][0]
    assert slot["claim_inventory"]["draft_sha256"] == sha256_utf8(claim)
    assert slot["claim_inventory"]["claims"][0]["claim_id"] == compute_claim_id(claim)


def test_mocked_e2e_revision_rebuilds_inventory_per_draft_version(base_state, monkeypatch):
    """Graph-level mocked E2E: revision -> new draft_sha256 -> Call A rerun ->
    inventory rebuilt from scratch (new claim enters); no stale reuse."""
    import agent.graph as graph_mod

    claim_v1 = "Gradient descent minimizes a loss function."
    claim_new = "Momentum smooths parameter updates."
    draft_v1 = claim_v1
    draft_v2 = f"{claim_v1}\n\n{claim_new}"

    rows_v1 = json.dumps([_row(claim_v1, claim_v1)])
    rows_v2 = json.dumps([_row(claim_v1, claim_v1), _row(claim_new, claim_new)])
    analyzer_v1 = json.dumps({"observations": [{
        "claim_id": compute_claim_id(claim_v1),
        "support_quotes": [], "full_entailment": False, "blockers": [],
    }]})
    analyzer_v2 = json.dumps({"observations": [
        {"claim_id": compute_claim_id(claim_v1),
         "support_quotes": [], "full_entailment": False, "blockers": []},
        {"claim_id": compute_claim_id(claim_new),
         "support_quotes": [], "full_entailment": False, "blockers": []},
    ]})
    client = verify_llm_client(rows_v1, analyzer_v1)
    client._responses.extend([fake_response(rows_v2), fake_response(analyzer_v2)])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)

    drafts = [draft_v1, draft_v2]

    def fake_draft(state):
        text = drafts[min(state.get("iterations", 0), 1)]
        return {
            "draft_sections": {"problem_framing": text, "technical_dive": "",
                               "code_snippets": "", "takeaways": ""},
            "draft_markdown": text,
            "iterations": state.get("iterations", 0) + 1,
        }

    monkeypatch.setattr(graph_mod, "retrieve_node", lambda state: {})
    monkeypatch.setattr(graph_mod, "draft_node", fake_draft)
    monkeypatch.setattr(graph_mod, "reflect_node", lambda state: {
        "reflection_score": 8, "reflection_notes": "ok",
    })
    monkeypatch.setattr(graph_mod, "hitl_node", lambda state: {
        "hitl_status": "rejected", "hitl_feedback": None,
    })

    graph = graph_mod.build_graph()
    init = dict(base_state)
    init.update(iterations=0, grounding_report=[], verification_status="not_started",
                claim_inventory=None)
    result = graph.invoke(init)

    assert client.calls == 4  # (Call A + Call B) x 2 passes
    assert result["iterations"] == 2
    final_inv = result["claim_inventory"]
    assert final_inv["draft_sha256"] == sha256_utf8(draft_v2)
    texts = {c["claim_text"] for c in final_inv["claims"]}
    assert texts == {claim_v1, claim_new}  # new claim entered after revision
    metrics = result["iteration_metrics"]
    assert len(metrics) == 2
    assert metrics[0]["N"] == 1 and metrics[1]["N"] == 2
    # Both semantic passes failed (all unverified) -> held at HITL, never published.
    assert result["hitl_status"] == "rejected"
    assert result.get("html_output") is None


def test_hitl_api_payload_includes_inventory_summary(base_state, monkeypatch):
    claim = "Gradient descent minimizes a loss function."
    rows = [_row(claim, claim)]
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(json.dumps(rows)))
    verified = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "hitl-inv"})
    state = {
        **base_state, **verified, "iterations": 2, "hitl_status": "pending",
        "reflection_score": 8,
    }
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("HITL_MODE", "api")
    captured = {}
    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: captured.setdefault("p", payload) or {"action": "reject"})
    nodes.hitl_node(state)
    summary = captured["p"]["claim_inventory_summary"]
    assert summary["draft_sha256"] == sha256_utf8(claim)
    assert summary["counts"]["total"] == 1
    assert summary["critical_failures"] == 0
    # blocker-bearing unverified claim still cannot be approved through the gate
    assert nodes.route_after_hitl({**state, "hitl_status": "rejected"}) == END
