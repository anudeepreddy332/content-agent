"""Phase 4 final closure: live brief_requirements wiring through production entry path.

Proves caller-supplied requirements reach graph state, Call-A context, claim
inventory, material policy, and publication-safety gates — not hand-built policy
states alone. Deterministic, $0, no providers.
"""
from __future__ import annotations

import json
import uuid

import pytest
from langgraph.graph import END

import agent.graph as graph_mod
import agent.nodes as nodes
from agent.claim_inventory import compute_claim_id
from agent.material_policy import REQ_MISSING, REQ_SATISFIED, REQ_UNKNOWN
from main import _build_initial_state
from tests.conftest import fake_response, verify_llm_client

QUORUM_REQ = {
    "req_id": "REQ-QUORUM",
    "kind": "required_content",
    "mandatory": True,
    "description": "Explain why quorum requires seven replica acknowledgements.",
}


def _retrieve_with_content(*fragments: str):
    """Mock retrieve_node: seed WEB-001 with text containing all quoted claims."""
    content = "\n".join(fragments)

    def _fn(state):
        return {
            "web_sources": [{
                "title": "Source",
                "url": "https://example.com/source",
                "content": content,
                "score": 0.9,
            }],
        }

    return _fn


def _claim_row(
    claim_text,
    anchor_quote=None,
    *,
    material=True,
    satisfies_req_ids=None,
):
    return {
        "claim_text": claim_text,
        "anchor_quote": anchor_quote or claim_text,
        "section": "technical_dive",
        "claim_type": "factual",
        "material": material,
        "materiality_reason_code": None,
        "materiality_rationale": None,
        "satisfies_req_ids": satisfies_req_ids or [],
        "specificity": "substantive",
        "requires_citation": None,
    }


def _verified_analyzer(*claim_texts: str) -> str:
    return json.dumps({
        "observations": [
            {
                "claim_id": compute_claim_id(text),
                "support_quotes": [{"evidence_id": "WEB-001", "quote": text}],
                "full_entailment": True,
                "blockers": [],
            }
            for text in claim_texts
        ]
    })


def test_build_initial_state_wires_brief_requirements():
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    state = _build_initial_state(
        "Topic", "topic", "card-1", "Series", str(uuid.uuid4()),
        brief_requirements=reqs,
    )
    assert state["brief_requirements"] == reqs


def test_build_initial_state_omitted_requirements_default_empty():
    state = _build_initial_state(
        "Topic", "topic", "card-1", "Series", str(uuid.uuid4()),
    )
    assert state["brief_requirements"] == []


def test_build_initial_state_drops_malformed_requirements():
    state = _build_initial_state(
        "Topic", "topic", "card-1", "Series", str(uuid.uuid4()),
        brief_requirements=[
            {"req_id": "REQ-OK", "kind": "required_content", "mandatory": True,
             "description": "ok"},
            {"bad": "row"},
        ],
    )
    assert len(state["brief_requirements"]) == 1
    assert state["brief_requirements"][0]["req_id"] == "REQ-OK"


def test_production_path_requirements_reach_inventory_and_call_a_context(
    monkeypatch: pytest.MonkeyPatch,
):
    """Live path: _build_initial_state → graph → verify → inventory + Call-A message."""
    claim = "System A improves latency."
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    init = _build_initial_state(
        "Latency", "latency", "W-1", "Test", "live-req-inv",
        brief_requirements=reqs,
    )
    init.update(iterations=0, draft_markdown="", grounding_report=[],
                verification_status="not_started", claim_inventory=None)

    captured_messages: list[str] = []

    client = verify_llm_client(json.dumps([_claim_row(claim, satisfies_req_ids=["REQ-1"])]),
                               _verified_analyzer(claim))
    orig_create = client.chat.completions.create

    def capture_create(**kwargs):
        captured_messages.append(kwargs["messages"][-1]["content"])
        return orig_create(**kwargs)

    client.chat.completions.create = capture_create
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    monkeypatch.setattr(graph_mod, "retrieve_node", _retrieve_with_content(claim))
    monkeypatch.setattr(graph_mod, "draft_node", lambda state: {
        "draft_markdown": claim,
        "iterations": state.get("iterations", 0) + 1,
    })
    monkeypatch.setattr(graph_mod, "reflect_node", lambda state: {
        "reflection_score": 8, "reflection_notes": "ok",
    })
    monkeypatch.setattr(graph_mod, "hitl_node", lambda state: {
        "hitl_status": "rejected", "hitl_feedback": None,
    })

    result = graph_mod.build_graph().invoke(init)

    assert result["brief_requirements"] == reqs
    assert any("Brief requirements (JSON):" in msg for msg in captured_messages)
    assert "REQ-1" in captured_messages[0]
    inv = result["claim_inventory"]
    assert [r["req_id"] for r in inv["brief_requirements"]] == ["REQ-1"]
    mp = nodes.material_policy_result(result)
    assert mp.mandatory_requirement_states["REQ-1"] == REQ_UNKNOWN


def test_production_path_denominator_gaming_deletion_blocks_publication(
    monkeypatch: pytest.MonkeyPatch,
):
    """Mandatory R persists; deleting linked content makes R MISSING; no publish pass."""
    reqs = [{"req_id": "REQ-1", "kind": "required_content", "mandatory": True,
             "description": "must state the mechanism"}]
    claim_v1 = "System A improves latency."
    claim_v2 = "Unrelated replacement content."
    draft_v2 = claim_v2
    rows_v1 = json.dumps([_claim_row(claim_v1, satisfies_req_ids=["REQ-1"])])
    rows_v2 = json.dumps([_claim_row(claim_v2, material=False)])
    # v1: linked but unverified -> REQ_UNRESOLVED -> revision (not HITL hold).
    analyzer_v1 = json.dumps({
        "observations": [{
            "claim_id": compute_claim_id(claim_v1),
            "support_quotes": [],
            "full_entailment": False,
            "blockers": [],
        }]
    })
    analyzer_v2 = _verified_analyzer(claim_v2)

    client = verify_llm_client(rows_v1, analyzer_v1)
    client._responses.extend([fake_response(rows_v2), fake_response(analyzer_v2)])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)

    drafts = [claim_v1, draft_v2]

    def fake_draft(state):
        text = drafts[min(state.get("iterations", 0), 1)]
        return {"draft_markdown": text, "iterations": state.get("iterations", 0) + 1}

    init = _build_initial_state(
        "Latency", "latency", "W-2", "Test", "live-req-del",
        brief_requirements=reqs,
    )
    init.update(iterations=0, grounding_report=[], verification_status="not_started",
                claim_inventory=None)

    monkeypatch.setattr(
        graph_mod, "retrieve_node",
        _retrieve_with_content(claim_v1, draft_v2),
    )
    monkeypatch.setattr(graph_mod, "draft_node", fake_draft)
    monkeypatch.setattr(graph_mod, "reflect_node", lambda state: {
        "reflection_score": 8, "reflection_notes": "ok",
    })
    monkeypatch.setattr(graph_mod, "hitl_node", lambda state: {
        "hitl_status": "rejected", "hitl_feedback": None,
    })

    result = graph_mod.build_graph().invoke(init)

    assert result["iterations"] == 2
    assert result["brief_requirements"] == reqs
    assert [r["req_id"] for r in result["claim_inventory"]["brief_requirements"]] == ["REQ-1"]
    mp = nodes.material_policy_result(result)
    assert mp.mandatory_requirement_states["REQ-1"] == REQ_MISSING
    assert mp.missing_requirement_ids == ["REQ-1"]
    assert nodes.material_policy_accepted(result) is False
    assert nodes.publication_safety_pass(result) is False


def test_production_path_advisory_link_does_not_satisfy_semantic_requirement(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sonnet attack through wired path: false satisfies_req_ids → UNKNOWN/HITL, no bypass."""
    claim = "The cache TTL is 30 seconds."
    init = _build_initial_state(
        "Quorum", "quorum", "W-3", "Test", "live-req-adv",
        brief_requirements=[QUORUM_REQ],
    )
    init.update(iterations=0, draft_markdown="", grounding_report=[],
                verification_status="not_started", claim_inventory=None)

    client = verify_llm_client(
        json.dumps([_claim_row(claim, satisfies_req_ids=["REQ-QUORUM"])]),
        _verified_analyzer(claim),
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    monkeypatch.setattr(graph_mod, "retrieve_node", _retrieve_with_content(claim))
    monkeypatch.setattr(graph_mod, "draft_node", lambda state: {
        "draft_markdown": claim,
        "iterations": state.get("iterations", 0) + 1,
    })
    monkeypatch.setattr(graph_mod, "reflect_node", lambda state: {
        "reflection_score": 8, "reflection_notes": "ok",
    })
    monkeypatch.setattr(graph_mod, "hitl_node", lambda state: {
        "hitl_status": "rejected", "hitl_feedback": None,
    })

    result = graph_mod.build_graph().invoke(init)

    assert result["brief_requirements"] == [QUORUM_REQ]
    mp = nodes.material_policy_result(result)
    assert mp.mandatory_requirement_states["REQ-QUORUM"] == REQ_UNKNOWN
    assert nodes.material_policy_accepted(result) is False
    assert nodes.route_after_reflect(result) == "hitl"

    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    hitl = nodes.hitl_node(result)
    assert hitl["hitl_status"] == "rejected"
    assert nodes.route_after_hitl({**result, **hitl}) == END


def test_production_path_structural_requirement_satisfied_when_present(
    monkeypatch: pytest.MonkeyPatch,
):
    """Deterministic section_presence through wired input path can SATISFIED."""
    section = "## Code"
    claim = f"{section}\n\nprint('hi')"
    req = {"req_id": "REQ-SEC", "kind": "section_presence", "mandatory": True,
           "description": section}
    init = _build_initial_state(
        "Code article", "code-article", "W-4", "Test", "live-req-struct",
        brief_requirements=[req],
    )
    init.update(iterations=0, draft_markdown="", grounding_report=[],
                verification_status="not_started", claim_inventory=None)

    client = verify_llm_client(
        json.dumps([_claim_row("print('hi')", anchor_quote="print('hi')")]),
        _verified_analyzer("print('hi')"),
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    monkeypatch.setattr(graph_mod, "retrieve_node", _retrieve_with_content("print('hi')"))
    monkeypatch.setattr(graph_mod, "draft_node", lambda state: {
        "draft_markdown": claim,
        "iterations": state.get("iterations", 0) + 1,
    })
    monkeypatch.setattr(graph_mod, "reflect_node", lambda state: {
        "reflection_score": 8, "reflection_notes": "ok",
    })
    monkeypatch.setattr(graph_mod, "hitl_node", lambda state: {
        "hitl_status": "rejected", "hitl_feedback": None,
    })

    result = graph_mod.build_graph().invoke(init)

    assert result["brief_requirements"] == [req]
    mp = nodes.material_policy_result(result)
    assert mp.mandatory_requirement_states["REQ-SEC"] == REQ_SATISFIED
    assert nodes.material_policy_accepted(result) is True
