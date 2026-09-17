"""Slice 3 + Phase 4 Slice 2A: production verify_node tests (mocked transport only).

Call A is now the claim-inventory/materiality extractor (no sources shown);
Call B remains the sole semantic authority. Claim IDs are Python
content-derived — never positional, never model-generated.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from openai import APITimeoutError

import agent.nodes as nodes
from agent.claim_inventory import compute_claim_id
from agent.semantic_analyzer.status_engine import extract_span_text
from tests.conftest import FakeLLMClient, fake_response, openai_error, verify_llm_client

FIXTURES = Path(__file__).resolve().parents[1] / "evals/fixtures/hybrid_verifier_status_offline.json"


def _load_case(case_id: str) -> dict:
    return json.loads(FIXTURES.read_text())["cases"][f"{case_id}-COMPLETE"]


def _inventory_row(
    claim_text: str,
    *,
    anchor_quote: str | None = None,
    section: str | None = None,
    claim_type: str = "factual",
    material=True,
    specificity: str = "substantive",
    satisfies_req_ids: list[str] | None = None,
) -> dict:
    """Call-A (claim inventory) output row — the post-Slice-2A contract."""
    return {
        "claim_text": claim_text,
        "anchor_quote": anchor_quote if anchor_quote is not None else claim_text,
        "section": section,
        "claim_type": claim_type,
        "material": material,
        "materiality_reason_code": "core_technical_conclusion" if material is True else "incidental_detail",
        "materiality_rationale": "fixture",
        "satisfies_req_ids": satisfies_req_ids or [],
        "specificity": specificity,
        "requires_citation": None,
    }


def _span_obs_to_quote_obs(source_text: str, observation: dict, evidence_id: str, claim_id: str) -> dict:
    support_quotes = [
        {
            "evidence_id": evidence_id,
            "quote": extract_span_text(source_text, span["start"], span["end"]),
        }
        for span in observation["support_spans"]
    ]
    blockers = []
    for blocker in observation["blockers"]:
        blockers.append(
            {
                "kind": blocker["kind"],
                "explanation": blocker["explanation"],
                "evidence_quotes": [
                    {
                        "evidence_id": evidence_id,
                        "quote": extract_span_text(source_text, span["start"], span["end"]),
                    }
                    for span in blocker["evidence_spans"]
                ],
            }
        )
    return {
        "claim_id": claim_id,
        "support_quotes": support_quotes,
        "full_entailment": observation["full_entailment"],
        "blockers": blockers,
    }


def _fixture_state(case: dict) -> dict:
    return {
        "topic": "fixture",
        "slug": "fixture-slug",
        "card_id": "FIX",
        "series_context": "",
        "draft_sections": {},
        "draft_markdown": case["draft_text"],
        "web_sources": [
            {
                "title": case.get("asset_id", "fixture"),
                "url": "https://fixture.test/source",
                "content": case["source_text"],
                "score": 0.9,
            }
        ],
        "kb_results": [],
        "grounding_report": [],
        "grounding_score": 0.0,
        "claim_inventory": None,
        "brief_requirements": [],
        "reflection_score": 0,
        "reflection_notes": "",
        "reflection_provenance": {},
        "iterations": 1,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "run_id": f"verify-cutover-{case.get('asset_id', 'x')}",
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "latency_ms": {},
        "error_log": [],
        "iteration_metrics": [],
        "m4_feedback_claims": 0,
    }


def _run_fixture_case(
    monkeypatch,
    case_id: str,
    *,
    material=True,
    claim_type: str = "factual",
) -> dict:
    case = _load_case(case_id)
    state = _fixture_state(case)
    inventory = json.dumps([
        _inventory_row(case["draft_text"], material=material, claim_type=claim_type)
    ])
    quote_obs = _span_obs_to_quote_obs(
        case["source_text"],
        case["observation"],
        "WEB-001",
        compute_claim_id(case["draft_text"]),
    )
    analyzer = json.dumps({"observations": [quote_obs]})
    monkeypatch.setattr(
        nodes,
        "_get_client",
        lambda: verify_llm_client(inventory, analyzer),
    )
    return nodes.verify_node(state)


def test_p6_through_verify_node(monkeypatch):
    result = _run_fixture_case(monkeypatch, "P6")
    assert result["verification_status"] == "completed"
    row = result["grounding_report"][0]
    assert row["status"] == "weak"
    assert row["support_spans"][0]["start"] == 400
    assert row["support_spans"][0]["end"] == 458
    assert row["blockers"][0]["evidence_spans"][0]["start"] == 1500
    assert "confidence" not in row
    assert result["grounding_report"][0]["blockers"][0]["kind"] == "contradiction"
    assert nodes.semantic_verification_accepted({**_fixture_state(_load_case("P6")), **result}) is False


def test_p7_through_verify_node(monkeypatch):
    result = _run_fixture_case(monkeypatch, "P7")
    row = result["grounding_report"][0]
    assert row["status"] == "weak"
    assert row["support_spans"][0]["start"] == 0
    assert row["support_spans"][0]["end"] == 60
    assert row["blockers"][0]["evidence_spans"][0]["start"] == 2000
    assert row["blockers"][0]["kind"] == "limitation"
    assert nodes.semantic_verification_accepted({**_fixture_state(_load_case("P7")), **result}) is False


def test_p1_through_verify_node(monkeypatch):
    result = _run_fixture_case(monkeypatch, "P1")
    row = result["grounding_report"][0]
    assert row["status"] == "verified"
    assert row["full_entailment"] is True
    assert row["blockers"] == []
    assert row["support_spans"][0]["start"] == 1500
    assert row["support_spans"][0]["end"] == 1563
    assert nodes.semantic_verification_accepted({**_fixture_state(_load_case("P1")), **result}) is True


def test_materiality_does_not_override_engine(monkeypatch):
    """Call A marks P6 nonmaterial; the engine must still produce weak and the
    contradiction blocker must still block acceptance."""
    case = _load_case("P6")
    result = _run_fixture_case(monkeypatch, "P6", material=False)
    assert result["grounding_report"][0]["status"] == "weak"
    assert result["grounding_report"][0]["material"] is False
    assert result["grounding_report"][0]["blockers"][0]["kind"] == "contradiction"
    assert nodes.semantic_verification_accepted({**_fixture_state(case), **result}) is False
    slot = result["semantic_trace"]["iterations"][0]
    claim_id = compute_claim_id(case["draft_text"])
    assert slot["semantic_analyzer"]["engine_status_by_claim"][claim_id] == "weak"


def test_p1_unknown_materiality_still_engine_verified(monkeypatch):
    """Call A reports material=unknown; engine still verifies P1, and UNKNOWN
    materiality is preserved (never converted to false)."""
    case = _load_case("P1")
    result = _run_fixture_case(monkeypatch, "P1", material="unknown")
    assert result["grounding_report"][0]["status"] == "verified"
    assert result["grounding_report"][0]["material"] == "unknown"
    slot = result["semantic_trace"]["iterations"][0]
    claim_id = compute_claim_id(case["draft_text"])
    assert slot["semantic_analyzer"]["engine_status_by_claim"][claim_id] == "verified"


def test_full_evidence_visible_after_legacy_clip_boundaries(monkeypatch, base_state):
    """Call B sees FULL evidence text — the old 1500/2000-char Call-A clip is
    gone entirely (Call A now receives no sources at all)."""
    web_quote = "WEB-QUOTE-AFTER-1500-CHAR-BOUNDARY"
    kb_quote = "KB-QUOTE-AFTER-2000-CHAR-BOUNDARY"
    web_content = ("w" * 1600) + web_quote
    kb_text = ("k" * 2100) + kb_quote
    claim_text = "Claim needing late web quote."
    inventory = json.dumps([_inventory_row(claim_text)])
    analyzer = json.dumps(
        {
            "observations": [
                {
                    "claim_id": compute_claim_id(claim_text),
                    "support_quotes": [{"evidence_id": "WEB-001", "quote": web_quote}],
                    "full_entailment": True,
                    "blockers": [],
                }
            ]
        }
    )
    state = dict(base_state)
    state.update(
        {
            "draft_markdown": claim_text,
            "web_sources": [{"title": "late", "url": "https://clip.test", "content": web_content, "score": 0.9}],
            "kb_results": [{"text": kb_text, "source": "late.md", "chunk_index": 0}],
            "iterations": 1,
            "run_id": "clip-regression",
        }
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(inventory, analyzer))
    result = nodes.verify_node(state)
    assert result["verification_status"] == "completed"
    assert result["grounding_report"][0]["support_spans"][0]["text"] == web_quote
    manifest_len = len(result["semantic_trace"]["iterations"][0]["semantic_analyzer"]["evidence_ids"])
    assert manifest_len == 2


def test_empty_inventory_fails_closed_without_analyzer(monkeypatch, base_state):
    """Zero extracted claims => no usable factual inventory => inventory_failed;
    Call B must not run."""
    client = FakeLLMClient(response=fake_response("[]"))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "iterations": 1, "run_id": "empty-roster"})
    assert result["verification_status"] == "inventory_failed"
    assert result["grounding_report"] == []
    assert result["claim_inventory"]["claims"] == []
    assert client.calls == 1
    assert nodes.semantic_verification_accepted({**base_state, **result}) is False


def test_legacy_parse_failure_skips_analyzer(monkeypatch, base_state):
    client = FakeLLMClient(response=fake_response("not-json"))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "iterations": 1, "run_id": "legacy-parse"})
    assert result["verification_status"] == "parse_failed"
    assert client.calls == 1


def test_legacy_transport_failure_is_verification_error(monkeypatch, base_state):
    client = FakeLLMClient(errors=[openai_error(APITimeoutError, 0)] * 3)
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "iterations": 1, "run_id": "legacy-timeout"})
    assert result["verification_status"] == "verification_error"
    assert result["grounding_report"] == []
    assert nodes.semantic_verification_accepted({**base_state, **result}) is False


@pytest.mark.parametrize(
    "analyzer_payload,expected_status",
    [
        ("{not-json", "parse_failed"),
        (
            json.dumps(
                {
                    "observations": [
                        {
                            "claim_id": compute_claim_id("Gradient descent minimizes loss."),
                            "support_quotes": [{"evidence_id": "WEB-001", "quote": "missing"}],
                            "full_entailment": False,
                            "blockers": [],
                        }
                    ]
                }
            ),
            "parse_failed",
        ),
        (json.dumps({"observations": []}), "parse_failed"),
    ],
)
def test_analyzer_failures_fail_closed(monkeypatch, base_state, analyzer_payload, expected_status):
    claim = "Gradient descent minimizes loss."
    inventory = json.dumps([_inventory_row(claim)])
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(inventory, analyzer_payload))
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "fail-closed"})
    assert result["verification_status"] == expected_status
    assert not any(r.get("status") == "verified" for r in result["grounding_report"])
    assert nodes.semantic_verification_accepted({**base_state, **result}) is False


def test_analyzer_provider_exception_is_verification_error(monkeypatch, base_state):
    claim = "Gradient descent minimizes loss."
    inventory = json.dumps([_inventory_row(claim)])

    class BoomClient:
        calls = 0

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    BoomClient.calls += 1
                    if BoomClient.calls == 1:
                        return fake_response(inventory)
                    raise RuntimeError("transport down")

    monkeypatch.setattr(nodes, "_get_client", lambda: BoomClient())
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "provider-boom"})
    assert result["verification_status"] == "verification_error"
    assert nodes.semantic_verification_accepted({**base_state, **result}) is False


def test_uvr_uses_engine_statuses_only(monkeypatch, base_state):
    claim = "Unsupported claim about quantum gradients."
    inventory = json.dumps([_inventory_row(claim)])
    analyzer = json.dumps(
        {
            "observations": [
                {
                    "claim_id": compute_claim_id(claim),
                    "support_quotes": [],
                    "full_entailment": False,
                    "blockers": [],
                }
            ]
        }
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(inventory, analyzer))
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "uvr-engine"})
    assert result["verification_status"] == "completed"
    assert result["grounding_report"][0]["status"] == "unverified"
    assert nodes.unverified_rate(result["grounding_report"]) == 1.0
    assert nodes.semantic_verification_accepted({**base_state, **result}) is False


def test_grounding_score_from_engine_statuses(monkeypatch, base_state):
    claim = "Gradient descent minimizes loss."
    inventory = json.dumps([_inventory_row(claim)])
    analyzer = json.dumps(
        {
            "observations": [
                {
                    "claim_id": compute_claim_id(claim),
                    "support_quotes": [],
                    "full_entailment": False,
                    "blockers": [],
                }
            ]
        }
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(inventory, analyzer))
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "gs-engine"})
    assert result["grounding_score"] == 0.0
    assert "confidence" not in (result["grounding_report"][0] if result["grounding_report"] else {})


def test_specificity_preserved_from_inventory(monkeypatch, base_state):
    claim = "Gradient descent minimizes loss."
    inventory = json.dumps([_inventory_row(claim, specificity="substantive")])
    analyzer = json.dumps(
        {
            "observations": [
                {
                    "claim_id": compute_claim_id(claim),
                    "support_quotes": [],
                    "full_entailment": False,
                    "blockers": [],
                }
            ]
        }
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: verify_llm_client(inventory, analyzer))
    result = nodes.verify_node({**base_state, "draft_markdown": claim, "iterations": 1, "run_id": "spec-bridge"})
    assert result["grounding_report"][0]["specificity"] == "substantive"
