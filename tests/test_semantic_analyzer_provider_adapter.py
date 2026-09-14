"""Production semantic-analyzer provider adapter tests (mocked transport only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.semantic_analyzer.contract import build_claim_roster, build_evidence_manifest
from agent.semantic_analyzer.provider import analyze_semantic_evidence
from agent.semantic_analyzer.status_engine import extract_span_text

FIXTURES = Path(__file__).resolve().parents[1] / "evals/fixtures/hybrid_verifier_status_offline.json"


def _load_case(case_id: str) -> dict:
    return json.loads(FIXTURES.read_text())["cases"][f"{case_id}-COMPLETE"]


def _span_obs_to_quote_obs(source_text: str, observation: dict, evidence_id: str) -> dict:
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
        "claim_id": observation["claim_id"],
        "support_quotes": support_quotes,
        "full_entailment": observation["full_entailment"],
        "blockers": blockers,
    }


def _mock_llm(content: str):
    def _call(**kwargs):
        return {"content": content, "model": "test-model", "id": "resp-test"}

    return _call


def _run_case(case_id: str) -> dict:
    case = _load_case(case_id)
    evidence_id = "WEB-001"
    claim_id = case["claim_id"]
    quote_obs = _span_obs_to_quote_obs(case["source_text"], case["observation"], evidence_id)
    quote_obs["claim_id"] = claim_id
    raw = json.dumps({"observations": [quote_obs]})
    claims = [
        {
            "claim_id": claim_id,
            "claim_text": case["draft_text"],
            "claim_span": [0, len(case["draft_text"])],
        }
    ]
    result = analyze_semantic_evidence(
        request_id=case["request_id"],
        draft_text=case["draft_text"],
        claims=claims,
        web_sources=[
            {
                "url": f"https://fixture.test/{case_id.lower()}",
                "content": case["source_text"],
                "title": case_id,
            }
        ],
        kb_results=[],
        llm_call=_mock_llm(raw),
        model="test-model",
    )
    return {"result": result, "case": case, "quote_obs": quote_obs}


def test_p6_adapter_proof() -> None:
    row = _run_case("P6")
    result = row["result"]
    assert result.success is True
    assert result.failure_kind is None
    assert result.adjudication is not None
    assert result.adjudication.semantic_status_by_claim["P6.claim.1"] == "weak"
    report = result.grounding_report[0]
    assert report["status"] == "weak"
    assert report["support_spans"][0] == {
        "evidence_id": "WEB-001",
        "start": 400,
        "end": 458,
        "text": "Helix gate permits export when mode flag HG-ENABLE is set.",
    }
    assert report["blockers"][0]["evidence_spans"][0]["start"] == 1500
    assert report["blockers"][0]["evidence_spans"][0]["end"] == 1562
    assert "confidence" not in report


def test_p7_adapter_proof() -> None:
    row = _run_case("P7")
    result = row["result"]
    assert result.success is True
    report = result.grounding_report[0]
    assert report["status"] == "weak"
    assert report["full_entailment"] is False
    assert report["support_spans"][0]["start"] == 0
    assert report["support_spans"][0]["end"] == 60
    assert report["blockers"][0]["evidence_spans"][0]["start"] == 2000
    assert report["blockers"][0]["evidence_spans"][0]["end"] == 2048


def test_p1_adapter_proof() -> None:
    row = _run_case("P1")
    result = row["result"]
    assert result.success is True
    report = result.grounding_report[0]
    assert report["status"] == "verified"
    assert report["full_entailment"] is True
    assert report["blockers"] == []
    assert report["support_spans"][0]["start"] == 1500
    assert report["support_spans"][0]["end"] == 1563


def test_full_evidence_quote_after_legacy_clip_boundaries() -> None:
    web_quote = "WEB-QUOTE-AFTER-1500-CHAR-BOUNDARY"
    kb_quote = "KB-QUOTE-AFTER-2000-CHAR-BOUNDARY"
    web_content = ("w" * 1600) + web_quote
    kb_text = ("k" * 2100) + kb_quote
    claims = build_claim_roster(["Claim needing late web quote."])
    raw = json.dumps(
        {
            "observations": [
                {
                    "claim_id": "claim-001",
                    "support_quotes": [{"evidence_id": "WEB-001", "quote": web_quote}],
                    "full_entailment": True,
                    "blockers": [],
                }
            ]
        }
    )
    result = analyze_semantic_evidence(
        request_id="REQ-CLIP-TEST",
        draft_text="Claim needing late web quote.",
        claims=claims,
        web_sources=[{"url": "https://clip.test", "content": web_content}],
        kb_results=[{"text": kb_text, "source": "late.md", "chunk_index": 0}],
        llm_call=_mock_llm(raw),
        model="test-model",
    )
    assert result.success is True
    assert len(web_content) > 1500
    assert len(kb_text) > 2000
    assert result.evidence_manifest[0]["source_text"] == web_content
    assert result.grounding_report[0]["support_spans"][0]["text"] == web_quote


def test_nonexistent_quote_fails_closed() -> None:
    case = _load_case("P6")
    bad = json.dumps(
        {
            "observations": [
                {
                    "claim_id": case["claim_id"],
                    "support_quotes": [
                        {"evidence_id": "WEB-001", "quote": "Fabricated quote not in source."}
                    ],
                    "full_entailment": False,
                    "blockers": [],
                }
            ]
        }
    )
    result = analyze_semantic_evidence(
        request_id=case["request_id"],
        draft_text=case["draft_text"],
        claims=[{"claim_id": case["claim_id"], "claim_text": case["draft_text"], "claim_span": [0, len(case["draft_text"])]}],
        web_sources=[{"url": "https://p6", "content": case["source_text"]}],
        llm_call=_mock_llm(bad),
        model="test-model",
    )
    assert result.success is False
    assert result.failure_kind == "quote_binding"
    assert not result.any_verified


def test_ambiguous_quote_fails_closed() -> None:
    source = "repeat me repeat me"
    quote = "repeat"
    raw = json.dumps(
        {
            "observations": [
                {
                    "claim_id": "claim-001",
                    "support_quotes": [{"evidence_id": "WEB-001", "quote": quote}],
                    "full_entailment": True,
                    "blockers": [],
                }
            ]
        }
    )
    result = analyze_semantic_evidence(
        request_id="REQ-AMB",
        draft_text="repeat me",
        claims=build_claim_roster(["repeat me"]),
        web_sources=[{"url": "https://a", "content": source}],
        llm_call=_mock_llm(raw),
        model="test-model",
    )
    assert result.success is False
    assert result.failure_kind == "quote_binding"


def test_wrong_evidence_id_fails_closed() -> None:
    case = _load_case("P1")
    quote = extract_span_text(case["source_text"], 1500, 1563)
    raw = json.dumps(
        {
            "observations": [
                {
                    "claim_id": case["claim_id"],
                    "support_quotes": [{"evidence_id": "WEB-999", "quote": quote}],
                    "full_entailment": True,
                    "blockers": [],
                }
            ]
        }
    )
    result = analyze_semantic_evidence(
        request_id=case["request_id"],
        draft_text=case["draft_text"],
        claims=[{"claim_id": case["claim_id"], "claim_text": case["draft_text"], "claim_span": [0, len(case["draft_text"])]}],
        web_sources=[{"url": "https://p1", "content": case["source_text"]}],
        llm_call=_mock_llm(raw),
        model="test-model",
    )
    assert result.success is False
    assert result.failure_kind == "quote_binding"


def test_malformed_json_fails_closed() -> None:
    case = _load_case("P1")
    result = analyze_semantic_evidence(
        request_id=case["request_id"],
        draft_text=case["draft_text"],
        claims=[{"claim_id": case["claim_id"], "claim_text": case["draft_text"], "claim_span": [0, len(case["draft_text"])]}],
        web_sources=[{"url": "https://p1", "content": case["source_text"]}],
        llm_call=_mock_llm("{not-json"),
        model="test-model",
    )
    assert result.success is False
    assert result.failure_kind == "response_contract"


def test_missing_claim_observation_fails_closed() -> None:
    case = _load_case("P6")
    raw = json.dumps({"observations": []})
    result = analyze_semantic_evidence(
        request_id=case["request_id"],
        draft_text=case["draft_text"],
        claims=[{"claim_id": case["claim_id"], "claim_text": case["draft_text"], "claim_span": [0, len(case["draft_text"])]}],
        web_sources=[{"url": "https://p6", "content": case["source_text"]}],
        llm_call=_mock_llm(raw),
        model="test-model",
    )
    assert result.success is False
    assert result.failure_kind == "adjudication_invalid"
    assert not result.any_verified


def test_unsupported_blocker_kind_rejected() -> None:
    case = _load_case("P6")
    quote_obs = _span_obs_to_quote_obs(case["source_text"], case["observation"], "WEB-001")
    quote_obs["claim_id"] = case["claim_id"]
    quote_obs["blockers"][0]["kind"] = "unsupported_kind"
    raw = json.dumps({"observations": [quote_obs]})
    result = analyze_semantic_evidence(
        request_id=case["request_id"],
        draft_text=case["draft_text"],
        claims=[{"claim_id": case["claim_id"], "claim_text": case["draft_text"], "claim_span": [0, len(case["draft_text"])]}],
        web_sources=[{"url": "https://p6", "content": case["source_text"]}],
        llm_call=_mock_llm(raw),
        model="test-model",
    )
    assert result.success is False
    assert result.failure_kind == "response_contract"


def test_provider_exception_surfaces_as_provider_failure() -> None:
    def _boom(**kwargs):
        raise RuntimeError("transport down")

    case = _load_case("P1")
    result = analyze_semantic_evidence(
        request_id=case["request_id"],
        draft_text=case["draft_text"],
        claims=[{"claim_id": case["claim_id"], "claim_text": case["draft_text"], "claim_span": [0, len(case["draft_text"])]}],
        web_sources=[{"url": "https://p1", "content": case["source_text"]}],
        llm_call=_boom,
        model="test-model",
    )
    assert result.success is False
    assert result.failure_kind == "provider"
    assert "transport down" in (result.failure_detail or "")


def test_raw_offset_fields_rejected() -> None:
    case = _load_case("P1")
    raw = json.dumps(
        {
            "observations": [
                {
                    "claim_id": case["claim_id"],
                    "support_spans": [{"evidence_id": "WEB-001", "start": 0, "end": 5}],
                    "full_entailment": True,
                    "blockers": [],
                }
            ]
        }
    )
    result = analyze_semantic_evidence(
        request_id=case["request_id"],
        draft_text=case["draft_text"],
        claims=[{"claim_id": case["claim_id"], "claim_text": case["draft_text"], "claim_span": [0, len(case["draft_text"])]}],
        web_sources=[{"url": "https://p1", "content": case["source_text"]}],
        llm_call=_mock_llm(raw),
        model="test-model",
    )
    assert result.success is False
    assert result.failure_kind == "response_contract"


def test_manifest_builder_never_clips_web_content() -> None:
    long_content = "X" * 5000
    manifest = build_evidence_manifest(
        request_id="REQ",
        web_sources=[{"url": "https://long", "content": long_content}],
    )
    assert manifest[0]["source_text"] == long_content
    assert len(manifest[0]["source_text"]) == 5000
