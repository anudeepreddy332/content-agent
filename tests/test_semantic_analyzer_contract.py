"""Production semantic-analyzer contract builders. No providers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.semantic_analyzer.contract import (
    WEB_SOURCE_LIMIT,
    assert_manifest_source_integrity,
    build_claim_roster,
    build_evidence_manifest,
    legacy_verdict_rows_to_claim_roster,
)
from agent.semantic_analyzer.status_engine import sha256_utf8

FIXTURES = Path(__file__).resolve().parents[1] / "evals/fixtures/hybrid_verifier_status_offline.json"


def _pad(text: str, width: int) -> str:
    return text if len(text) >= width else ("-" * (width - len(text))) + text


def test_evidence_manifest_uses_full_web_and_kb_text() -> None:
    web_tail = "TARGET-WEB-QUOTE-AT-1600"
    web_content = _pad("", 1600) + web_tail
    kb_tail = "TARGET-KB-QUOTE-AT-2100"
    kb_text = _pad("", 2100) + kb_tail
    manifest = build_evidence_manifest(
        request_id="run-1:0:verify",
        web_sources=[{"url": "https://example.test/w", "content": web_content, "title": "W"}],
        kb_results=[{"text": kb_text, "source": "doc.md", "chunk_index": 0}],
    )
    assert len(manifest) == 2
    assert manifest[0]["source_text"] == web_content
    assert manifest[1]["source_text"] == kb_text
    assert web_tail in manifest[0]["source_text"]
    assert kb_tail in manifest[1]["source_text"]
    assert_manifest_source_integrity(manifest)
    assert sha256_utf8(manifest[0]["source_text"]) == manifest[0]["source_sha256"]


def test_source_count_policy_preserves_five_plus_five() -> None:
    web = [{"url": f"https://x/{i}", "content": f"web-{i}"} for i in range(10)]
    kb = [{"text": f"kb-{i}", "source": f"{i}.md"} for i in range(10)]
    manifest = build_evidence_manifest(
        request_id="req",
        web_sources=web,
        kb_results=kb,
    )
    assert len(manifest) == WEB_SOURCE_LIMIT + 5
    assert manifest[0]["evidence_id"] == "WEB-001"
    assert manifest[4]["evidence_id"] == "WEB-005"
    assert manifest[5]["evidence_id"] == "KB-001"
    assert manifest[9]["evidence_id"] == "KB-005"
    assert manifest[0]["source_text"] == "web-0"
    assert manifest[5]["source_text"] == "kb-0"


def test_legacy_bridge_ignores_status_and_confidence() -> None:
    rows = [
        {
            "claim": "Alpha claim.",
            "source_url": "https://example.test",
            "confidence": 0.99,
            "status": "verified",
            "specificity": "substantive",
        },
        {
            "claim": "Beta claim.",
            "source_url": None,
            "confidence": 0.1,
            "status": "unverified",
            "specificity": "generic",
        },
    ]
    roster = legacy_verdict_rows_to_claim_roster(rows)
    assert roster[0]["claim_id"] == "claim-001"
    assert roster[0]["claim_text"] == "Alpha claim."
    assert "status" not in roster[0]
    assert "confidence" not in roster[0]


def test_build_claim_roster_deterministic_ids() -> None:
    roster = build_claim_roster(["One.", "Two."])
    assert roster[0]["claim_id"] == "claim-001"
    assert roster[1]["claim_id"] == "claim-002"


def test_cross_request_id_on_manifest_entry_fails_adjudication() -> None:
    from agent.semantic_analyzer.contract import build_adjudication_envelope
    from agent.semantic_analyzer.status_engine import adjudicate_hybrid_verifier_observations

    manifest = build_evidence_manifest(
        request_id="REQ-A",
        web_sources=[{"url": "https://a", "content": "support text here"}],
    )
    manifest[0]["request_id"] = "REQ-B"
    claims = build_claim_roster(["support text here"])
    observations = [
        {
            "claim_id": "claim-001",
            "support_spans": [{"evidence_id": "WEB-001", "start": 0, "end": 17}],
            "full_entailment": True,
            "blockers": [],
        }
    ]
    envelope = build_adjudication_envelope(
        request_id="REQ-A",
        draft_text="support text here",
        claims=claims,
        evidence_manifest=manifest,
        observations=observations,
    )
    adjudication = adjudicate_hybrid_verifier_observations(envelope)
    assert adjudication.validity == "INVALID"


def test_fixture_manifest_integrity_p6() -> None:
    case = json.loads(FIXTURES.read_text())["cases"]["P6-COMPLETE"]
    manifest = build_evidence_manifest(
        request_id=case["request_id"],
        web_sources=[{"url": "https://p6.test", "content": case["source_text"], "title": "P6"}],
    )
    assert_manifest_source_integrity(manifest)
    assert len(case["source_text"]) > 1500
