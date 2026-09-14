"""Production contracts: evidence manifest, claim roster, analyzer input, grounding mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent.semantic_analyzer.status_engine import (
    AdjudicationResult,
    SemanticStatus,
    build_minimal_envelope,
    extract_span_text,
    sha256_utf8,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_ANALYZER_PROMPT_PATH = REPO_ROOT / "prompts" / "semantic_analyzer_system.md"

# Matches verify_node source-context selection: first N per channel, full text (no char clip).
WEB_SOURCE_LIMIT = 5
KB_SOURCE_LIMIT = 5
DEFAULT_SCHEMA_VERSION = 1

EvidenceKind = Literal["web", "kb"]


@dataclass(frozen=True)
class ProductionClaim:
    claim_id: str
    claim_text: str
    claim_span: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "claim_span": list(self.claim_span),
        }


def load_semantic_analyzer_system_prompt(
    path: Path | str = SEMANTIC_ANALYZER_PROMPT_PATH,
) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"semantic analyzer prompt missing: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _web_evidence_id(index: int) -> str:
    return f"WEB-{index:03d}"


def _kb_evidence_id(index: int) -> str:
    return f"KB-{index:03d}"


def build_evidence_manifest(
    *,
    request_id: str,
    web_sources: list[dict[str, Any]] | None = None,
    kb_results: list[dict[str, Any]] | None = None,
    web_limit: int = WEB_SOURCE_LIMIT,
    kb_limit: int = KB_SOURCE_LIMIT,
) -> list[dict[str, Any]]:
    """Convert retrieved production sources into a qualified evidence manifest.

    Uses FULL ``content`` / ``text`` — never ``_build_source_context`` clipping.
    Source-count policy matches current verifier (first ``web_limit`` web + first
    ``kb_limit`` KB); retrieval breadth is unchanged.
    """
    manifest: list[dict[str, Any]] = []
    for index, source in enumerate((web_sources or [])[:web_limit], start=1):
        source_text = source.get("content")
        if not isinstance(source_text, str):
            raise ValueError(f"web source {index} missing string content")
        manifest.append(
            {
                "evidence_id": _web_evidence_id(index),
                "request_id": request_id,
                "source_text": source_text,
                "source_sha256": sha256_utf8(source_text),
                "verifier_visible": True,
                "provenance": {
                    "kind": "web",
                    "url": source.get("url"),
                    "title": source.get("title"),
                    "score": source.get("score"),
                },
            }
        )
    for index, result in enumerate((kb_results or [])[:kb_limit], start=1):
        source_text = result.get("text")
        if not isinstance(source_text, str):
            raise ValueError(f"kb result {index} missing string text")
        manifest.append(
            {
                "evidence_id": _kb_evidence_id(index),
                "request_id": request_id,
                "source_text": source_text,
                "source_sha256": sha256_utf8(source_text),
                "verifier_visible": True,
                "provenance": {
                    "kind": "kb",
                    "source": result.get("source"),
                    "chunk_index": result.get("chunk_index"),
                    "distance": result.get("distance"),
                    "rrf_score": result.get("rrf_score"),
                },
            }
        )
    return manifest


def build_claim_roster(
    claim_texts: list[str],
    *,
    id_prefix: str = "claim",
) -> list[dict[str, Any]]:
    """Deterministic ordered claim roster for one verifier pass."""
    roster: list[dict[str, Any]] = []
    for index, claim_text in enumerate(claim_texts, start=1):
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise ValueError(f"claim {index} must be non-empty text")
        roster.append(
            ProductionClaim(
                claim_id=f"{id_prefix}-{index:03d}",
                claim_text=claim_text,
                claim_span=(0, len(claim_text)),
            ).to_dict()
        )
    return roster


def legacy_verdict_rows_to_claim_roster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase-3 bridge: extract claim text only from legacy verifier verdict rows.

    Ignores legacy ``status`` and ``confidence`` — they are NOT semantic authority.
    Claim inventory completeness remains UNKNOWN (Phase 4).
    """
    claim_texts = []
    for row in rows:
        claim = row.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("legacy verdict row missing non-empty claim text")
        claim_texts.append(claim)
    return build_claim_roster(claim_texts)


def build_analyzer_input(
    *,
    request_id: str,
    draft_text: str,
    claims: list[dict[str, Any]],
    evidence_manifest: list[dict[str, Any]],
    schema_version: int = DEFAULT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Logical analyzer request body (no gold, no expected statuses)."""
    return {
        "request_id": request_id,
        "schema_version": schema_version,
        "draft_text": draft_text,
        "draft_sha256": sha256_utf8(draft_text),
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "claim_span": list(claim["claim_span"]),
            }
            for claim in claims
        ],
        "evidence_manifest": [
            {
                "evidence_id": entry["evidence_id"],
                "request_id": entry["request_id"],
                "source_text": entry["source_text"],
                "source_sha256": entry["source_sha256"],
                "verifier_visible": entry.get("verifier_visible", True),
            }
            for entry in evidence_manifest
        ],
    }


def build_analyzer_user_message(analyzer_input: dict[str, Any]) -> str:
    return json.dumps(analyzer_input, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def build_analyzer_messages(
    analyzer_input: dict[str, Any],
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    prompt = system_prompt if system_prompt is not None else load_semantic_analyzer_system_prompt()
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": build_analyzer_user_message(analyzer_input)},
    ]


def build_adjudication_envelope(
    *,
    request_id: str,
    draft_text: str,
    claims: list[dict[str, Any]],
    evidence_manifest: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    schema_version: int = DEFAULT_SCHEMA_VERSION,
) -> dict[str, Any]:
    return build_minimal_envelope(
        request_id=request_id,
        schema_version=schema_version,
        draft_text=draft_text,
        claims=claims,
        evidence_manifest=evidence_manifest,
        observations=observations,
    )


def _manifest_by_id(manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["evidence_id"]: entry for entry in manifest}


def _attribution_from_evidence(entry: dict[str, Any]) -> tuple[str, str | None, str | None]:
    provenance = entry.get("provenance") or {}
    kind = provenance.get("kind")
    if kind == "web":
        url = provenance.get("url")
        return "web", str(url) if url else None, str(url) if url else None
    if kind == "kb":
        source = provenance.get("source") or "unknown"
        kb_ref = f"kb:{source}"
        return "kb", str(source), kb_ref
    return "none", None, None


def _primary_support_evidence_id(observation: dict[str, Any]) -> str | None:
    support_spans = observation.get("support_spans") or []
    if not support_spans:
        return None
    first = support_spans[0]
    evidence_id = first.get("evidence_id")
    return evidence_id if isinstance(evidence_id, str) else None


def adjudication_to_grounding_report(
    *,
    adjudication: AdjudicationResult,
    claims: list[dict[str, Any]],
    evidence_manifest: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map canonical adjudication to production grounding_report rows.

    Semantic ``status`` comes ONLY from ``adjudicate_hybrid_verifier_observations``.
    Does NOT fabricate LLM ``confidence`` — Slice 3 must decide compatibility for
    ``grounding_score`` (legacy mean confidence).
    """
    claim_by_id = {row["claim_id"]: row for row in claims}
    obs_by_id = {row["claim_id"]: row for row in observations}
    manifest_by_id = _manifest_by_id(evidence_manifest)
    report: list[dict[str, Any]] = []

    for claim_result in adjudication.claim_results:
        claim_id = claim_result.claim_id
        claim_row = claim_by_id.get(claim_id, {})
        observation = obs_by_id.get(claim_id, {})
        semantic_status: SemanticStatus | None = claim_result.semantic_status
        status = semantic_status if semantic_status is not None else "unverified"

        source_kind = "none"
        source_ref: str | None = None
        source_url: str | None = None
        primary_evidence_id = _primary_support_evidence_id(observation)
        if primary_evidence_id and primary_evidence_id in manifest_by_id:
            source_kind, source_url, source_ref = _attribution_from_evidence(
                manifest_by_id[primary_evidence_id]
            )

        support_spans = [
            {
                "evidence_id": span["evidence_id"],
                "start": span["start"],
                "end": span["end"],
                "text": extract_span_text(
                    manifest_by_id[span["evidence_id"]]["source_text"],
                    span["start"],
                    span["end"],
                )
                if span["evidence_id"] in manifest_by_id
                else "",
            }
            for span in observation.get("support_spans", [])
        ]
        blockers = []
        for blocker in observation.get("blockers", []):
            evidence_spans = []
            for span in blocker.get("evidence_spans", []):
                evidence_spans.append(
                    {
                        "evidence_id": span["evidence_id"],
                        "start": span["start"],
                        "end": span["end"],
                        "text": extract_span_text(
                            manifest_by_id[span["evidence_id"]]["source_text"],
                            span["start"],
                            span["end"],
                        )
                        if span["evidence_id"] in manifest_by_id
                        else "",
                    }
                )
            blockers.append(
                {
                    "kind": blocker.get("kind"),
                    "explanation": blocker.get("explanation"),
                    "evidence_spans": evidence_spans,
                }
            )

        report.append(
            {
                "claim_id": claim_id,
                "claim": claim_row.get("claim_text", ""),
                "status": status,
                "source_url": source_url,
                "source_kind": source_kind,
                "source_ref": source_ref,
                "full_entailment": observation.get("full_entailment"),
                "support_spans": support_spans,
                "blockers": blockers,
                "analysis_validity": claim_result.validity,
                "invalid_reasons": list(claim_result.invalid_reasons),
                "reason_codes": list(claim_result.reason_codes),
            }
        )
    return report


def assert_manifest_source_integrity(manifest: list[dict[str, Any]]) -> None:
    """Invariant: exposed source_text hashes match and bind to the same string."""
    for entry in manifest:
        source_text = entry["source_text"]
        declared = entry["source_sha256"]
        actual = sha256_utf8(source_text)
        if declared != actual:
            raise ValueError(
                f"manifest hash mismatch for {entry['evidence_id']}: "
                f"declared {declared}, actual {actual}"
            )
