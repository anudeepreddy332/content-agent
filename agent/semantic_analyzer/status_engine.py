"""Hybrid verifier status engine — Python owns final semantic status.

The semantic analyzer supplies structured observations only. This module validates
those observations against a frozen evaluation envelope and deterministically derives
``verified`` / ``weak`` / ``unverified``. ``INVALID`` is an analysis-validity outcome,
not a fourth semantic status.

Claim extraction completeness is a separate upstream qualification problem.
Every supplied factual claim receives the same verification contract; materiality is
a later publication-governance concern and is excluded from this engine.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

SEMANTIC_STATUSES = frozenset({"verified", "weak", "unverified"})
BLOCKER_KINDS = frozenset({"contradiction", "limitation"})
ALLOWED_OBSERVATION_FIELDS = frozenset(
    {"claim_id", "support_spans", "full_entailment", "blockers"}
)
ALLOWED_SPAN_FIELDS = frozenset({"evidence_id", "start", "end"})
ALLOWED_BLOCKER_FIELDS = frozenset({"kind", "evidence_spans", "explanation"})

AnalysisValidity = Literal["VALID", "INVALID"]
SemanticStatus = Literal["verified", "weak", "unverified"]


class HybridVerifierStatusError(Exception):
    """Base error for hybrid verifier status engine failures."""


class DeterministicFactVeto(Protocol):
    """Typed extension point for future exact-fact vetoes (IDs, versions, units, etc.).

    Implementations must be pure and must not parse natural-language prose.
    """

    def claim_ids(self) -> frozenset[str]:
        """Claim IDs this veto may apply to; empty means any claim."""

    def vetoes_full_entailment(
        self,
        *,
        claim: dict[str, Any],
        observation: dict[str, Any],
        envelope: dict[str, Any],
    ) -> bool:
        """Return True when a deterministic fact check blocks full entailment."""


# Production adapters register here in future slices; offline qualification uses none.
DETERMINISTIC_FACT_VETOS: list[DeterministicFactVeto] = []


def sha256_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def codepoint_length(text: str) -> int:
    return len(text)


def extract_span_text(source_text: str, start: int, end: int) -> str:
    return source_text[start:end]


@dataclass(frozen=True)
class SpanRef:
    evidence_id: str
    start: int
    end: int


@dataclass
class ClaimAdjudication:
    claim_id: str
    validity: AnalysisValidity
    semantic_status: SemanticStatus | None = None
    invalid_reasons: list[str] = field(default_factory=list)
    consistency_diagnostics: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    meaningful_positive_support: bool | None = None


@dataclass
class AdjudicationResult:
    request_id: str
    validity: AnalysisValidity
    claim_results: list[ClaimAdjudication]
    invalid_reasons: list[str] = field(default_factory=list)

    @property
    def semantic_status_by_claim(self) -> dict[str, SemanticStatus | None]:
        return {row.claim_id: row.semantic_status for row in self.claim_results}


def _require(condition: bool, reason: str, *, bucket: list[str]) -> None:
    if not condition:
        bucket.append(reason)


def _observation_signature(observation: dict[str, Any]) -> str:
    return sha256_utf8(json.dumps(observation, sort_keys=True, ensure_ascii=True))


def _validate_span_ref(
    span: Any,
    *,
    evidence_by_id: dict[str, dict[str, Any]],
    request_id: str,
    invalid_reasons: list[str],
    prefix: str,
) -> SpanRef | None:
    if not isinstance(span, dict):
        invalid_reasons.append(f"{prefix}_malformed_span")
        return None
    extra = set(span) - ALLOWED_SPAN_FIELDS
    if extra:
        invalid_reasons.append(f"{prefix}_unknown_span_field")
        return None
    if "evidence_id" not in span or "start" not in span or "end" not in span:
        invalid_reasons.append(f"{prefix}_missing_span_field")
        return None
    evidence_id = span["evidence_id"]
    start = span["start"]
    end = span["end"]
    if not isinstance(evidence_id, str):
        invalid_reasons.append(f"{prefix}_invalid_evidence_id_type")
        return None
    if not isinstance(start, int) or isinstance(start, bool):
        invalid_reasons.append(f"{prefix}_invalid_start_type")
        return None
    if not isinstance(end, int) or isinstance(end, bool):
        invalid_reasons.append(f"{prefix}_invalid_end_type")
        return None
    entry = evidence_by_id.get(evidence_id)
    if entry is None:
        invalid_reasons.append(f"{prefix}_unknown_evidence_id")
        return None
    if entry.get("request_id") != request_id:
        invalid_reasons.append(f"{prefix}_cross_request_evidence_reference")
        return None
    expected_hash = entry.get("source_sha256")
    if expected_hash and sha256_utf8(entry["source_text"]) != expected_hash:
        invalid_reasons.append(f"{prefix}_stale_evidence_hash")
        return None
    if not entry.get("verifier_visible", True):
        invalid_reasons.append(f"{prefix}_invisible_evidence")
        return None
    if start < 0:
        invalid_reasons.append(f"{prefix}_negative_span_start")
        return None
    if end <= start:
        invalid_reasons.append(f"{prefix}_zero_or_negative_length_span")
        return None
    source_len = codepoint_length(entry["source_text"])
    if end > source_len:
        invalid_reasons.append(f"{prefix}_out_of_bounds_span")
        return None
    return SpanRef(evidence_id=evidence_id, start=start, end=end)


def _validate_blocker(
    blocker: Any,
    *,
    evidence_by_id: dict[str, dict[str, Any]],
    request_id: str,
    invalid_reasons: list[str],
    index: int,
) -> dict[str, Any] | None:
    prefix = f"blocker_{index}"
    if not isinstance(blocker, dict):
        invalid_reasons.append(f"{prefix}_malformed")
        return None
    extra = set(blocker) - ALLOWED_BLOCKER_FIELDS
    if extra:
        invalid_reasons.append(f"{prefix}_unknown_field")
        return None
    kind = blocker.get("kind")
    if kind not in BLOCKER_KINDS:
        invalid_reasons.append(f"{prefix}_unknown_kind")
        return None
    explanation = blocker.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        invalid_reasons.append(f"{prefix}_missing_explanation")
        return None
    raw_spans = blocker.get("evidence_spans")
    if not isinstance(raw_spans, list):
        invalid_reasons.append(f"{prefix}_malformed_evidence_spans")
        return None
    validated_spans: list[SpanRef] = []
    for span_index, span in enumerate(raw_spans):
        validated = _validate_span_ref(
            span,
            evidence_by_id=evidence_by_id,
            request_id=request_id,
            invalid_reasons=invalid_reasons,
            prefix=f"{prefix}_span_{span_index}",
        )
        if validated is not None:
            validated_spans.append(validated)
    if kind == "contradiction" and not validated_spans:
        invalid_reasons.append(f"{prefix}_contradiction_requires_evidence_span")
        return None
    return {
        "kind": kind,
        "evidence_spans": validated_spans,
        "explanation": explanation.strip(),
    }


def _validate_observation_row(
    observation: Any,
    *,
    evidence_by_id: dict[str, dict[str, Any]],
    request_id: str,
    invalid_reasons: list[str],
) -> dict[str, Any] | None:
    if not isinstance(observation, dict):
        invalid_reasons.append("observation_malformed")
        return None
    extra = set(observation) - ALLOWED_OBSERVATION_FIELDS
    if extra:
        invalid_reasons.append("observation_unknown_field")
        return None
    for required in ALLOWED_OBSERVATION_FIELDS:
        if required not in observation:
            invalid_reasons.append(f"observation_missing_{required}")
            return None
    claim_id = observation["claim_id"]
    if not isinstance(claim_id, str) or not claim_id:
        invalid_reasons.append("observation_invalid_claim_id")
        return None
    full_entailment = observation["full_entailment"]
    if not isinstance(full_entailment, bool):
        invalid_reasons.append("observation_invalid_full_entailment_type")
        return None
    support_raw = observation["support_spans"]
    blockers_raw = observation["blockers"]
    if not isinstance(support_raw, list):
        invalid_reasons.append("observation_malformed_support_spans")
        return None
    if not isinstance(blockers_raw, list):
        invalid_reasons.append("observation_malformed_blockers")
        return None

    support_spans: list[SpanRef] = []
    for index, span in enumerate(support_raw):
        validated = _validate_span_ref(
            span,
            evidence_by_id=evidence_by_id,
            request_id=request_id,
            invalid_reasons=invalid_reasons,
            prefix=f"support_span_{index}",
        )
        if validated is not None:
            support_spans.append(validated)

    blockers: list[dict[str, Any]] = []
    for index, blocker in enumerate(blockers_raw):
        validated = _validate_blocker(
            blocker,
            evidence_by_id=evidence_by_id,
            request_id=request_id,
            invalid_reasons=invalid_reasons,
            index=index,
        )
        if validated is not None:
            blockers.append(validated)

    return {
        "claim_id": claim_id,
        "support_spans": support_spans,
        "full_entailment": full_entailment,
        "blockers": blockers,
    }


def _claim_lookup(envelope: dict[str, Any]) -> dict[str, dict[str, Any]]:
    claims = envelope.get("claims") or []
    return {claim["claim_id"]: claim for claim in claims if isinstance(claim, dict)}


def _evidence_lookup(envelope: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = envelope.get("evidence_manifest") or []
    return {
        entry["evidence_id"]: entry
        for entry in manifest
        if isinstance(entry, dict) and "evidence_id" in entry
    }


def _has_deterministic_fact_veto(
    *,
    claim: dict[str, Any],
    observation: dict[str, Any],
    envelope: dict[str, Any],
) -> bool:
    claim_id = claim["claim_id"]
    for veto in DETERMINISTIC_FACT_VETOS:
        scoped = veto.claim_ids()
        if scoped and claim_id not in scoped:
            continue
        if veto.vetoes_full_entailment(claim=claim, observation=observation, envelope=envelope):
            return True
    return False


def derive_claim_status(
    *,
    claim: dict[str, Any],
    observation: dict[str, Any],
    envelope: dict[str, Any],
    row_invalid_reasons: list[str],
) -> ClaimAdjudication:
    claim_id = claim["claim_id"]
    if row_invalid_reasons:
        return ClaimAdjudication(
            claim_id=claim_id,
            validity="INVALID",
            invalid_reasons=list(row_invalid_reasons),
        )

    meaningful_support = len(observation["support_spans"]) > 0
    full_entailment = observation["full_entailment"]
    blockers = observation["blockers"]
    has_contradiction = any(b["kind"] == "contradiction" for b in blockers)
    has_limitation = any(b["kind"] == "limitation" for b in blockers)
    has_blocker = bool(blockers)
    fact_veto = _has_deterministic_fact_veto(
        claim=claim, observation=observation, envelope=envelope
    )

    consistency: list[str] = []
    reason_codes: list[str] = []

    if meaningful_support:
        reason_codes.append("MEANINGFUL_SUPPORT_PRESENT")
    else:
        reason_codes.append("NO_MEANINGFUL_SUPPORT")

    if full_entailment:
        reason_codes.append("FULL_ENTAILMENT_CLAIMED")
    if has_contradiction:
        reason_codes.append("BLOCKER_CONTRADICTION")
    if has_limitation:
        reason_codes.append("BLOCKER_LIMITATION")
    if fact_veto:
        reason_codes.append("DETERMINISTIC_FACT_VETO")

    if not meaningful_support and full_entailment:
        return ClaimAdjudication(
            claim_id=claim_id,
            validity="INVALID",
            semantic_status=None,
            invalid_reasons=["no_support_with_full_entailment_true"],
            reason_codes=reason_codes,
            meaningful_positive_support=meaningful_support,
        )

    if meaningful_support and full_entailment and has_blocker:
        consistency.append("full_entailment_true_with_blockers_downgraded_to_weak")

    if (
        meaningful_support
        and full_entailment
        and not has_blocker
        and not fact_veto
    ):
        return ClaimAdjudication(
            claim_id=claim_id,
            validity="VALID",
            semantic_status="verified",
            reason_codes=reason_codes,
            meaningful_positive_support=meaningful_support,
        )

    if meaningful_support and (
        not full_entailment or has_blocker or fact_veto
    ):
        return ClaimAdjudication(
            claim_id=claim_id,
            validity="VALID",
            semantic_status="weak",
            consistency_diagnostics=consistency,
            reason_codes=reason_codes,
            meaningful_positive_support=meaningful_support,
        )

    return ClaimAdjudication(
        claim_id=claim_id,
        validity="VALID",
        semantic_status="unverified",
        reason_codes=reason_codes,
        meaningful_positive_support=meaningful_support,
    )


def adjudicate_hybrid_verifier_observations(envelope: dict[str, Any]) -> AdjudicationResult:
    """Pure adjudication over a frozen envelope plus analyzer observations."""
    invalid_reasons: list[str] = []
    request_id = envelope.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        invalid_reasons.append("missing_request_id")
        request_id = "UNKNOWN"

    claims = envelope.get("claims")
    observations = envelope.get("observations")
    if not isinstance(claims, list) or not claims:
        invalid_reasons.append("missing_claim_roster")
        return AdjudicationResult(
            request_id=request_id,
            validity="INVALID",
            claim_results=[],
            invalid_reasons=invalid_reasons,
        )
    if not isinstance(observations, list):
        invalid_reasons.append("missing_observations")
        return AdjudicationResult(
            request_id=request_id,
            validity="INVALID",
            claim_results=[],
            invalid_reasons=invalid_reasons,
        )

    claim_by_id = _claim_lookup(envelope)
    if len(claim_by_id) != len(claims):
        invalid_reasons.append("duplicate_claim_in_roster")

    expected_claim_ids = set(claim_by_id)
    evidence_by_id = _evidence_lookup(envelope)

    validated_rows: dict[str, dict[str, Any]] = {}
    row_invalid: dict[str, list[str]] = {}
    signatures: dict[str, str] = {}

    for observation in observations:
        row_errors: list[str] = []
        claim_id_hint = observation.get("claim_id") if isinstance(observation, dict) else None
        validated = _validate_observation_row(
            observation,
            evidence_by_id=evidence_by_id,
            request_id=request_id,
            invalid_reasons=row_errors,
        )
        if validated is None:
            if isinstance(claim_id_hint, str) and claim_id_hint:
                row_invalid.setdefault(claim_id_hint, []).extend(row_errors)
                if claim_id_hint not in expected_claim_ids:
                    invalid_reasons.append(f"extra_claim_row:{claim_id_hint}")
            else:
                invalid_reasons.extend(row_errors)
            continue
        claim_id = validated["claim_id"]
        if claim_id not in expected_claim_ids:
            row_errors.append("unknown_claim_id")
            invalid_reasons.append(f"extra_claim_row:{claim_id}")
        if claim_id in validated_rows:
            signature = _observation_signature(observation)
            if signatures.get(claim_id) != signature:
                row_errors.append("conflicting_duplicate_claim_row")
            else:
                row_errors.append("duplicate_claim_row")
        else:
            signatures[claim_id] = _observation_signature(observation)
        if row_errors:
            row_invalid[claim_id] = row_errors
            validated_rows.pop(claim_id, None)
        else:
            validated_rows[claim_id] = validated

    missing_claims = sorted(expected_claim_ids - set(validated_rows) - set(row_invalid))
    if missing_claims:
        invalid_reasons.extend(f"missing_claim_row:{claim_id}" for claim_id in missing_claims)

    claim_results: list[ClaimAdjudication] = []
    overall_invalid = bool(invalid_reasons) or bool(row_invalid)

    for claim_id in sorted(expected_claim_ids):
        claim = claim_by_id[claim_id]
        if claim_id in row_invalid:
            overall_invalid = True
            claim_results.append(
                ClaimAdjudication(
                    claim_id=claim_id,
                    validity="INVALID",
                    invalid_reasons=row_invalid[claim_id],
                )
            )
            continue
        if claim_id not in validated_rows:
            overall_invalid = True
            claim_results.append(
                ClaimAdjudication(
                    claim_id=claim_id,
                    validity="INVALID",
                    invalid_reasons=[f"missing_claim_row:{claim_id}"],
                )
            )
            continue
        result = derive_claim_status(
            claim=claim,
            observation=validated_rows[claim_id],
            envelope=envelope,
            row_invalid_reasons=[],
        )
        if result.validity == "INVALID":
            overall_invalid = True
        claim_results.append(result)

    return AdjudicationResult(
        request_id=request_id,
        validity="INVALID" if overall_invalid else "VALID",
        claim_results=claim_results,
        invalid_reasons=sorted(set(invalid_reasons)),
    )


def build_minimal_envelope(
    *,
    request_id: str,
    schema_version: int,
    draft_text: str,
    claims: list[dict[str, Any]],
    evidence_manifest: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct an envelope with envelope-owned identity fields."""
    return {
        "schema_version": schema_version,
        "request_id": request_id,
        "draft_text": draft_text,
        "draft_sha256": sha256_utf8(draft_text),
        "claims": claims,
        "evidence_manifest": evidence_manifest,
        "observations": observations,
    }
