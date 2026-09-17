"""Deterministic exact-quote → canonical span binding for semantic analyzer observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

QuoteBindingFailure = Literal[
    "empty_quote",
    "invalid_evidence_id",
    "quote_not_found",
    "ambiguous_quote",
]

ALLOWED_QUOTE_FIELDS = frozenset({"evidence_id", "quote"})
ALLOWED_EXTERNAL_OBSERVATION_FIELDS = frozenset(
    {"claim_id", "support_quotes", "full_entailment", "blockers"}
)
ALLOWED_EXTERNAL_BLOCKER_FIELDS = frozenset({"kind", "evidence_quotes", "explanation"})
FORBIDDEN_EXTERNAL_OBSERVATION_FIELDS = frozenset(
    {"support_spans", "start", "end", "status", "materiality", "confidence", "publication_decision", "routing"}
)


class QuoteBindingError(ValueError):
    """Exact quote could not be bound to a unique span in bound source_text."""

    def __init__(self, reason: QuoteBindingFailure, *, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail
        message = reason if detail is None else f"{reason}:{detail}"
        super().__init__(message)


@dataclass(frozen=True)
class QuoteBindingResult:
    evidence_id: str
    quote: str
    start: int
    end: int
    bound_text: str
    match_count: int


def find_exact_quote_matches(source_text: str, quote: str) -> list[tuple[int, int]]:
    """Return zero-based half-open [start,end) offsets for each exact quote occurrence."""
    if quote == "":
        return []
    matches: list[tuple[int, int]] = []
    search_from = 0
    while search_from <= len(source_text):
        idx = source_text.find(quote, search_from)
        if idx == -1:
            break
        matches.append((idx, idx + len(quote)))
        search_from = idx + 1
    return matches


def bind_quote_to_span(
    *,
    evidence_id: str,
    quote: str,
    source_text: str,
) -> QuoteBindingResult:
    """Bind one exact quote to a unique canonical span. No normalization or fuzzy matching."""
    if not isinstance(quote, str) or quote == "":
        raise QuoteBindingError("empty_quote")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise QuoteBindingError("invalid_evidence_id", detail=evidence_id or "missing")

    matches = find_exact_quote_matches(source_text, quote)
    if len(matches) == 0:
        raise QuoteBindingError("quote_not_found", detail=evidence_id)
    if len(matches) > 1:
        raise QuoteBindingError("ambiguous_quote", detail=evidence_id)

    start, end = matches[0]
    bound_text = source_text[start:end]
    if bound_text != quote:
        raise QuoteBindingError("quote_not_found", detail="bound_text_mismatch")
    return QuoteBindingResult(
        evidence_id=evidence_id,
        quote=quote,
        start=start,
        end=end,
        bound_text=bound_text,
        match_count=1,
    )


def bind_quote_against_manifest(
    *,
    evidence_id: str,
    quote: str,
    evidence_manifest: list[dict[str, Any]],
) -> QuoteBindingResult:
    evidence_by_id = {entry["evidence_id"]: entry for entry in evidence_manifest}
    entry = evidence_by_id.get(evidence_id)
    if entry is None:
        raise QuoteBindingError("invalid_evidence_id", detail=evidence_id)
    return bind_quote_to_span(
        evidence_id=evidence_id,
        quote=quote,
        source_text=entry["source_text"],
    )


def convert_quote_observation_to_canonical(
    observation: dict[str, Any],
    evidence_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert one quote-based analyzer observation into canonical span observations."""
    support_spans: list[dict[str, Any]] = []
    for index, quote_row in enumerate(observation["support_quotes"]):
        result = bind_quote_against_manifest(
            evidence_id=quote_row["evidence_id"],
            quote=quote_row["quote"],
            evidence_manifest=evidence_manifest,
        )
        support_spans.append(
            {"evidence_id": result.evidence_id, "start": result.start, "end": result.end}
        )

    blockers: list[dict[str, Any]] = []
    for blocker in observation["blockers"]:
        evidence_spans: list[dict[str, Any]] = []
        for quote_row in blocker["evidence_quotes"]:
            result = bind_quote_against_manifest(
                evidence_id=quote_row["evidence_id"],
                quote=quote_row["quote"],
                evidence_manifest=evidence_manifest,
            )
            evidence_spans.append(
                {"evidence_id": result.evidence_id, "start": result.start, "end": result.end}
            )
        blockers.append(
            {
                "kind": blocker["kind"],
                "evidence_spans": evidence_spans,
                "explanation": blocker["explanation"],
            }
        )

    return {
        "claim_id": observation["claim_id"],
        "support_spans": support_spans,
        "full_entailment": observation["full_entailment"],
        "blockers": blockers,
    }


def convert_quote_observations_to_canonical(
    observations: list[dict[str, Any]],
    evidence_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        convert_quote_observation_to_canonical(observation, evidence_manifest)
        for observation in observations
    ]
