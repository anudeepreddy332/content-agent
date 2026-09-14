"""Production semantic-analyzer response contract validation (quote-based, no offsets)."""
from __future__ import annotations

import json
from typing import Any

from agent.semantic_analyzer.quote_binding import (
    ALLOWED_EXTERNAL_BLOCKER_FIELDS,
    ALLOWED_EXTERNAL_OBSERVATION_FIELDS,
    ALLOWED_QUOTE_FIELDS,
)
from agent.semantic_analyzer.status_engine import BLOCKER_KINDS

FORBIDDEN_ANALYZER_FIELDS = frozenset(
    {
        "status",
        "materiality",
        "confidence",
        "publication_decision",
        "routing",
        "support_spans",
        "evidence_spans",
        "start",
        "end",
    }
)


class ResponseContractError(ValueError):
    """Analyzer response violates the frozen JSON contract."""


def _json_load_no_duplicate_keys(raw: str) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ResponseContractError("duplicate_json_keys")
        return dict(pairs)

    return json.loads(raw, object_pairs_hook=hook)


def _validate_response_quote(quote: Any, *, prefix: str) -> None:
    if not isinstance(quote, dict):
        raise ResponseContractError(f"{prefix}_not_object")
    extra = set(quote) - ALLOWED_QUOTE_FIELDS
    if extra:
        raise ResponseContractError(f"{prefix}_unknown_field")
    for required in ALLOWED_QUOTE_FIELDS:
        if required not in quote:
            raise ResponseContractError(f"{prefix}_missing_{required}")
        if quote[required] is None:
            raise ResponseContractError(f"{prefix}_null_{required}")
    evidence_id = quote["evidence_id"]
    text = quote["quote"]
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ResponseContractError(f"{prefix}_invalid_evidence_id")
    if not isinstance(text, str):
        raise ResponseContractError(f"{prefix}_invalid_quote_type")
    if text == "":
        raise ResponseContractError(f"{prefix}_empty_quote")


def parse_analyzer_response(raw: str) -> dict[str, Any]:
    """Parse and structurally validate the quote-based analyzer response contract."""
    if not isinstance(raw, str) or not raw.strip():
        raise ResponseContractError("empty_response")

    text = raw.strip()
    if "```" in text:
        raise ResponseContractError("markdown_fence_rejected")

    try:
        parsed = _json_load_no_duplicate_keys(text)
    except json.JSONDecodeError as exc:
        if exc.msg == "Extra data":
            raise ResponseContractError("trailing_prose") from exc
        raise ResponseContractError(f"malformed_json:{exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ResponseContractError("top_level_not_object")

    extra_top = set(parsed) - {"observations"}
    if extra_top:
        raise ResponseContractError("unknown_top_level_field")
    if "observations" not in parsed:
        raise ResponseContractError("missing_observations")

    observations = parsed["observations"]
    if not isinstance(observations, list):
        raise ResponseContractError("observations_not_array")

    validated_rows: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        prefix = f"observation_{index}"
        if observation is None:
            raise ResponseContractError(f"{prefix}_null")
        if not isinstance(observation, dict):
            raise ResponseContractError(f"{prefix}_not_object")
        forbidden = FORBIDDEN_ANALYZER_FIELDS
        if forbidden.intersection(observation):
            raise ResponseContractError(f"{prefix}_forbidden_field")
        extra = set(observation) - ALLOWED_EXTERNAL_OBSERVATION_FIELDS
        if extra:
            raise ResponseContractError(f"{prefix}_unknown_field")
        for required in ALLOWED_EXTERNAL_OBSERVATION_FIELDS:
            if required not in observation:
                raise ResponseContractError(f"{prefix}_missing_{required}")
            if observation[required] is None:
                raise ResponseContractError(f"{prefix}_null_{required}")

        claim_id = observation["claim_id"]
        if not isinstance(claim_id, str) or not claim_id:
            raise ResponseContractError(f"{prefix}_invalid_claim_id")

        full_entailment = observation["full_entailment"]
        if not isinstance(full_entailment, bool):
            raise ResponseContractError(f"{prefix}_invalid_full_entailment_type")

        support_quotes = observation["support_quotes"]
        blockers = observation["blockers"]
        if not isinstance(support_quotes, list):
            raise ResponseContractError(f"{prefix}_malformed_support_quotes")
        if not isinstance(blockers, list):
            raise ResponseContractError(f"{prefix}_malformed_blockers")

        for quote_index, quote_row in enumerate(support_quotes):
            _validate_response_quote(quote_row, prefix=f"{prefix}_support_quote_{quote_index}")

        for blocker_index, blocker in enumerate(blockers):
            blocker_prefix = f"{prefix}_blocker_{blocker_index}"
            if not isinstance(blocker, dict):
                raise ResponseContractError(f"{blocker_prefix}_not_object")
            if forbidden.intersection(blocker):
                raise ResponseContractError(f"{blocker_prefix}_forbidden_field")
            extra_blocker = set(blocker) - ALLOWED_EXTERNAL_BLOCKER_FIELDS
            if extra_blocker:
                raise ResponseContractError(f"{blocker_prefix}_unknown_field")
            for required in ALLOWED_EXTERNAL_BLOCKER_FIELDS:
                if required not in blocker:
                    raise ResponseContractError(f"{blocker_prefix}_missing_{required}")
                if blocker[required] is None:
                    raise ResponseContractError(f"{blocker_prefix}_null_{required}")
            kind = blocker["kind"]
            if kind not in BLOCKER_KINDS:
                raise ResponseContractError(f"{blocker_prefix}_unknown_kind")
            if not isinstance(blocker["explanation"], str) or not blocker["explanation"].strip():
                raise ResponseContractError(f"{blocker_prefix}_missing_explanation")
            evidence_quotes = blocker["evidence_quotes"]
            if not isinstance(evidence_quotes, list):
                raise ResponseContractError(f"{blocker_prefix}_malformed_evidence_quotes")
            for quote_index, quote_row in enumerate(evidence_quotes):
                _validate_response_quote(
                    quote_row,
                    prefix=f"{blocker_prefix}_quote_{quote_index}",
                )

        validated_rows.append(observation)

    return {"observations": validated_rows}
