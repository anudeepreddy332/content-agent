"""Production semantic-analyzer provider adapter (injected transport, no experiment runner)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent.semantic_analyzer.contract import (
    build_adjudication_envelope,
    build_analyzer_input,
    build_analyzer_messages,
    build_evidence_manifest,
    adjudication_to_grounding_report,
)
from agent.semantic_analyzer.quote_binding import (
    QuoteBindingError,
    convert_quote_observations_to_canonical,
)
from agent.semantic_analyzer.response_contract import (
    ResponseContractError,
    parse_analyzer_response,
)
from agent.semantic_analyzer.status_engine import AdjudicationResult, adjudicate_hybrid_verifier_observations

AnalyzerFailureKind = Literal[
    "provider",
    "response_contract",
    "quote_binding",
    "adjudication_invalid",
]


class LLMTransport(Protocol):
    """Injected production LLM boundary (Slice 3 passes ``agent.nodes._llm_call``)."""

    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> Any:
        ...


@dataclass
class SemanticAnalyzerResult:
    request_id: str
    success: bool
    failure_kind: AnalyzerFailureKind | None = None
    failure_detail: str | None = None
    raw_response: str | None = None
    returned_model: str | None = None
    provider_response_id: str | None = None
    usage: dict[str, Any] | None = None
    latency_ms: float | None = None
    quote_observations: list[dict[str, Any]] = field(default_factory=list)
    canonical_observations: list[dict[str, Any]] = field(default_factory=list)
    adjudication: AdjudicationResult | None = None
    grounding_report: list[dict[str, Any]] = field(default_factory=list)
    evidence_manifest: list[dict[str, Any]] = field(default_factory=list)

    @property
    def any_verified(self) -> bool:
        return any(row.get("status") == "verified" for row in self.grounding_report)


def _extract_transport_payload(response: Any) -> tuple[str, str | None, str | None, dict[str, Any] | None]:
    """Normalize OpenAI-compatible chat completion objects or test doubles."""
    if isinstance(response, str):
        return response, None, None, None
    if isinstance(response, dict):
        content = response.get("content")
        if not isinstance(content, str):
            raise ValueError("transport dict missing string content")
        return (
            content,
            response.get("model"),
            response.get("id") or response.get("provider_response_id"),
            response.get("usage"),
        )
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("transport response missing choices")
    message = choices[0].message
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise ValueError("transport response missing assistant content")
    usage_obj = getattr(response, "usage", None)
    usage = None
    if usage_obj is not None:
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }
    return (
        content,
        getattr(response, "model", None),
        getattr(response, "id", None),
        usage,
    )


def analyze_semantic_evidence(
    *,
    request_id: str,
    draft_text: str,
    claims: list[dict[str, Any]],
    web_sources: list[dict[str, Any]] | None = None,
    kb_results: list[dict[str, Any]] | None = None,
    llm_call: LLMTransport,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    schema_version: int = 1,
    response_format: dict[str, str] | None = None,
) -> SemanticAnalyzerResult:
    """End-to-end semantic analysis: manifest → LLM → bind → adjudicate → grounding_report."""
    result = SemanticAnalyzerResult(request_id=request_id, success=False)
    evidence_manifest = build_evidence_manifest(
        request_id=request_id,
        web_sources=web_sources,
        kb_results=kb_results,
    )
    result.evidence_manifest = evidence_manifest

    analyzer_input = build_analyzer_input(
        request_id=request_id,
        draft_text=draft_text,
        claims=claims,
        evidence_manifest=evidence_manifest,
        schema_version=schema_version,
    )
    messages = build_analyzer_messages(analyzer_input, system_prompt=system_prompt)

    started = time.time()
    try:
        transport_response = llm_call(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format or {"type": "json_object"},
        )
    except Exception as exc:
        result.failure_kind = "provider"
        result.failure_detail = str(exc)
        result.latency_ms = round((time.time() - started) * 1000, 3)
        return result

    result.latency_ms = round((time.time() - started) * 1000, 3)
    try:
        raw_content, returned_model, response_id, usage = _extract_transport_payload(transport_response)
    except Exception as exc:
        result.failure_kind = "provider"
        result.failure_detail = f"transport_shape:{exc}"
        return result

    result.raw_response = raw_content
    result.returned_model = returned_model
    result.provider_response_id = response_id
    result.usage = usage

    try:
        quote_contract = parse_analyzer_response(raw_content)
    except ResponseContractError as exc:
        result.failure_kind = "response_contract"
        result.failure_detail = str(exc)
        return result

    result.quote_observations = quote_contract["observations"]
    try:
        canonical_observations = convert_quote_observations_to_canonical(
            quote_contract["observations"],
            evidence_manifest,
        )
    except QuoteBindingError as exc:
        result.failure_kind = "quote_binding"
        result.failure_detail = str(exc)
        return result

    result.canonical_observations = canonical_observations
    envelope = build_adjudication_envelope(
        request_id=request_id,
        draft_text=draft_text,
        claims=claims,
        evidence_manifest=evidence_manifest,
        observations=canonical_observations,
        schema_version=schema_version,
    )
    adjudication = adjudicate_hybrid_verifier_observations(envelope)
    result.adjudication = adjudication

    if adjudication.validity != "VALID":
        result.failure_kind = "adjudication_invalid"
        result.failure_detail = ",".join(adjudication.invalid_reasons) or "invalid_adjudication"
        result.grounding_report = adjudication_to_grounding_report(
            adjudication=adjudication,
            claims=claims,
            evidence_manifest=evidence_manifest,
            observations=canonical_observations,
        )
        return result

    result.grounding_report = adjudication_to_grounding_report(
        adjudication=adjudication,
        claims=claims,
        evidence_manifest=evidence_manifest,
        observations=canonical_observations,
    )
    result.success = True
    return result
