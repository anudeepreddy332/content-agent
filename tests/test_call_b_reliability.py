"""Offline Call-B capacity, response-metadata, and semantic integrity regressions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from openai import APITimeoutError

import agent.nodes as nodes
from agent.claim_inventory import (
    ANCHOR_AMBIGUOUS, build_claim_inventory, compute_claim_id, inventory_critical_failures,
)
from agent.semantic_analyzer.provider import CALL_B_MAX_TOKENS, analyze_semantic_evidence
from agent.semantic_trace import empty_trace, record_semantic_analyzer
from tests.conftest import FakeLLMClient, fake_response, openai_error


def _large_case():
    # Independent propositions and exact, unique source sentences. Successful JSON
    # parsing alone is insufficient: the status engine must bind every quote.
    sentences = [f"Method {i} requires condition {i} for operation." for i in range(50)]
    draft = "\n".join(sentences)
    claims = [
        {"claim_id": f"claim-{i:03d}", "claim_text": sentence,
         "claim_span": [draft.index(sentence), draft.index(sentence) + len(sentence)]}
        for i, sentence in enumerate(sentences)
    ]
    observations = [
        {"claim_id": claim["claim_id"],
         "support_quotes": [{"evidence_id": "WEB-001", "quote": claim["claim_text"]}],
         "full_entailment": True, "blockers": []}
        for claim in claims
    ]
    return draft, claims, "\n".join(sentences), observations


def _response(content, *, finish=None, reasoning=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
            content=content, reasoning_content="private reasoning must not persist"))],
        model="deepseek-flash", id="offline-call-b",
        usage=SimpleNamespace(
            prompt_tokens=10_147, completion_tokens=2_000, total_tokens=12_147,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


def _analyze(response, *, observations=None):
    draft, claims, source, _ = _large_case()
    return analyze_semantic_evidence(
        request_id="offline-call-b", draft_text=draft, claims=claims,
        web_sources=[{"url": "https://fixture.invalid/source", "title": "fixture",
                      "content": source}],
        llm_call=lambda **kwargs: response, model="deepseek-flash",
    )


def test_historical_empty_response_retains_observable_metadata_without_reasoning_text():
    result = _analyze(_response("", reasoning=2_000))
    assert not result.success
    assert (result.failure_kind, result.failure_detail) == ("response_contract", "empty_response")
    assert result.finish_reason is None and result.response_length == 0
    assert result.parser_status == "failed" and result.parser_error == "empty_response"
    assert result.usage == {"prompt_tokens": 10_147, "completion_tokens": 2_000,
                            "total_tokens": 12_147, "reasoning_tokens": 2_000}
    trace = empty_trace({"run_id": "offline-call-b"})
    record_semantic_analyzer(
        trace, iteration=1, request_id="offline-call-b", success=False,
        failure_kind=result.failure_kind, failure_detail=result.failure_detail,
        returned_model=result.returned_model, provider_response_id=result.provider_response_id,
        usage=result.usage, finish_reason=result.finish_reason,
        response_length=result.response_length, parser_status=result.parser_status,
        parser_error=result.parser_error, max_tokens=result.max_tokens,
        raw_response=result.raw_response,
    )
    saved = trace["iterations"][0]["semantic_analyzer"]
    assert saved["termination"] == "unknown" and saved["finish_reason"] is None
    assert saved["response_length"] == 0 and saved["parser_status"] == "failed"
    assert saved["request"]["max_tokens"] == CALL_B_MAX_TOKENS == 8_000
    assert "private reasoning" not in json.dumps(saved)


def test_null_assistant_content_is_empty_output_with_provider_metadata():
    result = _analyze(_response(None, finish="length", reasoning=2_000))
    assert not result.success and result.failure_detail == "provider_length"
    assert result.response_length == 0 and result.parser_error == "empty_response"
    assert result.usage["reasoning_tokens"] == 2_000


def test_missing_content_field_is_transport_failure_not_empty_output():
    result = _analyze({"model": "deepseek-flash", "id": "bad-shape"})
    assert not result.success and result.failure_kind == "provider"
    assert result.failure_detail == "transport_shape:transport dict missing content field"
    assert result.parser_status == "not_run" and result.response_length is None


@pytest.mark.parametrize("content", ["", '{"observations": ['])
def test_explicit_length_fails_closed_and_preserves_parser_truth(content):
    result = _analyze(_response(content, finish="length", reasoning=2_000))
    assert not result.success and result.failure_detail == "provider_length"
    assert result.finish_reason == "length" and result.parser_status == "failed"
    assert result.parser_error in ("empty_response", "malformed_json:Expecting value")
    assert result.adjudication is None and result.grounding_report == []


def test_parseable_length_termination_cannot_gain_semantic_acceptance():
    _, _, _, observations = _large_case()
    content = json.dumps({"observations": observations})
    result = _analyze(_response(content, finish="length"))
    assert result.parser_status == "ok" and result.parser_error is None
    assert not result.success and result.failure_detail == "provider_length"
    assert result.adjudication is None and result.grounding_report == []


def test_complete_large_roster_requires_all_exact_grounded_observations():
    _, _, _, observations = _large_case()
    content = json.dumps({"observations": observations})
    result = _analyze(_response(content, finish="stop"))
    assert result.max_tokens == CALL_B_MAX_TOKENS
    assert result.parser_status == "ok" and result.response_length == len(content)
    assert result.success and result.adjudication is not None
    assert len(result.grounding_report) == 50
    assert all(row["status"] == "verified" for row in result.grounding_report)

    missing = _analyze(_response(json.dumps({"observations": observations[:-1]}), finish="stop"))
    assert missing.parser_status == "ok" and not missing.success
    assert missing.failure_kind == "adjudication_invalid"
    assert missing.grounding_report == [] and not missing.any_verified

    bad_quote = [dict(row) for row in observations]
    bad_quote[0] = {**bad_quote[0], "support_quotes": [
        {"evidence_id": "WEB-001", "quote": "A quote absent from the source."}]}
    mismatched = _analyze(_response(json.dumps({"observations": bad_quote}), finish="stop"))
    assert mismatched.parser_status == "ok" and not mismatched.success
    assert mismatched.failure_kind == "quote_binding"


def test_sdk_serializes_large_call_b_budget_and_disabled_thinking():
    import httpx
    from openai import OpenAI

    draft, claims, source, observations = _large_case()
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-sdk-call-b", "object": "chat.completion", "created": 1,
            "model": nodes.DEEPSEEK_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({"observations": observations})}}],
            "usage": {"prompt_tokens": 10_147, "completion_tokens": 5_500,
                      "total_tokens": 15_647,
                      "completion_tokens_details": {"reasoning_tokens": 0}},
        })

    client = OpenAI(api_key="offline-test", base_url="https://example.invalid/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)),
                    max_retries=0)
    result = analyze_semantic_evidence(
        request_id="offline-sdk-call-b", draft_text=draft, claims=claims,
        web_sources=[{"url": "https://fixture.invalid/source", "title": "fixture",
                      "content": source}],
        llm_call=nodes._make_semantic_llm_transport(client), model=nodes.DEEPSEEK_MODEL,
    )
    assert len(requests) == 1 and result.success
    assert requests[0]["max_tokens"] == CALL_B_MAX_TOKENS
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert len(json.loads(requests[0]["messages"][1]["content"])["claims"]) == 50
    assert result.finish_reason == "stop" and result.usage["reasoning_tokens"] == 0


def test_missing_finish_reason_is_unknown_not_length_or_zero():
    _, _, _, observations = _large_case()
    result = _analyze(_response(json.dumps({"observations": observations}), reasoning=None))
    assert result.finish_reason is None and result.usage["reasoning_tokens"] is None
    assert result.parser_status == "ok" and result.success


def test_absent_usage_remains_unknown_in_diagnostics():
    _, _, _, observations = _large_case()
    result = _analyze({"content": json.dumps({"observations": observations}),
                       "model": "deepseek-flash", "id": "offline-no-usage"})
    assert result.success and result.finish_reason is None and result.usage is None
    trace = empty_trace({"run_id": "offline-no-usage"})
    record_semantic_analyzer(trace, iteration=1, request_id="offline-no-usage",
                             success=True, usage=result.usage,
                             finish_reason=result.finish_reason,
                             response_length=result.response_length,
                             parser_status=result.parser_status,
                             max_tokens=result.max_tokens)
    saved = trace["iterations"][0]["semantic_analyzer"]
    assert saved["usage"] is None and saved["termination"] == "unknown"


def test_anchor_ambiguity_from_october_13_remains_a_critical_failure():
    draft = ("Momentum uses a velocity term with gamma around 0.9.\n"
             "Takeaway: Momentum uses a velocity term with gamma around 0.9.")
    row = {"claim_text": "gamma is around 0.9.", "anchor_quote": "with gamma around 0.9.",
           "section": "technical_dive", "claim_type": "factual", "material": True,
           "materiality_reason_code": "core_technical_conclusion",
           "materiality_rationale": "The coefficient is an actionable value.",
           "satisfies_req_ids": [], "specificity": "substantive", "requires_citation": True}
    inventory = build_claim_inventory(run_id="offline", iteration=1, draft_markdown=draft,
                                      raw_claims=[row])
    assert inventory["claims"][0]["anchor_validity"] == ANCHOR_AMBIGUOUS
    assert len(inventory["claims"][0]["occurrences"]) == 2
    assert inventory_critical_failures(inventory) == inventory["claims"]


def test_verify_failure_cannot_reflect_or_reuse_stale_acceptance(base_state, monkeypatch):
    draft = base_state["draft_markdown"]
    anchor = "Gradient descent minimizes a loss function."
    inventory = [{"claim_text": anchor, "anchor_quote": anchor, "section": "technical_dive",
                  "claim_type": "factual", "material": True,
                  "materiality_reason_code": "core_technical_conclusion",
                  "materiality_rationale": "Central proposition", "satisfies_req_ids": [],
                  "specificity": "substantive", "requires_citation": True}]
    client = FakeLLMClient(responses=[fake_response(json.dumps(inventory)),
                                      _response("", finish="length", reasoning=2_000)])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "draft_markdown": draft, "iterations": 1,
             "quality_episode": 1, "quality_attempts": 2,
             "verification_status": "completed", "reflection_score": 10,
             "reflection_provenance": {"origin": "judge", "provider_called": True,
                                       "parse_status": "ok"}}
    result = nodes.verify_node(state)
    merged = {**state, **result}
    assert result["verification_status"] == "parse_failed"
    assert result["grounding_report"] == [] and result["grounding_score"] == 0.0
    assert result["semantic_trace"]["iterations"][0]["semantic_analyzer"]["termination"] == "provider_length"
    assert nodes.route_after_verify(merged) != "reflect"
    from agent import workflow
    assert not workflow.gate1_eligible(merged)
    assert workflow.verification_failure(merged)["terminal_status"] == "execution_failed"
    assert client.calls == 2


def test_incomplete_call_b_roster_has_no_partial_grounding_score(base_state, monkeypatch):
    claims = ["First sourced claim.", "Second sourced claim."]
    inventory = [
        {"claim_text": claim, "anchor_quote": claim, "section": "technical_dive",
         "claim_type": "factual", "material": True,
         "materiality_reason_code": "core_technical_conclusion",
         "materiality_rationale": "Required for the example", "satisfies_req_ids": [],
         "specificity": "substantive", "requires_citation": True}
        for claim in claims
    ]
    response = {"observations": [{
        "claim_id": compute_claim_id(claims[0]),
        "support_quotes": [{"evidence_id": "WEB-001", "quote": claims[0]}],
        "full_entailment": True, "blockers": [],
    }]}
    client = FakeLLMClient(responses=[fake_response(json.dumps(inventory)),
                                      _response(json.dumps(response), finish="stop")])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "draft_markdown": "\n".join(claims), "iterations": 1,
             "web_sources": [{"url": "https://fixture.invalid/source", "title": "fixture",
                              "content": "\n".join(claims), "score": 0.9}],
             "kb_results": []}
    result = nodes.verify_node(state)
    assert result["verification_status"] == "parse_failed"
    assert result["grounding_report"] == [] and result["grounding_score"] == 0.0
    assert not nodes.semantic_verification_accepted({**state, **result})
    assert nodes.route_after_verify({**state, **result}) != "reflect"


def test_call_b_transient_errors_retry_three_times_then_terminate(base_state, monkeypatch):
    anchor = "Gradient descent minimizes a loss function."
    inventory = [{"claim_text": anchor, "anchor_quote": anchor, "section": "technical_dive",
                  "claim_type": "factual", "material": True,
                  "materiality_reason_code": "core_technical_conclusion",
                  "materiality_rationale": "Central proposition", "satisfies_req_ids": [],
                  "specificity": "substantive", "requires_citation": True}]
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return fake_response(json.dumps(inventory))
        raise openai_error(APITimeoutError, 0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "iterations": 1, "quality_episode": 1, "quality_attempts": 1}
    result = nodes.verify_node(state)
    merged = {**state, **result}
    assert len(attempts) == 4  # one Call A, three existing transport attempts for Call B
    assert result["verification_status"] == "verification_error"
    assert result["grounding_report"] == []
    assert nodes.route_after_verify(merged) != "reflect"
    from agent import workflow
    assert workflow.verification_failure(merged)["terminal_status"] == "execution_failed"
