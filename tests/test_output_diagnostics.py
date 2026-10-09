"""Metadata-only structured-output diagnostics; all provider responses are fake."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import agent.graph as graph_mod
import agent.nodes as nodes
from agent.semantic_trace import (
    empty_trace, finish_output_diagnostic, new_output_diagnostic,
    record_output_diagnostic, sha256_utf8,
)
from main import _write_telemetry
from tests.conftest import FakeLLMClient, fake_response
from tests.workflow_fixtures import current_quality


def _response(content, *, finish=None, details=None, reasoning=None, with_reasoning=False):
    response = fake_response(content, tokens=8000)
    response.model = "returned-model"
    response.id = "completion-1"
    if finish is not None:
        response.choices[0].finish_reason = finish
    if with_reasoning:
        response.choices[0].message.reasoning_content = reasoning
    response.usage.prompt_tokens = 4000
    response.usage.completion_tokens = 4000
    response.usage.total_tokens = 8000
    if details is not None:
        response.usage.completion_tokens_details = SimpleNamespace(**details)
    return response


def _diag(response, *, stage="draft", original="x"):
    response.choices[0].message.content = original
    return new_output_diagnostic(
        stage=stage, iteration=1, quality_episode=2, quality_attempt=1,
        requested_model="requested-model", temperature=0.7,
        max_tokens=4000, response_format=None, response=response,
    )


def _records(state):
    return [d for slot in state["semantic_trace"]["iterations"]
            for d in slot.get("output_diagnostics", [])]


def test_finish_reason_length_stop_and_missing_usage_ceiling():
    length = _diag(_response("x", finish="length"))
    stop = _diag(_response("x", finish="stop"))
    missing = _diag(_response("x"))
    assert length["response"]["termination"] == "provider_length"
    assert stop["response"]["termination"] == "stop"
    assert missing["response"] == {
        "model": "returned-model", "id": "completion-1",
        "finish_reason": None, "termination": "unknown",
    }
    assert missing["usage"]["completion_tokens"] == missing["request"]["max_tokens"]


def test_usage_reasoning_and_unavailable_optional_fields():
    full = _diag(_response("x", details={"reasoning_tokens": 4000,
                                     "accepted_prediction_tokens": 2},
                           reasoning="private rationale", with_reasoning=True))
    assert full["usage"]["completion_tokens"] == 4000
    assert full["usage"]["reasoning_tokens"] == 4000
    assert full["usage"]["completion_token_details"]["accepted_prediction_tokens"] == 2
    assert full["reasoning_content"] == {"present": True, "length": len("private rationale")}
    assert "private rationale" not in json.dumps(full)
    absent = _diag(fake_response("x"))
    assert absent["response"]["model"] is None
    assert absent["response"]["id"] is None
    assert absent["usage"]["completion_token_details"] is None
    assert absent["usage"]["reasoning_tokens"] is None
    assert absent["reasoning_content"] == {"present": None, "length": None}


def test_content_metadata_and_sanitized_parse_failure():
    original = "  ```json\n{broken\n```  "
    parser_input = "{broken"
    diagnostic = _diag(_response(original), original=original)
    finish_output_diagnostic(diagnostic, parser_input, ValueError("secret error text"))
    assert diagnostic["content"] == {
        "original": {"length": len(original), "sha256": sha256_utf8(original)},
        "parser_input": {"length": len(parser_input), "sha256": sha256_utf8(parser_input)},
    }
    assert diagnostic["parse"] == {"status": "failed", "error_class": "ValueError"}
    assert original not in json.dumps(diagnostic)
    assert parser_input not in json.dumps(diagnostic)
    assert "secret error text" not in json.dumps(diagnostic)


def test_draft_success_failure_fences_request_neutrality(base_state, monkeypatch):
    content = json.dumps({key: key for key in (
        "problem_framing", "technical_dive", "code_snippets", "takeaways")})
    fenced = " \n```json\n" + content + "\n``` \n"
    client = FakeLLMClient(response=_response(fenced, finish="stop"))
    calls = []
    create = client.chat.completions.create

    def capture(**kwargs):
        calls.append(kwargs)
        return create(**kwargs)

    client.chat.completions.create = capture
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.draft_node(base_state)
    diagnostic = _records(result)[0]
    assert result["draft_status"] == "valid" and client.calls == 1
    assert diagnostic["parse"] == {"status": "ok", "error_class": None}
    assert diagnostic["content"]["original"]["sha256"] == sha256_utf8(fenced)
    assert diagnostic["content"]["parser_input"]["sha256"] == sha256_utf8(content)
    assert diagnostic["request"] == {"model": nodes.DEEPSEEK_MODEL,
                                     "temperature": nodes.DRAFT_TEMPERATURE,
                                     "max_tokens": 4000, "response_format": None}
    assert {k: v for k, v in calls[0].items() if k != "messages"} == {
        "model": nodes.DEEPSEEK_MODEL, "temperature": nodes.DRAFT_TEMPERATURE,
        "max_tokens": 4000, "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert len(calls[0]["messages"]) == 2 and "response_format" not in calls[0]

    broken = "{broken"
    failure_client = FakeLLMClient(response=_response(broken))
    monkeypatch.setattr(nodes, "_get_client", lambda: failure_client)
    failed = nodes.draft_node(base_state)
    assert failed["draft_status"] == "invalid" and failure_client.calls == 1
    assert failed["draft_sections"]["technical_dive"] == broken
    assert _records(failed)[0]["parse"] == {
        "status": "failed", "error_class": "JSONDecodeError"}


def test_openai_client_serializes_draft_thinking_toggle(base_state, monkeypatch):
    import httpx
    from openai import OpenAI

    requests = []
    content = json.dumps({key: key for key in (
        "problem_framing", "technical_dive", "code_snippets", "takeaways")})

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-draft", "object": "chat.completion", "created": 1,
            "model": nodes.DEEPSEEK_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 20, "total_tokens": 32},
        })

    transport = httpx.MockTransport(respond)
    client = OpenAI(api_key="offline-test", base_url="https://example.invalid/v1",
                    http_client=httpx.Client(transport=transport), max_retries=0)
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.draft_node(base_state)

    assert result["draft_status"] == "valid"
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["model"] == nodes.DEEPSEEK_MODEL
    assert requests[0]["temperature"] == nodes.DRAFT_TEMPERATURE
    assert requests[0]["max_tokens"] == 4000
    assert "response_format" not in requests[0]


def test_length_terminated_invalid_drafts_use_two_attempts_without_review(base_state, monkeypatch):
    client = FakeLLMClient(responses=[
        _response("", finish="length", details={"reasoning_tokens": 4000}),
        _response("{broken", finish="length", details={"reasoning_tokens": 2388}),
    ])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    monkeypatch.setattr(graph_mod, "retrieve_node", lambda state: {})
    verify = Mock(side_effect=AssertionError("invalid draft reached Verify"))
    review = Mock(side_effect=AssertionError("invalid draft reached Gate 1"))
    monkeypatch.setattr(graph_mod, "verify_node", verify)
    monkeypatch.setattr(graph_mod, "hitl_node", review)

    result = graph_mod.build_graph().invoke(base_state)
    diagnostics = _records(result)
    assert client.calls == 2
    assert result["quality_attempts"] == result["iterations"] == 2
    assert result["terminal_status"] == "execution_failed"
    assert result["verification_status"] == "not_started"
    assert [d["response"]["termination"] for d in diagnostics] == [
        "provider_length", "provider_length"]
    assert [d["parse"]["error_class"] for d in diagnostics] == [
        "JSONDecodeError", "JSONDecodeError"]
    verify.assert_not_called()
    review.assert_not_called()


def test_length_terminated_complete_json_cannot_reach_verify(base_state, monkeypatch):
    content = json.dumps({key: key for key in (
        "problem_framing", "technical_dive", "code_snippets", "takeaways")})
    client = FakeLLMClient(response=_response(content, finish="length"))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "quality_episode": 1, "quality_attempts": 0,
             "terminal_status": None}

    update = nodes.workflow.draft_step(state, run=nodes.draft_node)
    diagnostic = _records(update)[0]
    assert diagnostic["response"]["termination"] == "provider_length"
    assert diagnostic["parse"] == {"status": "ok", "error_class": None}
    assert update["draft_status"] == "invalid"
    assert nodes.route_after_draft({**state, **update}) != "verify"


@pytest.mark.parametrize("finish,termination", [
    ("stop", "stop"), (None, "unknown"),
])
def test_complete_json_without_length_still_reaches_verify(
    base_state, monkeypatch, finish, termination,
):
    content = json.dumps({key: key for key in (
        "problem_framing", "technical_dive", "code_snippets", "takeaways")})
    client = FakeLLMClient(response=_response(content, finish=finish))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "quality_episode": 1, "quality_attempts": 0,
             "terminal_status": None}

    update = nodes.workflow.draft_step(state, run=nodes.draft_node)
    diagnostic = _records(update)[0]
    assert diagnostic["response"]["termination"] == termination
    assert diagnostic["parse"] == {"status": "ok", "error_class": None}
    assert update["draft_status"] == "valid"
    assert nodes.route_after_draft({**state, **update}) == "verify"


def test_two_complete_length_responses_exhaust_budget_without_review(base_state, monkeypatch):
    content = json.dumps({key: key for key in (
        "problem_framing", "technical_dive", "code_snippets", "takeaways")})
    client = FakeLLMClient(responses=[
        _response(content, finish="length"), _response(content, finish="length"),
    ])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    monkeypatch.setattr(graph_mod, "retrieve_node", lambda state: {})
    verify = Mock(side_effect=AssertionError("length-terminated draft reached Verify"))
    review = Mock(side_effect=AssertionError("length-terminated draft reached Gate 1"))
    monkeypatch.setattr(graph_mod, "verify_node", verify)
    monkeypatch.setattr(graph_mod, "hitl_node", review)

    result = graph_mod.build_graph().invoke(base_state)
    diagnostics = _records(result)
    assert client.calls == result["quality_attempts"] == result["iterations"] == 2
    assert result["terminal_status"] == "execution_failed"
    assert result["verification_status"] == "not_started"
    assert [d["response"]["termination"] for d in diagnostics] == [
        "provider_length", "provider_length"]
    assert [d["parse"] for d in diagnostics] == [
        {"status": "ok", "error_class": None}] * 2
    verify.assert_not_called()
    review.assert_not_called()


def test_inventory_october_pattern_and_success(base_state, monkeypatch):
    state = {**base_state, "iterations": 1, "quality_episode": 2, "quality_attempts": 1}
    response = _response("", details={"reasoning_tokens": 4000})
    client = FakeLLMClient(response=response)
    calls = []
    create = client.chat.completions.create

    def capture(**kwargs):
        calls.append(kwargs)
        return create(**kwargs)

    client.chat.completions.create = capture
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    failed = nodes.verify_node(state)
    diagnostic = _records(failed)[0]
    assert failed["verification_status"] == "parse_failed" and client.calls == 1
    assert diagnostic["stage"] == "claim_inventory"
    assert diagnostic["response"]["termination"] == "unknown"
    assert diagnostic["usage"]["completion_tokens"] == 4000
    assert diagnostic["usage"]["reasoning_tokens"] == 4000
    assert diagnostic["content"]["original"] == {"length": 0, "sha256": sha256_utf8("")}
    assert diagnostic["parse"] == {"status": "failed", "error_class": "ClaimInventoryError"}
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}

    client = FakeLLMClient(response=fake_response("[]"))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    parsed = nodes.verify_node(state)
    assert parsed["verification_status"] == "inventory_failed" and client.calls == 1
    assert _records(parsed)[0]["parse"] == {"status": "ok", "error_class": None}


def _inventory_row():
    claim = "Gradient descent minimizes a loss function."
    return {
        "claim_text": claim, "anchor_quote": claim, "section": "technical_dive",
        "claim_type": "factual", "material": True,
        "materiality_reason_code": "core_technical_conclusion",
        "materiality_rationale": "fixture", "satisfies_req_ids": [],
        "specificity": "substantive", "requires_citation": None,
    }


def test_openai_client_serializes_inventory_thinking_toggle(base_state, monkeypatch):
    import httpx
    from openai import OpenAI

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-inventory", "object": "chat.completion", "created": 1,
            "model": nodes.DEEPSEEK_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "[]"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        })

    client = OpenAI(api_key="offline-test", base_url="https://example.invalid/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)),
                    max_retries=0)
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "iterations": 1})
    assert result["verification_status"] == "inventory_failed"
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["model"] == nodes.DEEPSEEK_MODEL
    assert requests[0]["temperature"] == 0.1
    assert requests[0]["max_tokens"] == 4000
    assert "response_format" not in requests[0]


@pytest.mark.parametrize("finish,expected_termination", [
    ("stop", "stop"), (None, "unknown"),
])
def test_complete_inventory_without_length_reaches_call_b(
    base_state, monkeypatch, finish, expected_termination,
):
    content = json.dumps([_inventory_row()])
    client = FakeLLMClient(response=_response(content, finish=finish))
    calls = []
    create = client.chat.completions.create

    def capture(**kwargs):
        calls.append(kwargs)
        return create(**kwargs)

    client.chat.completions.create = capture
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "iterations": 1})
    diagnostic = _records(result)[0]
    assert result["verification_status"] == "completed"
    assert result["claim_inventory"]["claims"]
    assert client.calls == 2
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "extra_body" not in calls[1]
    assert diagnostic["response"]["termination"] == expected_termination
    assert diagnostic["parse"] == {"status": "ok", "error_class": None}


def test_length_terminated_parseable_inventory_is_rejected_before_call_b(
    base_state, monkeypatch, tmp_path,
):
    content = json.dumps([_inventory_row()])
    client = FakeLLMClient(response=_response(content, finish="length"))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "iterations": 1, "quality_episode": 1, "quality_attempts": 1}
    result = nodes.verify_node(state)
    diagnostic = _records(result)[0]
    slot = result["semantic_trace"]["iterations"][0]
    assert client.calls == 1
    assert result["verification_status"] == "inventory_failed"
    assert result["claim_inventory"] is None
    assert slot["verifier_raw"]["parser_status"] == "ok"
    assert slot["verifier_raw"]["parse_error"] is None
    assert slot["verifier_raw"]["pre_dedup_rows"] == [_inventory_row()]
    assert diagnostic["parse"] == {"status": "ok", "error_class": None}
    assert diagnostic["response"]["termination"] == "provider_length"
    assert diagnostic["content"]["original"]["sha256"] == sha256_utf8(content)
    assert nodes.route_after_verify({**state, **result}) != "reflect"
    monkeypatch.chdir(tmp_path)
    loaded = json.loads(_write_telemetry({**state, **result}).read_text())
    persisted = loaded["semantic_trace_v1"]["iterations"][0]
    assert persisted["output_diagnostics"][0] == diagnostic
    assert persisted["verifier_raw"]["parser_status"] == "ok"
    assert "extra_body" not in json.dumps(diagnostic)


@pytest.mark.parametrize("content,error_class", [
    ("", "ClaimInventoryError"), ("{broken", "ClaimInventoryError"),
])
def test_length_terminated_unparseable_inventory_preserves_parse_failure(
    base_state, monkeypatch, content, error_class,
):
    client = FakeLLMClient(response=_response(content, finish="length"))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node({**base_state, "iterations": 1})
    diagnostic = _records(result)[0]
    assert result["verification_status"] == "parse_failed"
    assert result["claim_inventory"] is None and client.calls == 1
    assert diagnostic["response"]["termination"] == "provider_length"
    assert diagnostic["parse"] == {"status": "failed", "error_class": error_class}


def test_two_length_terminated_inventories_exhaust_without_review(base_state, monkeypatch):
    import agent.graph as graph_mod

    content = json.dumps([_inventory_row()])
    client = FakeLLMClient(responses=[
        _response(content, finish="length"), _response(content, finish="length"),
    ])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    monkeypatch.setattr(graph_mod, "retrieve_node", lambda state: {})
    monkeypatch.setattr(graph_mod, "draft_node", lambda state: {
        "draft_markdown": base_state["draft_markdown"], "draft_status": "valid",
        "draft_sections": {key: key for key in (
            "problem_framing", "technical_dive", "code_snippets", "takeaways")},
    })
    reflect = Mock(side_effect=AssertionError("failed inventory reached reflection"))
    review = Mock(side_effect=AssertionError("failed inventory reached Gate 1"))
    monkeypatch.setattr(graph_mod, "reflect_node", reflect)
    monkeypatch.setattr(graph_mod, "hitl_node", review)
    result = graph_mod.build_graph().invoke(base_state)
    assert client.calls == result["quality_attempts"] == 2
    assert result["terminal_status"] == "execution_failed"
    assert result["verification_status"] == "inventory_failed"
    assert result["claim_inventory"] is None
    assert [d["parse"]["status"] for d in _records(result)] == ["ok", "ok"]
    reflect.assert_not_called()
    review.assert_not_called()


def test_reflection_success_failure_and_stage_identity(base_state, monkeypatch):
    state = {**base_state, "iterations": 1, "verification_status": "completed",
             "quality_episode": 3, "quality_attempts": 2,
             "grounding_report": [{"claim": "fact", "status": "verified", "blockers": []}]}
    state = current_quality(state, build_inventory=True)
    state["quality_episode"], state["quality_attempts"] = 3, 2
    state["verification_identity"] = nodes.workflow.identity(state)
    client = FakeLLMClient(response=_response('{"score": 8, "notes": "ok"}', finish="stop"))
    calls = []
    create = client.chat.completions.create

    def capture(**kwargs):
        calls.append(kwargs)
        return create(**kwargs)

    client.chat.completions.create = capture
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    ok = nodes.reflect_node(state)
    assert ok["reflection_score"] == 8 and client.calls == 1
    diagnostic = _records(ok)[0]
    assert (diagnostic["stage"], diagnostic["quality_episode"],
            diagnostic["quality_attempt"]) == ("reflection", 3, 2)
    assert diagnostic["parse"]["status"] == "ok"
    assert "extra_body" not in calls[0]

    client = FakeLLMClient(response=_response("not JSON"))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    failed = nodes.reflect_node(state)
    assert failed["reflection_provenance"]["parse_status"] == "failed"
    assert failed["reflection_score"] == 7 and client.calls == 1
    assert _records(failed)[0]["parse"]["error_class"] == "JSONDecodeError"


def test_distinct_attempts_roundtrip_and_no_new_raw_fields(base_state, monkeypatch, tmp_path):
    first = _diag(_response("first-secret-response"), original="first-secret-response")
    finish_output_diagnostic(first, "first-secret-response")
    second = _diag(_response("second-secret-response"), stage="claim_inventory",
                   original="second-secret-response")
    second["iteration"] = 2
    finish_output_diagnostic(second, "second-secret-response")
    trace = empty_trace(base_state)
    record_output_diagnostic(trace, iteration=1, diagnostic=first)
    same_attempt_reflection = _diag(_response("reflection-secret"), stage="reflection",
                                    original="reflection-secret")
    finish_output_diagnostic(same_attempt_reflection, "reflection-secret")
    record_output_diagnostic(trace, iteration=1, diagnostic=same_attempt_reflection)
    record_output_diagnostic(trace, iteration=2, diagnostic=second)
    assert [d["stage"] for d in trace["iterations"][0]["output_diagnostics"]] == [
        "draft", "reflection"]
    assert trace["iterations"][1]["output_diagnostics"][0]["stage"] == "claim_inventory"
    state = {**deepcopy(base_state), "semantic_trace": trace}
    monkeypatch.chdir(tmp_path)
    loaded = json.loads(_write_telemetry(state).read_text())
    diagnostics = [d for slot in loaded["semantic_trace_v1"]["iterations"]
                   for d in slot["output_diagnostics"]]
    assert diagnostics == [first, same_attempt_reflection, second]
    # The same metadata-only state is what a graph checkpointer serializes.
    checkpoint_blob = json.dumps(state["semantic_trace"])
    diagnostic_blob = json.dumps(diagnostics)
    for secret in ("first-secret-response", "reflection-secret", "second-secret-response"):
        assert secret not in checkpoint_blob and secret not in diagnostic_blob
    assert set(first) == {"schema", "stage", "iteration", "quality_episode",
                          "quality_attempt", "request", "response", "usage",
                          "content", "parse", "reasoning_content"}


def test_diagnostics_are_not_projected_into_poll_or_sse(base_state, monkeypatch):
    import api.server as srv

    diagnostic = _diag(_response("private-response"), original="private-response")
    finish_output_diagnostic(diagnostic, "private-response")
    trace = empty_trace(base_state)
    record_output_diagnostic(trace, iteration=1, diagnostic=diagnostic)
    delta = {"semantic_trace": trace, "iterations": 1}
    assert srv._node_headline("draft", delta) == {"iterations": 1}
    run_id = "metadata-public-projection"
    monkeypatch.setitem(srv.REGISTRY, run_id, {
        "status": "completed", "error": None, "result": {**base_state, **delta},
    })
    poll = srv.get_run(run_id)
    assert "semantic_trace" not in json.dumps(poll)
    assert "output_diagnostics" not in json.dumps(poll)
    assert "private-response" not in json.dumps(poll)
