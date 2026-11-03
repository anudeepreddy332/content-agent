"""Offline SDK response contracts for draft accounting."""

import json

import httpx
import pytest
from openai import OpenAI

from agent import nodes, workflow
from config import DEEPSEEK_INPUT_COST_PER_M, DEEPSEEK_OUTPUT_COST_PER_M


@pytest.mark.parametrize("total_field", ["missing", "null"])
def test_draft_accepts_component_usage_when_total_absent(
    base_state, monkeypatch, total_field,
):
    sections = {key: key for key in (
        "problem_framing", "technical_dive", "code_snippets", "takeaways",
    )}
    requests = []

    def respond(request):
        requests.append(request)
        usage = {"prompt_tokens": 37, "completion_tokens": 63}
        if total_field == "null":
            usage["total_tokens"] = None
        return httpx.Response(200, json={
            "id": "offline-draft", "object": "chat.completion", "created": 1,
            "model": nodes.DEEPSEEK_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(sections),
            }}],
            "usage": usage,
        })

    client = OpenAI(
        api_key="offline-only", base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "quality_episode": 1, "quality_attempts": 0,
             "terminal_status": None, "error_log": []}

    update = workflow.draft_step(state, run=nodes.draft_node)

    assert update["draft_status"] == "valid"
    assert update.get("terminal_status") is None
    assert update["draft_markdown"]
    assert update["total_tokens"] == 100
    expected_cost = (37 * DEEPSEEK_INPUT_COST_PER_M
                     + 63 * DEEPSEEK_OUTPUT_COST_PER_M) / 1_000_000
    assert update["total_cost_usd"] == pytest.approx(expected_cost)
    diagnostic = update["semantic_trace"]["iterations"][0]["output_diagnostics"][0]
    assert diagnostic["usage"]["total_tokens"] is None  # provider-reported field
    assert diagnostic["usage"]["accounted_total_tokens"] == 100
    assert diagnostic["usage"]["total_tokens_source"] == "component_sum"
    assert nodes.route_after_draft({**state, **update}) == "verify"
    assert len(requests) == 1  # no application or SDK retry


@pytest.mark.parametrize("usage,content", [
    (None, json.dumps({key: key for key in (
        "problem_framing", "technical_dive", "code_snippets", "takeaways",
    )})),
    ({"prompt_tokens": 37, "completion_tokens": None, "total_tokens": None},
     json.dumps({key: key for key in (
         "problem_framing", "technical_dive", "code_snippets", "takeaways",
     )})),
    ({"prompt_tokens": 37, "completion_tokens": 63, "total_tokens": 100}, None),
])
def test_incomplete_response_fails_closed_without_retry(
    base_state, monkeypatch, usage, content,
):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-draft", "object": "chat.completion", "created": 1,
            "model": nodes.DEEPSEEK_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content,
            }}],
            "usage": usage,
        })

    client = OpenAI(
        api_key="offline-only", base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "quality_episode": 1, "quality_attempts": 0,
             "terminal_status": None, "error_log": []}
    update = workflow.draft_step(state, run=nodes.draft_node)

    assert update["terminal_status"] == "execution_failed"
    assert update["draft_status"] == "invalid"
    assert "draft_sections" not in update
    assert len(requests) == 1
    failure = json.loads(update["error_log"][-1].removeprefix("[draft] "))
    assert failure["failure_phase"] == "provider_response_handling"
    assert failure["source_file"] == "agent/nodes.py"
