"""Rejected provider drafts must leave only safe metadata in durable outputs."""

import json
import sqlite3
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from agent import graph, nodes, workflow
from agent.semantic_trace import sha256_utf8
from main import _build_initial_state, _write_telemetry
from tests.conftest import FakeLLMClient, fake_response


CANARY = "sk-private-Authorization-Bearer-prompt"
SECTIONS = ("problem_framing", "technical_dive", "code_snippets", "takeaways")


@pytest.mark.parametrize("content,finish", [
    ("{broken " + CANARY, "stop"),
    (json.dumps({"problem_framing": CANARY}), "stop"),
    (json.dumps({"problem_framing": CANARY, "technical_dive": 42,
                 "code_snippets": "code", "takeaways": "takeaway"}), "stop"),
    (json.dumps({key: CANARY for key in SECTIONS}), "length"),
])
def test_rejected_draft_keeps_only_safe_metadata(
    base_state, monkeypatch, capfd, content, finish,
):
    response = fake_response(content)
    response.choices[0].finish_reason = finish
    client = FakeLLMClient(response=response)
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    update = workflow.draft_step(base_state, run=nodes.draft_node)
    output = capfd.readouterr()

    assert update["draft_status"] == "invalid"
    assert nodes.route_after_draft({**base_state, **update}) != "verify"
    assert CANARY not in json.dumps(update) + output.out + output.err
    assert "raw_preview" not in output.out + output.err
    assert client.calls == 1
    diagnostic = update["semantic_trace"]["iterations"][-1]["output_diagnostics"][-1]
    assert diagnostic["content"]["original"] == {
        "length": len(content), "sha256": sha256_utf8(content),
    }
    assert diagnostic["response"]["finish_reason"] == finish
    assert diagnostic["usage"]["total_tokens"] == 100
    assert diagnostic["rejection"]["failure_phase"] == (
        "provider_response_handling" if finish == "length" else "json_parsing_validation"
    )
    assert diagnostic["rejection"]["exception_class"] == (
        None if finish == "length" else "JSONDecodeError" if content.startswith("{broken") else "ValueError"
    )
    assert diagnostic["rejection"]["run_id"] == base_state["run_id"]
    assert diagnostic["rejection"]["timestamp"].endswith("+00:00")


def test_valid_draft_preserves_full_content_and_verification_route(base_state, monkeypatch):
    content = json.dumps({key: CANARY + key for key in SECTIONS})
    response = fake_response(content)
    response.choices[0].finish_reason = "stop"
    client = FakeLLMClient(response=response)
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    update = workflow.draft_step(base_state, run=nodes.draft_node)
    assert update["draft_status"] == "valid"
    assert CANARY in update["draft_markdown"]
    slot = update["semantic_trace"]["iterations"][-1]
    assert slot["draft"]["draft_markdown"] == update["draft_markdown"]
    assert nodes.route_after_draft({**base_state, **update}) == "verify"
    assert client.calls == 1


def test_rejected_draft_has_no_canary_in_logs_checkpoint_or_telemetry(
    tmp_path, monkeypatch, capfd,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nodes, "_load_system_prompt", lambda: "offline prompt")
    monkeypatch.setattr(graph, "retrieve_node", lambda state: {})
    verify = Mock(side_effect=AssertionError("invalid draft reached verification"))
    monkeypatch.setattr(graph, "verify_node", verify)
    response = fake_response("{broken " + CANARY)
    response.choices[0].finish_reason = "stop"
    client = FakeLLMClient(response=response)
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    run_id = "invalid-draft-canary-offline"
    state = _build_initial_state("Gradient Descent", "gradient-descent",
                                 "standalone", "Learning Log", run_id)
    connection = sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False)
    saver = SqliteSaver(connection)
    result = graph.build_graph(checkpointer=saver).invoke(
        state, {"configurable": {"thread_id": run_id}},
    )
    logs = capfd.readouterr()
    recovered = graph.build_graph(checkpointer=saver).get_state(
        {"configurable": {"thread_id": run_id}},
    ).values
    telemetry = _write_telemetry(recovered)

    assert result["terminal_status"] == "execution_failed"
    assert recovered["draft_status"] == "invalid"
    assert client.calls == 2  # frozen quality-episode budget
    verify.assert_not_called()
    assert CANARY not in logs.out + logs.err
    assert "raw_preview" not in logs.out + logs.err
    assert CANARY not in json.dumps(recovered)
    assert CANARY.encode() not in (tmp_path / "checkpoints.sqlite").read_bytes()
    assert CANARY not in telemetry.read_text()
    connection.close()


@pytest.mark.parametrize("content", [None, ""])
def test_empty_response_keeps_safe_metadata(
    base_state, monkeypatch, content,
):
    response = fake_response(content)
    response.choices[0].finish_reason = "stop"
    client = FakeLLMClient(response=response)
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    update = workflow.draft_step(base_state, run=nodes.draft_node)
    if content is None:
        assert update["terminal_status"] == "execution_failed"
    else:
        assert update["draft_status"] == "invalid"
    if content is None:
        entry = json.loads(update["error_log"][-1].removeprefix("[draft] "))
        assert entry["response_content"] == {"length": None, "sha256": None}
        assert entry["finish_reason"] == "stop"
        assert entry["usage"]["total_tokens"] == 100
    assert client.calls == 1


def test_post_response_exception_retains_only_safe_metadata(base_state, monkeypatch):
    content = json.dumps({key: CANARY for key in SECTIONS})
    response = fake_response(content)
    response.choices[0].finish_reason = "stop"
    client = FakeLLMClient(response=response)
    monkeypatch.setattr(nodes, "_get_client", lambda: client)

    def fail_record(*args, **kwargs):
        raise RuntimeError(CANARY)

    monkeypatch.setattr(nodes, "record_draft", fail_record)
    update = workflow.draft_step(base_state, run=nodes.draft_node)
    entry = json.loads(update["error_log"][-1].removeprefix("[draft] "))
    assert update["terminal_status"] == "execution_failed"
    assert entry["failure_phase"] == "post_response_accounting_telemetry"
    assert entry["response_content"] == {
        "length": len(content), "sha256": sha256_utf8(content),
    }
    assert entry["finish_reason"] == "stop"
    assert entry["usage"]["total_tokens"] == 100
    assert CANARY not in json.dumps(update)
    assert client.calls == 1
