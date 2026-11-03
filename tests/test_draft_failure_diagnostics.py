"""Offline draft failure observability without provider content or secrets."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from agent import graph, nodes, workflow
from main import _build_initial_state, _write_telemetry
from tests.conftest import FakeLLMClient, fake_response


SECRET = "sk-private-Authorization-Bearer-prompt"
SECTIONS = {key: key for key in (
    "problem_framing", "technical_dive", "code_snippets", "takeaways",
)}


def _raise_secret(*args, **kwargs):
    raise RuntimeError(SECRET)


@pytest.mark.parametrize("phase,client_calls", [
    ("client_prompt_initialization", 0),
    ("source_context_construction", 0),
    ("provider_request", 1),
    ("provider_response_handling", 1),
    ("json_parsing_validation", 1),
    ("post_response_accounting_telemetry", 1),
])
def test_draft_exception_phase_is_safe_and_fail_closed(
    base_state, monkeypatch, capsys, phase, client_calls,
):
    client = FakeLLMClient(response=fake_response(json.dumps(SECTIONS)))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    if phase == "client_prompt_initialization":
        monkeypatch.setattr(nodes, "_get_client", _raise_secret)
    elif phase == "source_context_construction":
        monkeypatch.setattr(nodes, "_build_web_source_context", _raise_secret)
    elif phase == "provider_request":
        client = FakeLLMClient(errors=[RuntimeError(SECRET)])
        monkeypatch.setattr(nodes, "_get_client", lambda: client)
    elif phase == "provider_response_handling":
        monkeypatch.setattr(nodes, "new_output_diagnostic", _raise_secret)
    elif phase == "json_parsing_validation":
        monkeypatch.setattr(nodes, "json", SimpleNamespace(
            loads=_raise_secret, JSONDecodeError=json.JSONDecodeError,
        ))
    else:
        monkeypatch.setattr(nodes, "record_draft", _raise_secret)

    state = {**base_state, "quality_episode": 1, "quality_attempts": 0,
             "terminal_status": None, "error_log": []}
    result = workflow.draft_step(state, run=nodes.draft_node)
    diagnostic = json.loads(result["error_log"][-1].removeprefix("[draft] "))

    assert result["terminal_status"] == "execution_failed"
    assert result["draft_status"] == "invalid"
    assert result["quality_attempts"] == 1
    assert "draft_sections" not in result
    assert client.calls == client_calls
    assert diagnostic["run_id"] == state["run_id"]
    assert diagnostic["stage"] == "draft"
    assert diagnostic["failure_phase"] == phase
    assert diagnostic["exception_class"] == "RuntimeError"
    assert diagnostic["reason"] == workflow._DRAFT_FAILURE_REASONS[phase]
    assert diagnostic["timestamp"].endswith("+00:00")
    captured = capsys.readouterr()
    assert SECRET not in str(result) + captured.out + captured.err


def test_failure_diagnostic_survives_sqlite_restart_and_telemetry(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(graph, "retrieve_node", lambda state: {})
    monkeypatch.setattr(nodes, "_get_client", _raise_secret)
    run_id = "draft-failure-offline"
    state = _build_initial_state("Gradient Descent", "gradient-descent",
                                 "standalone", "Learning Log", run_id)
    connection = sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False)
    saver = SqliteSaver(connection)
    current = graph.build_graph(checkpointer=saver)
    result = current.invoke(state, {"configurable": {"thread_id": run_id}})
    assert result["terminal_status"] == "execution_failed"
    assert len(result["error_log"]) == 1

    recovered = graph.build_graph(checkpointer=saver).get_state(
        {"configurable": {"thread_id": run_id}}
    ).values
    assert recovered["error_log"] == result["error_log"]
    telemetry = _write_telemetry(recovered)
    saved = json.loads(telemetry.read_text())
    assert saved["error_log"] == result["error_log"]
    assert SECRET not in telemetry.read_text()
    assert recovered["draft_status"] == "invalid"
    assert recovered["verification_status"] == "not_started"
    assert recovered["hitl_status"] == "pending"
    connection.close()


def test_valid_draft_has_no_failure_diagnostic(base_state, monkeypatch):
    client = FakeLLMClient(response=fake_response(json.dumps(SECTIONS)))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    state = {**base_state, "quality_episode": 1, "quality_attempts": 0,
             "terminal_status": None, "error_log": []}
    result = workflow.draft_step(state, run=nodes.draft_node)
    assert result["draft_status"] == "valid"
    assert result.get("terminal_status") is None
    assert result.get("error_log", []) == []
    assert client.calls == 1
