"""Offline proof of the one-observation Call B recovery boundary."""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agent import call_b_journal, nodes, workflow
from agent.claim_inventory import compute_claim_id
from tests.conftest import FakeLLMClient, fake_response


CLAIM = "Gradient descent minimizes a loss function."
SOURCE = f"Evidence states: {CLAIM} A small step follows the negative gradient."


def _inventory() -> str:
    return json.dumps([{
        "claim_text": CLAIM, "anchor_quote": CLAIM,
        "section": "technical_dive", "claim_type": "factual",
        "material": True, "materiality_reason_code": "core_technical_conclusion",
        "materiality_rationale": "central proposition", "satisfies_req_ids": [],
        "specificity": "substantive", "requires_citation": True,
    }])


def _observation(*, quote: str = CLAIM, entailed: bool = True) -> str:
    return json.dumps({"observations": [{
        "claim_id": compute_claim_id(CLAIM),
        "support_quotes": ([{"evidence_id": "WEB-001", "quote": quote}]
                           if quote else []),
        "full_entailment": entailed, "blockers": [],
    }]})


def _state(base_state, *, run_id="durable-retry", iteration=1):
    return {**base_state, "run_id": run_id, "iterations": iteration,
            "quality_attempts": 1, "quality_episode": 1,
            "draft_status": "valid",
            "draft_markdown": CLAIM,
            "draft_sections": {key: CLAIM for key in (
                "problem_framing", "technical_dive", "code_snippets", "takeaways")},
            "web_sources": [{"title": "fixture", "url": "https://fixture.invalid/source",
                             "content": SOURCE, "score": 0.9}],
            "kb_results": []}


def _client(monkeypatch, *call_b_responses):
    client = FakeLLMClient(responses=[fake_response(_inventory()),
                                      *(fake_response(x) for x in call_b_responses)])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    return client


def test_quote_not_found_recovers_once_on_same_draft_and_roster(base_state, monkeypatch):
    client = _client(monkeypatch, _observation(quote="fabricated quote"), _observation())
    state = _state(base_state)
    result = nodes.verify_node(state)
    assert client.calls == 3  # Call A, original Call B, one additional Call B
    assert result["verification_status"] == "completed"
    assert result["grounding_report"][0]["status"] == "verified"
    assert result["total_tokens"] == 300
    assert result["provider_usage_status"] == "known"
    attempts = result["semantic_trace"]["call_b_observation_attempts"]
    assert [x["attempt"] for x in attempts] == ["initial", "retry"]
    assert attempts[0]["quote_failure"]["evidence_id"] == "WEB-001"
    assert attempts[0]["quote_failure"]["exact_match_count"] == 0
    assert "fabricated quote" not in json.dumps(result)
    assert b"fabricated quote" not in call_b_journal._path().read_bytes()
    assert call_b_journal.lookup(state["run_id"], "retry")["state"] == "responded"


def test_invalid_retry_fails_closed_without_redraft(base_state, monkeypatch):
    client = _client(monkeypatch, _observation(quote="bad 1"),
                     _observation(quote="bad 2"))
    state = _state(base_state)
    result = workflow.verify_step(state, run=nodes.verify_node)
    assert client.calls == 3
    assert result["terminal_status"] == "execution_failed"
    assert result["verification_status"] == "verification_error"
    assert result["grounding_report"] == []
    assert result["total_tokens"] == 300
    assert "quality_attempts" not in result and state["quality_attempts"] == 1


def test_valid_negative_judgment_is_not_rerolled(base_state, monkeypatch):
    client = _client(monkeypatch, _observation(quote="", entailed=False))
    state = _state(base_state)
    result = workflow.verify_step(state, run=nodes.verify_node)
    assert client.calls == 2
    assert call_b_journal.lookup(state["run_id"], "retry") is None
    assert result["grounding_report"][0]["status"] != "verified"
    assert result.get("terminal_status") is None
    assert nodes.route_after_verify({**state, **result}) == "draft"


def test_retry_cap_survives_new_iteration(base_state, monkeypatch):
    state = _state(base_state)
    _client(monkeypatch, _observation(quote="bad 1"), _observation())
    first = nodes.verify_node(state)
    assert first["verification_status"] == "completed"
    second_state = _state(base_state, iteration=2)
    second_state.update(call_b_accounted_requests=first["call_b_accounted_requests"],
                        total_tokens=first["total_tokens"],
                        total_cost_usd=first["total_cost_usd"])
    client = _client(monkeypatch, _observation(quote="bad again"))
    second = nodes.verify_node(second_state)
    assert client.calls == 2
    assert second["terminal_status"] == "execution_failed"
    assert call_b_journal.lookup(state["run_id"], "retry")["verify_key"].endswith(":1:verify")


def test_replay_after_initial_reservation_blocks_all_provider_calls(base_state, monkeypatch):
    state = _state(base_state)
    client = _client(monkeypatch, _observation())
    def crash_before_transport(_client):
        def transport(**_kwargs):
            raise SystemExit("simulated process death")
        return transport
    monkeypatch.setattr(nodes, "_make_semantic_llm_transport", crash_before_transport)
    with pytest.raises(SystemExit):
        nodes.verify_node(state)
    assert client.calls == 1  # only Call A before reservation
    row = call_b_journal.lookup(state["run_id"], "initial", f"{state['run_id']}:1:verify")
    assert row["state"] == "reserved"
    monkeypatch.setattr(nodes, "_get_client", lambda: pytest.fail("replay made provider call"))
    replay = nodes.verify_node(state)
    assert replay["terminal_status"] == "execution_failed"
    assert replay["provider_usage_status"] == "unknown"
    assert replay["total_tokens"] == 100  # Call A known usage survived
    again = nodes.verify_node({**state, **replay})
    assert again["provider_usage_status"] == "unknown"
    assert again["total_tokens"] == 100


def test_crash_after_response_before_usage_commit_is_unknown(base_state, monkeypatch):
    state = _state(base_state)
    client = _client(monkeypatch, _observation())
    def crash_on_record(**_kwargs):
        raise SystemExit("after transport before durable usage")
    monkeypatch.setattr(call_b_journal, "record_response", crash_on_record)
    with pytest.raises(SystemExit):
        nodes.verify_node(state)
    assert client.calls == 2
    replay = nodes.verify_node(state)
    assert replay["provider_usage_status"] == "unknown"
    assert replay["total_tokens"] == 100
    assert replay["terminal_status"] == "execution_failed"


def test_crash_after_usage_commit_reconciles_once(base_state, monkeypatch):
    state = _state(base_state)
    _client(monkeypatch, _observation())
    def crash_after_usage(_raw):
        raise SystemExit("after durable usage before graph checkpoint")
    monkeypatch.setattr("agent.semantic_analyzer.provider.parse_analyzer_response",
                        crash_after_usage)
    with pytest.raises(SystemExit):
        nodes.verify_node(state)
    replay = nodes.verify_node(state)
    assert replay["terminal_status"] == "execution_failed"
    assert replay["provider_usage_status"] == "known"
    assert replay["total_tokens"] == 200
    again = nodes.verify_node({**state, **replay})
    assert again["total_tokens"] == 200


def test_initial_reservation_failure_prevents_call_b(base_state, monkeypatch):
    state = _state(base_state)
    client = _client(monkeypatch, _observation())
    def no_commit(**_kwargs):
        raise OSError("reservation persistence unavailable")
    monkeypatch.setattr(call_b_journal, "reserve", no_commit)
    result = nodes.verify_node(state)
    assert client.calls == 1
    assert result["terminal_status"] == "execution_failed"
    assert call_b_journal.lookup(state["run_id"], "initial", f"{state['run_id']}:1:verify") is None


def test_crash_during_retry_never_redispatches(base_state, monkeypatch):
    state = _state(base_state)
    client = _client(monkeypatch, _observation(quote="bad"))
    def create(**kwargs):
        client.calls += 1
        if client._responses:
            return client._responses.pop(0)
        raise SystemExit("process dies during retry transport")
    client.chat.completions.create = create
    with pytest.raises(SystemExit):
        nodes.verify_node(state)
    assert client.calls == 3
    assert call_b_journal.lookup(state["run_id"], "retry")["state"] == "reserved"
    monkeypatch.setattr(nodes, "_get_client", lambda: pytest.fail("restart dispatched provider"))
    replay = nodes.verify_node(state)
    assert replay["terminal_status"] == "execution_failed"
    assert replay["provider_usage_status"] == "unknown"
    assert replay["total_tokens"] == 200  # Call A plus original Call B


def test_crash_after_retry_usage_preserves_all_known_cost(base_state, monkeypatch):
    state = _state(base_state)
    _client(monkeypatch, _observation(quote="bad"), _observation())
    from agent.semantic_analyzer import provider
    original_parse = provider.parse_analyzer_response
    calls = 0
    def crash_on_second_parse(raw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SystemExit("retry usage saved, graph checkpoint absent")
        return original_parse(raw)
    monkeypatch.setattr(provider, "parse_analyzer_response", crash_on_second_parse)
    with pytest.raises(SystemExit):
        nodes.verify_node(state)
    replay = nodes.verify_node(state)
    assert replay["total_tokens"] == 300
    assert replay["provider_usage_status"] == "known"
    assert replay["terminal_status"] == "execution_failed"


def test_unknown_response_usage_is_not_numeric_zero(base_state, monkeypatch):
    state = _state(base_state)
    missing_usage = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=_observation()),
                                 finish_reason="stop")], usage=None,
        model="offline", id="no-usage",
    )
    client = FakeLLMClient(responses=[fake_response(_inventory()), missing_usage])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node(state)
    assert client.calls == 2
    assert result["terminal_status"] == "execution_failed"
    assert result["provider_usage_status"] == "unknown"
    assert result["total_tokens"] == 100
    row = call_b_journal.lookup(state["run_id"], "initial", f"{state['run_id']}:1:verify")
    assert row["usage_status"] == "unknown" and row["total_tokens"] is None
    assert row["estimated_cost_usd"] is None


def test_component_sum_usage_records_its_source(base_state, monkeypatch):
    state = _state(base_state)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=_observation()),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=60, completion_tokens=40, total_tokens=None,
                              completion_tokens_details=None),
        model="offline", id="component-usage",
    )
    client = FakeLLMClient(responses=[fake_response(_inventory()), response])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    result = nodes.verify_node(state)
    assert result["verification_status"] == "completed"
    assert result["total_tokens"] == 200
    row = call_b_journal.lookup(state["run_id"], "initial", f"{state['run_id']}:1:verify")
    assert row["total_tokens"] == 100 and row["total_tokens_source"] == "component_sum"


def test_unknown_claim_id_cannot_trigger_quote_retry(base_state, monkeypatch):
    state = _state(base_state)
    observation = json.dumps({"observations": [{
        "claim_id": "forged-claim", "support_quotes": [{"evidence_id": "WEB-001", "quote": "bad"}],
        "full_entailment": False, "blockers": [],
    }]})
    client = _client(monkeypatch, observation, _observation())
    result = nodes.verify_node(state)
    assert client.calls == 2
    assert result["terminal_status"] == "execution_failed"
    assert call_b_journal.lookup(state["run_id"], "retry") is None


def test_cost_gate_includes_both_known_calls_before_retry(base_state, monkeypatch):
    state = _state(base_state)
    state["total_cost_usd"] = nodes.COST_GATE_USD - 0.00001
    client = _client(monkeypatch, _observation(quote="bad"), _observation())
    result = nodes.verify_node(state)
    assert client.calls == 2
    assert result["terminal_status"] == "resource_exhausted"
    assert call_b_journal.lookup(state["run_id"], "retry") is None


@pytest.mark.parametrize("mutation", ["draft", "manifest"])
def test_changed_identity_prevents_observation_retry(base_state, monkeypatch, mutation):
    state = _state(base_state)
    client = _client(monkeypatch, _observation(quote="bad"), _observation())
    original_create = client.chat.completions.create
    def create(**kwargs):
        response = original_create(**kwargs)
        if client.calls == 2:
            if mutation == "draft":
                state["draft_markdown"] += " changed"
            else:
                state["web_sources"][0]["content"] += " changed"
        return response
    client.chat.completions.create = create
    result = nodes.verify_node(state)
    assert client.calls == 2
    assert result["terminal_status"] == "execution_failed"
    assert call_b_journal.lookup(state["run_id"], "retry") is None


def test_sqlite_unique_retry_reservation_across_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL_B_JOURNAL_PATH", str(tmp_path / "journal.sqlite"))
    kwargs = dict(run_id="concurrent", kind="retry", verify_key="v1",
                  request_id="r1", draft_sha256="d" * 64,
                  roster_sha256="r" * 64, manifest_sha256="m" * 64)
    def reserve():
        try:
            call_b_journal.reserve(**kwargs)
            return "reserved"
        except call_b_journal.ReservationExists:
            return "blocked"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve(), range(2)))
    assert sorted(results) == ["blocked", "reserved"]
    assert call_b_journal.lookup("concurrent", "retry")["state"] == "reserved"
    with sqlite3.connect(tmp_path / "journal.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM call_b_requests").fetchone()[0] == 1


def test_actual_graph_sqlite_restart_blocks_replay(base_state, monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    import agent.graph as graph_mod

    monkeypatch.setattr(graph_mod, "retrieve_node", lambda _state: {})
    monkeypatch.setattr(graph_mod, "draft_node", lambda _state: {
        "draft_markdown": CLAIM, "draft_status": "valid",
        "draft_sections": {key: CLAIM for key in (
            "problem_framing", "technical_dive", "code_snippets", "takeaways")},
    })
    state = _state(base_state, run_id="sqlite-restart", iteration=0)
    state.update(quality_episode=0, quality_attempts=0, draft_status="not_started")
    client = _client(monkeypatch, _observation())
    original_create = client.chat.completions.create
    def crash_call_b(**kwargs):
        if client.calls == 1:
            raise SystemExit("crash during first Call B")
        return original_create(**kwargs)
    client.chat.completions.create = crash_call_b
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": state["run_id"]}}
    first_connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    first = graph_mod.build_graph(checkpointer=SqliteSaver(first_connection))
    with pytest.raises(SystemExit):
        first.invoke(state, config)
    assert client.calls == 1  # Call A returned; Call B died before counter increment
    first_connection.close()

    monkeypatch.setattr(nodes, "_get_client", lambda: pytest.fail("graph replay made provider call"))
    second_connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    restarted = graph_mod.build_graph(checkpointer=SqliteSaver(second_connection))
    result = restarted.invoke(None, config)
    assert result["terminal_status"] == "execution_failed"
    assert result["provider_usage_status"] == "unknown"
    assert result["total_tokens"] == 100
    assert not restarted.get_state(config).next
    second_connection.close()
