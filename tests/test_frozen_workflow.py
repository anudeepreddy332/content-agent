"""A-M frozen workflow regressions: production graph/routes/HITL, no providers."""
import builtins
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agent.graph as graph_mod
import agent.nodes as nodes
from agent import workflow
from agent.claim_inventory import build_claim_inventory
from agent.html_policy import sha256_utf8
from tests.conftest import FakeLLMClient, fake_response
from tests.test_citation_policy import _claim_row

CLAIM = "Gradient descent minimizes the loss."
SECTIONS = dict(problem_framing=CLAIM, technical_dive="Follow the gradient.",
                code_snippets="x = x - step * gradient", takeaways="Choose a small step.")


def accepted_verification(state):
    inv = build_claim_inventory(run_id=state["run_id"], iteration=state["iterations"],
                                draft_markdown=state["draft_markdown"],
                                raw_claims=[_claim_row(CLAIM, material=False)], brief_requirements=[])
    return {"claim_inventory": inv, "verification_status": "completed", "grounding_report": [{
        "claim_id": inv["claims"][0]["claim_id"], "claim": CLAIM, "status": "verified",
        "analysis_validity": "VALID", "blockers": [], "support_spans": [],
    }]}


def judge(score=8, origin="judge"):
    return {"reflection_score": score, "reflection_provenance": {
        "origin": origin, "provider_called": True, "parse_status": "ok",
    }}


@pytest.fixture
def harness(monkeypatch, base_state):
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("GIT_PUSH_ENABLED", "false")
    monkeypatch.setattr(builtins, "input", Mock(side_effect=AssertionError("No CLI input")))
    monkeypatch.setattr(nodes, "_get_client", Mock(side_effect=AssertionError("No provider call")))
    calls = []
    plan = {"verify": [], "reflect": [], "draft": []}

    def retrieve(state):
        calls.append("retrieve")
        return {}

    def draft(state):
        calls.append("draft")
        status = plan["draft"].pop(0) if plan["draft"] else "valid"
        return {"draft_status": status, "draft_sections": dict(SECTIONS),
                "draft_markdown": nodes._assemble_markdown(state["topic"], SECTIONS)}

    def verify(state):
        calls.append("verify")
        status = plan["verify"].pop(0) if plan["verify"] else "pass"
        if status != "pass":
            if status == "fail":
                result = accepted_verification(state)
                result["grounding_report"][0]["status"] = "unverified"
                return result
            return {"verification_status": status, "claim_inventory": None, "grounding_report": []}
        return accepted_verification(state)

    def reflect(state):
        calls.append("reflect")
        return plan["reflect"].pop(0) if plan["reflect"] else judge()

    def html(state):
        calls.append("html_gen")
        text = "<html><body><p>Trusted previous render.</p></body></html>"
        return {"html_output": text, "html_sha256": sha256_utf8(text)}

    def layout(state):
        calls.append("html_revise")
        raise RuntimeError("discarded layout revision")

    def git(state):
        calls.append("git")
        return {"git_status": "dry_run"}

    for name, fn in [('retrieve', retrieve), ('draft', draft), ('verify', verify), ('reflect', reflect),
                     ('html_gen', html), ('html_revise', layout), ('git', git)]:
        monkeypatch.setattr(graph_mod, name + '_node', fn)
    state = {**base_state, "iterations": 0, "quality_episode": 0, "quality_attempts": 0,
             "content_feedback_count": 0, "html_revision_attempts": 0,
             "terminal_status": None, "terminal_message": None}
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "frozen-matrix"}}
    return graph, config, state, plan, calls


def test_A_B_verify_failures_never_reflect_or_review(harness):
    graph, config, state, plan, calls = harness
    plan['verify'] = ['fail', 'fail']
    result = graph.invoke(state, config)
    assert calls == ['retrieve', 'draft', 'verify', 'draft', 'verify']
    assert result['quality_attempts'] == 2
    assert result['terminal_status'] == 'terminal_quality_exhausted'
    assert result['terminal_message'] == workflow.QUALITY_MESSAGE
    assert not graph.get_state(config).next
    assert '__interrupt__' not in result


def test_C_D_fresh_verify_failure_does_not_reflect(harness):
    graph, config, state, plan, calls = harness
    plan['verify'] = ['pass', 'fail']
    plan['reflect'] = [judge(3)]
    result = graph.invoke(state, config)
    assert calls == ['retrieve', 'draft', 'verify', 'reflect', 'draft', 'verify']
    assert result['terminal_status'] == 'terminal_quality_exhausted'
    assert '__interrupt__' not in result


def test_E_reflection_exhaustion(harness):
    graph, config, state, plan, calls = harness
    plan['reflect'] = [judge(3), judge(3)]
    result = graph.invoke(state, config)
    assert calls.count('reflect') == 2
    assert result['terminal_status'] == 'terminal_quality_exhausted'
    assert '__interrupt__' not in result


def test_F_real_gate_requires_explicit_decision(harness):
    graph, config, state, plan, calls = harness
    result = graph.invoke(state, config)
    assert result['__interrupt__'][0].value['type'] == 'hitl_review'
    assert workflow.gate1_eligible(graph.get_state(config).values)
    assert 'html_gen' not in calls
    rejected = graph.invoke(Command(resume={'action': 'reject'}), config)
    assert rejected['terminal_status'] == 'terminal_rejected'
    assert 'html_gen' not in calls and 'git' not in calls


@pytest.mark.parametrize('stage,status,expected', [
    ('draft', 'invalid', 'execution_failed'),
    ('verify', 'parse_failed', 'execution_failed'),
    ('verify', 'verification_error', 'execution_failed'),
])
def test_G_execution_failure_cannot_reach_reflect_or_gate(harness, stage, status, expected):
    graph, config, state, plan, calls = harness
    plan[stage] = [status, status]
    result = graph.invoke(state, config)
    assert result['terminal_status'] == expected
    assert 'reflect' not in calls and '__interrupt__' not in result


def test_G_real_inventory_parse_failure(harness, monkeypatch):
    _, config, state, _, calls = harness
    client = FakeLLMClient(responses=[fake_response('malformed'), fake_response('malformed')])
    monkeypatch.setattr(nodes, '_get_client', lambda: client)
    monkeypatch.setattr(graph_mod, 'verify_node', nodes.verify_node)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'execution_failed'
    assert client.calls == 2 and 'reflect' not in calls


def test_H_approval_rechecks_current_identity(harness, monkeypatch):
    graph, config, state, _, calls = harness
    graph.invoke(state, config)
    approve = nodes.hitl_approve_update
    def stale_approval(current):
        return approve({**current, 'reflection_identity': {'draft_sha256': 'stale'}})
    monkeypatch.setattr(nodes, 'hitl_approve_update', stale_approval)
    result = graph.invoke(Command(resume={'action': 'approve'}), config)
    assert result['terminal_status'] == 'execution_failed'
    assert result['hitl_status'] == 'rejected'
    assert 'html_gen' not in calls


def test_I_J_M_feedback_new_episode_independent_of_lifetime(harness):
    graph, config, state, plan, calls = harness
    state['iterations'] = 75
    plan['verify'] = ['fail', 'pass']
    graph.invoke(state, config)
    snap = graph.get_state(config).values
    assert snap['iterations'] == 77 and snap['quality_attempts'] == 2
    for episode in (2, 3):
        plan['verify'] = ['fail', 'pass']
        result = graph.invoke(Command(resume={'action': 'feedback', 'feedback': 'Clarify.'}), config)
        assert result['__interrupt__'][0].value['type'] == 'hitl_review'
        snap = graph.get_state(config).values
        assert snap['quality_episode'] == episode
        assert snap['content_feedback_count'] == episode - 1
        assert snap['quality_attempts'] == 2
        assert snap['iterations'] == 75 + episode * 2
        assert workflow.gate1_eligible(snap)
    before = len(calls)
    result = graph.invoke(Command(resume={'action': 'feedback', 'feedback': 'Third.'}), config)
    assert result['terminal_status'] == 'content_revision_limit'
    assert len(calls) == before


def test_K_L_layout_is_separate_failed_attempts_count(harness):
    graph, config, state, _, calls = harness
    graph.invoke(state, config)
    result = graph.invoke(Command(resume={'action': 'approve'}), config)
    assert result['__interrupt__'][0].value['type'] == 'hitl_html_review'
    initial = graph.get_state(config).values
    content_calls = calls.count('draft'), calls.count('verify'), calls.count('reflect')
    for attempt in (1, 2):
        result = graph.invoke(Command(resume={'action': 'feedback', 'feedback': 'Spacing.'}), config)
        assert result['__interrupt__'][0].value['type'] == 'hitl_html_review'
        snap = graph.get_state(config).values
        assert snap['html_revision_attempts'] == attempt
        assert snap['html_output'] == initial['html_output']
        assert snap['quality_attempts'] == initial['quality_attempts']
        assert (calls.count('draft'), calls.count('verify'), calls.count('reflect')) == content_calls
    result = graph.invoke(Command(resume={'action': 'feedback', 'feedback': 'Third.'}), config)
    assert result['terminal_status'] == 'layout_revision_limit'
    assert 'git' not in calls


def test_trusted_previous_layout_remains_explicitly_approvable(harness):
    graph, config, state, _, calls = harness
    graph.invoke(state, config)
    graph.invoke(Command(resume={'action': 'approve'}), config)
    graph.invoke(Command(resume={'action': 'feedback', 'feedback': 'Spacing.'}), config)
    result = graph.invoke(Command(resume={'action': 'approve'}), config)
    assert result['git_status'] == 'dry_run' and calls.count('git') == 1


@pytest.mark.parametrize('provenance', [None, {}, judge(9, 'fallback')['reflection_provenance']])
def test_missing_or_fallback_reflection_never_qualifies(harness, provenance):
    graph, config, state, plan, _ = harness
    plan['reflect'] = [{'reflection_score': 9, 'reflection_provenance': provenance}] * 2
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'execution_failed'
    assert '__interrupt__' not in result


@pytest.mark.parametrize('streaming', [False, True])
def test_api_terminal_surface_real_graph(harness, monkeypatch, streaming):
    import queue
    import api.server as server
    from fastapi.testclient import TestClient
    graph, _, state, plan, _ = harness
    plan['verify'] = ['fail', 'fail']
    rid = state['run_id']
    monkeypatch.setattr(server, 'GRAPH', graph)
    monkeypatch.setattr(server, '_write_telemetry', lambda s: None)
    monkeypatch.setattr(server, 'REGISTRY', {rid: {'status': 'queued', 'initial_state': state,
        'interrupt_payload': None, 'result': None, 'error': None, 'events': queue.Queue()}})
    monkeypatch.setenv('API_BEARER_TOKEN', 'test-workflow-token')
    (server._advance_streaming if streaming else server._advance)(rid, state, False)
    with TestClient(server.app) as client:
        response = client.get('/runs/' + rid, headers={'Authorization': 'Bearer test-workflow-token'})
    assert response.json()['status'] == 'terminal_quality_exhausted'
    assert response.json()['summary']['terminal_message'] == workflow.QUALITY_MESSAGE
    assert 'review' not in response.json()


@pytest.mark.parametrize('raw', ['{"problem_framing": []}', 'malformed'])
def test_real_invalid_structured_draft_consumes_episode(harness, monkeypatch, raw):
    _, config, state, _, calls = harness
    client = FakeLLMClient(response=fake_response(raw), analyzer_autofill=False)
    monkeypatch.setattr(nodes, '_get_client', lambda: client)
    monkeypatch.setattr(graph_mod, 'draft_node', nodes.draft_node)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, config)
    assert client.calls == 2 and result['quality_attempts'] == 2
    assert result['terminal_status'] == 'execution_failed'
    assert 'verify' not in calls and 'reflect' not in calls
    assert '__interrupt__' not in result


def test_verifier_cannot_replace_current_draft(harness, monkeypatch):
    _, config, state, _, calls = harness
    def stale_verify(current):
        return {**accepted_verification(current), 'draft_markdown': 'Changed by verifier.'}
    monkeypatch.setattr(graph_mod, 'verify_node', stale_verify)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'execution_failed'
    assert 'reflect' not in calls and '__interrupt__' not in result


def test_cumulative_resource_limit_cannot_reset_with_episode(harness):
    from config import COST_GATE_USD
    graph, config, state, _, calls = harness
    state['total_cost_usd'] = COST_GATE_USD
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'resource_exhausted'
    assert calls == ['retrieve']
    assert result['total_cost_usd'] == COST_GATE_USD


def test_unknown_material_policy_is_terminal_not_review(harness, monkeypatch):
    _, config, state, _, calls = harness
    def unknown(current):
        update = accepted_verification(current)
        update['claim_inventory']['claims'][0]['material'] = 'unknown'
        return update
    monkeypatch.setattr(graph_mod, 'verify_node', unknown)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'policy_blocked'
    assert 'reflect' not in calls and '__interrupt__' not in result


def test_initial_renderer_failure_is_terminal(harness, monkeypatch):
    _, config, state, _, calls = harness
    def broken(current):
        raise RuntimeError('render failed')
    monkeypatch.setattr(graph_mod, 'html_gen_node', broken)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    graph.invoke(state, config)
    result = graph.invoke(Command(resume={'action': 'approve'}), config)
    assert result['terminal_status'] == 'rendering_failed'
    assert '__interrupt__' not in result and 'git' not in calls


def test_gate_review_renderer_failure_is_terminal(harness, monkeypatch):
    graph, config, state, _, _ = harness
    from agent.html_policy import PolicyError
    monkeypatch.setattr(nodes, 'render_markdown_review_html', Mock(side_effect=PolicyError('render failed')))
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'rendering_failed'
    assert '__interrupt__' not in result


def test_real_semantic_analyzer_execution_failure(harness, monkeypatch):
    import json
    _, config, state, _, calls = harness
    rows = [_claim_row(CLAIM, material=False)]
    client = FakeLLMClient(response=fake_response(json.dumps(rows)), analyzer_autofill=False)
    monkeypatch.setattr(nodes, '_get_client', lambda: client)
    analyzer = Mock(side_effect=RuntimeError('semantic execution failed'))
    monkeypatch.setattr(nodes, 'analyze_semantic_evidence', analyzer)
    monkeypatch.setattr(graph_mod, 'verify_node', nodes.verify_node)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'execution_failed'
    assert analyzer.call_count == 1 and client.calls == 1
    assert 'reflect' not in calls and '__interrupt__' not in result
