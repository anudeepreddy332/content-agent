"""Permanent audit reproductions at production graph/API/CLI seams; no providers."""
import json
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from openai import APITimeoutError

import agent.graph as graph_mod
import agent.nodes as nodes
from agent import workflow
from tests.test_frozen_workflow import harness as frozen_harness, accepted_verification, judge, CLAIM


harness = frozen_harness


def test_G_real_reflection_timeout_is_nonrepairable(harness, monkeypatch, tmp_path):
    _, config, state, _, calls = harness
    transport = Mock(side_effect=APITimeoutError(request=httpx.Request('POST', 'https://provider.invalid')))
    monkeypatch.setattr(nodes, '_get_client', lambda: object())
    monkeypatch.setattr(nodes, '_llm_call', transport)
    def reflect(current):
        calls.append('reflect')
        return nodes.reflect_node(current)
    monkeypatch.setattr(graph_mod, 'reflect_node', reflect)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'execution_failed'
    assert calls == ['retrieve', 'draft', 'verify', 'reflect']
    assert transport.call_count == 1 and '__interrupt__' not in result
    assert result['reflection_provenance']['origin'] == 'unavailable'
    import main
    monkeypatch.chdir(tmp_path)
    record = json.loads(main._write_telemetry(result).read_text())
    assert record['terminal_status'] == 'execution_failed'
    assert record['reflection_provenance']['origin'] == 'unavailable'


def test_B_valid_low_reflection_still_repairs(harness, monkeypatch, tmp_path):
    graph, config, state, plan, calls = harness
    plan['reflect'] = [judge(3), judge(8)]
    updates = list(graph.stream(state, config, stream_mode='updates'))
    first = next(update['reflect'] for update in updates if 'reflect' in update)
    assert first.get('terminal_status') is None
    assert first['reflection_provenance']['origin'] == 'judge'
    assert first['reflection_score'] == 3
    import main
    monkeypatch.chdir(tmp_path)
    record = json.loads(main._write_telemetry({**state, **first}).read_text())
    assert record['terminal_status'] is None and record['reflection_score'] == 3
    assert record['reflection_provenance']['origin'] == 'judge'
    assert calls.count('draft') == 2 and calls.count('verify') == 2
    assert graph.get_state(config).tasks[0].interrupts[0].value['type'] == 'hitl_review'


@pytest.mark.parametrize('blocker', ['unknown_materiality', 'unknown_requirement', 'invalid_material', 'roster_integrity', 'citation'])
def test_H_nonrepairable_policy_precedes_semantic_retry(harness, monkeypatch, blocker):
    _, config, state, _, calls = harness
    def verify(current):
        calls.append('verify')
        if calls.count('verify') > 1:
            return accepted_verification(current)
        update = accepted_verification(current)
        row = update['grounding_report'][0]
        row['status'] = 'unverified'
        inv = update['claim_inventory']
        if blocker == 'unknown_materiality':
            inv['claims'][0]['material'] = 'unknown'
        elif blocker == 'unknown_requirement':
            from agent.claim_inventory import build_claim_inventory
            from tests.test_citation_policy import _claim_row
            reqs = [{'req_id': 'REQ-X', 'kind': 'semantic', 'mandatory': True,
                     'description': 'Explain convergence.'}]
            inv = build_claim_inventory(run_id=current['run_id'], iteration=current['iterations'],
                draft_markdown=current['draft_markdown'], brief_requirements=reqs,
                raw_claims=[_claim_row(CLAIM, material=True, satisfies_req_ids=['REQ-X']),
                            _claim_row('Follow the gradient.', material=False)])
            update['claim_inventory'] = inv
            update['grounding_report'] = [
                {**row, 'claim_id': inv['claims'][0]['claim_id'], 'status': 'verified'},
                {**row, 'claim_id': inv['claims'][1]['claim_id'], 'claim': 'Follow the gradient.'}]
            assert nodes.material_policy_result({**current, **update}).material_safety_state == 'unknown_materiality'
        elif blocker == 'invalid_material':
            inv['claims'][0]['material'] = True
            row['analysis_validity'] = 'INVALID'
        elif blocker == 'roster_integrity':
            row['claim_id'] = 'wrong-roster'
        return update
    monkeypatch.setattr(graph_mod, 'verify_node', verify)
    if blocker == 'citation':
        draft = graph_mod.draft_node
        def cited_draft(current):
            update = draft(current)
            update['draft_markdown'] += ' [KB-999]'
            return update
        monkeypatch.setattr(graph_mod, 'draft_node', cited_draft)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'policy_blocked'
    assert calls == ['retrieve', 'draft', 'verify']
    assert '__interrupt__' not in result and result['quality_attempts'] == 1
    # The harness's next draft/verifier would pass. It must never be invoked.


def test_E_semantic_failure_without_nonrepairable_blocker_repairs(harness):
    graph, config, state, plan, calls = harness
    plan['verify'] = ['fail', 'pass']
    result = graph.invoke(state, config)
    assert calls == ['retrieve', 'draft', 'verify', 'draft', 'verify', 'reflect']
    assert result['__interrupt__'][0].value['type'] == 'hitl_review'


def archive_graph(harness, monkeypatch):
    _, config, state, _, calls = harness
    def html(current):
        text = '<html><body><p>Approved article.</p></body></html>'
        return {'html_output': text, 'html_sha256': nodes.sha256_utf8(text), 'html_filename': 'article.html'}
    monkeypatch.setattr(graph_mod, 'html_gen_node', html)
    monkeypatch.setattr(graph_mod, 'git_node', nodes.git_node)
    return graph_mod.build_graph(checkpointer=InMemorySaver()), config, state, calls


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_I_L_real_archive_api_projection(harness, monkeypatch, tmp_path, streaming, fail):
    import api.server as server
    graph, _, state, _ = archive_graph(harness, monkeypatch)
    monkeypatch.chdir(tmp_path)
    if fail:
        (tmp_path / 'outputs').mkdir()
        # Real filesystem failure: the archive directory is a regular file.
        (tmp_path / 'outputs/articles').write_text('not a directory')
    rid = state['run_id']
    monkeypatch.setattr(server, 'GRAPH', graph)
    monkeypatch.setattr(server, '_write_telemetry', lambda s: None)
    monkeypatch.setattr(server, 'REGISTRY', {rid: {'status': 'queued', 'initial_state': state,
        'interrupt_payload': None, 'result': None, 'error': None, 'events': queue.Queue()}})
    monkeypatch.setenv('API_BEARER_TOKEN', 'p1-test-token')
    advance = server._advance_streaming if streaming else server._advance
    advance(rid, state, False)
    advance(rid, Command(resume={'action': 'approve'}), False)
    advance(rid, Command(resume={'action': 'approve'}), True)
    with TestClient(server.app) as client:
        body = client.get('/runs/' + rid, headers={'Authorization': 'Bearer p1-test-token'}).json()
    result = server.REGISTRY[rid]['result']
    assert result['git_status'] == ('failed' if fail else 'dry_run')
    assert body['status'] == ('execution_failed' if fail else 'complete')
    assert body['summary']['terminal_status'] == ('execution_failed' if fail else None)
    if fail:
        assert any('archive write failed' in e for e in result['error_log'])
    else:
        assert (tmp_path / 'outputs/articles/article.html').read_text() == result['html_output']
    if streaming:
        events = list(server.REGISTRY[rid]['events'].queue)
        done = next(e for e in reversed(events) if isinstance(e, dict) and e.get('event') == 'done')
        assert done['status'] == body['status']
        assert done['summary']['terminal_status'] == body['summary']['terminal_status']


@pytest.mark.parametrize('status', ['terminal_quality_exhausted', 'execution_failed', None])
def test_K_L_cli_terminal_presentation(harness, monkeypatch, tmp_path, status):
    import main
    import agent.kb_backend as kb
    import agent.kb_backend.selector as selector
    _, _, state, _, _ = harness
    result = {**state, 'git_status': 'dry_run'}
    if status:
        result.update(workflow.terminal(status))
    monkeypatch.setattr(main, 'build_graph', lambda: SimpleNamespace(invoke=lambda initial: result))
    monkeypatch.setattr(main, '_write_telemetry', lambda current: tmp_path / 'telemetry.json')
    monkeypatch.setattr(kb, 'warmup', lambda: {})
    monkeypatch.setattr(selector, 'abort_on_invalid_kb_backend', lambda: None)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'dummy-never-used')
    monkeypatch.setenv('TAVILY_API_KEY', 'dummy-never-used')
    output = CliRunner().invoke(main.cli, ['run', '--topic', 'Test note'])
    assert output.exit_code == 0, output.output
    if status:
        assert 'Run complete.' not in output.output
        assert status in output.output
        assert result['terminal_message'] in output.output
    else:
        assert 'Run complete.' in output.output


def test_real_unusable_reflection_response_is_execution_failure(harness, monkeypatch):
    from tests.conftest import fake_response
    _, config, state, _, calls = harness
    monkeypatch.setattr(nodes, '_get_client', lambda: object())
    transport = Mock(return_value=fake_response('unusable judge response'))
    monkeypatch.setattr(nodes, '_llm_call', transport)
    def reflect(current):
        calls.append('reflect')
        return nodes.reflect_node(current)
    monkeypatch.setattr(graph_mod, 'reflect_node', reflect)
    graph = graph_mod.build_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, config)
    assert result['terminal_status'] == 'execution_failed'
    assert calls == ['retrieve', 'draft', 'verify', 'reflect']
    assert transport.call_count == 1 and '__interrupt__' not in result
    assert result['reflection_provenance']['origin'] == 'fallback'


@pytest.mark.parametrize('streaming', [False, True])
def test_legacy_failed_git_result_cannot_project_success(monkeypatch, base_state, streaming):
    import api.server as server
    result = {**base_state, 'git_status': 'failed'}
    graph = SimpleNamespace(invoke=lambda *args: result,
        stream=lambda *args, **kwargs: iter([('values', result)]),
        get_state=lambda *args: SimpleNamespace(tasks=(), values=result))
    rid = base_state['run_id']
    monkeypatch.setattr(server, 'GRAPH', graph)
    monkeypatch.setattr(server, '_write_telemetry', lambda state: None)
    monkeypatch.setattr(server, 'REGISTRY', {rid: {'status': 'queued', 'initial_state': base_state,
        'interrupt_payload': None, 'result': None, 'error': None, 'events': queue.Queue()}})
    (server._advance_streaming if streaming else server._advance)(rid, base_state, False)
    assert server.REGISTRY[rid]['status'] == 'execution_failed'
    assert server.REGISTRY[rid]['result']['terminal_status'] == 'execution_failed'
