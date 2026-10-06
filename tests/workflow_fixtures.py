"""Explicit current-quality artifacts for tests predating episode bindings."""
from agent import workflow
from agent.claim_inventory import build_claim_inventory


def current_quality(state, *, build_inventory=False):
    state = dict(state)
    state.update(quality_episode=1, quality_attempts=min(state.get("iterations", 1), 2), draft_status='valid', terminal_status=None)
    state['draft_sections'] = {key: 'Synthetic section.' for key in (
        'problem_framing', 'technical_dive', 'code_snippets', 'takeaways')}
    if build_inventory and state.get('grounding_report'):
        report = [dict(row) for row in state['grounding_report']]
        state['draft_markdown'] = '\n'.join('[' + row['claim'] + ']' for row in report)
        inv = build_claim_inventory(run_id=state['run_id'], iteration=state['iterations'],
            draft_markdown=state['draft_markdown'], brief_requirements=[], raw_claims=[{
                'claim_text': row['claim'], 'anchor_quote': '[' + row['claim'] + ']', 'section': 'technical_dive',
                'claim_type': 'factual', 'material': False, 'materiality_reason_code': None,
                'materiality_rationale': None, 'satisfies_req_ids': [], 'specificity': 'substantive',
                'requires_citation': None,
            } for row in report])
        by_text = {claim['claim_text']: claim['claim_id'] for claim in inv['claims']}
        for row in report:
            row['claim_id'] = by_text[row['claim']]
        state.update(claim_inventory=inv, grounding_report=report)
    elif isinstance(state.get('claim_inventory'), dict):
        state['claim_inventory'] = {**state['claim_inventory'], 'run_id': state['run_id'],
                                    'iteration': state['iterations']}
    state['verification_identity'] = workflow.identity(state)
    state['reflection_identity'] = workflow.identity(state)
    state['reflection_provenance'] = {'origin': 'judge', 'provider_called': True, 'parse_status': 'ok'}
    return state


def verification_for_report(state, report):
    """Author a current mock inventory without changing the draft under test."""
    inv = build_claim_inventory(run_id=state['run_id'], iteration=state['iterations'],
        draft_markdown=state['draft_markdown'], brief_requirements=state.get('brief_requirements', []),
        raw_claims=[{'claim_text': row['claim'], 'anchor_quote': state['draft_markdown'],
            'section': 'technical_dive', 'claim_type': 'factual', 'material': False,
            'materiality_reason_code': None, 'materiality_rationale': None,
            'satisfies_req_ids': [], 'specificity': 'substantive', 'requires_citation': None,
        } for row in report])
    by_text = {claim['claim_text']: claim['claim_id'] for claim in inv['claims']}
    return {'claim_inventory': inv, 'verification_status': 'completed',
            'grounding_report': [{**row, 'claim_id': by_text[row['claim']]} for row in report]}
