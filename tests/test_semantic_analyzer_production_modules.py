"""Canonical production module ownership and compatibility re-export tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent.semantic_analyzer.quote_binding as canonical_quote_binding
import agent.semantic_analyzer.status_engine as canonical_status_engine
import scripts.hybrid_verifier_status_engine as script_status_engine
import scripts.semantic_analyzer_quote_binding as script_quote_binding
from agent.semantic_analyzer.response_contract import parse_analyzer_response
from agent.semantic_analyzer.response_contract import ResponseContractError as CanonicalResponseContractError


def test_status_engine_scripts_reexport_is_canonical() -> None:
    assert (
        script_status_engine.adjudicate_hybrid_verifier_observations
        is canonical_status_engine.adjudicate_hybrid_verifier_observations
    )
    assert script_status_engine.BLOCKER_KINDS is canonical_status_engine.BLOCKER_KINDS


def test_quote_binding_scripts_reexport_is_canonical() -> None:
    assert script_quote_binding.bind_quote_to_span is canonical_quote_binding.bind_quote_to_span
    assert script_quote_binding.QuoteBindingError is canonical_quote_binding.QuoteBindingError


def test_response_contract_rejects_raw_offsets() -> None:
    raw = (
        '{"observations": [{"claim_id": "X", "support_spans": [{"evidence_id": "E", "start": 0, "end": 1}],'
        ' "full_entailment": false, "blockers": []}]}'
    )
    with pytest.raises(CanonicalResponseContractError, match="forbidden_field"):
        parse_analyzer_response(raw)
