#!/usr/bin/env bash
# PR eval-gate: deterministic current semantic contract ($0, zero provider calls).
#
# Replaces legacy evals/verifier_golden_test.py as merge authority. Exercises Call-B
# observation parsing, exact quote binding, Python status engine, verify_node cutover,
# and blocker-aware publication acceptance — not LLM-emitted status/confidence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec uv run pytest \
  tests/test_hybrid_verifier_status_engine.py \
  tests/test_semantic_analyzer_quote_binding.py \
  tests/test_semantic_analyzer_contract.py \
  tests/test_semantic_analyzer_provider_adapter.py \
  tests/test_verify_node_semantic_cutover.py \
  tests/test_blocker_aware_semantic_acceptance.py \
  tests/test_phase4_acceptance_remediation.py::test_cases_abcd_blocker_blocks_acceptance_and_revision_routing \
  tests/test_phase4_acceptance_remediation.py::test_case_j_legacy_confidence_cannot_launder_blocker \
  tests/test_phase4_acceptance_remediation.py::test_grounding_score_cannot_alter_routing_when_blocker_present \
  tests/test_phase4_acceptance_remediation.py::test_grounding_score_cannot_alter_routing_when_accepted \
  tests/test_phase4_acceptance_remediation.py::test_case_j_call_a_materiality_cannot_override_engine_p6 \
  -q --tb=short "$@"
