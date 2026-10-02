# PHASE5-AUTH-PROTOCOL-1 implementation checklist

This working checklist freezes the implementation and test ownership for the
authoritative qualification protocol.  It is derived from the Phase 5E5
authorization brief and is deliberately limited to the release orchestrator;
it does not change retrieval, ranking, packing, thresholds, or provider use.

| Gap | Implementation owner | Invariant closed | Regression owner |
| --- | --- | --- | --- |
| G01 | `canonical_raw`, `qualified_canonical_source_map`, completed verifier | Raw rows are resolved against the independently derived canonical source map; archive intervals and content cannot define score truth. | `test_phase5e1_qualification_runner.py::test_canonical_adapter_mapping_fails_closed`; public Qdrant control uses the independently loaded qualified source map. |
| G02 | `_verify_protocol_archive` | Runtime identity and provenance are derived from raw evidence and trusted runtime expectations; stored pass flags are claims only. | `test_phase5_auth_protocol.py::test_completed_verifier_recomputes_provenance` |
| G03 | `run_authoritative_qualification`, A04 writer/verifier | A durable fresh pre-retrieval real preflight precedes every retrieval call. | `test_phase5_auth_protocol.py::test_initial_preflight_uses_uncached_live_validator`, `test_initial_artifacts_are_durable_before_query_one`, `test_archived_preflight_cannot_substitute_pass_flags` |
| G04 | `_protocol_execution`, A05 journal | Execution 1 reaches a durable COMPLETE boundary before execution 2 starts. | `test_phase5_auth_protocol.py::test_execution_one_completes_before_execution_two` |
| G05 | `_protocol_execution`, A05/A07/A08 writers | Per-query intent, backend invocation, raw results, score, and completion are journaled in causal order; journal and archived raw evidence agree exactly. | `test_phase5_auth_protocol.py::test_every_backend_invocation_is_one_shot`, `test_completed_verifier_rejects_missing_invocation_grammar`, `test_local_validation_precedes_qdrant_invocation` |
| G06 | `reserve_protocol_archive`, `write_durable_protocol_json` | Attempt, oracle identity, contract, and query plan are exclusively created and directory-synced before retrieval. | `test_phase5_auth_protocol.py::test_initial_artifacts_are_durable_before_query_one` |
| G07 | durable file helpers | Every transition requires unbuffered complete write, file fsync, and directory fsync. | `test_phase5_auth_protocol.py::test_durable_writes_sync_file_and_directory`, `test_durable_io_failure_classes`, `test_reservation_durability_failures` |
| G08 | state machine and preterminal verifier | Preterminal evaluation can return only `READY_TO_COMMIT`; synthetic defaults cannot create success. | `test_phase5_auth_protocol.py::test_only_completed_verifier_owns_pass`, `test_readiness_verification_prevents_corrupt_materialization_commit` |
| G09 | `verify_completed_protocol_archive` | Only independent completed-archive re-verification may return `PHASE-5E1A-HOLDOUT-PASS`. | `test_phase5_auth_protocol.py::test_only_completed_verifier_owns_pass` |
| G10 | completed verifier recomputation | Determinism, packed serialization, fingerprints, and duplicate identities/outcomes are recomputed from raw evidence. | `test_phase5_auth_protocol.py::test_completed_verifier_recomputes_determinism_and_duplicates` |
| G11 | `run_authoritative_qualification` | Production uses the fixed one-shot archive location; callers cannot select an archive root. | `test_phase5_auth_protocol.py::test_production_archive_root_is_not_caller_selectable` |
| G12 | approval/control validation | Approved implementation and control identities, including model/tokenizer pins, are verified before oracle parsing. | `test_phase5_auth_protocol.py::test_authorization_and_control_checks_precede_oracle_parse` |
| G13 | approval/contract/oracle validation | Execution and contract authorization precede holdout eligibility and canonical gold validation. | `test_phase5_auth_protocol.py::test_fixture_eligibility_is_after_authorization` |
| G14 | path, custody, and read helpers | All protocol files are regular private files under the reserved archive and are read stably without symlink traversal. | `test_phase5_auth_protocol.py::test_protocol_path_custody_rejects_unsafe_files` |
| G15 | lifecycle exception handler | Any lifecycle failure writes a truthful durable INCOMPLETE or FAIL state and never leaves stale success residue. | `test_phase5_auth_protocol.py::test_transition_failure_never_creates_verified_pass` |
| G16 | A02 writer and CLI | Archives carry only oracle identity/reference, use private modes, and do not print sealed paths or bytes. | `test_phase5_auth_protocol.py::test_archive_never_copies_oracle_bytes` |
| G17 | public production-path test seam | Tests exercise the actual state machine, readiness verification, and completed verifier. | `test_phase5_auth_protocol.py::test_public_production_path_uses_real_state_machine` |

The required artifacts are A01 `attempt.json`, A02 `oracle_identity.json`, A03
`contract.json`, A04 `pre_retrieval_preflight.json`, A05
`execution_journal.jsonl`, A06 `executions.json`, A07
`raw_backend_outputs.json`, A08 `execution_2_raw_backend_outputs.json`, A09
`backend_per_query.json`, A10 `per_query.json`, A11 `parity.json`, A12
`provenance.json`, A13 `determinism.json`, A14 `score_truth.json`, A15
`summary.json`, A16 `final_live_preflight.json`, A17
`authoritative_verification.json`, A18 `terminal_state.json`, A19
`disposition.json`, and A20 `manifest.json`.  A21 is the in-memory result of
completed-archive verification and is never archive content.

The state sequence is S0 `NO_ATTEMPT`, S1 `AUTHORIZED`, S2 `CONTROL_BOUND`,
S3 `ORACLE_IDENTIFIED`, S4 `PREFLIGHTED`, S5 `RESERVED`, S6
`ATTEMPT_MATERIALIZED`, S7 `EXECUTION_1`, S8 `EXECUTION_1_DURABLE`, S9
`EXECUTION_2`, S10 `EXECUTION_2_DURABLE`, S11 `SCORED`, S12 `GATED`, S13
`FINAL_PREFLIGHTED`, S14 `READY_TO_COMMIT`, S15 `VERIFICATION_PERSISTED`, S16
`TERMINALIZED`, S17 `COMMITTED_UNVERIFIED`, S18 `REVERIFIED`, S19 `PASS`.
Only S17 to S18 to S19 can produce PASS.  Every other terminal route is
durably INCOMPLETE or FAIL.
