# PHASE5-AUTH-PROTOCOL-1 validation closure

Validation starts at `85aaf1c3c30c1f6f45fb2c6e2c645d9260b1e65e`;
canonical comparison base is `7a2ca5b4f4a44a8d5c906267b9f90163502ff862`.
All mechanism inputs are public/synthetic and operator/storage seams are
test-isolated. No production approval record or sealed fixture is used.
The [G01–G17 checklist](PHASE5_AUTH_PROTOCOL_1_CHECKLIST.md) maps implementation
locations and named regression owners. Names below belong to
`tests/test_phase5_auth_protocol.py` unless another module is specified.

## Production transition failure matrix

`test_transition_failure_never_creates_verified_pass` injects each named case
in the real `run_authoritative_qualification` orchestration. Core readiness,
state machine, score truth and completed verifier are real on the positive
control. A negative test injects a fault at the named boundary deliberately.
All 19 parameter cases pass. An exception before reservation leaves NO ATTEMPT;
execution/persistence uncertainty leaves a rejected incomplete attempt. Once
manifest durability completes, a verification failure leaves
COMMITTED_UNVERIFIED. An existing terminal artifact is never overwritten to
repair a failed attempt.

| Parameter | Production transition exercised | Expected surviving state | Result |
| --- | --- | --- | --- |
| approval | Fixed operator approval loading | NO ATTEMPT | PASS |
| execution_control | Approved HEAD binding | NO ATTEMPT | PASS |
| control_bytes | Approved implementation/control bytes | NO ATTEMPT | PASS |
| oracle | Authorized fixture snapshot | NO ATTEMPT | PASS |
| runtime | Trusted runtime derivation | NO ATTEMPT | PASS |
| initial_preflight | Fresh initial live validation | NO ATTEMPT | PASS |
| reservation | Exclusive fixed attempt reservation | NO ATTEMPT | PASS |
| execution_transition | Execution 1 COMPLETE to execution 2 start | INCOMPLETE; execution 1 journal survives | PASS |
| materialization | Execution 2 COMPLETE to A06 materialization | INCOMPLETE; both execution journals survive | PASS |
| determinism | Two-run comparison | INCOMPLETE; query journal survives | PASS |
| score_truth | Per-query score computation | INCOMPLETE; raw backend evidence survives | PASS |
| gates | Aggregate/gate computation | INCOMPLETE; completed query evidence survives | PASS |
| final_preflight | Fresh final live validation | INCOMPLETE; A01–A15 survive | PASS |
| readiness | Persisted preterminal verification | INCOMPLETE; A16 survives | PASS |
| terminal_verification | A17 persistence | INCOMPLETE; preterminal evidence survives | PASS |
| terminal_state | A18 persistence | INCOMPLETE; A17 survives | PASS |
| disposition | A19 persistence | Incomplete archive; existing unverified A18 retained | PASS |
| manifest | A20 construction/commit | Incomplete archive; no committed success | PASS |
| completed_verification | Independent completed-archive verification | COMMITTED_UNVERIFIED; immutable archive | PASS |

Additional required production transitions:

| Named test | Cases | Transition / expected surviving evidence | Result |
| --- | ---: | --- | --- |
| `test_every_backend_invocation_is_one_shot` | 20 | Every local/Qdrant call in both executions; exactly one intent, no retry, predecessor raw evidence survives, INCOMPLETE | PASS |
| `test_every_query_completion_abrupt_death_preserves_predecessors` | 20 | Abrupt death before/after all ten query-completion boundaries; exact completed prefix survives, no future completion or manifest | PASS |
| `test_production_vs_development_authority_separation` | 1 | Development archive cannot grant authoritative release PASS | PASS |
| `test_truthful_gate_fail_is_distinct_from_incomplete` | 1 | Actual gate rejection has FAIL disposition, distinct from interrupted execution | PASS |
| `test_readiness_verification_prevents_corrupt_materialization_commit` | 1 | Persisted corruption rejects readiness before manifest | PASS |
| `test_final_preflight_exception_after_manifest_never_repairs_archive` | 1 | Live verification I/O failure leaves COMMITTED_UNVERIFIED; no post-manifest write | PASS |

There are **59 lifecycle/backend/query-boundary failure injections**, plus the
authority separation and explicit semantic/ordering controls above.

## Durable I/O matrix

`test_durable_io_failure_classes`: **41 parameter cases**, all PASS.
Equivalent JSON artifacts use the same exclusive unbuffered byte writer;
representatives cover initial identity, aggregate materialization, readiness,
terminal state and final manifest. The append-only journal is a separate class.

| Class representative | Pre-open | Collision | Partial write | File fsync | Directory fsync | Death before fsync | Death after fsync |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A01 attempt.json | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| A05 execution_journal.jsonl | PASS | N/A append | PASS | PASS | PASS | PASS | PASS |
| A06 executions.json | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| A17 authoritative_verification.json | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| A18 terminal_state.json | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| A20 manifest.json | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

`test_reservation_durability_failures`: **5 cases**, all PASS: pre-create,
exclusive collision, parent-directory sync failure, abrupt death before and
after reservation durability. Together: **46 durable-I/O failure cases**.
`test_durable_writes_sync_file_and_directory` additionally observes both fsyncs
for JSON, journal and manifest (3 cases).

Buffered flush failure is **not applicable**: these production primitives use
`os.write` directly, with complete-write looping and `os.fsync`; there is no
buffered stream/flush operation to inject. Partial-write and fsync failures
exercise the actual production operations. Journal collision is not an
exclusive-create operation; the exclusive attempt-reservation collision test
owns journal custody. This is an operation-level traceability distinction,
not an omitted failure mode or an invented successful flush.

Observed manifest persistence failure preserves bytes as
`manifest.incomplete.json`, invalidates completion, and never overwrites the
attempt. Abrupt death returns no verdict; all archive verdicts require fresh
independent verification, never an on-disk PASS flag.

## Artifacts, positive control and repairs

The original `test_phase5e1_qualification_runner.py::test_protocol_artifact_mutation_matrix_rejects_all_a01_a19_cases`
retains **133/133** rejections, including ordinary-hash refresh for semantic
mutations. `test_seven_manifest_cases` adds seven manifest-specific rejections.
Additional regressions reject missing invocation intent, forged provenance,
fabricated packed fingerprints, identity-forged preflight flags and nonprivate
custody for every A01–A20 file.

`test_public_production_path_uses_real_state_machine` runs the actual orchestration
to PASS; `test_only_completed_verifier_owns_pass` proves rejection at A21 cannot
grant PASS. `test_manifest_is_last_and_completed_verifier_is_independent`
observes final live preflight, readiness verification, A17/A18/A19 persistence,
manifest-last commit and independent final verification in causal order.
`test_initial_artifacts_are_durable_before_query_one` and
`test_execution_one_completes_before_execution_two` observe the earlier
durability boundaries.

Required regressions exposed these bounded production defects, now repaired:

- Incomplete invocation journal grammar was accepted; replay now requires the
  exact query/intent/raw/completion grammar and agreement with archived raw data.
- Local evidence was not validated before Qdrant invocation; raw evidence is
  now durably stored and validated before the next backend call.
- Truthful gate FAIL was labeled INCOMPLETE; failure disposition now preserves
  the actual state.
- Persisted readiness artifacts were not verified before manifest; semantic
  preterminal verification now returns READY_TO_COMMIT only.
- Provenance flags, packed fingerprints and preflight pass flags could substitute
  for independent truth; their evidence is now recomputed/validated from trusted
  runtime, canonical units and raw results.
- Artifact/fixture custody lacked complete private stable-read checks; reads
  now bind private owner/modes, no-follow open and stable descriptor/path identity.
- HEAD alone did not bind changed control bytes; approved control bytes are
  checked before fixture parsing.
- Manifest durability failure could leave completion evidence; failed commit
  bytes are preserved under a rejected incomplete name.
- Final verifier live I/O errors escaped; they now fail closed without repair.
- Healthy live hydration timings differed between independent reads; only
  elapsed `hydrate_ms` is excluded from identity equality, with controls retained.
- Initial preflight used cached startup state; it now invokes the existing fresh
  network-backed validator (`test_initial_preflight_uses_uncached_live_validator`).

## Public Qdrant and regression evidence

`scripts/phase5e5b_public_protocol_validation.py` starts fresh Qdrant **1.9.2**,
rebuilds **159** public qualified points and validates the serving alias, model
and collection fingerprint through real adapters. Public Q01/Q02 traverse both
executions, durable query protocol, final preflight, readiness, manifest and
independent completed verification. The resulting mechanism PASS is **not a
sealed holdout result**. The 33-query public regression has local and Qdrant
packed recall **0.93939394**, with **zero** seed, expanded, packed ID/fingerprint,
dense, BM25 or hybrid parity mismatches.

Focused protocol/legacy suite: **300 passed**. Full ordinary offline suite:
**1755 passed**; security/HITL/publication/browser and serving regressions are
included. Fork-based abrupt-death tests emit Python deprecation warnings about
forking a multithreaded test process; their child exit/status and durable
predecessor assertions pass.

## Ruff provenance

Identical command on canonical source export and candidate:
`ruff check scripts tests --output-format json`. Normalization uses relative
file, rule, row, column and message. Both contain **32** identical violations;
**0 new**, **0 resolved**. Modified Python files pass Ruff. The exact unchanged
location/rule set is below (every listed location has the same message at base
and candidate); archived/unrelated files were not edited.

| File | Rule / row:column |
| --- | --- |
| scripts/archive/m4_analyze.py | E702 47:43, 48:44 |
| scripts/archive/p2_3_analyze.py | E702 20:69 |
| scripts/check_telemetry_fields.py | F401 4:8 |
| scripts/ingest.py | E402 31:1 |
| scripts/phase5a2e_cswp.py | F401 8:8, 11:8 |
| scripts/phase5d4c_docker_acceptance.py | E402 23:1, 24:1, 25:1, 26:1, 32:1, 33:1, 34:1, 35:1, 36:1, 37:1, 38:1, 39:1 |
| tests/test_artifact_equivalence.py | F401 5:21, 12:76 |
| tests/test_browser_security.py | E702 127:24, 127:50, 127:77 |
| tests/test_index_update.py | E702 14:34, 24:34, 33:34 |
| tests/test_phase5d2_cswp_compiler.py | F401 8:18, 9:21, 19:5, 24:5, 30:43 |

Protected corpus/retrieval/model/index/contract inputs have empty diffs against
both starting candidate and canonical base. No push, main merge, paid provider
call, production approval modification, V1 access or V2 open/parse/hash/retrieval/
execution occurs in this validation.
