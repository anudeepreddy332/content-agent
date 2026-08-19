# P0-2 Verifier and Evaluation Integrity Architecture

_Audit date: 2026-08-19. Status: frozen implementation specification, independent review required._

## 1. Decision

`P0-2-CURSOR-OR-CODEX-READY`

P0-2 is split into two sequential, causally isolated missions:

1. **P0-2a — fail-closed evaluation scope, cardinality, and release-evidence binding**
2. **P0-2b — verifier status, confidence, completeness, and routing semantics**

Only P0-2a is specified here. P0-2b remains blocked on a separate architecture
mission because production claim completeness is not yet defined. MCP and A2A are optional workflow
infrastructure, are out of scope, and do not block Content Agent product hardening.

## 2. Repository and canonical-state verification

- Repository: `git@github.com:anudeepreddy332/content-agent.git`
- Expected main supplied to the audit: `ca29d32b4869269daa47142615d298580a577a77`
- Fetched `origin/main`: `ca29d32b4869269daa47142615d298580a577a77`
- Independent `git ls-remote --heads origin main`: the same exact SHA
- Audit worktree: clean, dedicated branch `chore/p0-2-evaluation-integrity-architecture`
- Product/runtime base for any later implementation: exact
  `ca29d32b4869269daa47142615d298580a577a77`
- `ca29d32` is a documentation-only descendant of validated P0-1 integration `d0be0a7`; P0-1 is
  closed and was not reopened.

The user's unrelated files in the original stale worktree were not touched. `FREEZE.md` was not
modified. This audit made no runtime/product change and made no provider call.

## 3. Relevant history and protections already present

The following commits are ancestors of `ca29d32` and must not be reimplemented or weakened:

| Commit | Existing protection |
| --- | --- |
| `fef583d1adccfcbb39335e5c975aaf2ba558be97` | Strict verifier verdict schema and optional exact count for known-cardinality callers |
| `16d3e7fc5174e5ca6d732d69c5c20860b0a0e202` | Deterministic extracted-verdict contract tests |
| `30fd3edf9de96cf75e570ddb4a9a96d10299fef9` | Corrected retrieval-evaluation semantics integration |
| `0daaf71462885482dd7c1ce8d72f8d0f9376f6c2` | Exact-run telemetry loading, explicit verification status, fail-closed incomplete telemetry |
| `e3c2cd80f542d93dfbc2b65200697b4dde14b225` | API crash telemetry marked `upstream_failed` |

Historical paired-benchmark provenance commit
`6fc797c2b393df4052999569790c789a5a6a640b` was also inspected. It is **not** an ancestor of current
main and is not current product code. Its experiment-digest patterns are evidence, not an
implementation dependency or authorization to import that branch.

Current correct protections include:

- successful subprocess exit required;
- `RUN_ID` extracted from that exact subprocess;
- only `outputs/runs/<RUN_ID>.json` loaded;
- telemetry `run_id` and requested topic must match;
- required telemetry fields and attribution/chunk coherence checked;
- non-`completed` verification is invalid;
- zero verdicts are unscorable and fail unless a smoke topic explicitly opts into no claims;
- known-cardinality verifier fixtures can use `expected_count`.

These protections do not validate selected evaluation scope, full release cardinality, duplicate
units, final-SHA identity, uniform prompt/config identity, or report identity.

## 4. Deterministic reproduction record

All probes used mocks or paths that select no work. No Tavily, DeepSeek, Qdrant, deployment, or paid
provider was called.

| Required path | Exact observed current behavior | Classification |
| --- | --- | --- |
| Unknown topic ID | `scripts/benchmark.py --id 999 --gate` selected 0, printed `Valid : 0/0`, printed CI PASS, exit 0 | `PROVEN DEFECT` |
| Zero selected units | Same reproduction; no subprocess call occurred | `PROVEN DEFECT` |
| Missing expected unit | Mocked `--limit 1 --gate` evaluated 1 of canonical 20 and passed | `PROVEN DEFECT` |
| Duplicate/incoherent unit | A two-entry manifest with duplicate topic ID ran twice and passed | `PROVEN DEFECT` |
| `unverified + 0.99` | One unverified verdict produced `grounding_score=0.99` and normal route `hitl` | `PROVEN DEFECT` |
| Valid empty verifier array | `[]` produced `verification_status=completed`, `N=0`, score 0 | `PROVEN DEFECT` plus completeness unknown |
| Verifier parse failure | Invalid JSON produced `parse_failed`, `N=0`, score 0 | Correct classification locally; ceiling routing remains defective |
| Max-iteration route | Empty/parse-failed verification at `MAX_ITERATIONS` routed to `hitl` | `PROVEN DEFECT` |
| Cost-ceiling route | Failed/incomplete state at cost ceiling routed to `hitl` | `PROVEN DEFECT`; HTML node later refuses a paid generation at the ceiling |
| Approved HITL afterward | `hitl_status=approved` routed each such state to `html_gen` | `PROVEN DEFECT`, mitigated by human review |

Additional limit probes:

- `--limit 0`: falsy branch, so all 20 topics were selected; unsafe surprise-call behavior.
- `--limit -1`: Python slice selected 19 of 20 and the gate passed.
- `--limit 999`: selected all 20 without rejecting the incoherent requested cardinality.

The focused existing suite passed `45/45`. The full exact-baseline suite passed `215` tests with
three warnings, including the four pinned-Chromium security tests. The first sandboxed full-suite
attempt could not bind loopback or launch Chromium; rerunning with those local OS capabilities
enabled passed. That was an environment restriction, not a product failure.

## 5. Defect inventory and classification

### 5.1 Evaluation integrity

1. Empty selection can satisfy the CI gate: `PROVEN DEFECT`.
2. Partial/smoke selection is indistinguishable from release qualification: `PROVEN DEFECT`.
3. Zero, negative, and out-of-range limits are not validated: `PROVEN DEFECT`.
4. Manifest cardinality, ordered identity, required fields, and duplicate IDs/topics/slugs are not
   preflight-validated: `PROVEN DEFECT`.
5. Aggregate completeness, unique run IDs, and exact ordered result-unit identity are not enforced:
   `PROVEN DEFECT`.
6. Report lacks exact code SHA, canonical manifest identity, uniform prompt/config identity, and an
   artifact digest: `MISSING ENTERPRISE CAPABILITY` that is causally part of P0-2a.
7. Stale telemetry, missing exact telemetry, run-ID mismatch, topic mismatch, incomplete verifier
   status, and most telemetry shape failures: already protected; regression boundary.

### 5.2 Verifier semantics

1. Arithmetic mean confidence ignores verdict status, so high-confidence `unverified` claims can
   satisfy the grounding route: `PROVEN DEFECT`.
2. Valid `[]` is labelled `completed`: `PROVEN DEFECT` when accepted as verified completeness, but
   distinguishing a genuine no-claim draft from extraction omission is unresolved.
3. Max-iteration and cost ceilings precede semantic verification acceptance and route to HITL;
   approval routes to HTML generation: `PROVEN DEFECT`, with mandatory HITL and the HTML cost gate as
   risk mitigations, not semantic correctness.
4. Human approval can reduce unattended-publication risk but cannot convert incomplete machine
   verification into a completed machine-verification fact.

### 5.3 Verifier completeness

The production verifier receives a complete draft plus sources and extracts an unknown-sized claim
set. The repository has no independent expected production claim set. Global exact count is
therefore not justified. Empty or incomplete output can mean either no material claims or silent
omission.

Classification: `UNKNOWN REQUIRING EXPERIMENT`.

A later P0-2b architecture mission must predeclare labeled known-claim and explicit no-claim
fixtures, an independent claim-candidate reference, material/generic claim rules, false-negative
measurement, and acceptance gates. It must not guess a global count or tune until green.

## 6. Causal ordering

P0-2a and P0-2b are not one runtime-coupled change. `scripts/benchmark.py` can be corrected and
tested with fully synthetic subprocess/telemetry records without changing `agent/nodes.py`.
Conversely, any P0-2b causal experiment would rely on evaluation evidence whose current scope and
release identity can false-pass.

Therefore P0-2a must come first. It creates the trustworthy measurement substrate for P0-2b while
isolating deterministic evaluator correctness from model behavior.

## 7. First implementation mission

Mission name:

`P0-2a — FAIL-CLOSED EVALUATION SCOPE, CARDINALITY, AND RELEASE-EVIDENCE BINDING`

### 7.1 Isolated variable

Only benchmark selection, manifest validation, smoke/release qualification semantics, aggregate
completeness, exact-SHA/config/prompt provenance, and report integrity may change.

### 7.2 Exact files allowed

- `scripts/benchmark.py`
- `tests/test_evaluation_integrity.py`
- `.github/workflows/eval.yml`
- new `evals/benchmark_release_contract.json`

No other file may change. If implementation requires another file, stop and return to the
Architect.

### 7.3 Exact files and variables forbidden

Forbidden files include `agent/nodes.py`, `agent/state.py`, `agent/graph.py`, `main.py`, `config.py`,
all prompt files, `evals/topics.json`, verifier golden data, retrieval/chunking/embedding code,
publication/deployment code, `FREEZE.md`, and historical decisions.

Forbidden variables include verifier parsing/scoring/routing, prompts, models, temperatures,
retrieval, corpus, Qdrant contents, HITL behavior, UVR threshold `0.15`, grounding floor `0.60`,
reflection threshold, cost threshold, and any provider retry policy.

## 8. Frozen correct behavior

### 8.1 Release contract

Add `evals/benchmark_release_contract.json` with exactly:

```json
{
  "schema_version": 1,
  "manifest_id": "content-agent-release-topics-v1",
  "manifest_path": "evals/topics.json",
  "manifest_sha256": "2e1a502834252f8c7fe7a3b1136efb9f949d3392f493e6e2102838316b0c7b08",
  "expected_topic_count": 20,
  "ordered_topic_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}
```

The loader must reject missing or extra contract fields, wrong types, unsupported schema, missing
manifest, raw-byte SHA mismatch, non-list manifest, count mismatch, order mismatch, duplicate ID,
duplicate topic, duplicate slug, non-positive/non-integer ID, or missing/blank `topic`, `slug`,
`card_id`, `series`, or `category`. `card_id` is not globally unique by design.

The exact count `20`, ordered IDs `1..20`, manifest bytes/digest, required fields, and release
semantics are immutable properties of `content-agent-release-topics-v1`, not permanent product
assumptions. Future deliberate fixture evolution must add an explicitly reviewed successor such as
`content-agent-release-topics-v2`, with its own schema/contract version as appropriate, manifest
identity and SHA, expected cardinality, ordered identities, semantics, and review evidence. V1 must
never be loosened or mutated in place, and historical V1 evidence must remain interpretable against
the unchanged V1 contract. The loader must never dynamically bless whatever currently exists in
`evals/topics.json` as V1.

All manifest and CLI preflight checks occur before `subprocess.run`.

### 8.2 Explicit CLI modes

The mode is mandatory; there is no ambiguous default.

Smoke form:

```text
python scripts/benchmark.py --mode smoke (--id N | --limit N) [--gate]
```

- Exactly one selector is required.
- `N` must be a positive integer.
- Unknown ID, zero/negative limit, or limit above 20 is a Click usage error with exit 2.
- Empty selection is always exit 2 before subprocess/provider calls.
- A smoke gate can pass/fail its selected units, but its report qualification is always
  `NON_RELEASE` and output must say `SMOKE GATE: PASS — NON-RELEASE` when green.

Release form:

```text
python scripts/benchmark.py --mode release --gate --expected-code-sha <40-lowercase-hex>
```

- `--id` and `--limit` are forbidden.
- `--gate` and `--expected-code-sha` are required.
- Release qualification is CI-only and main-only. Require `GITHUB_ACTIONS=true`,
  `GITHUB_REF=refs/heads/main`, nonempty `GITHUB_RUN_ID`, `GITHUB_RUN_ATTEMPT`, and
  `GITHUB_WORKFLOW_REF`, and require `GITHUB_SHA` to equal `--expected-code-sha`. A feature or
  architecture branch can run only smoke/non-release evidence; it can never emit release PASS.
- Resolve `git rev-parse HEAD`; require exact equality to both the supplied SHA and `GITHUB_SHA`.
- Require no tracked unstaged or staged diff before the first subprocess. Untracked benchmark
  outputs do not invalidate the checkout.
- Select the exact full ordered 20-topic manifest only.
- Any preflight failure exits 2 before provider calls and cannot emit release PASS.

### 8.3 Exact release-unit acceptance

After execution, release qualification requires all of the following:

1. exactly 20 result records;
2. result topic IDs exactly `[1..20]` in manifest order, with no missing, extra, or duplicate unit;
3. 20 nonempty unique run IDs;
4. every subprocess exit status is zero;
5. every exact telemetry record passes all existing validation;
6. every telemetry topic and run ID matches its invocation;
7. every `verification_status` is exactly `completed`;
8. every run is scorable with total verdict count greater than zero; `allow_zero_claims` can never
   earn release qualification;
9. every run has the same expected `prompt_version` and exact `prompt_hashes` resolved before calls;
10. every per-run UVR is at most the unchanged `0.15` threshold;
11. code, manifest, configuration, and report identities below are present and coherent.

One failure makes release qualification `FAIL` and process exit 1. Aggregate means cannot hide a
unit failure.

Immediately after all release units finish and before finalizing the evidence report or printing
release PASS, rerun `git rev-parse HEAD`, `git diff --quiet`, and `git diff --cached --quiet`. The
final HEAD must equal the expected, GitHub, and preflight SHA; tracked unstaged and staged state must
still be clean. Record separate `preflight_code_identity` and `final_code_identity` objects, each
containing resolved SHA plus staged/unstaged cleanliness. Any mid-run HEAD or tracked-state drift
makes `release_qualification=FAIL` and exit 1. Do not lock the repository; detect drift and fail
closed. The evidence digest binds both attestations, proving the same exact code identity before the
run and immediately before PASS.

`release_qualification=PASS` means only that this benchmark evidence contract passed. It is not an
overall product-release, merge, deployment, verifier-golden, or production-readiness approval.

### 8.4 Configuration and evidence identity

Resolve one safe evaluation-configuration object before calls containing:

- `DEEPSEEK_MODEL`;
- SHA-256 of the effective `DEEPSEEK_BASE_URL` string, never credentials;
- SHA-256 of the effective `QDRANT_URL` string, never credentials;
- `QDRANT_COLLECTION`, `QDRANT_EMBEDDING_DIM`, and `KB_N_RESULTS`;
- `TAVILY_MAX_RESULTS` and `TAVILY_MIN_AVG_SCORE`;
- `PROMPT_VERSION` and sorted `PROMPT_HASHES`;
- `GROUNDING_FLOOR` and `UVR_THRESHOLD`.

Never persist API keys, authorization headers, or raw secret-bearing URLs. Compute
`evaluation_config_sha256` over UTF-8 canonical JSON using sorted keys, compact separators, and
`ensure_ascii=False`. Pass explicit effective DeepSeek model/base URL and Qdrant URL/collection in
the child environment so the child cannot silently resolve a different `.env` value.

The final report payload must contain exactly these decision-critical sections:

- schema version `benchmark-evidence-v2`;
- UTC timestamp and mode;
- `release_qualification`: `PASS`, `FAIL`, or `NON_RELEASE`;
- gate requested;
- GitHub Actions ref/run/workflow identity;
- expected code SHA plus separate preflight and final code/clean-state attestations;
- release-contract and selected-manifest identity/count/order;
- safe evaluation configuration plus its digest;
- ordered result records with exact run IDs and telemetry;
- aggregate metrics;
- ordered gate failures.

Compute `evidence_sha256` over canonical JSON of the complete payload before adding the digest field.
Write atomically through a sibling temporary file plus `os.replace`, reread it, recompute the digest,
and validate the schema before printing any PASS. A missing, truncated, mutated, or self-inconsistent
report can never pass.

### 8.5 Workflow integration

Update the manual `.github/workflows/eval.yml` dispatch to expose an explicit `smoke`/`release`
choice, defaulting to smoke. Smoke passes the positive topic-count input. Before dependency setup,
ingestion, or any provider-bearing gate, an exact workflow guard must reject `mode=release` unless
`${{ github.ref }}` is exactly `refs/heads/main`. The release path passes no selector and supplies
`${{ github.sha }}` as the expected code SHA; the benchmark independently revalidates the same ref
and SHA environment. Step names and artifact output must state whether evidence is non-release or
release. A manual dispatch against a feature/architecture ref cannot earn release PASS. Do not
automatically start this paid workflow during implementation.

## 9. Deterministic gates

Add tests that prove, with a subprocess sentinel asserting zero calls on preflight failure:

1. unknown ID and empty selection fail;
2. zero, negative, and out-of-range limits fail;
3. mode is mandatory; selectors are mutually exclusive; release forbids selectors and requires gate
   plus exact SHA and complete main-branch GitHub Actions identity;
4. empty/malformed contract or manifest, digest/count/order mismatch, missing fields, and duplicate
   ID/topic/slug fail;
5. a parameterized V1 immutability case proves changed count, IDs, or manifest bytes cannot keep
   `content-agent-release-topics-v1` and pass;
6. a one-topic smoke pass remains `NON_RELEASE`;
7. a 19/20 or 21/20 release result set fails;
8. missing, extra, duplicate, or out-of-order unit and duplicate/blank run ID fail;
9. nonzero subprocess, missing RUN_ID, missing/stale telemetry, run/topic mismatch, or malformed
   telemetry fail;
10. `parse_failed`, `skipped_cost_gate`, `upstream_failed`, `unknown`, or unscorable/zero-verdict unit
   fails release;
11. mixed prompt version/hash or configuration identity fails;
12. wrong expected SHA, non-main/missing GitHub ref identity, or tracked dirty state fails before
    calls; a workflow-structure test must prove the non-main release guard precedes provider-bearing
    steps and `${{ github.sha }}` is passed as the expected SHA;
13. UVR `0.00` and `0.15` pass while any value above `0.15` fails;
14. a fully synthetic exact 20-unit release passes once and only once;
15. report digest validates, while field mutation, truncation, or missing digest fails;
16. failure after calls writes `FAIL`, never PASS; preflight failure emits no release-qualified report;
17. no report path or logged configuration contains injected API-key/URL credentials;
18. parameterized post-run drift simulates a changed final HEAD, final staged change, and final
    unstaged change after clean preflight; each writes/returns release `FAIL`, exits nonzero, and
    never prints release PASS.

Retain and rerun every existing `tests/test_evaluation_integrity.py` protection.

## 10. Dependency-integration and regression gates

Run, in order, without provider credentials:

```text
uv sync --frozen
uv run pytest tests/test_evaluation_integrity.py -q
uv run pytest tests/test_verifier_contract.py tests/test_failure_injection.py -q
uv run pytest tests/ -q
uv run ruff check --select E9,F63,F7,F82 .
git diff --check
```

Expected baseline before implementation is `45` focused tests and `215` full tests. The final count
must increase only by the new deterministic P0-2a tests. Existing P0-1 browser/security, API,
publication, verifier-contract, and failure-injection tests must remain green.

## 11. Evaluation and AI causal gates

P0-2a changes deterministic evaluation infrastructure only. No paid AI evaluation, prompt-quality
run, model call, Qdrant ingestion, or production service is required to prove isolated correctness.

Do not run the manual paid workflow during implementation. After merge is separately authorized and
completed, any future release claim requires one predeclared full release run on the exact final
integration SHA. That later run needs explicit provider-cost authorization and is evidence for the
release gate only; it is not implementation or merge approval.

No AI causal validation applies because model behavior, prompts, retrieval, and thresholds are
forbidden from changing. P0-2b will require its own predeclared causal experiment.

## 12. Failure and abuse cases

Fail closed on malformed JSON, missing contract, empty manifest/selection/result set, duplicate or
unknown IDs, reordered units, missing expected units, invalid limits, duplicate run IDs, stale or
mismatched telemetry, incomplete/unscorable verification, mixed prompt/config identity, wrong or
dirty code identity before or after execution, non-main/missing CI ref identity, report
write/read/digest failure, and secret-bearing evidence.

Smoke, partial, historical, branch-local, unbound, `N=0`, or non-final-SHA evidence must never be
described as release PASS.

## 13. Kill conditions

Stop implementation and return `P0-2a-ARCHITECTURE-BLOCKED` if:

- `origin/main` or the implementation base differs from exact `ca29d32`;
- the independently authorized final architecture object cannot be resolved exactly or is not the
  one documentation-only child of reviewed architecture commit `5451b53`;
- the canonical 20-topic manifest or its supplied digest cannot be reproduced;
- another file is required beyond the four-file allowlist;
- a prompt, model, threshold, retrieval, verifier, HITL, publication, deployment, or corpus change is
  needed;
- deterministic tests would require real provider credentials/calls;
- any release PASS can occur with zero, partial, duplicate, incomplete, unscorable, stale, mixed, or
  unbound evidence;
- identity binding would expose a secret;
- passing needs retry-until-green, threshold movement, or suppression of a regression;
- unrelated work overlaps an allowed file;
- the full deterministic regression or fatal Ruff tier fails for a change-caused reason.

## 14. Git and ownership discipline

The Implementation Agent must:

1. fetch remote refs and verify `origin/main` remains exact `ca29d32`;
2. verify the architecture branch contains exact reviewed commit `5451b53` and that the exact final
   architecture SHA supplied with human authorization is the branch head and has `5451b53` as its
   parent;
3. read the specification without checkout mutation using
   `git show <AUTHORIZED_FINAL_ARCHITECTURE_SHA>:docs/P0_2_VERIFIER_EVALUATION_INTEGRITY_ARCHITECTURE.md`
   or an equivalent exact-object read;
4. create a clean isolated worktree and branch `fix/p0-2a-evaluation-integrity` from exact
   `ca29d32`, never from the unmerged documentation branch;
5. reproduce the zero-topic control before editing;
6. change only the four allowed files;
7. commit and push the branch;
8. report exact commit, parent, diff, test/lint evidence, and zero provider calls;
9. stop for independent review.

The Implementer must not merge, push main, publish, deploy, call providers, change thresholds, or
approve its own work. Human approval is required before merge.

## 15. Required Implementer response

Return exactly:

1. exact fetched `origin/main` and implementation base;
2. branch/worktree and dirty-state verification;
3. pre-fix reproduction and exit code;
4. files changed;
5. behavior implemented;
6. deterministic test commands and exact counts;
7. full regression and fatal Ruff results;
8. evidence that preflight failures made zero subprocess/provider calls;
9. evidence that synthetic full release is the only release PASS path;
10. evidence that smoke is always non-release;
11. report/config/manifest/code identity evidence;
12. threshold/prompt/model/retrieval invariance confirmation;
13. provider calls and spend: exactly zero;
14. pushed commit SHA and parent;
15. `READY FOR INDEPENDENT REVIEW — DO NOT MERGE`, or the exact kill condition and
    `P0-2a-ARCHITECTURE-BLOCKED`.

## 16. Fresh-thread implementation prompt

Use the following prompt only after independent review and human authorization:

> You are the IMPLEMENTATION AGENT (Cursor or Codex) for the public repository
> `https://github.com/anudeepreddy332/content-agent`. Assume no prior conversation context.
>
> Implement exactly `P0-2a — FAIL-CLOSED EVALUATION SCOPE, CARDINALITY, AND RELEASE-EVIDENCE
> BINDING`. Fetch remote refs. Require `origin/main` to be exact
> `ca29d32b4869269daa47142615d298580a577a77`. Verify exact reviewed architecture commit
> `5451b53b11a07e41c7a0c7d9f5e8f526cc55131a` exists on
> `origin/chore/p0-2-evaluation-integrity-architecture`; verify the exact final architecture SHA
> supplied with human authorization is that branch's head and its parent is exact `5451b53`.
> Read the frozen specification with
> `git show <AUTHORIZED_FINAL_ARCHITECTURE_SHA>:docs/P0_2_VERIFIER_EVALUATION_INTEGRITY_ARCHITECTURE.md`.
> Do not check out or implement from the documentation branch. If any identity differs, stop with
> `P0-2a-ARCHITECTURE-BLOCKED`. Create a clean isolated worktree and branch
> `fix/p0-2a-evaluation-integrity` from exact product main `ca29d32`. Reproduce
> `python scripts/benchmark.py --id 999 --gate` selecting zero topics and exiting 0 before editing.
>
> You may change only `scripts/benchmark.py`, `tests/test_evaluation_integrity.py`,
> `.github/workflows/eval.yml`, and new `evals/benchmark_release_contract.json`. Implement every
> frozen CLI, immutable V1 manifest, scope/cardinality, main-only GitHub release ref, preflight and
> post-run code/clean-state attestation, exact-SHA, prompt/config, aggregate, report-digest,
> smoke/release, atomic-write, secret-exclusion, and fail-closed rule in the architecture document.
> Do not modify `agent/nodes.py`, `main.py`, `config.py`, prompts, `evals/topics.json`, thresholds,
> verifier behavior, retrieval, corpus, HITL, publication, deployment, `FREEZE.md`, or historical
> decisions. Do not make provider calls or run the paid workflow.
>
> Add every frozen deterministic case, then run the focused suite, verifier/failure regressions,
> full tests, fatal Ruff tier, and `git diff --check` exactly as specified. Do not lower thresholds,
> retry until green, hide failures, expand scope, merge, publish, deploy, or approve your own work.
> Commit and push the implementation branch, then return the 15-item response required by the
> architecture document and stop with `READY FOR INDEPENDENT REVIEW — DO NOT MERGE`. If any kill
> condition occurs, stop without redesign and return `P0-2a-ARCHITECTURE-BLOCKED` with evidence.

## 17. Confidence

Numerical confidence: `0.98`.

The deterministic P0-2a root defect, causal isolation, correct first mission, gates, and stop
conditions are frozen. Confidence is not 1.00 because the later P0-2b production completeness
contract deliberately remains unresolved and requires its own experiment.
