# content-agent — Production Readiness Audit

Scope: read-only audit of the committed codebase on `main` (tip `80e6a83`) plus the
demo/cloud-publish work merged from `feature/demo-ui`. Every claim below cites the file(s)
it was verified against. Ratings: **COMPLETE**, **PARTIAL** (gap stated precisely), **MISSING**.

---

## 1. Eval & benchmarking — PARTIAL

**What exists and is real:**
- `evals/verifier_golden_test.py` — 12 fixed claims (verbatim/paraphrase/absent/false ×
  substantive/generic), each verified independently against a known source. Real exit code:
  `sys.exit(0 if (ground_ok >= 11 and spec_ok >= 10) else 1)` (line 117). This is a genuine
  gate, not a printed suggestion.
- `scripts/benchmark.py --gate` — runs N topics from `evals/topics.json` (20 topics) through
  the real CLI, computes per-run UVR, and **fails the process** (`sys.exit(1)`, lines 168-183)
  if any run errors outright or any run's UVR > 0.15. This is the locked grounding gate from
  `DECISIONS.md`/`CLAUDE.md`, enforced in code, not just documented.
- `scripts/check_telemetry_fields.py` — asserts every required telemetry field is present and
  that derived fields are internally consistent (attribution counts sum to claim count,
  `source_kind` present on every claim, `chunk_index` present on every KB result). Real
  `sys.exit(1)` on failure.
- `.github/workflows/eval.yml` wires all three of the above as sequential gates ("Gate 1/2/3")
  plus a Qdrant service container and KB ingest step. This is a legitimate, runnable eval
  pipeline — not vestigial.

**The gap:** `eval.yml` is `on: workflow_dispatch` only (line 3) — it costs real API money
(documented as "~$0.07" in the workflow name) and is **never triggered automatically**, not on
push, not on PR. The only workflow that runs automatically on every push/PR is `ci.yml`
(`.github/workflows/ci.yml`), which runs fatal-tier lint (`ruff check --select E9,F63,F7,F82`,
line 20) and `pytest tests/` (line 36) — and every test in `tests/` is mocked at $0 (confirmed
by `tests/conftest.py`'s `FakeLLMClient`, and `os.environ["API_SYNC"]="1"` in
`tests/test_api_stream.py`/`test_api_publish.py`). **None of the 54 tests in `tests/` touch
grounding quality, the golden verifier fixture, or the benchmark gate.** A PR that regresses
the verifier prompt or the grounding rubric can merge to `main` with a fully green CI badge,
because the only thing that would catch it (`eval.yml`) requires a human to manually click
"Run workflow."

**Also ad hoc, wired into nothing:** `evals/prompt_evals/{test_draft,test_verify,test_reflect}_prompt.py`
(schema-stability checks, run manually per their own docstrings) and `scripts/retrieval_eval.py`
(25-query golden retrieval set, recall@k) are real and well-built but are not referenced by
either GitHub workflow — they are run by hand and the results live in commit history /
DECISIONS.md narrative, not in an enforced gate.

**Verdict:** the eval harnesses themselves are complete, correctly designed, and have real
enforced exit codes — this is well above the median portfolio project. What's missing is
automatic enforcement: the grounding-quality gate is opt-in, not a merge requirement.

---

## 2. Observability — PARTIAL

**What exists and is real:**
- Structured logging via `structlog` with `JSONRenderer` to stdout (`observability/logger.py`,
  all 8 lines) — every node (`draft_node`, `retrieve_node`, `verify_node`, `reflect_node`,
  `git_node`, `html_gen_node`, `html_revise_node`, the routers, `api/server.py`) logs through
  `get_logger(name)` and consistently includes `run_id` on every call site (verified across
  `agent/nodes.py`).
- Prompt-version hashing is real and automatic: `config.py` lines 79-99 compute a SHA-256
  hash per prompt file at import time (`PROMPT_HASHES`) and a combined `PROMPT_VERSION`; it
  fails loudly (`FileNotFoundError`) if a prompt file is missing. This is stamped into every
  telemetry record (`main.py` `_write_telemetry`, lines 48-54) — there is no way for a prompt
  to change silently without the hash changing.
- Telemetry completeness: `main.py::_write_telemetry` (lines 31-141) writes per-run JSON with
  run_id, full grounding_report, full web_sources/kb_results (truncated to 2000 chars each,
  not just counts), per-iteration `iteration_metrics` (so a 2-iteration revise loop is fully
  reconstructable, not just the final draft), attribution breakdown, cost, tokens, latency per
  node, and `error_log`. `scripts/check_telemetry_fields.py` enforces 27 required fields exist
  — this is a genuine, tested reconstructability contract, not just a wish.
- Per-iteration evidence: `agent/nodes.py::verify_node` (lines 793-815) appends a full snapshot
  of that iteration's grounding_report to `iteration_metrics`, specifically because (per the
  inline comment) the top-level `grounding_report` only reflects the *last* iteration — a real,
  previously-identified reconstructability bug that was fixed, not just claimed fixed.

**The gaps:**
- `tools/query_kb.py` uses bare `print()` for its three error paths (lines 329, 337, 340)
  instead of the structlog logger every other module uses. These errors (Qdrant search
  failure, missing rank_bm25, BM25 failure) will not appear in the structured JSON log stream
  in production — they go to stdout as plain text, breaking the "every run reconstructable
  from logs" property for retrieval-layer failures specifically.
- No log correlation/request ID beyond the manually-passed `run_id` kwarg — there is no
  middleware that auto-attaches it, so any log line where a developer forgets to pass
  `run_id=...` (none currently, but nothing enforces it) silently drops out of reconstructability.
- No external log sink: logs go to container stdout only (`docker-compose.prod.yml` has no
  logging driver override, `docs/deploy/DEPLOY.md` line 55 explicitly says structlog-to-stdout "IS the log
  sink; a platform collector ingests it" — i.e. log persistence beyond `docker logs` is assumed
  to be the operator's problem and is not configured anywhere in this repo).
- B4 limitation (documented, not hidden): the in-memory `REGISTRY` in `api/server.py` is not
  rehydrated from the SqliteSaver checkpoint on restart (`FREEZE.md` item 1, `docs/deploy/DEPLOY.md`
  lines 42-52) — a paused run's HTTP-visible state is lost on restart even though the
  underlying LangGraph checkpoint survives. This is a real, acknowledged gap in
  reconstructability for in-flight (not completed) runs specifically.

**Verdict:** completed runs are fully reconstructable from telemetry + logs (and this is
verified, not just claimed, by `check_telemetry_fields.py`). The two real gaps are the
inconsistent logging in the KB layer and unrehydrated in-flight run state after a restart.

---

## 3. Monitoring — MISSING

A repo-wide search for `sentry|prometheus|grafana|datadog|pagerduty|alertmanager|uptime`
across `*.py`, `*.md`, `*.yml`, `*.toml` returns zero matches. There is:
- No metrics endpoint (no `/metrics`, no Prometheus client import anywhere).
- No alerting of any kind — not on cost-gate breaches, not on `git_status: failed`, not on
  repeated `unverified` spikes, not on the API process being down.
- No uptime check / external healthcheck wiring. `GET /health` exists (`api/server.py` line
  249) and the Docker `HEALTHCHECK` directive uses it (`Dockerfile` line 38-39), but that only
  drives `docker compose ps` / container restart policy (`restart: unless-stopped` in both
  compose files) — it does not notify a human.
- No dashboard. The only place aggregate numbers surface is `scripts/benchmark.py`'s printed
  summary and the JSON files under `outputs/benchmark_results/` — there is no Grafana/Looker/
  anything consuming them.

**Verdict:** genuinely missing, not partially built. For a single-operator demo/portfolio
project this is a defensible scope cut (documented nowhere as a cut, though — it's just
absent), but it is the single most "not production" item in this audit. Anyone running this
unattended would only discover an outage by trying the demo URL or `docker compose ps`.

---

## 4. Reliability — COMPLETE (for the scope it covers)

**What exists and is real, with test evidence:**
- Retry policy on every LLM call: `agent/nodes.py::_llm_call` (lines 60-91) retries exactly
  `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError` up to 3
  attempts with exponential backoff (2s→4s→8s), and explicitly does **not** retry
  `AuthenticationError`/`BadRequestError` (these can never succeed). Verified by
  `tests/test_failure_injection.py::test_llm_call_no_retry_on_auth_error` (asserts exactly 1
  call) and `test_llm_call_retries_rate_limit_then_reraises` (asserts exactly 3 calls then
  reraise). Tavily has its own identical retry wrapper (`tools/web_search.py::_call_tavily`,
  lines 67-88).
- Cost gates enforced at every LLM-leading edge, not just documented: `verify_node` (line 695),
  `reflect_node` (line 846), `html_gen_node` (line 1180) all check
  `total_cost_usd >= COST_GATE_USD` and skip the LLM call entirely (not just skip the loop).
  `route_after_reflect` (line 1659) also checks the gate independently so a costly draft can't
  loop again even if grounding is bad. Verified by
  `test_verify_cost_gate_skips_llm_entirely` (asserts the mocked client's `.create` is never
  called) and `test_route_cost_gate_forces_hitl_over_revision`.
- Graceful degradation, proven not asserted: `retrieve_node` survives Tavily raising
  `ConnectionError` (logs to `error_log`, returns `[]`, does not crash) and Qdrant being down
  (`tools/query_kb.py::_collection_exists` catches `Exception` broadly and returns `False`,
  so `query_kb` returns `[]`). Both directions tested independently
  (`test_retrieve_tavily_errors_logged_not_fatal`, `test_query_kb_qdrant_down_returns_empty`,
  `test_retrieve_survives_kb_down` — the last one specifically proves a KB outage does not
  affect the (still-working) web path).
- Malformed LLM output handled without crashing: both `draft_node` and `verify_node` catch
  `json.JSONDecodeError`/parse failures and degrade to a placeholder/empty result rather than
  raising — but the raw output is preserved in the degraded output for debugging
  (`test_draft_malformed_output_degrades_gracefully` asserts the raw text survives into
  `draft_markdown`). The verifier's tolerant `_extract_json_array` (lines 107-136) has its own
  three tests including a true-negative (`test_extract_json_array_raises_on_garbage` — garbage
  in must still raise, not silently return `[]`, so a parse failure is distinguishable from a
  genuine empty result upstream).
- The content-freeze guard in `html_revise_node` (lines 1333-1388) is a reliability mechanism,
  not just a content rule: it computes a word-multiset diff between the original and
  LLM-revised HTML and **discards** the revision if drift exceeds 2 words or the output isn't
  valid-looking HTML, falling back to the original — this is tested
  (`tests/test_api.py::test_html_revise_discards_content_change` and
  `test_html_revise_applies_layout_only`) and protects against a misbehaving model corrupting
  already-approved content, not just a prompt-following nicety.
- `git_node` wraps every git operation in try/except (`GitCommandError` and bare `Exception`,
  lines 1602-1608) and the `finally` block restores the original branch even on failure
  (lines 1610-1617) — a failed publish does not leave the target repo on a feature branch.

**One scope-bounded note (not a gap in what's tested, a gap in what's covered):** the new
`POST /ui/runs/{id}/publish` endpoint (`api/server.py` lines 422-460) has its own retry-free
`subprocess.run(..., timeout=30)` for the actual `git push` and converts both a timeout and a
non-zero exit into an HTTP error — but unlike the LLM/Tavily paths, a transient push failure
(e.g. a momentary network blip to GitHub) is not retried, only surfaced. This is tested for
correctness (`tests/test_api_publish.py`, 11 tests covering 404/409/500/200 paths) but not for
resilience — a flaky network during the live demo means clicking "Publish" again, not an
automatic retry. Given this is a human-triggered, human-watched action (not a background
process), the lack of retry here is a reasonable design choice, not an oversight, but it is the
one LLM/Tavily-style failure mode in the new code that doesn't get the same treatment as the
rest of the pipeline.

**Verdict:** COMPLETE for everything the test suite claims to cover, and the test suite's
claims are independently verifiable in the source (not just asserted in test names). This is
the strongest section of the audit.

---

## 5. Security — PARTIAL

**Covered and verified:**
- Slug sanitization is allowlist-based, by construction: `main.py::_make_slug` (lines 144-150)
  strips everything outside `[a-z0-9-]`, collapses repeats, strips leading/trailing `-`, caps
  at 80 chars. This reaches the filesystem (`html_gen_node`'s `archive_path`, `git_node`'s
  `dest_path = repo_path/filename`) and git branch/tag names (`feature/article-{slug}`,
  `v-{date}-{slug}`). Six dedicated regression tests in `tests/test_slug.py` cover path
  traversal (`../../etc/passwd` → `etc-passwd`), git-ref/shell injection
  (`feature/../../main; rm -rf /` → `feature-main-rm-rf`), a closed-output-alphabet fuzz test
  over five adversarial inputs including a null byte, and the empty-slug-must-be-rejected case
  (enforced by both the CLI's `click.UsageError` and the API's `HTTPException(422, ...)` in
  `api/server.py` lines 256-258 and 322-324).
- Auth is fail-closed, not fail-open: `api/server.py::require_auth` (lines 216-222) raises
  `503` if `API_BEARER_TOKEN` isn't configured at all (refuses to silently allow unauthenticated
  access) and uses `hmac.compare_digest` (constant-time comparison, not `==`) for the token
  check. The same fail-closed pattern is independently re-implemented for the two endpoints
  that take the token via query param instead of header (`ui_events`, `ui_preview`) because
  `EventSource`/plain links can't set headers — both are tested
  (`tests/test_api_stream.py::test_sse_requires_token`, the analogous preview test in
  `test_api_publish.py`).
- Network isolation: `docker-compose.prod.yml` gives Qdrant no `ports:` mapping at all (line
  8-19) — it is reachable only as `qdrant:6333` inside the compose network, never from the
  host or internet. The demo override (`docker-compose.demo.yml`) goes further and removes the
  app's own host port binding via the compose `!reset` tag (verified by running
  `docker compose config` during this audit — the merged output for `app` has no `ports:` key
  at all), so in the demo deployment Caddy is the *only* process with a public-facing port.
- Secrets never reach the image or git history: `.dockerignore` excludes `.env` (line 1) and
  `*.md` generally; `.gitignore` excludes `.env`; `.env.example` documents required vars without
  values. The fork's push credential is documented (`docs/deploy/DEPLOY_DEMO.md` lines 33-36) to live only
  in `fork-clone/.git/config` on the VM, explicitly never in `.env` or a compose file — this is
  a stated design constraint, not yet independently verifiable from this repo alone since the
  fork-clone directory is host-side and out of scope.
- Non-root container: `Dockerfile` line 32 creates `uid 10001` and runs as that user (line 33).
  Note this line changed recently (`git log -p` on `Dockerfile` shows commit `884ec8b`,
  "Modified Run useradd command with app .cache added") from
  `chown -R appuser:appuser /app` to the narrower
  `chown -R appuser:appuser /app/.cache /home/appuser` — this is a *more* least-privilege
  change (the bind-mounted `outputs/`/`fork-clone/` directories get their permissions from the
  host-side `chown -R 10001:10001` in `docs/deploy/DEPLOY.md`/`docs/deploy/DEPLOY_DEMO.md`, not from the image), and the
  commit message states it was tested against a real EC2 deployment.
- Prompt-injection surface is documented, not silently ignored: retrieved web/KB content and
  the user-supplied topic both flow into LLM prompts unsanitized (by design — sanitizing would
  defeat the purpose of grounding), and the standing mitigation is that **HITL is mandatory on
  both gates** (`agent/nodes.py::hitl_node`/`hitl_html_node` both check
  `HITL_AUTO_APPROVE` and default to interrupt-and-wait; the API forces
  `os.environ["HITL_AUTO_APPROVE"] = "0"` at import time, `api/server.py` line 42) — a human
  reads every draft and every rendered HTML before anything reaches git, so an injected
  instruction in a retrieved source has to survive human review to do anything. This is the
  documented mitigation in `agent.md`/`DECISIONS.md`/`CLAUDE.md` and it is mechanically true in
  the current code (verified by reading `hitl_node`/`hitl_html_node` directly, not just citing
  the docs).

**Gaps:**
- No rate limiting anywhere in `api/server.py`. The bearer token gates *who* can call
  `POST /ui/runs`, but not *how often* — a leaked or guessed-adjacent token (the comparison is
  constant-time, but tokens aren't rotated or scoped) could spin up unbounded runs, each
  costing real DeepSeek/Tavily money. The single-worker `EXECUTOR` (`max_workers=1`) bounds
  *concurrency* but not *queue depth* or *total spend* — `COST_GATE_USD` caps spend per-run, not
  per-token or per-day.
- No CORS configuration at all (no `CORSMiddleware` in `api/server.py`). This defaults to
  same-origin-only from a browser's perspective, which is *accidentally* safe for this
  same-origin SPA, but it's an absence, not a deliberate policy — there's nothing stopping a
  future change from needing CORS and someone reflexively setting `allow_origins=["*"]`.
- `GET /ui/runs/{run_id}/events` and `GET /ui/runs/{run_id}/preview` pass the bearer token as a
  URL query parameter (`api/server.py` lines 361-369, 397-407) — this is explicitly flagged in
  the code's own comments as "Query-param tokens can land in access logs; acceptable for a
  single-user demo," which is an honest, documented tradeoff rather than a hidden gap, but it
  is a real gap if this token is ever reused for anything beyond the single-operator demo.
- The publish endpoint's `subprocess.run(["git", "push", remote, "main"], ...)` (line 446)
  builds an argv list (not a shell string), so it is not shell-injectable via `remote` even
  though `remote` comes from `os.environ.get("PUBLISH_REMOTE", "origin")` — but `PUBLISH_REMOTE`
  is operator-set environment, not request input, so this was never a realistic injection
  vector in the first place; noting it because the task asked for the injection surface to be
  characterized, not because it's a finding.

**Verdict:** the security-sensitive paths that were clearly identified and worked (slug
sanitization, auth, network isolation, non-root, prompt-injection mitigation via mandatory
HITL) are genuinely solid and test-backed. The gaps are the ones that weren't explicitly in
scope for any milestone (rate limiting, token scoping/rotation, CORS posture) — absent rather
than broken.

---

## 6. Deployment — PARTIAL

**Covered and verified:**
- Two deployment runbooks exist and are detailed enough to follow: `docs/deploy/DEPLOY.md` (single-VM,
  loopback-only, SSH-tunnel access — the original freeze posture) and `docs/deploy/DEPLOY_DEMO.md`
  (adds Caddy + a public port for the demo, written and verified as part of this session's
  prior work). Both correctly call out the uid-10001 bind-mount chown requirement and the B4
  registry-volatility limitation.
- Rollback is scripted, not just described: `scripts/rollback_publish.sh` reverts the
  publish's `--no-ff` merge via `git revert` (handles both the merge-commit case with `-m 1`
  and the plain-commit fallback case, lines 43-47), requires the operator to type the slug to
  confirm (line 39-40), and explicitly does **not** auto-push — it prints the push command for
  the operator to run (lines 57-58), preserving the same human-gated-push posture as publish
  itself. **Caveat:** this script has no automated test (it's bash, and the test suite is
  Python/pytest-only) — its correctness is attested in `DECISIONS.md` (2026-06-16 entry) as
  having been manually validated against a fork twice, including a real bug fix to the original
  version (the old `rm -f` was deleting articles on a *modification* rollback when it should
  have restored the prior version). It works because it was debugged against a real repo, not
  because it's covered by CI.
- Publish-posture safety is real, not just asserted: `git_node` (`agent/nodes.py` lines
  1463-1626) has no code path that calls `git push` — grep confirms `push` does not appear
  anywhere in `git_node`'s body. The only `git push` in the entire codebase is the new
  `ui_publish` endpoint (`api/server.py` line 446), which is a separate, explicitly
  human-triggered HTTP endpoint gated on `git_status` already being `merged`/`tagged_and_merged`
  (i.e., `git_node` already ran and already refused to push). This cleanly preserves the
  "agent can merge locally, only a human pushes" property end-to-end, including in the new
  cloud-publish surface — the new endpoint is additive risk surface, but it does not weaken the
  no-autonomous-publish guarantee, because it still requires a separate explicit POST that only
  succeeds after a human has already approved both HITL gates.
- The EC2/Caddy path is real, not aspirational: `docker-compose.demo.yml` was validated during
  this audit with `docker compose config` (Compose v5.1.4) — the merged config correctly shows
  `app` with no `ports:` key and a `caddy` service on 80/443 reverse-proxying to `app:8000`.
  The commit history (`884ec8b`, `80e6a83`, dated after the demo work) and their commit messages
  ("Demo works on local host and also deployed on aws ec2 instance. tested and working.")
  indicate this was actually run against a live EC2 instance, not just written and never
  exercised — though this audit cannot independently re-verify a live EC2 box from the repo
  alone.

**Gaps:**
- **No Docker Hub / registry path exists.** A repo-wide search for
  `docker hub|dockerhub|docker\.io|registry|ECR` (excluding the unrelated B4 "in-memory
  registry" hits) finds nothing. There is no CI job that builds and pushes an image anywhere
  (`ci.yml` only lints and tests Python, it never runs `docker build`). The deployment model in
  both `docs/deploy/DEPLOY.md` and `docs/deploy/DEPLOY_DEMO.md` is "clone the repo onto the VM and `docker compose
  build` from source" — meaning every deploy rebuilds from a git checkout on the target
  machine rather than pulling a pre-built, scanned, versioned image. This works, but it is not
  the registry-based deployment path the task description's framing ("EC2/Caddy/Docker Hub")
  implies should exist; if a Docker Hub path was intended, it has not been built.
- No staging environment in the literal sense. `FREEZE.md` already documents this as a known,
  accepted gap ("B6 staging/publishing validation — INTENT MET via fork-based validation + HITL
  promotion gate; literal separate staging-branch environment DEFERRED post-freeze") — the fork
  itself (`themachinist-website-fork`, Netlify-deployed) plays the staging role for content, but
  there is no staging deployment of the *application* itself (no separate compose/VM that gets
  exercised before a prod-equivalent deploy).
- Rollback is scripted but not drilled in CI — there is no test (manual or automated) that runs
  `rollback_publish.sh` as part of any pipeline; its correctness rests entirely on the two
  manual fork validations recorded in `DECISIONS.md`. A future change to `git_node`'s commit
  message format or branch-naming convention could silently break the script's `--grep` match
  (line 25) with nothing catching it until an actual rollback is needed.
- B4 (registry volatility on restart) and single-worker throughput are both carried forward as
  accepted limitations rather than fixed — correctly disclosed in `FREEZE.md`, but still true
  today; nothing in the EC2/demo work changes this.

**Verdict:** the publish-safety property (the single most important "production" guarantee in
this project) is genuinely solid and verifiably preserved end-to-end including in the new cloud
path. The deployment *mechanics* (runbook quality, rollback scripting, network posture) are
good. What's missing is the registry/image-supply-chain piece and CI-verified rollback.

---

## Pending items — prioritized

| # | Item | Blocks production? | Priority | Rough effort |
|---|------|---------------------|----------|---------------|
| 1 | Wire a cheap subset of the eval gate into per-PR CI (at minimum: `evals/verifier_golden_test.py`, ~$0.02) so a grounding regression can't merge silently | **Yes** | Do now | 0.5 day (add a `pull_request`-triggered job with a `secrets`-gated cost; or a scheduled nightly run against `main` as a cheaper compromise) |
| 2 | Any monitoring/alerting at all — at minimum an external uptime check on `/health` and an alert on `git_status: failed` in telemetry | **Yes**, for unattended operation; **No** for a supervised demo | Do now if this runs unattended; defer if always human-watched | 0.5–1 day (e.g. a free uptime-checker pinging `/health`, plus a log-based alert) |
| 3 | Fix `tools/query_kb.py`'s three `print()` error paths to use the structlog logger like every other module | No | Do now (cheap) | 30 minutes |
| 4 | Decide and either build or explicitly drop the Docker Hub / image-registry deploy path | No | Defer (current source-build-on-VM path works) | 1 day if built (CI image build + push + VM pulls instead of building) |
| 5 | Rate limiting / per-token spend cap on `POST /runs` and `POST /ui/runs` | No (single-operator demo; token isn't public) | Defer unless the token is ever shared beyond one operator | 0.5 day |
| 6 | Add a CI smoke test (or a documented manual checklist with no test) that exercises `scripts/rollback_publish.sh` against a throwaway fork repo | No | Defer | 0.5 day |
| 7 | Rehydrate the API's in-memory `REGISTRY` from the SqliteSaver checkpoint on startup (closes the B4 limitation) | No (documented, mitigated by "don't restart mid-review") | Defer (already scoped as post-freeze in `FREEZE.md`) | 1–2 days |
| 8 | Explicit CORS policy decision (even if the answer is "same-origin only, by design") documented in `api/server.py` | No | Defer | 1 hour |
| 9 | Token-in-query-param exposure on `/ui/runs/{id}/events` and `/preview` — rotate or scope tokens if this demo is ever exposed beyond one operator's own session | No (already disclosed in-code as an accepted tradeoff) | Defer | n/a unless threat model changes |

**Overall read:** this is a notably more rigorous project than most portfolio pieces — the
reliability and security work in particular is backed by real tests that independently verify
the claims, not just docstrings asserting them. The honest gaps are concentrated in exactly the
places the task description predicted: monitoring is absent, and the eval gates — while real
and well-built — are not load-bearing in the automated pipeline that actually decides what
reaches `main`.
