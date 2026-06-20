# content-agent — Production Readiness Audit (FINAL)

Scope: read-only audit of `main` at tip `9a1ee17` (cleanup + docs-sync + LangSmith-fixes all
merged), re-run after the prior audit's four "do-now" fixes landed (tag `v7-audit-fixes`) and
after LangSmith tracing was made default-on with token/cost attached. Every claim cites the
file(s) it was verified against. Ratings: **COMPLETE**, **PARTIAL** (gap stated precisely),
**MISSING**.

Audit date: 2026-06-20. Suite state at audit time: `uv run pytest tests/` → **62 passed** (9.2s,
$0, verified this session). Working tree carried 15 files of uncommitted ruff-style cleanups
(import-splitting, unused-import removal, no-placeholder f-string fixes) — **behavior-neutral**,
confirmed by reading the diff; they are not part of this audit's findings but should be committed
or stashed so `main`'s working tree is clean.

This supersedes the prior audit (tip `80e6a83`). What changed since then is summarized per
section under **Since last audit**.

---

## Executive summary — the three questions

### a. Is this truly production-grade? Would a big-tech CTO accept it as a portfolio piece?

**Yes, as a single-operator, supervised system — and it is well above the median portfolio
project.** The distinguishing feature is not the feature list; it is the *discipline*. Reliability
and security claims are backed by tests that independently verify the behavior in source (retry
policy asserts exact attempt counts, cost gates assert the LLM client is never called, slug
sanitization has a closed-alphabet fuzz test, content-freeze discards drifting revisions). The
publish path has a real, end-to-end-preserved safety property (the agent can merge locally but
*cannot* push — every live publish needs a human `git push`), and that property is verifiably
intact even in the new cloud-publish endpoint. The project tracks a calibrated primary metric
(SV) with prompt-hash comparability, logs structured JSON per node, traces cross-node with
token/cost, and ships two real runbooks plus a scripted, debugged-against-a-fork rollback.

A CTO would *not* mistake it for a multi-tenant SaaS — and it doesn't claim to be one. Within its
declared scope (one operator, one article at a time, human-in-the-loop twice), it is engineered
to a professional standard, with its limitations named first-class in `FREEZE.md` rather than
hidden. The honesty of the gap disclosure is itself a senior signal.

### b. Can I claim professional experience with this? What seniority does it demonstrate?

**Yes.** This demonstrates **senior / staff-level individual-contributor** judgment, specifically
in applied-ML / LLM systems engineering. The evidence isn't "I built an agent" — most candidates
can say that. It's:
- *Experimental rigor*: pre-registered A/B experiments with noise bands, frozen-cache protocols,
  and rejected hypotheses kept on record (query reformulation, blind re-roll, MAX_ITERATIONS>2).
  This is the single rarest thing to find in a portfolio and it's the strongest interview asset.
- *Metric design under adversarial conditions*: recognizing that UVR-alone rewards vagueness and
  constructing SV as a no-loss co-condition is a staff-level measurement insight.
- *Production hardening as a first-class concern*: failure-injection suite, enforced CI gates with
  real exit codes, durable HITL via a checkpointer, non-root network-isolated containers, a
  human-gated publish safeguard.

Honest framing for a resume/interview: *"Designed, hardened, and deployed a grounded LLM content
pipeline — single-agent LangGraph, FastAPI + durable human-in-the-loop, containerized to EC2 with
TLS — with a measured grounding metric enforced in CI and a no-autonomous-publish safety property."*
Do **not** claim multi-tenant scale, high availability, or unattended autonomous operation — the
system is explicitly none of those, and a sharp interviewer will find the single bearer token and
the `max_workers=1` executor in five minutes. Claiming the scope you actually built is stronger
than overclaiming; the rigor is what carries the seniority signal.

### c. Biggest gaps blocking real-world multi-tenant / HA use — fix now vs. defer?

The system is architected for *one* operator. Going multi-tenant/HA is not a hardening pass; it is
a re-architecture. The load-bearing blockers:

| Gap | Why it blocks multi-tenant / HA | Before demo? |
|---|---|---|
| **Single shared bearer token, no per-tenant identity/scoping** (`api/server.py::require_auth`) | No tenant isolation, no per-tenant quotas, no revocation of one tenant without rotating everyone | **Defer** — the demo is single-operator; this is a re-architecture, not a fix |
| **No rate limiting / per-day spend cap** (`COST_GATE_USD` caps *per-run*, not per-token/day/tenant) | One caller can spin unbounded paid runs; no cost ceiling across runs | **Defer** for supervised demo; **required** before any shared exposure |
| **`max_workers=1` single executor + single VM** (`api/server.py`, `DECISIONS.md` B4) | No horizontal scale, no HA; runs serialize; the box is a single point of failure | **Defer** — documented in `FREEZE.md`; correct scope cut for a demo |
| **In-memory `REGISTRY` not rehydrated on restart** (B4, `FREEZE.md` item 1) | A restart/redeploy 404s in-flight paused runs (checkpoint survives, HTTP view doesn't); fatal for always-on multi-tenant | **Defer** — mitigation is "drain before restart"; real fix is post-freeze |
| **No automated alerting** (uptime check is an operator-provisioned manual runbook) | Unattended operation has no failure signal beyond a human poking the URL | **Defer** for *supervised* demo; **fix** before unattended/multi-tenant |
| **No CI image build/scan; image built by hand via `buildx`** | No vulnerability scanning or immutable, provenance-tracked supply chain | **Defer** — acceptable for a demo; needed for real prod |

**None of these block a supervised demo.** They block leaving it running unattended for multiple
strangers. Every one is already disclosed in `FREEZE.md` / `DECISIONS.md` as an accepted scope
cut — the audit confirms they are *accurately* disclosed, not understated.

**Recommendation: make no further code changes before the demo.** The four prior do-now items are
done; the remaining gaps are all correct deferrals for the single-operator supervised scope. The
only housekeeping item is committing/stashing the working-tree lint cleanup. Spending demo-prep
time on rate limiting or registry rehydration would be polishing capabilities the demo does not
exercise.

---

## 1. Eval & benchmarking — PARTIAL (materially improved)

**Since last audit:** the prior audit's #1 do-now ("a grounding-regression gate that actually runs
on PRs") **shipped**. `.github/workflows/ci.yml` now has an `eval-gate` job (lines 46-91) that runs
`evals/verifier_golden_test.py` on every `pull_request` to `main`, secret-gated (skips cleanly with
a `::notice::` on forks/clones without `DEEPSEEK_API_KEY`), ~$0.02/run. The golden fixture's real
exit code (`sys.exit(0 if ground_ok>=11 and spec_ok>=10 else 1)`) makes it merge-blocking.

**What's real:** the three enforced eval harnesses (`verifier_golden_test.py`, `benchmark.py
--gate` with per-run UVR≤0.15, `check_telemetry_fields.py`) all have genuine `sys.exit(1)` gates,
not printed advisories. `eval.yml` wires all three behind a Qdrant service container. This is a
legitimate, runnable eval pipeline.

**Residual gaps:**
- The PR gate covers **only** the verifier fixture (8 LLM calls). The full 20-topic SV/UVR
  benchmark (`benchmark.py --gate`) is still `workflow_dispatch`-only — a draft-prompt regression
  that lowers SV across topics without tripping the 12-claim verifier fixture is still not caught
  automatically. This is a defensible cost/noise trade (SV has a ±7 noise band at n≤3, so gating
  it would flake), but it means the *comprehensive* grounding gate remains manual.
- The PR gate is merge-blocking **only if** the repo's branch-protection rule lists `eval-gate` as
  a required status check — the workflow itself notes this (ci.yml lines 83-85). That's a GitHub
  setting outside the repo; the audit cannot verify it is configured.

**Verdict:** the cheap, high-frequency regression guard is now load-bearing on PRs — a real
improvement over "opt-in only." The comprehensive guard stays manual by design. **PARTIAL**, but
the gap narrowed from "no automatic grounding gate at all" to "only the cheap one is automatic."

---

## 2. Observability — COMPLETE (for the project's scope)

**Since last audit:** two of the prior gaps closed. (1) `tools/query_kb.py`'s three bare `print()`
error paths are now `log.error`/`log.warning` through the shared structlog logger (lines 333, 341,
345) — retrieval-layer failures now land in the structured JSON stream like every other module.
(2) **LangSmith tracing is implemented and default-on when configured** (`observability/tracing.py`,
`agent/nodes.py::_get_client`): `wrap_openai()` wraps the DeepSeek client when tracing is on so
every `_llm_call` emits a traced "llm" run, and `_attach_usage_to_current_run` layers DeepSeek's
real per-token cost onto each run's `usage_metadata` (LangSmith has no `deepseek-chat` price entry,
so cost would otherwise read zero). This adds cross-node trace visualization + per-call latency and
token/cost breakdown — exactly the layer structlog-to-stdout did not provide.

**What's real (carried forward, re-verified):** structured JSON logging per node with `run_id` on
every call site; automatic prompt-hash versioning stamped into every telemetry record (no silent
prompt drift); per-run JSON with full grounding_report, sources, per-iteration `iteration_metrics`,
attribution, cost, tokens, per-node latency, error_log; `check_telemetry_fields.py` enforces the
27-field reconstructability contract with a real exit code.

**Residual gaps (both documented, neither hidden):**
- No external log sink is *configured in the repo* — logs go to container stdout; `docs/deploy/
  DEPLOY.md` states stdout "IS the log sink; a platform collector ingests it," making persistence
  the operator's responsibility. `docker-compose.prod.yml` sets no logging driver.
- B4: the in-memory `REGISTRY` is not rehydrated from the checkpoint on restart, so in-flight
  paused runs lose their HTTP-visible state across a restart (the LangGraph checkpoint survives).
  Real, acknowledged, mitigated by "drain before restart."
- Tracing is **off unless `LANGCHAIN_API_KEY` + `LANGCHAIN_PROJECT` are set** — not in the prod/
  demo compose or `.env.example`, so it's per-operator opt-in. On the live demo box it is on.

**Verdict:** completed-run reconstructability is fully verified (`check_telemetry_fields.py`), the
KB-layer logging inconsistency is fixed, and cross-node tracing with token/cost now exists. The two
residuals are scoped, documented limitations, not defects. **COMPLETE** for this project's scope.

---

## 3. Monitoring — PARTIAL (up from MISSING)

**Since last audit:** the prior audit's #2 do-now produced (a) an **uptime-check runbook**
(`docs/deploy/DEPLOY_DEMO.md` "Uptime monitoring" section, lines ~146-170) walking the operator
through provisioning a free HTTP monitor against `/health`; (b) **ERROR-level, greppable
publish-failure logs** — `api.publish_failed`/`ui.publish_failed` (`api/server.py` lines 122, 214,
473), `git.command_error`/`git.unexpected_error` (`agent/nodes.py` lines 1635, 1639); and (c) when
LangSmith is on, per-node latency and per-run cost are visible in its dashboards.

**What's still missing in-repo:**
- No automated alerting — the uptime check is *provisioned by hand* by the operator in a
  third-party service; nothing in the repo fires an alert on cost-gate breach, `git_status:
  failed`, an `unverified` spike, or the process being down.
- No metrics endpoint (`/metrics`), no Prometheus client, no in-repo dashboard.
- `GET /health` + the Docker `HEALTHCHECK` drive only `restart: unless-stopped`; they don't notify
  a human on their own.

**Verdict:** the *building blocks and the runbook* now exist, and failure events are greppable in
the structured logs — a real move off "genuinely absent." But the actual monitoring surface is
still operator-provisioned and manual, with no alerting wired in the repo. For a *supervised* demo
this is fine. For unattended or multi-tenant operation it remains the single most "not production"
area. **PARTIAL** — honestly, the lower end of PARTIAL.

---

## 4. Reliability — COMPLETE (for the scope it covers)

**Since last audit:** unchanged in substance; re-verified. The LangSmith wrapping is additive and
does not touch the retry/cost-gate/degradation paths (`_llm_call`'s `@retry` decorator and the
cost gates are intact; the wrap happens at client construction, the retry wraps the call).

**What's real, with test evidence (re-verified in source):** exact-3-attempt exponential-backoff
retry on transient LLM errors only, never on auth/bad-request (`_llm_call`, asserted by exact
call-count tests); cost gates at every LLM-leading edge that skip the call entirely (asserted the
mocked client is never invoked); graceful degradation on Tavily/Qdrant outage and malformed model
JSON (each independently tested, raw output preserved for debugging); the content-freeze
word-multiset guard in `html_revise_node` that discards drifting revisions (tested both
directions); `git_node` wrapping every git op in try/except with a `finally` that restores the
original branch on failure.

**Scope note (unchanged):** the human-triggered `POST /ui/runs/{id}/publish` does a retry-free
`subprocess.run(..., timeout=30)` for the actual `git push` — a flaky network means clicking
"Publish" again, not an auto-retry. Reasonable for a human-watched action; tested for correctness
(11 tests), not resilience.

**Verdict:** **COMPLETE** for everything the suite covers, and the suite's claims are independently
verifiable in source. Still the strongest section. (Availability/scale limits — single worker,
single VM, registry volatility — are Deployment/HA concerns below, not logic-reliability gaps.)

---

## 5. Security — PARTIAL (unchanged)

**Since last audit:** no security item was a do-now and none changed. Re-verified intact:
allowlist slug sanitization reaching filesystem + git refs (6 regression tests incl. path-traversal
and a null-byte fuzz); fail-closed bearer auth with `hmac.compare_digest` and a `503` when the
token is unconfigured (refuses fail-open); Qdrant network-isolated with no host port; demo compose
removes the app's host port so Caddy is the only public surface; secrets excluded from image and
git; non-root `uid 10001` container; prompt-injection mitigated by *mandatory* HITL on both gates
(the API forces `HITL_AUTO_APPROVE=0` at import) — verified mechanically, not just cited.

**Gaps (unchanged, none addressed — none were in scope for any milestone):**
- No rate limiting / per-token / per-day spend cap anywhere in `api/server.py`.
- No CORS policy (accidentally same-origin-safe for this SPA, but an absence, not a decision).
- Bearer token passed as a URL query param on `/ui/runs/{id}/events` and `/preview` (EventSource/
  plain links can't set headers) — flagged in-code as an accepted single-operator tradeoff.
- Single shared, unrotated, unscoped bearer token — no per-caller identity.

**Verdict:** the security-sensitive paths that were *in scope* (sanitization, auth, isolation,
non-root, injection-via-HITL) are solid and test-backed. The gaps are the multi-tenant/exposure
concerns (rate limiting, token scoping, CORS) — absent rather than broken, and the correct things
to *defer* for a single-operator demo but *fix* before shared exposure. **PARTIAL.**

---

## 6. Deployment — PARTIAL (improved)

**Since last audit:** the prior audit's #4 ("decide and either build or drop the Docker Hub path")
**resolved toward build-and-document**. `docs/deploy/DEPLOY_DEMO.md` and `docs/CHEATSHEET_AWS.md`
document the registry path as the primary deploy: build once on a fast machine via
`docker buildx --platform linux/arm64 --push` to `anudeepreddy332/content-agent:demo`, then the EC2
box only ever *pulls* — avoiding rebuilding the embedding-model-baking image on a small VM. This
is the path the live demo actually uses (`DECISIONS.md` 2026-06-19).

**What's real (carried forward):** two detailed runbooks (`DEPLOY.md` loopback/SSH-tunnel freeze
posture, `DEPLOY_DEMO.md` Caddy + public port); scripted human-gated rollback
(`rollback_publish.sh`, debugged against a fork, no auto-push); the publish-safety property
preserved end-to-end (grep confirms `git push` appears nowhere in `git_node`; the only push is the
separate, human-triggered `ui_publish` endpoint gated on `git_status` already being merged); the
live EC2/Caddy/sslip.io topology validated via `docker compose config` (app has no host port,
Caddy is the sole public surface).

**Residual gaps:**
- The registry path is **documented and manually executed**, not **CI-automated or scanned**.
  `ci.yml` never runs `docker build`; there is no image vulnerability scan, no immutable
  digest-pinned promotion. The supply chain is "an operator runs `buildx` by hand and pushes."
- No staging environment for the *application* (the website fork plays content-staging; the app
  itself has no pre-prod deploy that gets exercised first).
- `rollback_publish.sh` has no automated test — a future change to `git_node`'s commit-message
  format or branch naming could silently break its `--grep` match; correctness rests on two manual
  fork validations in `DECISIONS.md`.
- B4 registry volatility and single-worker throughput carried forward as accepted limitations.

**Verdict:** the publish-safety guarantee — the single most important production property here — is
genuinely solid and verifiably preserved end-to-end including in the cloud path. The registry gap
from the last audit is closed at the "documented + live-executed" level, but not at the "CI-built,
scanned, immutable" level. **PARTIAL**, improved.

---

## Gaps table — current state (prioritized)

| # | Item | Blocks production? | Status since last audit | Priority |
|---|------|---------------------|--------------------------|----------|
| 1 | PR-CI grounding gate (verifier fixture) | — | **DONE** (`eval-gate` job) | needs branch-protection rule to be merge-blocking |
| 2 | query_kb `print()` → structlog | — | **DONE** | — |
| 3 | Uptime-check runbook + ERROR-level publish-failure logs | — | **DONE** (runbook + greppable logs) | — |
| 4 | Docker Hub / registry deploy path | — | **DONE** (documented + live) | not CI-automated/scanned |
| 5 | LangSmith cross-node tracing + token/cost | — | **DONE** (default-on when configured) | off unless creds set |
| 6 | Full 20-topic SV/UVR benchmark in automatic CI | No | Unchanged (manual `workflow_dispatch`) | Defer (noise-band trade) |
| 7 | Automated alerting (cost/`git_status: failed`/down) | Yes for unattended; No for supervised | Partially mitigated (greppable ERROR logs + runbook) | Do now *if* unattended; else Defer |
| 8 | Rate limiting / per-day spend cap on run endpoints | No (single-operator) | Unchanged | Defer until token shared |
| 9 | CI image build + vulnerability scan + immutable promotion | No | Unchanged | Defer (manual buildx works) |
| 10 | CI-drilled `rollback_publish.sh` (throwaway-fork smoke) | No | Unchanged | Defer |
| 11 | Rehydrate API `REGISTRY` from checkpoint on restart (B4) | No (mitigated) | Unchanged | Defer (post-freeze) |
| 12 | Per-tenant identity / token scoping & rotation | No (single-operator) | Unchanged | Defer (re-architecture) |
| 13 | Horizontal scale / HA (`max_workers=1`, single VM) | No (one-article editorial use) | Unchanged | Defer (re-architecture) |
| 14 | Explicit CORS policy decision documented | No | Unchanged | Defer (1h) |

**Overall read:** all four of the prior audit's do-now items shipped and are verified in source,
plus LangSmith tracing added a real observability layer. Eval moved from "no automatic grounding
gate" to "cheap gate is automatic"; Observability is effectively complete for scope; Monitoring
moved off MISSING but is still manual/operator-provisioned; Reliability and the publish-safety
property remain the strongest, test-backed guarantees. The honest remaining gaps are exactly the
multi-tenant / HA / unattended-operation items — all correctly disclosed as accepted scope cuts,
none of which block a supervised single-operator demo.

---

## Final verdict

**This project is production-ready for a supervised, single-operator demo and is a strong
senior/staff-level portfolio showcase; the only pre-demo blocker is housekeeping (commit or stash
the working-tree lint cleanup), and every remaining gap — automated alerting, rate limiting,
horizontal scale/HA, per-tenant identity, CI-built/scanned images — is an accurately-disclosed,
correctly-deferred requirement for unattended multi-tenant operation, not for the demo.**
