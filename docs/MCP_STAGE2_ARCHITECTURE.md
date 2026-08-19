# Stage 2 MCP Engineering-Evidence Architecture

Status: `ARCHITECT PROPOSAL — INDEPENDENT REVIEW REQUIRED`

Decision: `MCP-CURSOR-READY`

Canonical repository tip investigated: `ca29d32b4869269daa47142615d298580a577a77`

Product-runtime integration retained as the accepted runtime boundary:
`d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3`

Date: 2026-08-19

## 1. Current repository and canonical-state verification

The remote gate passed.

- `git fetch origin main` completed before architecture selection.
- Fetched `origin/main` and `FETCH_HEAD` both resolved to exact SHA
  `ca29d32b4869269daa47142615d298580a577a77`.
- The remote is `git@github.com:anudeepreddy332/content-agent.git`.
- `ca29d32` is a documentation-only descendant of the validated P0-1 integration
  `d0be0a77f1f9a2c53fbe3743d852552f4fa6b0f3` through `94226dc`.
- A clean isolated worktree and branch were created from exact `ca29d32` at
  `/private/tmp/content-agent-stage2-mcp-architecture` on
  `chore/stage2-mcp-architecture`.
- The user's primary checkout was not changed. It remains at local `main` SHA `61de06d` and has
  two unrelated untracked files; those files were not read, modified, staged, or removed.
- No MCP implementation or configuration existed in the canonical tree. A case-insensitive search
  found only the accepted Stage 2 sequence in canonical documentation, not an MCP server.

The canonical documents call `d0be0a7` the current canonical product/runtime integration, while the
public branch tip is now `ca29d32` after two documentation-only closeout commits. This proposal uses
the terms precisely: `ca29d32` is the implementation base for Stage 2; `d0be0a7` remains the
validated P0-1 runtime integration inside that history.

The P0-2a control was independently reproduced at `ca29d32` with zero selected topics. The command
exited `0` after printing `Valid: 0/0`, `Scorable UVR runs: 0/0`, and `CI GATE: PASS`. The focused
evaluation-integrity regression suite passed `14/14` in a new lockfile-synchronized environment.
The first attempt using the primary checkout's older virtual environment was invalid because it did
not contain the canonical P0-1 `nh3` dependency; no result from that attempt is claimed.

## 2. MCP problem statement

MCP is solving an engineering-evidence handoff problem, not a Content Agent product problem.

Today, Architect, Reviewer, and Implementer can all use Git and a shell, but they exchange long prose
that can omit the exact checkout, silently refer to stale output, or summarize a command without
exposing its exit status and provenance. Repository state, command execution, and evidence identity
are repeatedly reconstructed by each agent.

The proposed MCP boundary gives all three roles the same small, machine-readable evidence plane:

```text
fixed repository/worktree + frozen mission policy
        -> bounded read/inspection/check tools
        -> schema-validated evidence records
        -> independently reproducible evidence bundle
```

It does not edit product files, decide architecture, approve work, merge, publish, deploy, call model
providers, or replace Git.

## 3. Current workflow weaknesses MCP is meant to reduce

1. Prose handoffs can name a branch without binding evidence to its exact HEAD.
2. A command summary can omit nonzero exit status, timeout, output truncation, or stale artifacts.
3. Agents can read different worktrees or assume that local `main` equals fetched `origin/main`.
4. File-scope validation is manual and can miss untracked or mid-run changes.
5. The repository has structured telemetry, but engineering validation lacks a common response
   envelope and bundle manifest.
6. Native shells invite free-form commands, inherited secrets, and accidental network access when
   the task only needs a deterministic check.
7. Repository content is untrusted model input. A broad file or shell tool would enlarge the prompt-
   injection and exfiltration surface.
8. Reviewer re-execution is possible today but not expressed as one repeatable contract.

MCP does not remove the need for independent re-execution. A local evidence digest is not a remote
attestation and must not be described as tamper-proof.

## 4. Primary-source MCP findings

The selected protocol target is MCP revision `2026-07-28`, with legacy-client compatibility.

- MCP remains a host/client/server architecture. Hosts own connection permission, user consent, and
  lifecycle; servers should stay focused and independently bounded.
- Revision `2026-07-28` is stateless. Every request carries its protocol version, client identity,
  and capabilities. `server/discover` replaces mandatory initialization for modern clients.
- `2025-11-25` and earlier are initialization/session-era protocols. A dual-era server can serve
  both, which is necessary because current host documentation has not converged on one era.
- stdio is newline-delimited JSON-RPC between a client-launched subprocess and the host. Protocol
  output is stdout-only; operational logs belong on stderr. Closing stdin is the graceful shutdown
  signal.
- Streamable HTTP is valid but adds an endpoint, origin validation, authentication, OAuth, and
  multi-user operations that Stage 2 does not need.
- Tools are model-controlled functions. They support JSON Schema inputs, structured JSON results,
  optional output schemas, protocol errors, and execution errors.
- Resources are application-controlled context identified by URIs. Their user experience remains
  host-specific. They are not required for this MVP.
- Prompts are user-controlled templates. They would blur the Architect's ownership of architecture
  and add no evidence capability, so none are proposed.
- For HTTP, MCP authorization is OAuth-oriented. For stdio, the specification says not to use that
  HTTP authorization flow; credentials come from the process environment. This design passes no
  credentials and treats the operator-controlled process launch as the local trust boundary.
- Current Python SDK `mcp==2.0.0` is the stable v2 line, supports Python 3.14, implements the
  `2026-07-28` protocol, and also serves initialization-era clients.

Primary references:

- [MCP 2026-07-28 architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP versioning and compatibility](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP stdio transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [MCP Python SDK 2.0.0](https://pypi.org/project/mcp/2.0.0/)

## 5. Target-agent MCP compatibility findings

### Codex and ChatGPT desktop

Official OpenAI documentation says the ChatGPT desktop app, Codex CLI, and Codex IDE extension share
MCP configuration and support local stdio and Streamable HTTP servers. Codex can allowlist tools,
set per-server and per-tool approval modes, and configure startup/tool timeouts. The local machine
used for this audit has `codex-cli 0.145.0`, and its `codex mcp` command is available.

Current OpenAI documentation still describes reading server `instructions` during initialization.
Therefore the server must be dual-era; security may not depend on an instruction being read.

ChatGPT web is different: it uses remote MCP-backed plugin tools and does not read local Codex MCP
configuration. A Reviewer operating only in ChatGPT web cannot consume this local stdio server.
That is an explicit Stage 2 block, not a reason to deploy a premature remote server. The Reviewer
must use an MCP-capable local Codex/ChatGPT desktop/IDE host for this phase.

Reference: [OpenAI MCP documentation](https://developers.openai.com/codex/mcp).

### Cursor

Current Cursor documentation supports stdio, SSE, and Streamable HTTP. It supports tools and
project-local `.cursor/mcp.json`; tool invocations require approval by default. The locally installed
Cursor is version `3.16.29`. Cursor's published capability table does not establish a portable
resource experience, so the architecture depends on tools only.

Reference: [Cursor MCP documentation](https://docs.cursor.com/context/model-context-protocol).

### Compatibility conclusion

The common proven denominator is local stdio plus tools. Resources, prompts, roots, sampling,
elicitation, tasks, subscriptions, and HTTP auth are not required. Actual Codex and Cursor connection
smokes remain mandatory implementation gates; a documentation claim is not a connection test.

## 6. Chosen MCP architecture

Build one logical Python package, `engineering_mcp`, instantiated as one local stdio process per
agent host and target worktree.

```text
Codex/ChatGPT desktop or Cursor
        | stdio, one client -> one process
        v
engineering_mcp (trusted, versioned evidence plane)
        |-- fixed target repository root
        |-- frozen mission policy + digest
        |-- read-only Git adapter
        |-- tracked-file reader
        |-- fixed deterministic check runner
        `-- evidence records under outputs/mcp_evidence/
```

There is one codebase and one tool surface, not separate Architect, Reviewer, and Implementer
servers. This is safe only because the initial surface has no source mutation, Git mutation,
approval, provider, network, or arbitrary-shell capability. Each launch is still bound to a target
worktree and a mission policy. An `actor_role` configured at launch is provenance, not authorization.

For P0-2a, the independently validated Stage 2 server code must run unchanged. Its server-code
manifest digest and policy digest are included in every response. The P0-2a mission gets a separately
reviewed policy file; it does not get new server code merely to add a check ID.

## 7. Transport and deployment decision

Choose local stdio.

Reasons:

- both target local hosts support it;
- the host owns process start/stop and user consent;
- there is no listening port, DNS-rebinding surface, OAuth system, remote tenancy, or shared state;
- repository and worktree paths stay local;
- stderr can carry bounded operational logs without corrupting protocol stdout;
- a one-user local process matches the project's present supervised single-operator posture.

The launcher is `uv run --frozen --offline python -m engineering_mcp ...` from a synchronized
project environment. `--offline` prevents dependency resolution from becoming implicit network
activity. The server must also work when invoked directly as `.venv/bin/python -m engineering_mcp`.

Remote HTTP, a shared service, containers, OAuth, and cloud deployment are roadmap options only.
Move to them only if a later multi-machine need is evidenced; A2A compatibility is not a Stage 2
selection criterion.

## 8. Server ownership and lifecycle

- Human repository operator: owns installation, host configuration, target path, process launch,
  and any approval to fetch, branch, commit, push, merge, or deploy.
- Architect: owns the mission policy, allowed scope, check definitions, and acceptance gates.
- Implementer: implements the independently reviewed server specification and later edits product
  code with native editor/Git tools; it does not expand the policy.
- Reviewer: validates the server at an exact SHA and independently re-runs evidence operations.
- MCP host: starts one server subprocess, presents tool approvals, sends requests, and closes stdin
  on shutdown.

No agent approves its own server or product change. Server `instructions` summarize safety and stop
rules but grant no authority.

## 9. Role and capability separation

All roles get the same five-tool evidence surface. The separation is in what MCP deliberately lacks:

- Architect and Reviewer cannot mutate through MCP.
- Implementer cannot mutate through MCP either; it uses Cursor's native editing under the reviewed
  file scope.
- No tool can record `approved`, change a threshold, change the mission policy, or merge work.
- Read and check evidence is available to all roles, which makes independent reproduction possible.
- Host-side tool allowlists and approval prompts are defense in depth, not the server security
  boundary.

Separate capability servers or credentials would add ceremony without changing authority in a local
stdio process. If a future MCP version adds write or remote capabilities, it must be a new
independently reviewed architecture with separate authorization, not an expansion of this server.

## 10. Exact initial MCP capability inventory

All input schemas use JSON Schema with `type: object`, required fields as stated, bounded strings and
arrays, and `additionalProperties: false`. The target repository root and policy path are launch
configuration, never model-supplied tool arguments.

### 10.1 `repo_snapshot`

Consumer: all roles. Classification: read-only.

Input:

```json
{
  "expected_head_sha": "optional lowercase 40-hex SHA",
  "require_clean": false
}
```

Both properties are optional; `require_clean` defaults to `false`. When an expected SHA is known,
the caller must supply it. Omitting it permits state discovery but is recorded explicitly in the
evidence envelope and never satisfies an exact-SHA acceptance gate.

Output result fields:

```json
{
  "branch": "string or null",
  "head_sha": "40-hex",
  "origin_url": "normalized expected repository identity",
  "origin_main_sha": "40-hex or null",
  "is_clean": true,
  "tracked_changes": [{"status": "two-character Git XY status", "path": "repo-relative"}],
  "untracked_paths": ["repo-relative or secret-path-redacted marker"],
  "status_digest": "sha256",
  "server_code_digest": "sha256",
  "policy_digest": "sha256",
  "lockfile_digest": "sha256"
}
```

Allowed scope: Git metadata for the fixed target worktree and normalized `origin` URL only. It does
not fetch. Timeout: 10 seconds. A HEAD mismatch, wrong remote, unsupported repository, or required-
clean dirty state returns `BLOCKED`.

Why MCP: one call binds state, policy, server version, and a stable dirty-state digest instead of
requiring agents to interpret several shell outputs.

### 10.2 `read_tracked_file`

Consumer: all roles. Classification: read-only; result content is explicitly untrusted.

Input:

```json
{
  "path": "repo-relative POSIX path, 1..256 characters",
  "start_line": 1,
  "end_line": 400
}
```

`start_line` and `end_line` are positive integers; at most 400 lines and 64 KiB are returned per
call. The full file must be a Git-tracked regular UTF-8 text file no larger than 256 KiB. Symlinks,
`.git`, ignored/untracked files, binary content, secret-like paths, traversal, and resolved paths
outside the root are rejected.

`path` is required. `start_line` and `end_line` are optional and default to the first bounded
window; supplying one requires supplying both.

Output result fields: `path`, index blob ID, full-worktree-file SHA-256, selected line bounds,
`truncated`, `content_trust: "UNTRUSTED_REPOSITORY_CONTENT"`, and `content`. Timeout: 5 seconds.

Why MCP: it proves which blob and root produced the text and narrows the prompt-injection and secret
surface compared with an unrestricted filesystem read.

### 10.3 `inspect_change`

Consumer: all roles. Classification: read-only.

Input:

```json
{
  "base_sha": "lowercase 40-hex",
  "target": "HEAD|WORKTREE",
  "scope_id": "identifier present in the frozen mission policy",
  "include_patch": false
}
```

`base_sha`, `target`, and `scope_id` are required. `include_patch` is optional and defaults to
`false`.

Output result fields: `base_sha`, `head_sha`, `merge_base_sha`, `target`, ordered changed-path
records, additions/deletions, `change_set_sha256`, allowed paths, violations, and optional patch
metadata. For `WORKTREE`, the change set is the union of base-to-index, index-to-worktree, and
untracked paths. Allowed-scope untracked regular files are represented by path, byte length, and
full-content SHA-256 so new implementation files cannot disappear from scope or identity checks;
secret-like paths, special files, symlinks, and out-of-scope untracked paths fail closed without
returning their content. Patch output is off by default, labeled untrusted, limited to 256 KiB, and
withheld if secret scanning fires. `change_set_sha256` covers a canonical manifest of the complete
tracked diff plus the full allowed untracked-file identities, not only the displayed prefix.
Timeout: 30 seconds.

Only exact SHAs and `HEAD|WORKTREE` are accepted; no revision expression or arbitrary Git option is
accepted. A scope violation returns `FAIL` and may never be converted to a warning.

Why MCP: changed-file enforcement is evaluated against an Architect-owned policy rather than an
allowed-path list supplied by the implementation agent.

### 10.4 `run_check`

Consumer: all roles. Classification: deterministic execution plus evidence-directory writes.

Input:

```json
{
  "check_id": "identifier present in the frozen mission policy",
  "expected_head_sha": "lowercase 40-hex",
  "expected_status_digest": "sha256"
}
```

There is no command, argument, environment, path, selector, or timeout field. The policy maps the
ID to an exact argv, timeout, output limit, environment class, and expected artifact set.

Output result fields: check ID, exact fixed argv, environment-class name, interpreter and dependency
inventory digests, pre/post HEAD and status digests, exit code, timeout/output-limit flags,
stdout/stderr SHA-256 and bounded tails, artifact records, semantic result digest, and duration.
The full bounded streams are retained under the evidence ID. Any pre/post state change invalidates
the check. Timeout is policy-specific and never more than 1,200 seconds. stdout and stderr are each
limited to 1 MiB.

Why MCP: it removes free-form shell arguments, scrubs credentials, binds results to exact state, and
makes nonzero/timeout/truncation conditions explicit.

### 10.5 `build_evidence_bundle`

Consumer: all roles. Classification: evidence-directory write only.

Input:

```json
{
  "evidence_ids": ["1..50 opaque IDs"],
  "expected_head_sha": "lowercase 40-hex"
}
```

Both properties are required. Mission identity is taken only from the launch-bound policy; it is
not caller-selectable.

The server reads only its own evidence records. It refuses unknown IDs or a mix of repository,
server-code, policy, HEAD, or worktree-status identities. It writes an ordered JSON manifest under
`outputs/mcp_evidence/bundles/<bundle_id>/`, copies the bounded records without filesystem links,
and returns bundle ID, relative manifest path, manifest SHA-256, evidence IDs, and consistency
result. Timeout: 30 seconds; bundle limit: 25 MiB.

Why MCP: the Reviewer receives a manifest of exact operation evidence rather than a prose-selected
set of excerpts.

## 11. Common evidence and provenance schema

Every valid tool invocation returns the same envelope as `structuredContent` and as serialized JSON
in a text content block for legacy clients:

```json
{
  "schema_version": "content-agent.mcp-evidence.v1",
  "operation": "repo_snapshot|read_tracked_file|inspect_change|run_check|build_evidence_bundle",
  "status": "PASS|FAIL|BLOCKED",
  "code": "stable_machine_code",
  "summary": "short non-authoritative text",
  "evidence_id": "opaque UUID",
  "request_id": "opaque UUID",
  "mission_id": "policy mission ID",
  "actor_role": "architect|reviewer|implementer",
  "repository": {
    "id": "github.com/anudeepreddy332/content-agent",
    "head_sha_before": "40-hex",
    "head_sha_after": "40-hex",
    "status_digest_before": "sha256",
    "status_digest_after": "sha256"
  },
  "server": {
    "package_version": "semver",
    "code_digest": "sha256",
    "protocol_versions": ["2026-07-28", "legacy versions actually served"],
    "policy_id": "string",
    "policy_digest": "sha256"
  },
  "timing": {
    "started_at": "UTC RFC3339",
    "finished_at": "UTC RFC3339",
    "duration_ms": 0
  },
  "result": {},
  "artifacts": [{"path": "approved relative path", "sha256": "hex", "bytes": 0}],
  "warnings": [],
  "redactions": [],
  "semantic_result_digest": "sha256"
}
```

`semantic_result_digest` excludes request IDs, timestamps, and elapsed time so deterministic results
can be compared. It includes the complete normalized result, not displayed tails.

This is provenance, not cryptographic identity attestation. The evidence directory and chained
audit log are locally mutable. Independent Reviewer reproduction at the exact commit is mandatory.

## 12. Read/write boundaries

Read permissions:

- fixed target worktree;
- Git-tracked regular files through `read_tracked_file`;
- read-only Git plumbing invoked through fixed argv;
- server-owned records under `outputs/mcp_evidence/`.

Write permissions:

- only `outputs/mcp_evidence/` and a per-check temporary directory below it;
- test tools may write only to the policy-declared ignored output/temp paths;
- no tracked file may change during a check.

Forbidden:

- arbitrary filesystem paths, `.git` file reads, `.env`, ignored/untracked file content, sockets,
  provider APIs, repository edits, Git index/ref/config mutations, and website checkout access.

The server must not import Content Agent runtime modules, which would load `.env`, model clients, or
retrieval dependencies as a side effect.

## 13. Command-execution policy

No shell is used. `shell=False` is mandatory. Every subprocess receives a fixed argv from the
trusted policy and runs in the fixed target root with a new process group.

Initial Stage 2 check IDs:

| Check ID | Exact command | Timeout |
| --- | --- | ---: |
| `mcp_protocol_smoke` | `.venv/bin/python -m engineering_mcp.probe` | 60 s |
| `mcp_unit` | `.venv/bin/python -m pytest tests/test_mcp_server.py tests/test_mcp_evidence.py -q -p no:cacheprovider` | 180 s |
| `mcp_security` | `.venv/bin/python -m pytest tests/test_mcp_security.py -q -p no:cacheprovider` | 180 s |
| `mcp_compatibility` | `.venv/bin/python -m pytest tests/test_mcp_compatibility.py -q -p no:cacheprovider` | 180 s |
| `fatal_lint` | `.venv/bin/ruff check --select E9,F63,F7,F82 .` | 120 s |
| `full_regression` | `.venv/bin/python -m pytest tests/ -q -p no:cacheprovider` | 1,200 s |

The policy stores argv arrays, not command strings. No model-supplied pytest selector, environment,
working directory, Git revision, or output path is permitted.

`fetch`, `ls-remote`, worktree creation, branch creation, editing, staging, committing, pushing,
merging, deployment, and paid/live evaluation remain native operator workflows outside MCP.

## 14. Filesystem and Git security model

At startup, the server:

1. resolves the configured repository root and requires it to equal `git rev-parse --show-toplevel`;
2. verifies the normalized `origin` identity;
3. loads a schema-validated policy and computes its digest;
4. computes a server-code manifest digest;
5. verifies the evidence directory resolves beneath the target root;
6. refuses symlinks in policy, server-code, or evidence paths;
7. records Git, Python, SDK, lockfile, and platform identity.

Permitted Git operations are read-only: `status --porcelain=v2`, `rev-parse`, `merge-base`,
`remote get-url`, `ls-files`, `cat-file`, `show`, and `diff` with server-constructed arguments and
`--no-ext-diff`. External diff/textconv helpers and user aliases are disabled. No command accepts a
leading-dash path or revision expression.

`run_check` takes a global nonblocking lock. A second execution returns `BLOCKED: CHECK_BUSY`; it is
not queued against potentially stale state. HEAD and worktree digests are captured before and after.
Tracked or relevant untracked changes during execution invalidate the result.

Destructive Git operations do not exist in the tool registry.

## 15. Secrets and network policy

- Host configurations pass no API key, bearer token, Git credential, cloud credential, or provider
  credential to the MCP server.
- The server does not import `config.py`, `dotenv`, Content Agent nodes, or provider clients.
- Startup fails if known Content Agent provider/publish credential variables are present. Broad
  token/key/secret/password variables are removed before child execution and never logged.
- Child commands receive a small constructed environment: deterministic locale/hash settings,
  a server-owned temp/home directory, and only explicitly required non-secret variables.
- Commands use the synchronized `.venv` directly; they do not install or update dependencies.
- Stage 2 exposes no HTTP client or Git-network command. `uv` launch is `--offline`.
- No check that requires external network or a provider credential is eligible for the initial
  policy. Such a requirement returns `BLOCKED` and requires a new architecture review.
- P0-2a's zero-provider proof is a deterministic test that asserts invalid selection fails before
  the benchmark subprocess/provider path, combined with absent provider credentials. This is not a
  claim of kernel-level network sandboxing.

The initial native stdio deployment does not provide OS-level network isolation. If independent
review requires hostile-code containment rather than controlled deterministic-check containment,
Stage 2 is blocked and must evaluate a network-denied container separately. Do not imply that
environment scrubbing is a network sandbox.

## 16. Error, timeout, cancellation, and output-limit policy

- Malformed JSON-RPC or a schema-invalid request uses a protocol error.
- Valid calls that cannot execute return the common envelope with `status: BLOCKED`, `isError: true`,
  and a stable code.
- A deterministic check with a nonzero exit returns `status: FAIL`, `isError: true`; it is evidence,
  not a reason for automatic retry.
- No retry is performed by the server. The caller must not retry until lucky.
- Timeout, cancellation, or output overflow terminates the entire subprocess group, waits for child
  cleanup, and returns `BLOCKED` with partial-stream digests and truncation flags.
- Unknown exceptions return a generic `INTERNAL_ERROR` without stack traces, absolute home paths, or
  environment values. Full sanitized diagnostics go only to stderr/evidence.
- Tool timeouts: snapshot 10 s, file read 5 s, change inspection 30 s, bundle 30 s, checks as frozen
  above. Host tool timeout must be configured above the longest permitted check.
- File response: 64 KiB/400 lines. Patch response: 256 KiB. stdout/stderr: 1 MiB each. Bundle: 25 MiB.

## 17. P0-2a workflow through the proposed MCP

P0-2a is not implemented in Stage 2. After independent MCP validation, its workflow is:

1. Operator fetches remote refs natively and creates an isolated implementation worktree/branch
   from the exact then-canonical MCP-integrated SHA.
2. Architect and Reviewer approve a P0-2a mission policy containing the exact base, allowed paths,
   and fixed deterministic check IDs. The policy does not change quality thresholds.
3. Cursor connects the already validated `engineering_mcp` server to that worktree.
4. `repo_snapshot` proves repository identity, exact HEAD, and clean starting state.
5. Architect/Reviewer use `read_tracked_file` for `scripts/benchmark.py`,
   `tests/test_evaluation_integrity.py`, `evals/topics.json`, and canonical governance documents.
6. Cursor edits with its native editor. MCP performs no mutation.
7. `inspect_change` proves the worktree changes only the frozen P0-2a scope.
8. `run_check` executes the pre-provider-call deterministic P0-2a tests with provider credentials
   absent. A zero-selection test must prove the benchmark subprocess/provider path was not invoked.
9. Cursor commits natively after local gates pass; MCP cannot commit.
10. `repo_snapshot`, `inspect_change`, focused P0-2a checks, fatal lint, and full regression are run
    against the exact candidate commit.
11. `build_evidence_bundle` binds the coherent evidence records to that commit.
12. An independent Reviewer uses a clean checkout and the validated server to re-run the critical
    operations. Implementer evidence alone cannot approve the transition.
13. Merge remains a separately authorized, exact-SHA native Git action.

The P0-2a mission may add a reviewed policy file and deterministic check IDs. It may not add a
free-form command facility or alter the Stage 2 server to obtain a passing result.

## 18. What remains outside MCP

- architecture decisions, priority, thresholds, and acceptance gates;
- independent review and approval;
- Git fetch, worktree/branch creation, source editing, stage/commit, PR, push, merge, and tags;
- GitHub Actions and branch protection;
- live/paid provider evaluation;
- product telemetry semantics and P0-2a/P0-2b fixes;
- publication and deployment;
- credentials, secrets, network access, and cloud services;
- A2A communication or orchestration;
- tamper-proof or centrally retained audit.

Native Git/CLI remains better for transparent, human-authorized mutations. MCP is used where a
cross-agent structured evidence contract adds value.

## 19. Threat model and mitigations

| Threat | Control |
| --- | --- |
| Repository prompt injection | Tracked-file-only bounded reads, explicit untrusted label, no automatic command derivation from content, human review. |
| Path traversal | POSIX relative-path schema, strict resolution, root containment, `lstat`, symlink rejection. |
| Symlink escape | No symlink reads/writes; evidence and policy paths must resolve under root. |
| Arbitrary shell/argument injection | No shell, fixed policy argv, no free args/env/cwd, reject leading-dash paths. |
| Malicious mission-policy edit | Policy digest in every record, fixed policy path at launch, scope checked before/after, independent clean-checkout rerun. |
| Server self-modification | Server-code manifest digest in every record; server files outside allowed product mission scope; Reviewer uses clean validated server code. |
| Secret disclosure | No ignored/untracked reads, secret-like path denylist, no credential env, output redaction, no env dump. |
| Network/provider call | No network tools, offline launch, credential absence, check-specific pre-call test. No false OS-sandbox claim. |
| Stale worktree | Expected HEAD/status inputs, pre/post digests, global execution lock, state-change invalidation. |
| Destructive Git | No mutating Git command in code or policy; fixed read-only verb tests. |
| Output flood | Stream caps, process-group termination, truncation flag, full digest over retained bounded bytes. |
| Hung or orphan process | Per-check timeout, cancellation handling, new process group, kill-and-wait proof. |
| Concurrent evidence race | One nonblocking execution lock; evidence IDs and atomic writes. |
| Evidence tampering | Hashes and chained records plus independent re-execution; explicitly not claimed as external attestation. |
| Role self-approval | No approval tool or state; Reviewer must rerun; operator controls merge. |
| Capability creep | Exactly five tools; new tool or mutation requires new architecture and independent review. |

## 20. Frozen deterministic MCP acceptance gates

All gates are mandatory. Failure is not permission to weaken the gate.

1. Base gate: exact implementation base `ca29d32`, correct origin, clean isolated worktree.
2. Scope gate: only the files listed in section 24 change; product runtime and canonical state files
   remain byte-identical.
3. Dependency gate: `mcp==2.0.0` is an exact dev dependency; `uv.lock` is synchronized; production
   `--no-dev` behavior is unchanged.
4. Protocol gate: the programmatic probe passes modern `2026-07-28` discovery/call and legacy
   initialization/call paths; tool list order and schemas are deterministic.
5. Codex gate: local Codex connects, lists exactly five tools, runs `repo_snapshot`, rejects a
   traversal read, and preserves structured results.
6. Cursor gate: local Cursor connects from project config, lists exactly five tools, approval is on,
   runs `repo_snapshot`, and surfaces a deliberate `BLOCKED` result correctly.
7. Web boundary gate: documentation states ChatGPT web cannot use this local server; no remote
   workaround is silently added.
8. Read isolation gate: absolute paths, `..`, percent/Unicode traversal, symlinks, `.git`, `.env`,
   ignored, untracked, binary, oversized, and outside-root files are rejected.
9. Git gate: only enumerated read-only Git verbs occur; aliases, hooks, external diff/textconv, and
   revision-expression injection cannot execute.
10. Command gate: unknown check IDs and extra arguments are rejected; no shell invocation exists;
    child cwd/env/argv match policy.
11. Secret gate: sentinel secrets in parent environment do not reach child, results, stderr, logs, or
    bundles; known Content Agent credentials cause fail-closed startup.
12. Timeout/output gate: injected hang and output flood terminate the complete process group and
    return explicit `BLOCKED` evidence with no surviving child.
13. Stale/concurrency gate: mid-check tracked change invalidates evidence; a concurrent check returns
    `CHECK_BUSY` and does not run.
14. Evidence gate: every call validates against its output schema; full content/log/diff digests
    match artifacts; mixed-identity bundles are rejected; same semantic input produces the same
    semantic digest.
15. No-network-scope gate: implementation contains no HTTP client/network Git operation; the server
    runs with no provider credentials and makes no external call in deterministic probes.
16. Focused gate: all MCP unit, evidence, security, and compatibility tests pass.
17. Regression gate: existing fatal Ruff tier and `pytest tests/` pass from the final candidate SHA.
18. Operability gate: startup/shutdown, interrupted stdin, invalid policy, missing venv, and read-only
    target failures are explicit and leave no corrupt evidence record.
19. Independent-review gate: Reviewer repeats critical host, security, scope, and evidence tests at
    the exact candidate SHA. Implementer self-report is insufficient.
20. No merge/deploy gate: passing this implementation mission still does not authorize merge,
    deployment, P0-2a, or A2A.

## 21. MCP failure and kill conditions

Return `ARCHITECTURE-BLOCKED` and stop if:

- fetched `origin/main` differs from the authorized base;
- Codex or Cursor cannot reliably connect to stdio or consume the five tool schemas;
- the Reviewer is restricted to ChatGPT web and cannot use an MCP-capable local host;
- arbitrary shell, arbitrary file reads, Git mutation, product-runtime coupling, or provider
  credentials become necessary for basic usefulness;
- target-root containment, policy immutability checks, or process cleanup cannot be enforced;
- structured evidence cannot bind to pre/post exact SHA and worktree state;
- a test needs a threshold change, retry-until-green, or P0-2a product change;
- implementation requires Streamable HTTP, remote hosting, OAuth, A2A, or multi-user state;
- full regression or any security gate fails;
- the capability surface exceeds the five frozen tools without a new architecture review.

After the real P0-2a mission, simplify or abandon MCP if any critical evidence fact—exact HEAD,
changed-file scope, deterministic exit status, no-provider precondition, or coherent bundle—still
requires an unstructured agent claim; if a stale/out-of-scope result can pass; or if Reviewer setup
and evidence review are materially slower than the native Git/CLI baseline without reducing a real
handoff error. Do not keep MCP merely because it has been built.

## 22. Performance and operability expectations

- warmed startup: at most 3 seconds;
- idle RSS: at most 150 MiB; the server must not import torch, LangGraph, Qdrant, browser, or Content
  Agent runtime modules;
- `repo_snapshot`: p95 at most 2 seconds on this repository;
- `read_tracked_file`: p95 at most 1 second for an allowed file;
- `inspect_change` without patch: p95 at most 5 seconds;
- `run_check` orchestration overhead: at most 500 ms excluding the child command;
- one check at a time; busy calls fail immediately rather than queue;
- evidence records are atomic, relative-path-only, bounded, and recoverable after server restart;
- stdio stdout contains protocol frames only; sanitized operational logs use stderr;
- no background daemon remains after host shutdown.

These are infrastructure budgets, not Content Agent product-performance claims.

## 23. Logging and audit strategy

Each tool call atomically writes its validated evidence JSON and bounded streams under
`outputs/mcp_evidence/operations/<evidence_id>/`. A process audit JSONL records start, finish,
status, repository/policy/server digests, artifact digests, and a previous-record digest. Logs never
contain raw environment values or unrestricted file content.

The response returns the evidence ID and semantic digest immediately. `build_evidence_bundle`
creates an ordered manifest of selected records. stderr is for sanitized operator diagnostics and
must never be parsed as evidence.

The hash chain detects accidental alteration when an earlier digest is retained, but the local
agent can control the filesystem. Therefore the audit strategy relies on exact-SHA independent
Reviewer re-execution, not a claim of immutability or non-repudiation.

## 24. Exact files and components expected for implementation

Allowed new files:

- `engineering_mcp/__init__.py`
- `engineering_mcp/__main__.py`
- `engineering_mcp/server.py`
- `engineering_mcp/policy.py`
- `engineering_mcp/repository.py`
- `engineering_mcp/runner.py`
- `engineering_mcp/evidence.py`
- `engineering_mcp/schemas.py`
- `engineering_mcp/probe.py`
- `mcp/policies/schema.json`
- `mcp/policies/stage2.json`
- `tests/test_mcp_server.py`
- `tests/test_mcp_evidence.py`
- `tests/test_mcp_security.py`
- `tests/test_mcp_compatibility.py`
- `.codex/config.toml`
- `.cursor/mcp.json`
- `docs/MCP_STAGE2_RUNBOOK.md`

Allowed existing-file changes:

- `pyproject.toml`: add exact `mcp==2.0.0` to the dev dependency group only;
- `uv.lock`: lock synchronization only;
- `.gitignore`: only if a new evidence ignore is needed; prefer the already ignored `outputs/`.

No `.github/workflows/ci.yml` change is needed because existing lint and `pytest tests/` discovery
already cover the new code. If implementation proves otherwise, report `ARCHITECTURE-BLOCKED`
instead of widening scope.

Server modules must depend only on the standard library, Git subprocesses, Pydantic already present,
and the pinned MCP SDK. They must not import product modules.

## 25. Explicit out of scope

- any change to `agent/`, `api/`, `static/`, `prompts/`, `tools/`, `evals/`, `main.py`, `config.py`,
  `scripts/benchmark.py`, evaluation behavior, or provider behavior;
- changes to `PROJECT_STATUS.md`, `architecture.md`, `DECISIONS.md`,
  `docs/EXPERIMENT_LEDGER.md`, `FREEZE.md`, or `README.md`;
- P0-2a or P0-2b implementation;
- A2A design or implementation;
- source editing, branch/worktree creation, commit, merge, push, publish, or deployment through MCP;
- remote/cloud MCP, HTTP/SSE transport, OAuth, multi-user authorization, registry publication;
- arbitrary shell, generic Git, generic filesystem, browser, provider, database, or website tools;
- MCP resources/prompts/tasks/apps, UI, sampling, roots, elicitation, subscriptions;
- production Content Agent runtime integration or Docker image change;
- paid/live calls.

## 26. Tracked architecture artifact

This file is the only tracked artifact produced by the architecture mission. It is deliberately
separate from the accepted product architecture and does not mark Stage 2 validated. The isolated
branch/commit identity is reported in the Architect's handoff; no merge or deployment is authorized.

## 27. Decision

`MCP-CURSOR-READY`

The problem, transport, five-tool capability surface, mutation boundary, security model, schemas,
implementation files, deterministic gates, and stop conditions are frozen. Cursor has no material
architecture decision to make. Connection and security outcomes remain validation gates, not open
design choices.

## 28. Fresh-thread Cursor implementation specification

The following prompt is self-contained and assumes zero prior conversation knowledge.

---

### CONTENT AGENT — STAGE 2 MCP IMPLEMENTATION MISSION

You are the Implementation Agent in a fresh Cursor thread.

Repository: `https://github.com/anudeepreddy332/content-agent`

Exact authorized base: `ca29d32b4869269daa47142615d298580a577a77`

Role: implementation only.

Workflow: `SPECIFICATION -> IMPLEMENT -> TEST -> MEASURE -> REPORT -> STOP`.

Do not redesign architecture, change thresholds, approve your own work, merge, push, deploy, publish,
make paid/live calls, implement P0-2a/P0-2b, or design A2A.

#### Mandatory preflight

1. Fetch `origin/main`.
2. Verify fetched `origin/main` is exact
   `ca29d32b4869269daa47142615d298580a577a77`. Otherwise report
   `ARCHITECTURE-BLOCKED` and stop.
3. Create a clean isolated worktree/branch from that exact SHA. Do not use or modify a dirty user
   checkout.
4. Read completely: `PROJECT_STATUS.md`, `architecture.md`, `DECISIONS.md`,
   `docs/EXPERIMENT_LEDGER.md`, `README.md`, and `docs/MCP_STAGE2_ARCHITECTURE.md`.
5. Verify no existing MCP implementation overlaps the authorized files.

#### Objective

Implement the local stdio `engineering_mcp` engineering-evidence server specified in
`docs/MCP_STAGE2_ARCHITECTURE.md` without changing Content Agent product behavior.

Use exact dev dependency `mcp==2.0.0`. The server must support modern MCP `2026-07-28` and legacy
initialization-era clients through the SDK. It exposes exactly five tools in deterministic order:

1. `repo_snapshot`
2. `read_tracked_file`
3. `inspect_change`
4. `run_check`
5. `build_evidence_bundle`

Implement the exact input/output schemas, common evidence envelope, timeouts, output limits, status
codes, and evidence behavior in sections 10-16 of the architecture document. Return structured JSON
plus serialized-JSON text fallback. Security must be enforced in code; server instructions are not
a security boundary.

#### Frozen architecture

- transport: local stdio only;
- one logical server package, one process per host/worktree;
- root and policy fixed at launch, never supplied by a tool call;
- no product-module imports;
- no source mutation, Git mutation, approval, provider, network, arbitrary filesystem, or arbitrary
  command capability;
- no shell; policy-defined fixed argv only;
- Git operations limited to the read-only verbs in section 14;
- writes only under `outputs/mcp_evidence/` and check temp paths;
- pre/post exact HEAD and worktree digest binding;
- no automatic retries;
- local hashes are provenance, not external attestation;
- user approval remains on in both hosts.

#### Allowed files

Create only:

- `engineering_mcp/__init__.py`
- `engineering_mcp/__main__.py`
- `engineering_mcp/server.py`
- `engineering_mcp/policy.py`
- `engineering_mcp/repository.py`
- `engineering_mcp/runner.py`
- `engineering_mcp/evidence.py`
- `engineering_mcp/schemas.py`
- `engineering_mcp/probe.py`
- `mcp/policies/schema.json`
- `mcp/policies/stage2.json`
- `tests/test_mcp_server.py`
- `tests/test_mcp_evidence.py`
- `tests/test_mcp_security.py`
- `tests/test_mcp_compatibility.py`
- `.codex/config.toml`
- `.cursor/mcp.json`
- `docs/MCP_STAGE2_RUNBOOK.md`

Modify only:

- `pyproject.toml` to add `mcp==2.0.0` to dev dependencies;
- `uv.lock` for the corresponding lock synchronization;
- `.gitignore` only if necessary, preferring existing ignored `outputs/`.

`docs/MCP_STAGE2_ARCHITECTURE.md` is read-only.

#### Forbidden files and behavior

Do not modify `agent/`, `api/`, `static/`, `prompts/`, `tools/`, `evals/`, `main.py`, `config.py`,
`scripts/benchmark.py`, `.github/workflows/`, Docker/deploy files, canonical state documents, README,
or any product runtime/evaluation behavior.

Do not add resources, prompts, roots, sampling, elicitation, tasks, UI, HTTP/SSE, OAuth, remote
deployment, A2A, Git mutation, file-write/edit tools, policy-edit tools, generic subprocess tools, or
provider calls.

#### Stage 2 mission policy

Implement a JSON Schema-validated `mcp/policies/stage2.json` with:

- mission ID and exact repository identity;
- exact authorized base;
- the allowed changed-file scope from section 24;
- exact five tool definitions/order;
- exact six check IDs, argv arrays, timeouts, 1 MiB stream limits, environment classes, and artifact
  rules from section 13;
- fixed evidence root `outputs/mcp_evidence`;
- policy and server-code digest reporting.

The agent must not be able to supply an allowed-path list or command arguments at call time.

#### Security implementation requirements

1. Strict root containment and regular-file checks; reject symlinks and traversal variants.
2. Git-tracked UTF-8 text reads only; secret-like, `.git`, ignored, untracked, binary, and oversized
   content denied.
3. `shell=False`; fixed argv; disable Git aliases, hooks, external diff/textconv and prompts.
4. Construct child environment; reject known Content Agent credentials; never log environment
   values.
5. New process group for checks; kill and wait for the entire group on timeout, cancellation, or
   output overflow.
6. Global nonblocking check lock; concurrent call returns `CHECK_BUSY`.
7. Atomic evidence writes, bounded logs, relative paths, output schemas, redaction tests.
8. Pre/post HEAD and status digests; mid-run change invalidates evidence.
9. No network client or Git network operation. Do not claim OS-level network sandboxing.
10. stdout contains MCP frames only; sanitized diagnostics use stderr.

#### Host configuration

Create project-local Codex and Cursor configs that launch with
`uv run --frozen --offline python -m engineering_mcp`, bind the project root and Stage 2 policy, pass
no credentials, expose exactly five tools, and preserve tool approval. Use only syntax established by
current official host documentation. If a portable project-root expression cannot be proven in an
actual host, document operator configuration in the runbook and report `ARCHITECTURE-BLOCKED` rather
than inventing it.

#### Required tests and validation order

Follow this order; stop at the first blocker:

1. Base/origin/clean-worktree verification.
2. Add the exact dependency and synchronize the lock.
3. Schema/policy unit tests.
4. Modern and legacy protocol probe.
5. Repository/read/change/evidence unit tests.
6. Traversal, symlink, `.git`, `.env`, ignored/untracked, binary, size, secret, command-injection,
   Git-injection, timeout, output-flood, orphan-child, concurrency, stale-state, mixed-bundle, and
   redaction security tests.
7. Local Codex connection smoke at the actual installed host.
8. Local Cursor connection smoke at the actual installed host.
9. Exact Stage 2 scope inspection.
10. Fatal Ruff tier.
11. Full `pytest tests/` regression from the final candidate state.
12. Repeat critical gates after committing so evidence binds to the exact candidate SHA.

Run deterministic tests before any operation that could call a paid provider. No paid/live provider
operation is authorized at all in this mission. Do not retry until lucky and do not weaken a test or
timeout after seeing a failure.

Apply every acceptance gate and kill condition in sections 20 and 21 of the architecture document.
In particular, ChatGPT web is not a substitute for local Reviewer access, and Streamable HTTP is not
a fallback.

#### Git discipline

- keep the isolated worktree clean except for authorized changes;
- do not touch unrelated user work;
- do not stage or commit until focused gates pass;
- create one coherent implementation commit only after validation;
- do not rebase, amend, squash, force, push, merge, tag, or deploy;
- after committing, report exact parent, candidate SHA, changed-file list, and `git diff --check`.

#### Stop conditions

Report `ARCHITECTURE-BLOCKED` and stop on any base drift, scope overlap, client incompatibility,
arbitrary-shell/filesystem need, network/provider need, product coupling, policy bypass, secret leak,
path/symlink escape, surviving child, stale-state false pass, schema incompatibility, full-regression
failure, or need to change the frozen architecture.

#### Required response

Return:

1. exact base, branch, and clean-worktree proof;
2. files created/changed;
3. server architecture and package/dependency version;
4. exact five tools and schemas implemented;
5. Codex actual connection/tool-call result;
6. Cursor actual connection/tool-call result;
7. focused test counts and commands;
8. security/failure-path test counts and outcomes;
9. fatal lint and full regression counts;
10. timeout/output/concurrency/stale-state evidence;
11. secret/network-boundary evidence and explicit non-claim of OS sandboxing;
12. evidence-bundle path/digest from the candidate SHA;
13. exact commit SHA, parent, and changed-file scope;
14. any warnings or unknowns;
15. final status `IMPLEMENTATION-READY-FOR-INDEPENDENT-REVIEW` or `ARCHITECTURE-BLOCKED`;
16. numerical confidence `0.00-1.00`.

Stop after reporting. Do not merge or begin P0-2a.

---

## 29. Confidence

`0.91`

The architecture is high-confidence for the present local, supervised workflow because it uses the
common host denominator and removes mutation from MCP. Residual uncertainty is concentrated in
actual dual-host interoperability and long-running tool-call behavior; both are frozen fail-closed
implementation gates rather than deferred design choices.
