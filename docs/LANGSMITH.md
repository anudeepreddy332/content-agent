# Optional LangSmith tracing

Off by default. Adds cross-node trace visualization (which node ran, how long, what it
returned) on top of — not instead of — the structlog JSON logs every run already produces.
There is no code change to enable or disable this beyond setting environment variables.

## Enable it

All three of these must be set, or tracing stays off:

```bash
LANGSMITH_TRACING=1
LANGCHAIN_API_KEY=<your LangSmith API key>
LANGCHAIN_PROJECT=<a project name, e.g. content-agent-dev>
```

Add them to `.env` (CLI picks them up automatically) or pass them to the container
(`docker-compose.prod.yml`'s `env_file: .env`, same as `DEEPSEEK_API_KEY`/`TAVILY_API_KEY`).

Get an API key at https://smith.langchain.com (free tier covers this project's run volume).

## What happens when it's on

`observability/tracing.py::setup_langsmith_tracing()` runs once at startup (CLI: inside
`main.py run`, before the graph is built; API: at `api/server.py` import time, before
`build_graph()`). If all three vars above are set, it normalizes `LANGSMITH_TRACING` to the
literal string `"true"` that `langchain_core`/`langsmith` check for internally, then returns.
From that point on, LangGraph's native LangChain-callback tracing activates process-wide —
every node in the graph (`retrieve`, `draft`, `verify`, `reflect`, `hitl`, `html_gen`,
`hitl_html`, `html_revise`, `git`) is traced automatically. No node code, no `graph.invoke()`
call, no prompt, and no `prompt_version` changes — tracing is observability, not logic.

## What happens when it's off (the default)

`setup_langsmith_tracing()` checks the three env vars and returns `False` immediately. It
never imports the `langsmith` client, never touches any other environment variable, and adds
no latency. Importing/booting the CLI or the API server works identically whether or not any
of these three vars are set — see `tests/test_tracing.py` for the proof (off-by-default,
partial-set-stays-off, and a fresh-process import check both with the flag unset and with
dummy credentials set).

## Coexistence with structlog

This does not replace `observability/logger.py`. Every node still logs its own structured
JSON line to stdout exactly as before — that's what the eval harnesses, `check_telemetry_fields.py`,
and the reconstructability story in `docs/PRODUCTION_READINESS.md` depend on. LangSmith tracing
is purely additive: a second, optional view of the same runs, useful for visually inspecting
the graph's execution tree rather than grepping JSON.
