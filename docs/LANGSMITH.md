# Optional LangSmith tracing

Off by default. Adds cross-node trace visualization (which node ran, how long, what it
returned, and per-LLM-call token usage/cost) on top of — not instead of — the structlog
JSON logs every run already produces. There is no code change to enable or disable this
beyond setting environment variables.

## Enable it

Tracing auto-enables once both of these are set — no extra flag required:

```bash
LANGCHAIN_API_KEY=<your LangSmith API key>
LANGCHAIN_PROJECT=<a project name, e.g. content-agent-dev>
```

Add them to `.env` (CLI picks them up automatically) or pass them to the container
(`docker-compose.prod.yml`'s `env_file: .env`, same as `DEEPSEEK_API_KEY`/`TAVILY_API_KEY`).

Get an API key at https://smith.langchain.com (free tier covers this project's run volume).

### Force it off

Set `LANGSMITH_TRACING=0` to keep tracing off even if `LANGCHAIN_API_KEY` and
`LANGCHAIN_PROJECT` are present (e.g. a shared `.env` with credentials you don't want
this run to use). Any other value of `LANGSMITH_TRACING`, or leaving it unset, has no
effect — it is not required to opt in.

## What happens when it's on

`observability/tracing.py::setup_langsmith_tracing()` runs once at startup (CLI: inside
`main.py run`, before the graph is built; API: at `api/server.py` import time, before
`build_graph()`). If both vars above are set (and the force-off override isn't), it
normalizes `LANGSMITH_TRACING` to the literal string `"true"` that `langchain_core`/
`langsmith` check for internally, then returns. From that point on, LangGraph's native
LangChain-callback tracing activates process-wide — every node in the graph (`retrieve`,
`draft`, `verify`, `reflect`, `hitl`, `html_gen`, `hitl_html`, `html_revise`, `git`) is
traced automatically. No node code, no `graph.invoke()` call, no prompt, and no
`prompt_version` changes — tracing is observability, not logic.

Additionally, `agent/nodes.py::_get_client()` wraps the DeepSeek client with LangSmith's
`wrap_openai()` when tracing is on, so every `_llm_call()` becomes its own traced "llm"
run with token usage attached (DeepSeek's API is OpenAI-compatible, so the same wrapper
applies). `_llm_call()` then layers this pipeline's own DeepSeek per-token pricing
(`DEEPSEEK_INPUT_COST_PER_M` / `DEEPSEEK_OUTPUT_COST_PER_M`) onto that run's
`usage_metadata`, since LangSmith's built-in price table has no entry for
`deepseek-chat` — without this, token counts would show but cost would read as zero.

## What happens when it's off (the default)

`setup_langsmith_tracing()` checks the env vars and returns `False` immediately. It
never imports the `langsmith` client, never touches any other environment variable, and
adds no latency. `is_tracing_enabled()` (used by `_get_client()`/`_llm_call()`) is a
plain env-var read — when tracing is off, the DeepSeek client is never wrapped and no
`langsmith` import happens on the LLM call path either. Importing/booting the CLI or the
API server works identically whether or not these vars are set — see
`tests/test_tracing.py` for the proof (off-by-default, force-off override,
partial-set-stays-off, and a fresh-process import check both with nothing set and with
dummy credentials set).

## Coexistence with structlog

This does not replace `observability/logger.py`. Every node still logs its own structured
JSON line to stdout exactly as before — that's what the eval harnesses, `check_telemetry_fields.py`,
and the reconstructability story in `docs/PRODUCTION_READINESS.md` depend on. LangSmith tracing
is purely additive: a second, optional view of the same runs, useful for visually inspecting
the graph's execution tree and per-call token usage/cost rather than grepping JSON.
