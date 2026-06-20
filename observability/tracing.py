"""
Opt-in LangSmith tracing — OFF by default, additive to structlog (never replaces it).

Tracing auto-enables when BOTH of these env vars are set:
    LANGCHAIN_API_KEY=<key>
    LANGCHAIN_PROJECT=<project name>

No extra flag is required. LANGSMITH_TRACING="0" is a force-off override: if set,
tracing stays off even when both vars above are present. Any other value of
LANGSMITH_TRACING (including unset) is ignored — it is not required to opt in.

If the key or project is missing (and the force-off override isn't set),
`setup_langsmith_tracing()` is a true no-op: it reads env vars and returns False,
with zero imports beyond the stdlib and zero side effects. There is no latency or
error-surface cost to leaving tracing off.

When enabled, this normalizes LANGSMITH_TRACING to the literal string "true" that
langchain_core/langsmith actually check for internally (langsmith.utils.get_env_var
requires the string "true", not "1"). Once set, LangGraph's native LangChain-callback-
based tracing picks up every node run automatically: no graph.invoke()/build_graph()
argument changes, no per-node instrumentation, no prompt or prompt_version change.
structlog keeps logging every node exactly as before — this is purely additive.
"""
import os

from observability.logger import get_logger

log = get_logger("tracing")


def setup_langsmith_tracing() -> bool:
    """Enable LangSmith tracing iff LANGCHAIN_API_KEY and LANGCHAIN_PROJECT are both
    set, unless LANGSMITH_TRACING="0" forces it off. Returns True if enabled, False
    if left off (the default). Never raises — a missing/invalid var just means no
    tracing, not a startup failure."""
    if os.environ.get("LANGSMITH_TRACING") == "0":
        return False
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return False
    project = os.environ.get("LANGCHAIN_PROJECT")
    if not project:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    log.info("tracing.langsmith_enabled", project=project)
    return True


def is_tracing_enabled() -> bool:
    """Cheap, import-free check of the current tracing state, intended to be called
    after setup_langsmith_tracing() has run at process startup. Used to decide
    whether to pay the (small) cost of wrapping the LLM client for usage/cost
    capture — never imports langsmith itself."""
    return os.environ.get("LANGSMITH_TRACING") == "true"
