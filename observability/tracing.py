"""
Opt-in LangSmith tracing — OFF by default, additive to structlog (never replaces it).

Tracing is enabled only when ALL three of these env vars are set:
    LANGSMITH_TRACING=1
    LANGCHAIN_API_KEY=<key>
    LANGCHAIN_PROJECT=<project name>

If any is missing, `setup_langsmith_tracing()` is a true no-op: it reads three
env vars and returns False, with zero imports beyond the stdlib and zero
side effects. There is no latency or error-surface cost to leaving tracing off.

When all three are set, this normalizes LANGSMITH_TRACING's value to the
literal string "true" that langchain_core/langsmith actually check for
(langsmith.utils.get_env_var requires the string "true", not "1" — "1" is
this project's own gate value, not LangSmith's). Once set, LangGraph's
native LangChain-callback-based tracing picks up every node run automatically:
no graph.invoke()/build_graph() argument changes, no per-node instrumentation,
no prompt or prompt_version change. structlog keeps logging every node exactly
as before — this is purely additive.
"""
import os

from observability.logger import get_logger

log = get_logger("tracing")


def setup_langsmith_tracing() -> bool:
    """Enable LangSmith tracing iff LANGSMITH_TRACING=1, LANGCHAIN_API_KEY, and
    LANGCHAIN_PROJECT are all set. Returns True if enabled, False if left off
    (the default). Never raises — a missing/invalid var just means no tracing,
    not a startup failure."""
    if os.environ.get("LANGSMITH_TRACING") != "1":
        return False
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return False
    project = os.environ.get("LANGCHAIN_PROJECT")
    if not project:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    log.info("tracing.langsmith_enabled", project=project)
    return True
