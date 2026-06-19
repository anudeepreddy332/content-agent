"""Opt-in LangSmith tracing (observability/tracing.py). Proves the OFF-by-default
gate and that importing api.server never crashes regardless of the flag. $0,
zero network — setup_langsmith_tracing() only ever sets env vars + logs; it
never imports or calls the langsmith client itself."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from observability.tracing import setup_langsmith_tracing

_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean_tracing_env(monkeypatch):
    """Every test starts with all three gate vars unset, regardless of the
    ambient environment (e.g. a developer's real .env)."""
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)


def test_disabled_by_default_with_nothing_set():
    assert setup_langsmith_tracing() is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_disabled_when_tracing_flag_missing(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "dummy-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "dummy-project")
    assert setup_langsmith_tracing() is False


def test_disabled_when_api_key_missing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "1")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "dummy-project")
    assert setup_langsmith_tracing() is False


def test_disabled_when_project_missing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "1")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "dummy-key")
    assert setup_langsmith_tracing() is False


def test_disabled_when_flag_not_exactly_one(monkeypatch):
    """Only the literal "1" gates this project's own check (distinct from the
    "true" string langsmith.utils itself checks for after normalization)."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "dummy-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "dummy-project")
    assert setup_langsmith_tracing() is False


def test_enabled_when_all_three_set_and_normalizes_flag(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "1")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "dummy-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "dummy-project")
    assert setup_langsmith_tracing() is True
    # Normalized to what langchain_core/langsmith actually check for.
    assert os.environ["LANGSMITH_TRACING"] == "true"


def _import_api_server_in_subprocess(env_overrides: dict) -> subprocess.CompletedProcess:
    """A fresh process, not a reload — proves the actual import path (including
    module-level setup_langsmith_tracing() + build_graph()) never crashes,
    independent of whatever other tests already did to the shared test session's
    cached api.server module."""
    env = {**os.environ, "API_SYNC": "1", **env_overrides}
    for v in _VARS:
        env.pop(v, None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "import api.server"],
        capture_output=True, text=True, timeout=60, env=env, cwd=str(_REPO_ROOT),
    )


def test_api_server_imports_with_tracing_unset():
    proc = _import_api_server_in_subprocess({})
    assert proc.returncode == 0, proc.stderr


def test_api_server_imports_with_tracing_dummy_credentials():
    """Dummy/invalid credentials must not crash import or graph setup -
    setup_langsmith_tracing() only sets env vars and logs; LangSmith's own
    client lazily makes network calls on an actual traced run, not at import
    time, so a bad key surfaces later (if at all), never here."""
    proc = _import_api_server_in_subprocess({
        "LANGSMITH_TRACING": "1",
        "LANGCHAIN_API_KEY": "dummy-key-not-real",
        "LANGCHAIN_PROJECT": "dummy-project",
    })
    assert proc.returncode == 0, proc.stderr
