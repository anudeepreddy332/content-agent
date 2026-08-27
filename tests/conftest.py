"""Shared fixtures for the B2 failure-injection suite. All LLM/Tavily/Qdrant
interactions are mocked — this suite costs $0 and must stay that way."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    """Neutralize tenacity's backoff (2s->4s->8s) so retry tests run in ms."""
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)

@pytest.fixture
def base_state():
    """Minimal valid AgentState — mirrors main.py's initial_state shape."""
    return {
        "topic": "Gradient Descent",
        "slug": "gradient-descent-test",
        "card_id": "B2-TEST",
        "series_context": "Test",
        "draft_sections": {},
        "draft_markdown": "## Test\nGradient descent minimizes a loss function.",
        "web_sources": [{"title": "t", "url": "https://example.com/gd",
                         "content": "gradient descent content", "score": 0.9}],
        "kb_results": [{"text": "kb chunk", "source": "gd.md",
                        "chunk_index": 0, "distance": 0.1, "rrf_score": 0.03}],
        "grounding_report": [],
        "grounding_score": 0.0,
        "reflection_score": 0,
        "reflection_notes": "",
        "reflection_provenance": {
            "origin": "unavailable",
            "reason": "not_run",
            "provider_called": False,
            "parse_status": "not_started",
        },
        "iterations": 0,
        "hitl_status": "pending",
        "hitl_feedback": None,
        "html_output": None,
        "html_filename": None,
        "article_body_html": None,
        "html_sha256": None,
        "html_policy_version": None,
        "approved_html_sha256": None,
        "git_commit_sha": None,
        "publish_expected_remote_sha": None,
        "publish_observed_remote_sha": None,
        "publish_status": None,
        "publish_error": None,
        "published_remote_sha": None,
        "published_live_url": None,
        "branch_name": None,
        "git_status": None,
        "run_id": "b2-test-run",
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "latency_ms": {},
        "error_log": [],
        "policy_diagnostics": [],
        "iteration_metrics": [],
        "m4_feedback_claims": 0,
    }

def fake_response(content: str, tokens: int = 100):
    """Duck-typed OpenAI ChatCompletion: .choices[0].message.content + .usage."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=tokens // 2,
                              completion_tokens=tokens // 2,
                              total_tokens=tokens),
    )

class FakeLLMClient:
    """Stands in for the OpenAI client. Raises from `errors` in order, then
    returns `response`. Counts attempts so retry semantics are assertable."""
    def __init__(self, response=None, errors=None):
        self.calls = 0
        self._response = response
        self._errors = list(errors or [])
        c = self

        class _Completions:
            def create(self, **kwargs):
                c.calls += 1
                if c._errors:
                    raise c._errors.pop(0)
                if c._response is None:
                    raise AssertionError("LLM was called but no response configured "
                                         "(sentinel for cost-gate tests)")
                return c._response

        self.chat = SimpleNamespace(completions=_Completions())


def openai_error(cls, status: int):
    """Build a real openai exception instance (SDK >=1.x signatures)."""
    import httpx
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    if cls.__name__ == "APITimeoutError":
        return cls(request=req)
    resp = httpx.Response(status, request=req)
    return cls("injected", response=resp, body=None)

