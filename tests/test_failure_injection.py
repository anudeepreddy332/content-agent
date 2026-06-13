"""B2: every documented fault mode degrades gracefully (or crashes LOUDLY where
it must — M4 lesson: 'fail loud' can be silently defeated by per-node except)."""
import pytest
from openai import AuthenticationError, RateLimitError, APITimeoutError

import agent.nodes as nodes
from tests.conftest import FakeLLMClient, fake_response, openai_error

# ---------- Fault 1: DeepSeek auth / timeout / rate limit ----------
def test_llm_call_no_retry_on_auth_error():
    client = FakeLLMClient(errors=[openai_error(AuthenticationError, 401)] * 3)
    with pytest.raises(AuthenticationError):
        nodes._llm_call(client, model="x", messages=[])
    assert client.calls == 1, "auth errors must NOT be retried (will never succeed)"

def test_llm_call_retries_rate_limit_then_reraises():
    client = FakeLLMClient(errors=[openai_error(RateLimitError, 429)] * 5)
    with pytest.raises(RateLimitError):
        nodes._llm_call(client, model="x", messages=[])
    assert client.calls == 3, "transient errors retry exactly 3 attempts, then reraise"

def test_llm_call_timeout_recovers_on_third_attempt():
    client = FakeLLMClient(
        response=fake_response("ok"),
        errors=[openai_error(APITimeoutError, 0), openai_error(APITimeoutError, 0)],
    )
    result = nodes._llm_call(client, model="x", messages=[])
    assert client.calls == 3
    assert result.choices[0].message.content == "ok"

def test_draft_auth_error_propagates_to_crash_handler(base_state, monkeypatch):
    """The crash path contract: draft must NOT swallow auth errors into graceful
    degradation — main.py's handler needs the raise to write crash telemetry."""
    client = FakeLLMClient(errors=[openai_error(AuthenticationError, 401)])
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    with pytest.raises(AuthenticationError):
        nodes.draft_node(base_state)


# ---------- Fault 2: Tavily empty / error ----------
def test_retrieve_tavily_empty_results(base_state, monkeypatch):
    calls = {"n": 0}

    def empty_search(query, max_results=5, force_refresh=False):
        calls["n"] += 1
        return []

    monkeypatch.setattr(nodes, "web_search", empty_search)
    monkeypatch.setattr(nodes, "query_kb", lambda query, n_results=5: base_state["kb_results"])

    result = nodes.retrieve_node(base_state)

    assert result["web_sources"] == []
    assert calls["n"] == 6, "3 first-pass queries + 3 force-refresh queries"
    assert any("low quality sources" in e for e in result["error_log"])
    assert result["kb_results"], "KB path must still run when web is empty"
    lm = result["latency_ms"]
    assert lm["retrieve"] == lm["retrieve_web"] + lm["retrieve_kb"], \
        "regression guard for the M5 retrieve_kb split fix"

def test_retrieve_tavily_errors_logged_not_fatal(base_state, monkeypatch):
    def exploding_search(query, max_results=5, force_refresh=False):
        raise ConnectionError("tavily unreachable")

    monkeypatch.setattr(nodes, "web_search", exploding_search)
    monkeypatch.setattr(nodes, "query_kb", lambda query, n_results=5: [])

    result = nodes.retrieve_node(base_state)  # must not raise

    tavily_errors = [e for e in result["error_log"] if "Tavily" in e]
    assert len(tavily_errors) >= 6, "3 first-pass + 3 refresh errors all captured"
    assert result["web_sources"] == []


# ---------- Fault 3: Qdrant down ----------
@pytest.fixture
def reset_query_kb_globals():
    import tools.query_kb as qk
    saved = (qk._client, qk._encoder, qk._bm25)
    yield qk
    qk._client, qk._encoder, qk._bm25 = saved


def test_query_kb_qdrant_down_returns_empty(reset_query_kb_globals):
    qk = reset_query_kb_globals

    class DeadQdrant:
        def get_collection(self, name):
            raise ConnectionError("qdrant unreachable")

    qk._client = DeadQdrant()
    assert qk.query_kb("anything", n_results=5) == [], \
        "_collection_exists must convert connection failure into empty results"


def test_retrieve_survives_kb_down(base_state, monkeypatch):
    monkeypatch.setattr(nodes, "web_search",
                        lambda query, max_results=5, force_refresh=False:
                        [{"title": "t", "url": f"https://e.com/{query[:10]}",
                          "content": "c", "score": 0.9}])
    monkeypatch.setattr(nodes, "query_kb", lambda query, n_results=5: [])
    result = nodes.retrieve_node(base_state)
    assert result["kb_results"] == []
    assert result["web_sources"], "web path unaffected by KB outage"


# ---------- Fault 4: malformed model JSON ----------
def test_extract_json_array_strips_fences():
    assert nodes._extract_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]

def test_extract_json_array_recovers_from_preamble():
    raw = 'Here is the analysis:\n[{"claim": "x"}]\nHope that helps!'
    assert nodes._extract_json_array(raw) == [{"claim": "x"}]

def test_extract_json_array_raises_on_garbage():
    with pytest.raises((ValueError, Exception)):
        nodes._extract_json_array("I cannot produce JSON, sorry.")


def test_verify_malformed_output_degrades_gracefully(base_state, monkeypatch):
    client = FakeLLMClient(response=fake_response("Sorry, no JSON from me today."))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)

    result = nodes.verify_node(base_state)  # must not raise

    assert result["grounding_report"] == []
    assert result["grounding_score"] == 0
    assert result["iteration_metrics"][-1]["N"] == 0, \
        "parse failure must still produce an iteration_metrics entry"


def test_draft_malformed_output_degrades_gracefully(base_state, monkeypatch):
    client = FakeLLMClient(response=fake_response("not json at all"))
    monkeypatch.setattr(nodes, "_get_client", lambda: client)

    result = nodes.draft_node(base_state)  # must not raise

    assert "[PARSE ERROR]" in result["draft_markdown"]
    assert "not json at all" in result["draft_markdown"], \
        "raw output preserved for debugging"


# ---------- Fault 5: cost-gate breach ----------
def test_verify_cost_gate_skips_llm_entirely(base_state, monkeypatch):
    sentinel = FakeLLMClient()  # raises AssertionError if .create is ever called
    monkeypatch.setattr(nodes, "_get_client", lambda: sentinel)
    base_state["total_cost_usd"] = nodes.COST_GATE_USD

    result = nodes.verify_node(base_state)

    assert sentinel.calls == 0, "cost gate must fire BEFORE the LLM call"
    assert result["grounding_report"] == []


def test_reflect_cost_gate_skips_llm(base_state, monkeypatch):
    sentinel = FakeLLMClient()
    monkeypatch.setattr(nodes, "_get_client", lambda: sentinel)
    base_state["total_cost_usd"] = nodes.COST_GATE_USD

    result = nodes.reflect_node(base_state)  # must not raise

    assert sentinel.calls == 0
    assert isinstance(result, dict)


def test_route_cost_gate_forces_hitl_over_revision(base_state):
    base_state.update(total_cost_usd=nodes.COST_GATE_USD,
                      iterations=0, reflection_score=2, grounding_score=0.1)
    assert nodes.route_after_reflect(base_state) == "hitl", \
        "cost gate must beat the revise gate even on a terrible draft"


def test_route_max_iterations_forces_hitl(base_state):
    base_state.update(iterations=nodes.MAX_ITERATIONS,
                      reflection_score=2, grounding_score=0.1)
    assert nodes.route_after_reflect(base_state) == "hitl"


