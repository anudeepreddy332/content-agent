"""P0-2b slice 2B: semantic_trace_v1 reconstructability. Deterministic, $0, no providers."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest
from langgraph.graph import END

import agent.nodes as nodes
from agent.semantic_trace import (
    canonical_json,
    embed_semantic_trace,
    empty_trace,
    record_draft,
    record_verify,
    reread_validate_trace,
    sha256_utf8,
    stamp_final,
    validate_trace,
)
from config import COST_GATE_USD, UVR_THRESHOLD
from main import _write_telemetry
from tests.conftest import FakeLLMClient, fake_response

SECRET = "sk-test-secret-do-not-emit-9f3a"
BEARER = "bearer-secret-do-not-emit-c1e2"


@pytest.fixture(autouse=True)
def _no_provider_or_langsmith(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "0")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)


def _draft_json(tag: str) -> str:
    return json.dumps({
        "problem_framing": f"framing-{tag}",
        "technical_dive": f"dive-{tag}",
        "code_snippets": f"code-{tag}",
        "takeaways": f"takeaways-{tag}",
    })


def _verdict(
    claim: str,
    status: str = "verified",
    source_url: str | None = "https://example.com/gd",
    confidence: float = 0.9,
    specificity: str = "substantive",
) -> dict:
    return {
        "claim": claim,
        "source_url": source_url,
        "confidence": confidence,
        "status": status,
        "specificity": specificity,
    }


def _merge(state: dict, update: dict) -> dict:
    merged = dict(state)
    merged.update(update)
    return merged


def _complete_trace(report: list | None = None) -> dict:
    """Accepted one-iteration trace with all four required objects and matching UVR_v1."""
    report = report or [_verdict(f"ok-{i}") for i in range(10)]
    trace = empty_trace({"run_id": "complete-run", "topic": "Gradient Descent"})
    record_draft(trace, iteration=1, draft_markdown="draft-a", revision_linkage=None)
    record_verify(
        trace,
        iteration=1,
        consumed=True,
        draft_markdown="draft-a",
        source_context="sources",
        user_message="Draft to verify:\ndraft-a\nAvailable sources:\nsources",
        verify_system_text="sys",
        raw_response="[]",
        parser_status="ok",
        pre_dedup_rows=[],
        dropped_rows=[],
        post_dedup_rows=[],
        post_attribution_rows=report,
        web_sources=[{"url": "https://example.com/gd", "content": "gradient descent content"}],
        kb_results=[{"source": "kb.md", "text": "kb text", "chunk_index": 0}],
    )
    stamp_final(
        trace,
        {
            "verification_status": "completed",
            "grounding_report": report,
            "hitl_status": "approved",
        },
        semantic_accepted=True,
    )
    trace["trace_status"] = "ok"
    return trace


def test_sha256_identical_content_is_stable_and_changes_with_edits():
    assert sha256_utf8("alpha") == sha256_utf8("alpha")
    assert sha256_utf8("alpha") != sha256_utf8("beta")


def test_malformed_and_missing_trace_fail_validation():
    assert validate_trace({})
    assert validate_trace(None)
    assert validate_trace("nope")
    broken = empty_trace({"run_id": "r", "topic": "t"})
    broken["iterations"] = [{
        "iteration": 1,
        "draft": {"draft_markdown": "hello", "draft_sha256": "deadbeef"},
        "verifier_raw": None,
        "verifier_input": None,
        "revision_linkage": None,
        "post_processing": None,
    }]
    errors = validate_trace(broken)
    assert any("draft_sha256" in e for e in errors)


def test_iteration_one_and_two_drafts_are_preserved_independently(base_state, monkeypatch):
    unverified = "Linear models can only draw straight-line boundaries."
    monkeypatch.setattr(
        nodes, "_get_client",
        lambda: FakeLLMClient(response=fake_response(_draft_json("one"))),
    )
    start = _merge(base_state, {"iterations": 0, "run_id": "trace-run"})
    d1 = nodes.draft_node(start)
    assert d1["iterations"] == 1
    md1 = d1["draft_markdown"]
    assert "framing-one" in md1
    slot1 = d1["semantic_trace"]["iterations"][0]
    assert slot1["iteration"] == 1
    assert slot1["draft"]["draft_markdown"] == md1
    assert slot1["draft"]["draft_sha256"] == sha256_utf8(md1)
    assert slot1["revision_linkage"] is None

    verify_raw = json.dumps([
        _verdict(unverified, status="unverified", source_url=None, confidence=0.2),
        _verdict("Gradient descent updates parameters."),
    ])
    monkeypatch.setattr(
        nodes, "_get_client",
        lambda: FakeLLMClient(response=fake_response(verify_raw)),
    )
    s_after_d1 = _merge(start, d1)
    v1 = nodes.verify_node(s_after_d1)
    assert v1["verification_status"] == "completed"
    vin = v1["semantic_trace"]["iterations"][0]["verifier_input"]
    assert vin["consumed"] is True
    assert vin["user_message"]
    assert md1 in vin["user_message"]
    assert vin["source_context"] in vin["user_message"]
    assert "gradient descent content" in vin["source_context"]
    raw = v1["semantic_trace"]["iterations"][0]["verifier_raw"]
    assert raw["raw_response"] == verify_raw
    assert raw["parser_status"] == "ok"
    assert raw["pre_dedup_rows"][0]["claim"] == unverified

    s_after_v1 = _merge(s_after_d1, v1)
    monkeypatch.setattr(
        nodes, "_get_client",
        lambda: FakeLLMClient(response=fake_response(_draft_json("two"))),
    )
    d2 = nodes.draft_node(s_after_v1)
    assert d2["iterations"] == 2
    md2 = d2["draft_markdown"]
    assert md1 != md2
    trace = d2["semantic_trace"]
    assert [item["iteration"] for item in trace["iterations"]] == [1, 2]
    assert trace["iterations"][0]["draft"]["draft_markdown"] == md1
    assert trace["iterations"][1]["draft"]["draft_markdown"] == md2
    assert trace["iterations"][0]["draft"]["draft_sha256"] != trace["iterations"][1]["draft"]["draft_sha256"]
    link = trace["iterations"][1]["revision_linkage"]
    assert link["source_iteration"] == 1
    assert link["next_iteration"] == 2
    assert unverified in link["targeted_unverified_claims"]
    assert unverified in link["grounding_feedback_block"]

    monkeypatch.setattr(
        nodes, "_get_client",
        lambda: FakeLLMClient(response=fake_response(json.dumps([
            _verdict("Gradient descent updates parameters."),
        ]))),
    )
    v2 = nodes.verify_node(_merge(s_after_v1, d2))
    final_trace = v2["semantic_trace"]
    assert final_trace["iterations"][0]["draft"]["draft_markdown"] == md1
    assert final_trace["iterations"][1]["draft"]["draft_markdown"] == md2
    assert final_trace["iterations"][0]["verifier_raw"]["raw_response"] == verify_raw


def test_duplicate_removal_and_post_attribution_are_reconstructable(base_state, monkeypatch):
    claim = "Gradient descent is an iterative optimization algorithm that minimizes a loss function."
    near = "Gradient descent is an iterative optimization algorithm that minimizes a loss function!"
    raw = json.dumps([
        _verdict(claim, source_url="https://example.com/gd"),
        _verdict(near, source_url="https://example.com/gd"),
        _verdict("Unrelated verified fact about line search.", source_url="https://example.com/gd"),
    ])
    monkeypatch.setattr(nodes, "_get_client", lambda: FakeLLMClient(response=fake_response(raw)))
    result = nodes.verify_node(_merge(base_state, {"iterations": 1, "run_id": "dedup-run"}))
    slot = result["semantic_trace"]["iterations"][0]
    assert slot["verifier_raw"]["raw_response"] == raw
    assert len(slot["verifier_raw"]["pre_dedup_rows"]) == 3
    dropped = slot["post_processing"]["dropped_rows"]
    assert dropped
    assert dropped[0]["reason"] == "near_duplicate"
    assert len(slot["post_processing"]["post_dedup_rows"]) == 2
    post = slot["post_processing"]["post_attribution_rows"]
    assert post == result["grounding_report"]
    assert all("source_kind" in row for row in post)
    assert slot["post_processing"]["counts"]["dropped"] == len(dropped)
    assert slot["post_processing"]["counts"]["pre_dedup"] == 3


def test_parse_failed_empty_report_and_cost_gate_remain_traceable(base_state, monkeypatch):
    monkeypatch.setattr(nodes, "_get_client", lambda: FakeLLMClient(response=fake_response("not-json")))
    parsed = nodes.verify_node(_merge(base_state, {"iterations": 1, "run_id": "parse-run"}))
    assert parsed["verification_status"] == "parse_failed"
    assert parsed["grounding_report"] == []
    slot = parsed["semantic_trace"]["iterations"][0]
    assert slot["verifier_raw"]["raw_response"] == "not-json"
    assert slot["verifier_raw"]["parser_status"] == "parse_failed"
    assert slot["verifier_input"]["consumed"] is True

    monkeypatch.setattr(
        nodes, "_get_client",
        lambda: FakeLLMClient(response=fake_response("[]")),
    )
    empty = nodes.verify_node(_merge(base_state, {"iterations": 1, "run_id": "empty-run"}))
    assert empty["verification_status"] == "completed"
    assert empty["grounding_report"] == []
    empty_slot = empty["semantic_trace"]["iterations"][0]
    assert empty_slot["verifier_raw"]["parser_status"] == "ok"
    assert empty_slot["verifier_raw"]["pre_dedup_rows"] == []
    assert nodes.semantic_verification_accepted(_merge(base_state, empty)) is False

    sentinel = FakeLLMClient()
    monkeypatch.setattr(nodes, "_get_client", lambda: sentinel)
    skipped = nodes.verify_node(_merge(base_state, {
        "iterations": 1,
        "total_cost_usd": COST_GATE_USD,
        "run_id": "gate-run",
    }))
    assert sentinel.calls == 0
    assert skipped["verification_status"] == "skipped_cost_gate"
    skip_slot = skipped["semantic_trace"]["iterations"][0]
    assert skip_slot["verifier_input"]["consumed"] is False
    assert skip_slot["verifier_raw"]["parser_status"] == "skipped_cost_gate"
    assert skip_slot["verifier_raw"]["raw_response"] is None


def test_accepted_and_uvr_failed_routing_unchanged(base_state, monkeypatch):
    ok_report = [_verdict(f"ok-{i}") for i in range(10)]
    accepted = _merge(base_state, {
        "iterations": 1,
        "verification_status": "completed",
        "grounding_report": ok_report,
        "grounding_score": 0.80,
        "reflection_score": 8,
        "run_id": "ok-run",
    })
    assert nodes.semantic_verification_accepted(accepted) is True
    assert nodes.route_after_reflect(accepted) == "hitl"
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    hitl_ok = nodes.hitl_node(accepted)
    assert hitl_ok["hitl_status"] == "approved"
    assert nodes.route_after_hitl(_merge(accepted, hitl_ok)) == "html_gen"

    bad_report = [_verdict("u1", status="unverified", source_url=None, confidence=0.99)] + [
        _verdict(f"v-{i}") for i in range(4)
    ]
    failed = _merge(base_state, {
        "iterations": 1,
        "verification_status": "completed",
        "grounding_report": bad_report,
        "grounding_score": 0.99,
        "reflection_score": 10,
        "run_id": "uvr-run",
    })
    assert nodes.unverified_rate(bad_report) > UVR_THRESHOLD
    assert nodes.semantic_verification_accepted(failed) is False
    assert nodes.route_after_reflect(failed) == "draft"
    hitl_bad = nodes.hitl_node(_merge(failed, {"iterations": 2}))
    assert hitl_bad["hitl_status"] == "rejected"
    assert nodes.route_after_hitl(_merge(failed, hitl_bad)) == END


def test_api_cli_hitl_fail_closed_unchanged(base_state, monkeypatch):
    failed = _merge(base_state, {
        "iterations": 2,
        "verification_status": "parse_failed",
        "grounding_report": [],
        "grounding_score": 0.99,
        "reflection_score": 10,
        "run_id": "hitl-run",
    })
    monkeypatch.setenv("HITL_AUTO_APPROVE", "1")
    auto = nodes.hitl_node(failed)
    assert auto["hitl_status"] == "rejected"
    assert nodes.route_after_hitl(_merge(failed, auto)) == END

    monkeypatch.setenv("HITL_AUTO_APPROVE", "0")
    monkeypatch.setenv("HITL_MODE", "api")
    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: {"action": "approve"})
    api = nodes.hitl_node(failed)
    assert api["hitl_status"] == "rejected"
    assert nodes.route_after_hitl(_merge(failed, api)) == END

    monkeypatch.setattr("langgraph.types.interrupt", lambda payload: {
        "action": "feedback", "feedback": "ground the unverified claims",
    })
    fb = nodes.hitl_node(failed)
    assert fb["hitl_status"] == "feedback"
    assert fb["hitl_feedback"] == "ground the unverified claims"
    assert nodes.route_after_hitl(_merge(failed, fb)) == "draft"
    assert fb["semantic_trace"]["hitl_events"][-1]["hitl_feedback"] == "ground the unverified claims"


def test_telemetry_reread_validation_and_failed_envelope(
    base_state, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    ok_report = [_verdict(f"ok-{i}") for i in range(10)]
    state = _merge(base_state, {
        "run_id": "tele-run",
        "verification_status": "completed",
        "grounding_report": ok_report,
        "grounding_score": 0.9,
        "hitl_status": "approved",
        "semantic_trace": empty_trace({"run_id": "tele-run", "topic": base_state["topic"]}),
    })
    record_draft(
        state["semantic_trace"],
        iteration=1,
        draft_markdown="draft-a",
        revision_linkage=None,
    )
    record_verify(
        state["semantic_trace"],
        iteration=1,
        consumed=True,
        draft_markdown="draft-a",
        source_context="sources",
        user_message="Draft to verify:\ndraft-a\nAvailable sources:\nsources",
        verify_system_text="sys",
        raw_response="[]",
        parser_status="ok",
        pre_dedup_rows=[],
        dropped_rows=[],
        post_dedup_rows=[],
        post_attribution_rows=ok_report,
        web_sources=state.get("web_sources"),
        kb_results=state.get("kb_results"),
    )
    path = _write_telemetry(state)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["verification_status"] == "completed"
    assert loaded["semantic_trace_status"] == "ok"
    assert validate_trace(loaded["semantic_trace_v1"]) == []
    assert loaded["semantic_trace_v1"]["iterations"][0]["draft"]["draft_markdown"] == "draft-a"

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["semantic_trace_v1"]["iterations"][0]["draft"]["draft_sha256"] = "nope"
    path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    reread_validate_trace(path, tampered, state)
    reread = json.loads(path.read_text(encoding="utf-8"))
    assert reread["verification_status"] == "completed"
    assert reread["semantic_trace_status"] == "failed"
    assert reread["semantic_trace_v1"]["trace_status"] == "failed"


def test_crash_path_trace_is_detectable_and_does_not_claim_success(
    base_state, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    crash = _merge(base_state, {
        "run_id": "crash-run",
        "verification_status": "upstream_failed",
        "error_log": ["pipeline crash: boom"],
        "semantic_trace": empty_trace({"run_id": "crash-run", "topic": base_state["topic"]}),
    })
    path = _write_telemetry(crash)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["verification_status"] == "upstream_failed"
    assert loaded["semantic_trace_status"] in {"incomplete", "absent", "failed"}
    assert loaded["semantic_trace_v1"]["trace_status"] != "ok"


def test_trace_does_not_emit_test_credentials(base_state, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    monkeypatch.setenv("API_BEARER_TOKEN", BEARER)
    monkeypatch.setattr(
        nodes, "_get_client",
        lambda: FakeLLMClient(response=fake_response(_draft_json("sec"))),
    )
    drafted = nodes.draft_node(_merge(base_state, {"run_id": "secret-run"}))
    blob = canonical_json(drafted["semantic_trace"])
    assert SECRET not in blob
    assert BEARER not in blob
    monkeypatch.setattr(
        nodes, "_get_client",
        lambda: FakeLLMClient(response=fake_response(json.dumps([_verdict("x")]))),
    )
    verified = nodes.verify_node(_merge(base_state, drafted))
    blob2 = canonical_json(verified["semantic_trace"])
    assert SECRET not in blob2
    assert BEARER not in blob2


def test_deterministic_repeated_construction_matches():
    def build():
        trace = empty_trace({"run_id": "det", "topic": "Gradient Descent"})
        record_draft(trace, iteration=1, draft_markdown="same-draft", revision_linkage=None)
        record_verify(
            trace,
            iteration=1,
            consumed=True,
            draft_markdown="same-draft",
            source_context="ctx",
            user_message="msg",
            verify_system_text="sys",
            raw_response="[]",
            parser_status="ok",
            pre_dedup_rows=[],
            post_attribution_rows=[],
        )
        return trace
    a, b = build(), build()
    assert canonical_json(a) == canonical_json(b)
    assert a["iterations"][0]["draft"]["draft_sha256"] == sha256_utf8("same-draft")


def test_hitl_feedback_is_linked_on_next_draft(base_state, monkeypatch):
    monkeypatch.setattr(
        nodes, "_get_client",
        lambda: FakeLLMClient(response=fake_response(_draft_json("rev"))),
    )
    state = _merge(base_state, {
        "iterations": 1,
        "hitl_feedback": "please cut the unsourced claim",
        "grounding_report": [_verdict("unsourced number 42", status="unverified", source_url=None)],
        "run_id": "hitl-fb-run",
    })
    drafted = nodes.draft_node(state)
    link = drafted["semantic_trace"]["iterations"][-1]["revision_linkage"]
    assert link["hitl_feedback"] == "please cut the unsourced claim"
    assert "unsourced number 42" in link["targeted_unverified_claims"]
    assert link["source_iteration"] == 1
    assert link["next_iteration"] == 2


def _mut_remove_verifier_input(trace: dict) -> None:
    del trace["iterations"][0]["verifier_input"]


def _mut_remove_verifier_raw(trace: dict) -> None:
    del trace["iterations"][0]["verifier_raw"]


def _mut_remove_post_processing(trace: dict) -> None:
    del trace["iterations"][0]["post_processing"]


def _mut_corrupt_verifier_input_draft_sha(trace: dict) -> None:
    trace["iterations"][0]["verifier_input"]["draft_sha256"] = "deadbeef" * 8


def _mut_verifier_input_draft_differs(trace: dict) -> None:
    vin = trace["iterations"][0]["verifier_input"]
    vin["draft_markdown"] = "other-draft"
    vin["draft_sha256"] = sha256_utf8("other-draft")


def _mut_corrupt_source_context_sha(trace: dict) -> None:
    trace["iterations"][0]["verifier_input"]["source_context_sha256"] = "deadbeef" * 8


def _mut_corrupt_user_message_sha(trace: dict) -> None:
    trace["iterations"][0]["verifier_input"]["user_message_sha256"] = "deadbeef" * 8


def _mut_corrupt_source_set_digest(trace: dict) -> None:
    trace["iterations"][0]["verifier_input"]["source_set"]["digest"] = "deadbeef" * 8


def _mut_corrupt_raw_sha(trace: dict) -> None:
    trace["iterations"][0]["verifier_raw"]["raw_sha256"] = "deadbeef" * 8


def _mut_corrupt_post_processing_counts(trace: dict) -> None:
    trace["iterations"][0]["post_processing"]["counts"]["post_attribution"] = 99


def _mut_corrupt_final_uvr_numerator(trace: dict) -> None:
    trace["final"]["uvr_numerator"] = 99


def _mut_deciding_iteration_missing(trace: dict) -> None:
    trace["final"]["deciding_iteration"] = 99


_CAUSAL_MUTATIONS = [
    (_mut_remove_verifier_input, "missing verifier_input"),
    (_mut_remove_verifier_raw, "missing verifier_raw"),
    (_mut_remove_post_processing, "missing post_processing"),
    (_mut_corrupt_verifier_input_draft_sha, "verifier_input.draft_sha256 mismatch"),
    (_mut_verifier_input_draft_differs, "verifier_input draft differs from iteration draft"),
    (_mut_corrupt_source_context_sha, "source_context_sha256 mismatch"),
    (_mut_corrupt_user_message_sha, "user_message_sha256 mismatch"),
    (_mut_corrupt_source_set_digest, "source_set.digest mismatch"),
    (_mut_corrupt_raw_sha, "raw_sha256 mismatch"),
    (_mut_corrupt_post_processing_counts, "post_processing.counts.post_attribution mismatch"),
    (_mut_corrupt_final_uvr_numerator, "uvr_numerator"),
    (_mut_deciding_iteration_missing, "deciding_iteration does not resolve"),
]


@pytest.mark.parametrize(
    "mutator,needle",
    _CAUSAL_MUTATIONS,
    ids=[
        "remove_verifier_input",
        "remove_verifier_raw",
        "remove_post_processing",
        "corrupt_verifier_input_draft_sha",
        "verifier_input_draft_differs",
        "corrupt_source_context_sha",
        "corrupt_user_message_sha",
        "corrupt_source_set_digest",
        "corrupt_raw_sha",
        "corrupt_post_processing_counts",
        "corrupt_final_uvr_numerator",
        "deciding_iteration_nonexistent",
    ],
)
def test_causal_mutations_fail_complete_validation(mutator, needle):
    good = _complete_trace()
    assert validate_trace(good, require_complete=True) == []
    broken = deepcopy(good)
    mutator(broken)
    errors = validate_trace(broken, require_complete=True)
    assert errors, f"expected complete validation to fail for {needle}"
    assert any(needle in e for e in errors), errors


def test_skipped_cost_gate_complete_trace_validates_without_fake_provider_io():
    trace = empty_trace({"run_id": "skip-run", "topic": "Gradient Descent"})
    record_draft(trace, iteration=1, draft_markdown="draft-a", revision_linkage=None)
    record_verify(
        trace,
        iteration=1,
        consumed=False,
        skip_reason="skipped_cost_gate",
        draft_markdown="draft-a",
    )
    stamp_final(
        trace,
        {
            "verification_status": "skipped_cost_gate",
            "grounding_report": [],
            "hitl_status": None,
        },
        semantic_accepted=False,
    )
    trace["trace_status"] = "ok"
    assert validate_trace(trace, require_complete=True) == []
    vin = trace["iterations"][0]["verifier_input"]
    raw = trace["iterations"][0]["verifier_raw"]
    assert vin["consumed"] is False
    assert vin["reason"] == "skipped_cost_gate"
    assert "source_context" not in vin
    assert "source_set" not in vin
    assert raw["raw_response"] is None
    assert raw["raw_sha256"] is None


def test_parse_failed_complete_trace_validates():
    trace = empty_trace({"run_id": "parse-complete", "topic": "Gradient Descent"})
    record_draft(trace, iteration=1, draft_markdown="draft-a", revision_linkage=None)
    record_verify(
        trace,
        iteration=1,
        consumed=True,
        draft_markdown="draft-a",
        source_context="sources",
        user_message="Draft to verify:\ndraft-a\nAvailable sources:\nsources",
        verify_system_text="sys",
        raw_response="not-json",
        parser_status="parse_failed",
        parse_error="json decode failed",
        pre_dedup_rows=[],
        dropped_rows=[],
        post_dedup_rows=[],
        post_attribution_rows=[],
        web_sources=[{"url": "https://example.com/gd", "content": "x"}],
        kb_results=[],
    )
    stamp_final(
        trace,
        {
            "verification_status": "parse_failed",
            "grounding_report": [],
            "hitl_status": None,
        },
        semantic_accepted=False,
    )
    trace["trace_status"] = "ok"
    assert validate_trace(trace, require_complete=True) == []
    slot = trace["iterations"][0]
    assert slot["verifier_input"]["consumed"] is True
    assert slot["verifier_raw"]["parser_status"] == "parse_failed"
    assert slot["verifier_raw"]["raw_response"] == "not-json"
    assert slot["verifier_raw"]["raw_sha256"] == sha256_utf8("not-json")


def test_accepted_normal_trace_validates_complete():
    trace = _complete_trace()
    assert validate_trace(trace, require_complete=True) == []
    slot = trace["iterations"][0]
    assert slot["draft"]["draft_markdown"] == "draft-a"
    assert isinstance(slot["verifier_input"], dict)
    assert isinstance(slot["verifier_raw"], dict)
    assert isinstance(slot["post_processing"], dict)
    assert trace["final"]["uvr"] == 0.0
    assert trace["final"]["uvr_numerator"] == 0
    assert trace["final"]["uvr_denominator"] == 10


def test_two_iteration_revision_trace_validates_complete():
    first = [_verdict("unsourced number 42", status="unverified", source_url=None, confidence=0.2)]
    second = [_verdict(f"ok-{i}") for i in range(10)]
    trace = empty_trace({"run_id": "rev-run", "topic": "Gradient Descent"})
    record_draft(trace, iteration=1, draft_markdown="draft-a", revision_linkage=None)
    record_verify(
        trace,
        iteration=1,
        consumed=True,
        draft_markdown="draft-a",
        source_context="sources",
        user_message="Draft to verify:\ndraft-a\nAvailable sources:\nsources",
        verify_system_text="sys",
        raw_response='[{"claim":"unsourced number 42"}]',
        parser_status="ok",
        pre_dedup_rows=first,
        dropped_rows=[],
        post_dedup_rows=first,
        post_attribution_rows=first,
        web_sources=[{"url": "https://example.com/gd", "content": "x"}],
        kb_results=[],
    )
    record_draft(
        trace,
        iteration=2,
        draft_markdown="draft-b",
        revision_linkage={
            "source_iteration": 1,
            "next_iteration": 2,
            "targeted_unverified_claims": ["unsourced number 42"],
            "grounding_feedback_block": "unsourced number 42",
            "hitl_feedback": None,
        },
    )
    record_verify(
        trace,
        iteration=2,
        consumed=True,
        draft_markdown="draft-b",
        source_context="sources",
        user_message="Draft to verify:\ndraft-b\nAvailable sources:\nsources",
        verify_system_text="sys",
        raw_response="[]",
        parser_status="ok",
        pre_dedup_rows=second,
        dropped_rows=[],
        post_dedup_rows=second,
        post_attribution_rows=second,
        web_sources=[{"url": "https://example.com/gd", "content": "x"}],
        kb_results=[],
    )
    stamp_final(
        trace,
        {
            "verification_status": "completed",
            "grounding_report": second,
            "hitl_status": "approved",
        },
        semantic_accepted=True,
    )
    trace["trace_status"] = "ok"
    assert validate_trace(trace, require_complete=True) == []
    assert [item["iteration"] for item in trace["iterations"]] == [1, 2]
    assert trace["iterations"][0]["draft"]["draft_markdown"] == "draft-a"
    assert trace["iterations"][1]["draft"]["draft_markdown"] == "draft-b"
    assert trace["final"]["deciding_iteration"] == 2
    assert trace["final"]["uvr"] == 0.0


def test_trace_failure_does_not_alter_article_semantic_acceptance_or_routing(base_state):
    ok_report = [_verdict(f"ok-{i}") for i in range(10)]
    state = _merge(base_state, {
        "iterations": 1,
        "verification_status": "completed",
        "grounding_report": ok_report,
        "grounding_score": 0.80,
        "reflection_score": 8,
        "hitl_status": "approved",
        "run_id": "routing-run",
        "semantic_trace": _complete_trace(ok_report),
    })
    del state["semantic_trace"]["iterations"][0]["verifier_input"]
    assert nodes.semantic_verification_accepted(state) is True
    assert nodes.route_after_reflect(state) == "hitl"
    assert nodes.route_after_hitl(state) == "html_gen"
    record = {"verification_status": state["verification_status"]}
    embed_semantic_trace(record, state, semantic_accepted=True)
    assert record["semantic_trace_status"] == "failed"
    assert record["verification_status"] == "completed"
    assert state["verification_status"] == "completed"
    assert nodes.semantic_verification_accepted(state) is True
    assert nodes.route_after_reflect(state) == "hitl"
    assert nodes.route_after_hitl(state) == "html_gen"


def test_malformed_trace_cannot_emerge_ok_through_telemetry_write_reread(
    base_state, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    ok_report = [_verdict(f"ok-{i}") for i in range(10)]
    trace = _complete_trace(ok_report)
    del trace["iterations"][0]["verifier_input"]
    state = _merge(base_state, {
        "run_id": "malformed-tele-run",
        "verification_status": "completed",
        "grounding_report": ok_report,
        "grounding_score": 0.9,
        "hitl_status": "approved",
        "semantic_trace": trace,
    })
    assert nodes.semantic_verification_accepted(state) is True
    path = _write_telemetry(state)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["verification_status"] == "completed"
    assert loaded["semantic_trace_status"] != "ok"
    assert loaded["semantic_trace_v1"]["trace_status"] != "ok"
    assert loaded["semantic_trace_status"] == "failed"

    # A file that already claims ok must also fail closed on reread.
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["semantic_trace_v1"] = _complete_trace(ok_report)
    del tampered["semantic_trace_v1"]["iterations"][0]["verifier_input"]
    tampered["semantic_trace_v1"]["trace_status"] = "ok"
    tampered["semantic_trace_status"] = "ok"
    path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
    reread_validate_trace(path, tampered, state)
    reread = json.loads(path.read_text(encoding="utf-8"))
    assert reread["verification_status"] == "completed"
    assert reread["semantic_trace_status"] != "ok"
    assert reread["semantic_trace_v1"]["trace_status"] != "ok"
