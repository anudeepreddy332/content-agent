"""Frozen quality episodes and terminal outcomes (D-2026-10-05-04).

Provider/policy semantics stay in their existing nodes. This module owns only
attempt accounting, current-artifact bindings, gate eligibility and destinations.
"""
from contextvars import ContextVar
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from config import COST_GATE_USD, REFLECTION_THRESHOLD
from agent.reviewer_identity import approval_origin_valid

QUALITY_ATTEMPTS = 2
CONTENT_FEEDBACK_LIMIT = 2
LAYOUT_REVISION_LIMIT = 2
QUALITY_MESSAGE = "Quality gates were not met within the allowed attempts. Please try again."

_DRAFT_FAILURE_REASONS = {
    "client_prompt_initialization": "draft_setup_failed",
    "source_context_construction": "source_context_failed",
    "provider_request": "provider_request_failed",
    "provider_response_handling": "provider_response_invalid",
    "json_parsing_validation": "draft_parse_failed",
    "post_response_accounting_telemetry": "post_response_processing_failed",
}
_DRAFT_FAILURE_PHASE = ContextVar("draft_failure_phase", default="client_prompt_initialization")
_DRAFT_RESPONSE_DIAGNOSTIC = ContextVar("draft_response_diagnostic", default=None)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def mark_draft_failure_phase(phase: str) -> None:
    """Record only a fixed, non-sensitive phase for the draft execution boundary."""
    if phase not in _DRAFT_FAILURE_REASONS:
        raise ValueError("unknown draft failure phase")
    _DRAFT_FAILURE_PHASE.set(phase)


def mark_draft_response_diagnostic(diagnostic: dict) -> None:
    """Retain safe response metadata for a later post-response exception."""
    _DRAFT_RESPONSE_DIAGNOSTIC.set({
        "response_content": diagnostic["content"]["original"].copy(),
        "finish_reason": diagnostic["response"]["finish_reason"],
        "usage": diagnostic["usage"].copy(),
    })


def _draft_failure_entry(state: dict, exc: Exception) -> str:
    phase = _DRAFT_FAILURE_PHASE.get()
    exception_class = type(exc).__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", exception_class):
        exception_class = "Exception"
    source = {"source_file": "agent/workflow.py", "source_function": "draft_step",
              "source_line": 0}
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        try:
            relative = Path(frame.f_code.co_filename).resolve().relative_to(_PROJECT_ROOT)
        except (OSError, ValueError):
            relative = None
        if relative is not None and relative.parts[0] == "agent":
            function = frame.f_code.co_name
            source = {
                "source_file": relative.as_posix(),
                "source_function": function if re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]{0,63}", function,
                ) else "unknown",
                "source_line": traceback.tb_lineno,
            }
        traceback = traceback.tb_next
    run_id = state.get("run_id")
    diagnostic = {
        "run_id": run_id if isinstance(run_id, str) else None,
        "stage": "draft",
        "failure_phase": phase,
        "exception_class": exception_class,
        "reason": _DRAFT_FAILURE_REASONS[phase],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **source,
    }
    response_diagnostic = _DRAFT_RESPONSE_DIAGNOSTIC.get()
    if response_diagnostic is not None:
        diagnostic.update(response_diagnostic)
    return "[draft] " + json.dumps(diagnostic, sort_keys=True)


def identity(state):
    return {
        "run_id": state.get("run_id"),
        "draft_sha256": sha256((state.get("draft_markdown") or "").encode()).hexdigest(),
        "episode": state.get("quality_episode"),
        "attempt": state.get("quality_attempts"),
        "iteration": state.get("iterations"),
    }


def resource_stopped(state):
    return state.get("total_cost_usd", 0) >= COST_GATE_USD


def terminal(status, message=None):
    return {"terminal_status": status, "terminal_message": message or (
        QUALITY_MESSAGE if status == "terminal_quality_exhausted" else status.replace("_", " ")
    )}


def draft_valid(state):
    sections = state.get("draft_sections")
    return (
        state.get("draft_status") == "valid"
        and isinstance(sections, dict)
        and all(isinstance(sections.get(k), str) and sections[k].strip()
                for k in ("problem_framing", "technical_dive", "code_snippets", "takeaways"))
        and isinstance(state.get("draft_markdown"), str)
        and bool(state["draft_markdown"].strip())
    )


def inventory_current(state):
    inventory = state.get("claim_inventory")
    return (
        isinstance(inventory, dict)
        and inventory.get("run_id") == state.get("run_id")
        and inventory.get("iteration") == state.get("iterations")
        and inventory.get("draft_sha256") == identity(state)["draft_sha256"]
    )


def verification_eligible(state):
    from agent import nodes
    return (
        not state.get("terminal_status") and not resource_stopped(state)
        and draft_valid(state) and inventory_current(state)
        and state.get("verification_identity") == identity(state)
        and nodes.semantic_verification_accepted(state)
        and nodes.material_policy_accepted(state)
        and nodes.citation_policy_accepted(state)
    )


def reflection_accepted(state):
    provenance = state.get("reflection_provenance")
    if not isinstance(provenance, dict):
        return False
    score = state.get("reflection_score")
    return (
        state.get("reflection_identity") == identity(state)
        and provenance.get("origin") == "judge"
        and provenance.get("provider_called") is True
        and provenance.get("parse_status") == "ok"
        and type(score) is int and REFLECTION_THRESHOLD <= score <= 10
    )


def gate1_eligible(state):
    """Single authority, recomputed both before interrupt and on approval."""
    return verification_eligible(state) and reflection_accepted(state)


def start_episode(state):
    if state.get("terminal_status"):
        return {}
    if resource_stopped(state):
        return terminal("resource_exhausted")
    return {
        "quality_episode": state.get("quality_episode", 0) + 1,
        "quality_attempts": 0,
        "draft_status": "not_started",
        "verification_identity": None, "reflection_identity": None,
        "approved_draft_identity": None, "hitl_status": "pending",
        "gate1_reviewer_id": None, "gate2_reviewer_id": None,
        "gate1_review_origin": None, "gate2_review_origin": None,
        "gate1_evidence_signoff": None, "initial_html_equivalence": None,
    }


def failure(state, outcome, *, repairable=True):
    if state.get("terminal_status"):
        return {}
    if resource_stopped(state):
        return terminal("resource_exhausted")
    if repairable and state.get("quality_attempts", 0) < QUALITY_ATTEMPTS:
        return {}
    return terminal(outcome)


def verification_failure(state):
    from agent import nodes
    if state.get("verification_identity") != identity(state):
        return failure(state, "execution_failed", repairable=False)
    if state.get("verification_status") != "completed":
        # Output/schema failure may use the one remaining draft. Exhausted
        # transport/execution errors terminate; transport retries stay in SDK nodes.
        return failure(state, "execution_failed", repairable=state.get("verification_status") in (
            "parse_failed", "inventory_failed",
        ))
    if not inventory_current(state) or nodes.inventory_critical_failures(state.get("claim_inventory")):
        return failure(state, "execution_failed", repairable=False)
    material = nodes.material_policy_result(state)
    material_ok = nodes.material_policy_passed(material)
    # UNKNOWN materiality/requirements, invalid material analyses and integrity
    # failures never authorize a stochastic retry, even alongside semantic rejection.
    if not material_ok and material.material_safety_state not in ("repairable", "exhausted"):
        return failure(state, "policy_blocked", repairable=False)
    if not nodes.citation_policy_accepted(state):
        return failure(state, "policy_blocked", repairable=False)
    if not nodes.semantic_verification_accepted(state):
        return failure(state, "terminal_quality_exhausted")
    if not material_ok:
        return failure(state, "policy_blocked")
    return failure(state, "execution_failed", repairable=False)


def draft_step(state, *, run):
    if state.get("terminal_status"):
        return {}
    if resource_stopped(state):
        return terminal("resource_exhausted")
    if state.get("quality_attempts", 0) >= QUALITY_ATTEMPTS:
        return terminal("terminal_quality_exhausted")
    attempt = state.get("quality_attempts", 0) + 1
    ordinal = state.get("iterations", 0) + 1
    phase_token = _DRAFT_FAILURE_PHASE.set("client_prompt_initialization")
    response_token = _DRAFT_RESPONSE_DIAGNOSTIC.set(None)
    try:
        update = run(state)
    except Exception as exc:
        return {"quality_attempts": attempt, "iterations": ordinal,
                "draft_status": "invalid",
                "error_log": [*(state.get("error_log") or []), _draft_failure_entry(state, exc)],
                **terminal("execution_failed")}
    finally:
        _DRAFT_FAILURE_PHASE.reset(phase_token)
        _DRAFT_RESPONSE_DIAGNOSTIC.reset(response_token)
    update = {**update, "iterations": ordinal, "quality_attempts": attempt,
              "verification_identity": None, "reflection_identity": None,
              "approved_draft_identity": None, "verification_status": "not_started",
              "claim_inventory": None, "grounding_report": [], "reflection_score": 0,
              "reflection_provenance": {}, "hitl_status": "pending",
              "gate1_reviewer_id": None, "gate2_reviewer_id": None,
              "gate1_review_origin": None, "gate2_review_origin": None,
              "gate1_evidence_signoff": None, "initial_html_equivalence": None}
    current = {**state, **update}
    if not draft_valid(current):
        update.update(failure(current, "execution_failed"))
    elif resource_stopped(current):
        update.update(terminal("resource_exhausted"))
    return update


def verify_step(state, *, run):
    if not draft_valid(state) or state.get("terminal_status"):
        return terminal("execution_failed")
    try:
        update = run(state)
    except Exception:
        return {"verification_status": "verification_error", **terminal("execution_failed")}
    update = {**update, "verification_identity": identity(state)}
    current = {**state, **update}
    if not verification_eligible(current):
        update.update(verification_failure(current))
    return update


def reflect_step(state, *, run):
    if not verification_eligible(state):
        return terminal("execution_failed")
    try:
        update = run(state)
    except Exception as exc:
        return {"reflection_score": 0, "reflection_identity": identity(state),
                "reflection_provenance": {"origin": "unavailable", "parse_status": "execution_failed",
                                          "provider_called": None, "reason": type(exc).__name__},
                "error_log": [*state.get("error_log", []), f"reflect: {type(exc).__name__}: {exc}"],
                **terminal("execution_failed", "Reflection execution failed.")}
    update = {**update, "reflection_identity": identity(state)}
    current = {**state, **update}
    provenance = current.get("reflection_provenance")
    score = current.get("reflection_score")
    executed = (isinstance(provenance, dict) and provenance.get("origin") == "judge"
                and provenance.get("provider_called") is True and provenance.get("parse_status") == "ok"
                and type(score) is int and 1 <= score <= 10)
    if resource_stopped(current):
        update.update(terminal("resource_exhausted"))
    elif not executed:
        update.update(terminal("execution_failed", "Reflection response was unusable."))
    elif not gate1_eligible(current):
        update.update(failure(current, "terminal_quality_exhausted"))
    return update


def html_step(state, *, run):
    from agent.evidence_signoff import receipt_valid
    if (not gate1_eligible(state) or state.get("approved_draft_identity") != identity(state)
            or not approval_origin_valid(state, 1) or not receipt_valid(state)):
        return terminal("execution_failed")
    try:
        update = run(state)
    except Exception:
        return terminal("rendering_failed")
    if resource_stopped({**state, **update}):
        update.update(terminal("resource_exhausted"))
    elif not update.get("html_output"):
        update.update(terminal("rendering_failed"))
    return update


def layout_step(state, *, run):
    if resource_stopped(state):
        return terminal("resource_exhausted")
    try:
        update = run(state)
    except Exception:
        # Accepted feedback already consumed an attempt; preserve the trusted
        # previous render, which can still be explicitly approved.
        update = {"html_feedback": None, "approved_html_sha256": None,
                  "gate2_reviewer_id": None, "gate2_review_origin": None,
                  "error_log": [*state.get("error_log", []), "html_revise: revision failed; kept original"]}
    if resource_stopped({**state, **update}):
        update.update(terminal("resource_exhausted"))
    return update


def content_feedback(state, note):
    if state.get("content_feedback_count", 0) >= CONTENT_FEEDBACK_LIMIT:
        return terminal("content_revision_limit")
    return {**start_episode(state), "content_feedback_count": state.get("content_feedback_count", 0) + 1,
            "hitl_status": "feedback", "hitl_feedback": note}


def layout_feedback(state, note):
    if state.get("html_revision_attempts", 0) >= LAYOUT_REVISION_LIMIT:
        return terminal("layout_revision_limit")
    return {"html_review_status": "changes", "html_feedback": note,
            "approved_html_sha256": None, "gate2_reviewer_id": None,
            "gate2_review_origin": None,
            "html_revision_attempts": state.get("html_revision_attempts", 0) + 1}


def project_terminal(state):
    """One terminal projection for legacy/ordinary graph results and consumers."""
    if state.get("git_status") == "failed" and not state.get("terminal_status"):
        return {**state, **terminal("execution_failed", "Local Git/archive action failed.")}
    return state


def terminal_api_status(state):
    state = project_terminal(state)
    if state.get("terminal_status"):
        return state["terminal_status"]
    if state.get("hitl_status") == "rejected" or state.get("html_review_status") == "rejected":
        return "terminal_rejected"
    return "complete"
