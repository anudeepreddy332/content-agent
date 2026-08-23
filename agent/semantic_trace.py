"""
semantic_trace_v1 — lean reconstructable evidence for the current
draft → verify → revision chain.

Persistence only. Does not change retrieval, prompts, models, thresholds,
routing, or HITL approval semantics. Does not claim retrieved source text
was verifier-visible unless that exact text was in the verifier user message.

Code SHA is not stamped: the runtime has no verified git identity today,
and production code must not shell out to git to invent one.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from config import DEEPSEEK_MODEL, PROMPT_HASHES, PROMPT_VERSION

SCHEMA = "semantic_trace_v1"
SCHEMA_VERSION = 1

CODE_IDENTITY = {
    "available": False,
    "reason": (
        "Runtime does not currently stamp a verified git SHA; "
        "production code does not shell out to git."
    ),
}


def sha256_utf8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_obj(obj: Any) -> str:
    return sha256_utf8(canonical_json(obj))


def source_set_identity(web_sources: list | None, kb_results: list | None) -> dict:
    """Identity of sources fed into the verifier context builder (first 5 each).

    This is not a claim that full retrieved payloads were model-visible.
    """
    web = list(web_sources or [])[:5]
    kb = list(kb_results or [])[:5]
    payload = {
        "web_urls": [s.get("url") for s in web],
        "kb_sources": [
            {"source": k.get("source"), "chunk_index": k.get("chunk_index", 0)}
            for k in kb
        ],
    }
    return {"payload": payload, "digest": digest_obj(payload)}


def empty_trace(state: dict | None = None) -> dict:
    state = state or {}
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_id": state.get("run_id"),
        "topic": state.get("topic"),
        "code_identity": dict(CODE_IDENTITY),
        "prompt_version": state.get("prompt_version") or PROMPT_VERSION,
        "prompt_hashes": dict(PROMPT_HASHES),
        "model": DEEPSEEK_MODEL,
        "iterations": [],
        "hitl_events": [],
        "final": None,
        "trace_status": "in_progress",
        "trace_error": None,
    }


def copy_trace(state: dict) -> dict:
    existing = state.get("semantic_trace")
    if not isinstance(existing, dict) or existing.get("schema") != SCHEMA:
        return empty_trace(state)
    return json.loads(canonical_json(existing))


def _slot(trace: dict, iteration: int) -> dict:
    for item in trace["iterations"]:
        if item.get("iteration") == iteration:
            return item
    slot = {
        "iteration": iteration,
        "draft": None,
        "revision_linkage": None,
        "verifier_input": None,
        "verifier_raw": None,
        "post_processing": None,
    }
    trace["iterations"].append(slot)
    return slot


def record_draft(
    trace: dict,
    *,
    iteration: int,
    draft_markdown: str,
    revision_linkage: dict | None,
) -> None:
    slot = _slot(trace, iteration)
    slot["draft"] = {
        "draft_markdown": draft_markdown,
        "draft_sha256": sha256_utf8(draft_markdown),
    }
    if revision_linkage is not None:
        slot["revision_linkage"] = revision_linkage
    trace["trace_status"] = "in_progress"


def record_verify(
    trace: dict,
    *,
    iteration: int,
    consumed: bool,
    skip_reason: str | None = None,
    draft_markdown: str = "",
    source_context: str | None = None,
    user_message: str | None = None,
    verify_system_text: str | None = None,
    web_sources: list | None = None,
    kb_results: list | None = None,
    raw_response: str | None = None,
    parser_status: str | None = None,
    parse_error: str | None = None,
    pre_dedup_rows: list | None = None,
    dropped_rows: list | None = None,
    post_dedup_rows: list | None = None,
    post_attribution_rows: list | None = None,
) -> None:
    slot = _slot(trace, iteration)
    if consumed:
        identity = source_set_identity(web_sources, kb_results)
        slot["verifier_input"] = {
            "consumed": True,
            "draft_markdown": draft_markdown,
            "draft_sha256": sha256_utf8(draft_markdown),
            "source_context": source_context,
            "source_context_sha256": sha256_utf8(source_context or ""),
            "user_message": user_message,
            "user_message_sha256": sha256_utf8(user_message or ""),
            "verify_system_hash_12": PROMPT_HASHES.get("verify_system"),
            "verify_system_sha256": sha256_utf8(verify_system_text or ""),
            "source_set": identity,
        }
        slot["verifier_raw"] = {
            "raw_response": raw_response,
            "raw_sha256": sha256_utf8(raw_response or ""),
            "parser_status": parser_status,
            "parse_error": parse_error,
            "pre_dedup_rows": pre_dedup_rows or [],
        }
        post_attr = post_attribution_rows or []
        pre = pre_dedup_rows or []
        post_dedup = post_dedup_rows if post_dedup_rows is not None else post_attr
        dropped = dropped_rows or []
        slot["post_processing"] = {
            "pre_dedup_count": len(pre),
            "dropped_rows": dropped,
            "post_dedup_rows": post_dedup,
            "post_attribution_rows": post_attr,
            "counts": {
                "pre_dedup": len(pre),
                "dropped": len(dropped),
                "post_dedup": len(post_dedup),
                "post_attribution": len(post_attr),
                "verified": sum(1 for r in post_attr if r.get("status") == "verified"),
                "weak": sum(1 for r in post_attr if r.get("status") == "weak"),
                "unverified": sum(1 for r in post_attr if r.get("status") == "unverified"),
            },
        }
    else:
        slot["verifier_input"] = {
            "consumed": False,
            "reason": skip_reason,
            "draft_markdown": draft_markdown,
            "draft_sha256": sha256_utf8(draft_markdown or ""),
        }
        slot["verifier_raw"] = {
            "raw_response": None,
            "raw_sha256": None,
            "parser_status": skip_reason or "skipped",
            "parse_error": None,
            "pre_dedup_rows": [],
        }
        slot["post_processing"] = {
            "pre_dedup_count": 0,
            "dropped_rows": [],
            "post_dedup_rows": [],
            "post_attribution_rows": [],
            "counts": {
                "pre_dedup": 0,
                "dropped": 0,
                "post_dedup": 0,
                "post_attribution": 0,
                "verified": 0,
                "weak": 0,
                "unverified": 0,
            },
        }
    trace["trace_status"] = "in_progress"


def record_hitl_event(trace: dict, status: str | None, feedback: str | None) -> None:
    trace.setdefault("hitl_events", []).append({
        "hitl_status": status,
        "hitl_feedback": feedback,
    })


def uvr_v1(report: list | None) -> tuple[float | None, int, int]:
    """Current UVR_v1: unverified / N, or None when not computable."""
    if not report:
        return None, 0, 0
    n_unverified = sum(1 for row in report if row.get("status") == "unverified")
    return n_unverified / len(report), n_unverified, len(report)


def stamp_final(
    trace: dict,
    state: dict,
    *,
    semantic_accepted: bool,
) -> None:
    report = list(state.get("grounding_report") or [])
    uvr, numerator, denominator = uvr_v1(report)
    hitl_status = state.get("hitl_status")
    iterations = trace.get("iterations") or []
    deciding = iterations[-1]["iteration"] if iterations else None
    html_eligible = bool(semantic_accepted and hitl_status == "approved")
    trace["final"] = {
        "verification_status": state.get("verification_status"),
        "grounding_report": report,
        "uvr_numerator": numerator,
        "uvr_denominator": denominator,
        "uvr": uvr,
        "semantic_accepted": semantic_accepted,
        "hitl_status": hitl_status,
        "html_review_status": state.get("html_review_status"),
        "git_status": state.get("git_status"),
        "html_gen_eligible": html_eligible,
        "claim_completeness": "unknown",
        "deciding_iteration": deciding,
    }


def validate_trace(trace: Any, *, require_complete: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(trace, dict):
        return ["trace is not an object"]
    if trace.get("schema") != SCHEMA:
        errors.append("missing or unknown schema")
    if trace.get("schema_version") != SCHEMA_VERSION:
        errors.append("missing or unknown schema_version")
    iterations = trace.get("iterations")
    if not isinstance(iterations, list):
        errors.append("iterations must be a list")
        return errors
    seen: set[int] = set()
    for item in iterations:
        if not isinstance(item, dict):
            errors.append("iteration slot is not an object")
            continue
        iteration = item.get("iteration")
        if not isinstance(iteration, int):
            errors.append("iteration id must be an int")
            continue
        if iteration in seen:
            errors.append(f"duplicate iteration id {iteration}")
        seen.add(iteration)
        draft = item.get("draft")
        if isinstance(draft, dict):
            md = draft.get("draft_markdown")
            digest = draft.get("draft_sha256")
            if not isinstance(md, str):
                errors.append(f"iteration {iteration} draft_markdown missing")
            elif digest != sha256_utf8(md):
                errors.append(f"iteration {iteration} draft_sha256 mismatch")
        linkage = item.get("revision_linkage")
        if isinstance(linkage, dict):
            src = linkage.get("source_iteration")
            if src is not None and not any(
                other.get("iteration") == src for other in iterations
            ):
                errors.append(
                    f"iteration {iteration} revision_linkage source_iteration "
                    f"{src} is missing"
                )
        raw = item.get("verifier_raw")
        if isinstance(raw, dict) and raw.get("raw_response") is not None:
            if raw.get("raw_sha256") != sha256_utf8(raw.get("raw_response") or ""):
                errors.append(f"iteration {iteration} raw_sha256 mismatch")
        vin = item.get("verifier_input")
        if isinstance(vin, dict) and vin.get("consumed") is True:
            if vin.get("user_message") is None:
                errors.append(f"iteration {iteration} missing consumed user_message")
            if vin.get("source_context") is None:
                errors.append(f"iteration {iteration} missing consumed source_context")
            um = vin.get("user_message")
            if isinstance(um, str) and vin.get("user_message_sha256") != sha256_utf8(um):
                errors.append(f"iteration {iteration} user_message_sha256 mismatch")
        if require_complete:
            if not isinstance(draft, dict):
                errors.append(f"iteration {iteration} missing draft")
            if not isinstance(item.get("verifier_raw"), dict):
                errors.append(f"iteration {iteration} missing verifier_raw")
    if require_complete:
        if trace.get("trace_status") == "failed":
            errors.append("trace_status is failed")
        if not isinstance(trace.get("final"), dict):
            errors.append("missing final decision")
        else:
            deciding = trace["final"].get("deciding_iteration")
            if deciding is not None and deciding not in seen:
                errors.append("final.deciding_iteration does not resolve")
            for item in iterations:
                link = item.get("revision_linkage")
                if not isinstance(link, dict):
                    continue
                nxt = link.get("next_iteration")
                if nxt is not None and nxt not in seen:
                    errors.append(
                        f"revision_linkage next_iteration {nxt} does not resolve"
                    )
    return errors


def failed_envelope(state: dict, errors: list[str], prior: dict | None = None) -> dict:
    trace = empty_trace(state)
    if isinstance(prior, dict) and prior.get("schema") == SCHEMA:
        trace["iterations"] = list(prior.get("iterations") or [])
        trace["hitl_events"] = list(prior.get("hitl_events") or [])
        trace["final"] = prior.get("final")
    trace["trace_status"] = "failed"
    trace["trace_error"] = list(errors)
    return trace


def embed_semantic_trace(record: dict, state: dict, *, semantic_accepted: bool) -> dict:
    """Attach semantic_trace_v1 to a telemetry record. Never mutates routing fields."""
    prior = state.get("semantic_trace")
    if not isinstance(prior, dict) or prior.get("schema") != SCHEMA:
        trace = empty_trace(state)
        trace["trace_status"] = "absent" if not prior else "failed"
        if prior and prior.get("schema") != SCHEMA:
            trace["trace_error"] = [
                "semantic_trace present but schema is not semantic_trace_v1"
            ]
        stamp_final(trace, state, semantic_accepted=semantic_accepted)
        record["semantic_trace_v1"] = trace
        record["semantic_trace_status"] = trace["trace_status"]
        return trace

    trace = json.loads(canonical_json(prior))
    stamp_final(trace, state, semantic_accepted=semantic_accepted)
    require_complete = bool(trace.get("iterations"))
    errors = validate_trace(trace, require_complete=require_complete)
    if errors:
        trace = failed_envelope(state, errors, prior=trace)
        stamp_final(trace, state, semantic_accepted=semantic_accepted)
    elif trace.get("trace_status") not in {"failed", "absent"}:
        trace["trace_status"] = "ok" if require_complete else "incomplete"
    record["semantic_trace_v1"] = trace
    record["semantic_trace_status"] = trace["trace_status"]
    return trace


def reread_validate_trace(path, record: dict, state: dict) -> None:
    """Reload the written artifact. A broken trace must not look successful."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record["semantic_trace_v1"] = failed_envelope(state, [f"reread failed: {exc}"])
        record["semantic_trace_status"] = "failed"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return
    trace = loaded.get("semantic_trace_v1")
    errors = validate_trace(trace, require_complete=False)
    status = trace.get("trace_status") if isinstance(trace, dict) else None
    if errors or status not in {"ok", "incomplete", "absent", "failed", "in_progress"}:
        failed = failed_envelope(
            state,
            errors or [f"invalid trace_status {status!r}"],
            prior=trace if isinstance(trace, dict) else None,
        )
        loaded["semantic_trace_v1"] = failed
        loaded["semantic_trace_status"] = "failed"
        path.write_text(json.dumps(loaded, indent=2), encoding="utf-8")
        return
    if loaded.get("semantic_trace_status") != status:
        loaded["semantic_trace_status"] = status
        path.write_text(json.dumps(loaded, indent=2), encoding="utf-8")
