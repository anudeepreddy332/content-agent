"""Deterministic Stage 2C evidence-exposure qualification harness. No providers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = REPO_ROOT / "evals" / "fixtures" / "evidence_exposure_2c.json"

EVALUATOR_ID = "evidence_exposure_2c"
PACK_ID = "evidence_exposure_2c"
SCHEMA_VERSION = 1
STAGE = "2c"
K = 1

# Frozen production prefix limits (mirrors agent/nodes.py; not imported to avoid coupling).
WEB_PREFIX_CHARS = 1500
KB_PREFIX_CHARS = 2000

ARM_A = "arm_a_current_prefix"
ARM_B = "arm_b_complete_source"
ARM_C = "arm_c_gold_relevant_context"
ARMS = (ARM_A, ARM_B, ARM_C)
CONSUMERS = ("draft", "verifier")
SOURCE_KINDS = frozenset({"web", "kb"})
SEMANTIC_LABELS = frozenset({"verified", "weak", "unverified"})

CASE_IDS = (
    "E2C-W01",
    "E2C-K01",
    "E2C-W02",
    "E2C-K02",
    "E2C-W03",
    "E2C-K03",
    "E2C-W04",
    "E2C-K04",
)


class EvidenceExposure2CError(ValueError):
    """Fixture pack or exposure asset is not evaluable."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceExposure2CError(message)


def prefix_limit(source_kind: str) -> int:
    if source_kind == "web":
        return WEB_PREFIX_CHARS
    if source_kind == "kb":
        return KB_PREFIX_CHARS
    raise EvidenceExposure2CError(f"unknown source kind {source_kind!r}")


def merge_windows(windows: list[list[int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    sorted_windows = sorted((int(s), int(e)) for s, e in windows)
    merged: list[tuple[int, int]] = [sorted_windows[0]]
    for start, end in sorted_windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def expose_arm_a(source_kind: str, source_text: str) -> tuple[str, int, str | None]:
    limit = prefix_limit(source_kind)
    exposed = source_text[:limit]
    reason = None
    if len(source_text) > limit:
        reason = f"prefix_truncated_at_{limit}"
    return exposed, limit, reason


def expose_arm_b(source_text: str) -> tuple[str, int, str | None]:
    return source_text, len(source_text), None


def expose_arm_c(source_text: str, required_windows: list[list[int]]) -> tuple[str, list[tuple[int, int]], str | None]:
    merged = merge_windows(required_windows)
    parts: list[str] = []
    for start, end in merged:
        _require(0 <= start < end <= len(source_text), "required window out of source bounds")
        parts.append(source_text[start:end])
    exposed = "\n---\n".join(parts)
    return exposed, merged, None


def window_visible_in_prefix(window: tuple[int, int], prefix_end: int) -> bool:
    start, end = window
    return end <= prefix_end


def window_visible_in_complete(window: tuple[int, int], source_len: int) -> bool:
    start, end = window
    return 0 <= start < end <= source_len


def window_visible_in_gold_context(window: tuple[int, int], merged_windows: list[tuple[int, int]]) -> bool:
    start, end = window
    for merged_start, merged_end in merged_windows:
        if start >= merged_start and end <= merged_end:
            return True
    return False


def requirement_visibility(
    *,
    arm: str,
    source_text: str,
    required_windows: list[list[int]],
    source_kind: str,
    exposed_text: str,
    gold_merged: list[tuple[int, int]] | None = None,
    prefix_end: int | None = None,
) -> dict[str, Any]:
    windows = [(int(s), int(e)) for s, e in required_windows]
    span_results: list[dict[str, Any]] = []
    all_visible = True
    for index, window in enumerate(windows):
        if arm == ARM_A:
            visible = window_visible_in_prefix(window, prefix_end or prefix_limit(source_kind))
            truncated = not visible
        elif arm == ARM_B:
            visible = window_visible_in_complete(window, len(source_text))
            truncated = not visible
        elif arm == ARM_C:
            _require(gold_merged is not None, "gold merged windows required for arm C")
            visible = window_visible_in_gold_context(window, gold_merged)
            truncated = not visible
        else:
            raise EvidenceExposure2CError(f"unknown arm {arm!r}")
        if not visible:
            all_visible = False
        span_results.append(
            {
                "window_index": index,
                "source_span": [window[0], window[1]],
                "visible": visible,
                "truncation_violation": truncated,
            }
        )
    substring_check = all(
        source_text[w[0] : w[1]] in exposed_text
        for w in windows
        if any(item["visible"] for item in span_results)
    )
    return {
        "all_required_windows_visible": all_visible,
        "partial_only": not all_visible and any(item["visible"] for item in span_results),
        "span_results": span_results,
        "substring_check": substring_check,
    }


def build_evidence_record(
    *,
    case_id: str,
    requirement_id: str,
    source_id: str,
    source_kind: str,
    source_text: str,
    arm: str,
    consumer: str,
    exposed_text: str,
    exposure_policy_id: str,
    original_offsets: list[tuple[int, int]],
    exposed_offsets: list[tuple[int, int]] | None,
    visibility: dict[str, Any],
    truncation_reason: str | None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "source_id": source_id,
        "source_kind": source_kind,
        "complete_source_text": source_text,
        "source_sha256": sha256_text(source_text),
        "original_character_length": len(source_text),
        "exposure_policy_id": exposure_policy_id,
        "consumer": consumer,
        "exposed_text": exposed_text,
        "exposed_text_sha256": sha256_text(exposed_text),
        "original_source_offsets": [[s, e] for s, e in original_offsets],
        "exposed_offsets": [[s, e] for s, e in (exposed_offsets or [])],
        "requirement_visibility": visibility,
        "truncation_reason": truncation_reason,
    }


def evaluate_requirement_at_rank(
    *,
    case: dict[str, Any],
    requirement: dict[str, Any],
    arm: str,
    consumer: str,
) -> dict[str, Any]:
    source_text = case["source_text"]
    source_kind = case["source_kind"]
    required_windows = requirement["required_windows"]

    if arm == ARM_A:
        exposed, prefix_end, truncation_reason = expose_arm_a(source_kind, source_text)
        exposure_policy_id = f"current_prefix_{source_kind}_{prefix_end}"
        visibility = requirement_visibility(
            arm=arm,
            source_text=source_text,
            required_windows=required_windows,
            source_kind=source_kind,
            exposed_text=exposed,
            prefix_end=prefix_end,
        )
        original_offsets = [(0, min(prefix_end, len(source_text)))]
        exposed_offsets = [(0, len(exposed))]
    elif arm == ARM_B:
        exposed, _, truncation_reason = expose_arm_b(source_text)
        exposure_policy_id = "complete_source"
        visibility = requirement_visibility(
            arm=arm,
            source_text=source_text,
            required_windows=required_windows,
            source_kind=source_kind,
            exposed_text=exposed,
        )
        original_offsets = [(0, len(source_text))]
        exposed_offsets = [(0, len(exposed))]
    elif arm == ARM_C:
        exposed, gold_merged, truncation_reason = expose_arm_c(source_text, required_windows)
        exposure_policy_id = "gold_relevant_context"
        visibility = requirement_visibility(
            arm=arm,
            source_text=source_text,
            required_windows=required_windows,
            source_kind=source_kind,
            exposed_text=exposed,
            gold_merged=gold_merged,
        )
        original_offsets = gold_merged
        exposed_offsets = [(0, len(exposed))]
    else:
        raise EvidenceExposure2CError(f"unknown arm {arm!r}")

    return {
        "requirement_id": requirement["id"],
        "arm": arm,
        "consumer": consumer,
        "recall_at_k": 1.0 if case["source_rank"] == 1 else 0.0,
        "requirement_recall": 1.0 if visibility["all_required_windows_visible"] else 0.0,
        "truncation_violation": any(item["truncation_violation"] for item in visibility["span_results"]),
        "evidence_record": build_evidence_record(
            case_id=case["id"],
            requirement_id=requirement["id"],
            source_id=case["source_id"],
            source_kind=source_kind,
            source_text=source_text,
            arm=arm,
            consumer=consumer,
            exposed_text=exposed,
            exposure_policy_id=exposure_policy_id,
            original_offsets=original_offsets,
            exposed_offsets=exposed_offsets,
            visibility=visibility,
            truncation_reason=truncation_reason,
        ),
    }


def metric_ratio(numerator: int, denominator: int, *, undefined_reason: str | None = None) -> dict[str, Any]:
    if denominator == 0:
        return {
            "value": None,
            "numerator": 0,
            "denominator": 0,
            "undefined": True,
            "undefined_reason": undefined_reason or "zero denominator",
        }
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "undefined": False,
        "undefined_reason": None,
    }


def validate_case(case: dict[str, Any]) -> None:
    _require(case["id"] in CASE_IDS, f"unknown case id {case.get('id')}")
    _require(case["source_kind"] in SOURCE_KINDS, "source_kind must be web or kb")
    _require(case["source_rank"] == 1, "Stage 2C requires rank-1 source")
    _require(isinstance(case["source_text"], str) and case["source_text"].isascii(), "source_text must be ASCII")
    _require(case.get("source_sha256") == sha256_text(case["source_text"]), "source_sha256 mismatch")
    _require(case.get("source_character_length") == len(case["source_text"]), "source_character_length mismatch")
    claim = case["claim"]
    _require(claim.get("claim_sha256") == sha256_text(claim["text"]), "claim_sha256 mismatch")
    _require(claim["independent_semantic_label"] in SEMANTIC_LABELS, "invalid semantic label")
    for requirement in case["requirements"]:
        _require(requirement.get("id"), "requirement id required")
        truth = [(int(s), int(e)) for s, e in requirement["truth_spans"]]
        windows = [(int(s), int(e)) for s, e in requirement["required_windows"]]
        for start, end in truth + windows:
            _require(0 <= start < end <= len(case["source_text"]), "span out of bounds")
        for t_start, t_end in truth:
            covered = any(w_start <= t_start and t_end <= w_end for w_start, w_end in windows)
            _require(covered, "truth span must be covered by required window")
    expected = case["expected_visibility"]
    for arm in ARMS:
        _require(arm in expected, f"missing expected_visibility for {arm}")


def validate_pack(pack: dict[str, Any]) -> None:
    _require(pack["pack_id"] == PACK_ID, "pack_id mismatch")
    _require(pack["evaluator_id"] == EVALUATOR_ID, "evaluator_id mismatch")
    _require(pack["schema_version"] == SCHEMA_VERSION, "schema_version mismatch")
    _require(pack.get("stage") == STAGE, "stage must be 2c")
    cases = pack["cases"]
    _require(len(cases) == len(CASE_IDS), f"expected {len(CASE_IDS)} cases")
    seen: set[str] = set()
    for case in cases:
        validate_case(case)
        _require(case["id"] not in seen, f"duplicate case id {case['id']}")
        seen.add(case["id"])
    _require(seen == set(CASE_IDS), "case catalog mismatch")


def load_pack(path: Path | str = DEFAULT_FIXTURES) -> dict[str, Any]:
    pack_path = Path(path)
    try:
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceExposure2CError(f"unreadable fixture pack {pack_path}: {error}") from error
    validate_pack(pack)
    return pack


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    requirements = case["requirements"]
    per_arm: dict[str, dict[str, Any]] = {}
    evidence_records: list[dict[str, Any]] = []

    for arm in ARMS:
        consumer_results: dict[str, Any] = {}
        for consumer in CONSUMERS:
            req_results = [
                evaluate_requirement_at_rank(case=case, requirement=requirement, arm=arm, consumer=consumer)
                for requirement in requirements
            ]
            evidence_records.extend(item["evidence_record"] for item in req_results)
            recall_num = sum(1 for item in req_results if item["requirement_recall"] == 1.0)
            trunc_num = sum(1 for item in req_results if item["truncation_violation"])
            consumer_results[consumer] = {
                "requirement_results": req_results,
                "gold_evidence_recall": metric_ratio(recall_num, len(req_results)),
                "truncation_violations": trunc_num,
                "all_requirements_visible": recall_num == len(req_results),
            }
        per_arm[arm] = consumer_results

    retrieval_recall = metric_ratio(1 if case["source_rank"] == 1 else 0, 1)

    return {
        "case_id": case["id"],
        "source_kind": case["source_kind"],
        "source_rank": case["source_rank"],
        "claim": case["claim"],
        "expected_visibility": case["expected_visibility"],
        "retrieval_gold_evidence_requirement_recall_at_k": retrieval_recall,
        "arms": per_arm,
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(arm: str, consumer: str, field: str) -> tuple[int, int]:
        num = 0
        den = 0
        if field == "retrieval":
            for result in results:
                den += 1
                if result["retrieval_gold_evidence_requirement_recall_at_k"]["value"] == 1.0:
                    num += 1
            return num, den
        for result in results:
            req_results = result["arms"][arm][consumer]["requirement_results"]
            den += len(req_results)
            if field == "recall":
                num += sum(1 for item in req_results if item["requirement_recall"] == 1.0)
            elif field == "truncation":
                num += sum(1 for item in req_results if item["truncation_violation"])
        return num, den

    retrieval_num, retrieval_den = collect(ARM_A, "draft", "retrieval")
    draft_a_num, draft_a_den = collect(ARM_A, "draft", "recall")
    verifier_a_num, verifier_a_den = collect(ARM_A, "verifier", "recall")
    trunc_a_num, trunc_a_den = collect(ARM_A, "draft", "truncation")

    return {
        "gold_evidence_requirement_recall_at_k.v1": metric_ratio(retrieval_num, retrieval_den),
        "draft_gold_evidence_recall.v1": metric_ratio(draft_a_num, draft_a_den),
        "verifier_gold_evidence_recall.v1": metric_ratio(verifier_a_num, verifier_a_den),
        "retrieved_to_draft_evidence_retention.v1": metric_ratio(draft_a_num, draft_a_den),
        "retrieved_to_verifier_evidence_retention.v1": metric_ratio(verifier_a_num, verifier_a_den),
        "relevant_span_truncation_violation_rate.v1": metric_ratio(trunc_a_num, trunc_a_den),
        "by_arm": {
            ARM_B: {
                "draft_gold_evidence_recall": metric_ratio(*collect(ARM_B, "draft", "recall")),
                "verifier_gold_evidence_recall": metric_ratio(*collect(ARM_B, "verifier", "recall")),
                "truncation_violation_rate": metric_ratio(*collect(ARM_B, "draft", "truncation")),
            },
            ARM_C: {
                "draft_gold_evidence_recall": metric_ratio(*collect(ARM_C, "draft", "recall")),
                "verifier_gold_evidence_recall": metric_ratio(*collect(ARM_C, "verifier", "recall")),
                "truncation_violation_rate": metric_ratio(*collect(ARM_C, "draft", "truncation")),
            },
        },
    }


def evaluate_pack(pack: dict[str, Any]) -> dict[str, Any]:
    validate_pack(pack)
    results = [evaluate_case(case) for case in pack["cases"]]
    metrics = aggregate_metrics(results)
    gates = evaluate_gates(results)
    return {
        "evaluator_id": EVALUATOR_ID,
        "pack_id": pack["pack_id"],
        "stage": STAGE,
        "k": K,
        "results": results,
        "metrics": metrics,
        "gates": gates,
        "stage_2c_pass": gates["all_pass"],
    }


def evaluate_gates(results: list[dict[str, Any]]) -> dict[str, Any]:
    gate_results: dict[str, Any] = {}
    all_pass = True

    for result in results:
        case_id = result["case_id"]
        if result["retrieval_gold_evidence_requirement_recall_at_k"]["value"] != 1.0:
            gate_results[f"{case_id}_retrieval_recall_at_1"] = False
            all_pass = False
        else:
            gate_results[f"{case_id}_retrieval_recall_at_1"] = True

        for arm in (ARM_B, ARM_C):
            for consumer in CONSUMERS:
                visible = result["arms"][arm][consumer]["all_requirements_visible"]
                key = f"{case_id}_{arm}_{consumer}_recall_1"
                gate_results[key] = visible
                if not visible:
                    all_pass = False

        for consumer in CONSUMERS:
            trunc = result["arms"][ARM_B][consumer]["truncation_violations"]
            gate_results[f"{case_id}_arm_b_{consumer}_truncation_zero"] = trunc == 0
            if trunc != 0:
                all_pass = False
            trunc_c = result["arms"][ARM_C][consumer]["truncation_violations"]
            gate_results[f"{case_id}_arm_c_{consumer}_truncation_zero"] = trunc_c == 0
            if trunc_c != 0:
                all_pass = False

        expected_a = result["expected_visibility"][ARM_A]
        for consumer in CONSUMERS:
            actual = result["arms"][ARM_A][consumer]["all_requirements_visible"]
            expected = expected_a[consumer]
            key = f"{case_id}_arm_a_{consumer}_expected_visibility"
            gate_results[key] = actual == expected
            if actual != expected:
                all_pass = False

    gate_results["all_pass"] = all_pass
    return gate_results


def build_frozen_fixtures() -> dict[str, Any]:
    """Build the eight frozen Stage 2C cases with deterministic ASCII sources."""

    def filler(n: int, char: str = ".") -> str:
        return char * n

    def case(
        case_id: str,
        title: str,
        source_kind: Literal["web", "kb"],
        source_id: str,
        source_url: str,
        source_text: str,
        claim_text: str,
        semantic_label: str,
        requirements: list[dict[str, Any]],
        expected_a_draft: bool,
        expected_a_verifier: bool,
    ) -> dict[str, Any]:
        return {
            "id": case_id,
            "title": title,
            "source_kind": source_kind,
            "source_rank": 1,
            "source_id": source_id,
            "source_url": source_url,
            "source_text": source_text,
            "source_sha256": sha256_text(source_text),
            "source_character_length": len(source_text),
            "claim": {
                "text": claim_text,
                "claim_sha256": sha256_text(claim_text),
                "material": True,
                "independent_semantic_label": semantic_label,
            },
            "requirements": requirements,
            "expected_visibility": {
                ARM_A: {"draft": expected_a_draft, "verifier": expected_a_verifier},
                ARM_B: {"draft": True, "verifier": True},
                ARM_C: {"draft": True, "verifier": True},
            },
        }

    w01_evidence = "NEXUS-7 bounded replay retains exactly 512 events per shard."
    w01_source = (
        "Project NEXUS-7 internal field manual. "
        + w01_evidence
        + " Operators must rotate shard logs weekly. "
        + filler(600, "-")
    )

    k01_evidence = "FluxCap array rebuild completes in 4096 milliseconds under nominal load."
    k01_source = (
        "FluxCap storage array service note FC-881. "
        + k01_evidence
        + " Maintenance window policy FC-12 applies. "
        + filler(900, "-")
    )

    w02_evidence = "Quorum latch engages only after seven replica acknowledgements."
    w02_source = filler(WEB_PREFIX_CHARS, "-") + w02_evidence

    k02_evidence = "Shard seal token ZETA-44 activates only after checkpoint 8192."
    k02_source = filler(KB_PREFIX_CHARS, "-") + k02_evidence

    w03_support = "Helix gate permits export when mode flag HG-ENABLE is set."
    w03_contra = "Unless mode flag HG-ENABLE is cleared, export remains blocked."
    w03_source = (
        filler(400, "-")
        + w03_support
        + " "
        + filler(WEB_PREFIX_CHARS - 400 - len(w03_support) - 1, "-")
        + w03_contra
    )

    k03_base = "Cache tier T3 admits writes when isolation level IL-2 holds."
    k03_qualifier = "unless cache mode is disabled via CM-OFF switch."
    k03_source = (
        k03_base
        + " "
        + filler(KB_PREFIX_CHARS - len(k03_base) - 1, "-")
        + k03_qualifier
    )

    w04_evidence = "Throughput threshold is 98304 tokens per second at batch size 64."
    w04_source = filler(WEB_PREFIX_CHARS, "-") + w04_evidence

    k04_evidence = "Vector index VIX-9 stores 65537 centroids for partition P0."
    k04_source = k04_evidence + filler(1200, "-") + "Marketing summary only."

    cases = [
        case(
            "E2C-W01",
            "web early support",
            "web",
            "SRC-W01",
            "https://example.test/nexus-7/manual",
            w01_source,
            "NEXUS-7 bounded replay retains exactly 512 events per shard.",
            "verified",
            [
                {
                    "id": "REQ-W01-1",
                    "truth_spans": [[w01_source.index(w01_evidence), w01_source.index(w01_evidence) + len(w01_evidence)]],
                    "required_windows": [[max(0, w01_source.index(w01_evidence) - 20), w01_source.index(w01_evidence) + len(w01_evidence) + 20]],
                }
            ],
            True,
            True,
        ),
        case(
            "E2C-K01",
            "KB early support",
            "kb",
            "SRC-K01",
            "kb://fluxcap/service-note-fc881",
            k01_source,
            "FluxCap array rebuild completes in 4096 milliseconds under nominal load.",
            "verified",
            [
                {
                    "id": "REQ-K01-1",
                    "truth_spans": [[k01_source.index(k01_evidence), k01_source.index(k01_evidence) + len(k01_evidence)]],
                    "required_windows": [[max(0, k01_source.index(k01_evidence) - 25), k01_source.index(k01_evidence) + len(k01_evidence) + 25]],
                }
            ],
            True,
            True,
        ),
        case(
            "E2C-W02",
            "web late support",
            "web",
            "SRC-W02",
            "https://example.test/quorum/latch",
            w02_source,
            "Quorum latch engages only after seven replica acknowledgements.",
            "verified",
            [
                {
                    "id": "REQ-W02-1",
                    "truth_spans": [[w02_source.index(w02_evidence), w02_source.index(w02_evidence) + len(w02_evidence)]],
                    "required_windows": [[
                        max(0, w02_source.index(w02_evidence) - 5),
                        min(len(w02_source), w02_source.index(w02_evidence) + len(w02_evidence) + 5),
                    ]],
                }
            ],
            False,
            False,
        ),
        case(
            "E2C-K02",
            "KB late support",
            "kb",
            "SRC-K02",
            "kb://zeta/shard-seal",
            k02_source,
            "Shard seal token ZETA-44 activates only after checkpoint 8192.",
            "verified",
            [
                {
                    "id": "REQ-K02-1",
                    "truth_spans": [[k02_source.index(k02_evidence), k02_source.index(k02_evidence) + len(k02_evidence)]],
                    "required_windows": [[
                        max(0, k02_source.index(k02_evidence) - 5),
                        min(len(k02_source), k02_source.index(k02_evidence) + len(k02_evidence) + 5),
                    ]],
                }
            ],
            False,
            False,
        ),
        case(
            "E2C-W03",
            "web late contradiction",
            "web",
            "SRC-W03",
            "https://example.test/helix/export-gate",
            w03_source,
            "Helix gate permits export when mode flag HG-ENABLE is set.",
            "weak",
            [
                {
                    "id": "REQ-W03-1",
                    "truth_spans": [
                        [w03_source.index(w03_support), w03_source.index(w03_support) + len(w03_support)],
                        [w03_source.index(w03_contra), w03_source.index(w03_contra) + len(w03_contra)],
                    ],
                    "required_windows": [
                        [max(0, w03_source.index(w03_support) - 10), w03_source.index(w03_support) + len(w03_support) + 10],
                        [max(0, w03_source.index(w03_contra) - 15), min(len(w03_source), w03_source.index(w03_contra) + len(w03_contra) + 15)],
                    ],
                }
            ],
            False,
            False,
        ),
        case(
            "E2C-K03",
            "KB late qualifier/condition",
            "kb",
            "SRC-K03",
            "kb://cache/tier-t3-policy",
            k03_source,
            "Cache tier T3 admits writes when isolation level IL-2 holds unless cache mode is disabled.",
            "verified",
            [
                {
                    "id": "REQ-K03-1",
                    "truth_spans": [
                        [k03_source.index(k03_base), k03_source.index(k03_base) + len(k03_base)],
                        [k03_source.index(k03_qualifier), k03_source.index(k03_qualifier) + len(k03_qualifier)],
                    ],
                    "required_windows": [
                        [0, len(k03_base) + 5],
                        [
                            max(0, k03_source.index(k03_qualifier) - 10),
                            min(len(k03_source), k03_source.index(k03_qualifier) + len(k03_qualifier) + 10),
                        ],
                    ],
                }
            ],
            False,
            False,
        ),
        case(
            "E2C-W04",
            "web late exact numeric/version/threshold evidence",
            "web",
            "SRC-W04",
            "https://example.test/throughput/threshold",
            w04_source,
            "Throughput threshold is 98304 tokens per second at batch size 64.",
            "verified",
            [
                {
                    "id": "REQ-W04-1",
                    "truth_spans": [[w04_source.index(w04_evidence), w04_source.index(w04_evidence) + len(w04_evidence)]],
                    "required_windows": [[
                        max(0, w04_source.index(w04_evidence) - 8),
                        min(len(w04_source), w04_source.index(w04_evidence) + len(w04_evidence) + 8),
                    ]],
                }
            ],
            False,
            False,
        ),
        case(
            "E2C-K04",
            "KB early decisive evidence plus irrelevant late content",
            "kb",
            "SRC-K04",
            "kb://vix/partition-p0",
            k04_source,
            "Vector index VIX-9 stores 65537 centroids for partition P0.",
            "verified",
            [
                {
                    "id": "REQ-K04-1",
                    "truth_spans": [[k04_source.index(k04_evidence), k04_source.index(k04_evidence) + len(k04_evidence)]],
                    "required_windows": [[0, len(k04_evidence) + 15]],
                }
            ],
            True,
            True,
        ),
    ]
    pack = {
        "pack_id": PACK_ID,
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "stage": STAGE,
        "description": "Frozen Stage 2C evidence-exposure qualification fixtures",
        "cases": cases,
    }
    validate_pack(pack)
    return pack


def write_frozen_fixtures(path: Path | str = DEFAULT_FIXTURES) -> None:
    pack = build_frozen_fixtures()
    Path(path).write_text(json.dumps(pack, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2C evidence exposure fixtures")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--write-fixtures", action="store_true", help="Regenerate frozen fixture JSON")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args(argv)
    if args.write_fixtures:
        write_frozen_fixtures(args.fixtures)
        print(f"wrote {args.fixtures}")
        return 0
    pack = load_pack(args.fixtures)
    report = evaluate_pack(pack)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(f"stage_2c_pass={report['stage_2c_pass']}")
        for result in report["results"]:
            case_id = result["case_id"]
            arm_a = result["arms"][ARM_A]["draft"]["all_requirements_visible"]
            arm_b = result["arms"][ARM_B]["draft"]["all_requirements_visible"]
            arm_c = result["arms"][ARM_C]["draft"]["all_requirements_visible"]
            print(f"{case_id}: A={arm_a} B={arm_b} C={arm_c}")
    return 0 if report["stage_2c_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
