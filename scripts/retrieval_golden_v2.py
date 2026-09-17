"""Deterministic validator for retrieval golden set v2.

V1 authority remains in scripts/retrieval_eval.py (GOLDEN_SET). This module validates
the separate v2 adjudication artifact only — no semantic model calls, no KB access.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ORACLE_PATH = REPO_ROOT / "evals/fixtures/retrieval_golden_v2.json"
BASELINE_IDENTITY_PATH = REPO_ROOT / "evals/fixtures/retrieval_baseline_a_identity.json"
V1_EVAL_PATH = REPO_ROOT / "scripts/retrieval_eval.py"
SEED_DOCS_DIR = REPO_ROOT / "kb/seed_docs"

ANSWERABILITY_VALUES = frozenset({"ANSWERABLE", "PARTIAL", "ABSENT"})
V1_VERDICT_VALUES = frozenset({"CORRECT", "INCOMPLETE", "WRONG", "AMBIGUOUS"})
VALID_GRADES = frozenset({1, 2})


class RetrievalGoldenV2Error(ValueError):
    """Retrieval golden v2 oracle or validation error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def load_v1_golden_set() -> list[dict[str, Any]]:
    src = V1_EVAL_PATH.read_text(encoding="utf-8")
    start = src.index("GOLDEN_SET = [")
    end = src.index("]\n\ndef run_retrieval_eval")
    return ast.literal_eval(src[start + len("GOLDEN_SET = ") : end + 1])


def load_oracle(path: Path | str = ORACLE_PATH) -> dict[str, Any]:
    oracle_path = Path(path)
    payload = json.loads(oracle_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "retrieval_golden_v2":
        raise RetrievalGoldenV2Error("invalid schema_version")
    return payload


def _manifest_sources(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = payload.get("corpus_manifest")
    if not isinstance(manifest, list):
        raise RetrievalGoldenV2Error("corpus_manifest must be a list")
    by_source: dict[str, dict[str, Any]] = {}
    for entry in manifest:
        source = entry.get("source")
        if not source:
            raise RetrievalGoldenV2Error("corpus_manifest entry missing source")
        if source in by_source:
            raise RetrievalGoldenV2Error(f"duplicate corpus_manifest source: {source}")
        by_source[source] = entry
    return by_source


def _validate_manifest_digest(payload: dict[str, Any]) -> None:
    manifest = payload.get("corpus_manifest")
    observed = sha256_text(canonical_json_dumps(manifest))
    declared = payload.get("corpus_manifest_digest")
    if observed != declared:
        raise RetrievalGoldenV2Error(
            f"corpus_manifest_digest mismatch: observed={observed} declared={declared}"
        )


def _read_source_lines(source: str, manifest_entry: dict[str, Any]) -> list[str]:
    rel_path = manifest_entry.get("path")
    if not rel_path:
        raise RetrievalGoldenV2Error(f"manifest entry for {source} missing path")
    file_path = REPO_ROOT / rel_path
    if not file_path.is_file():
        raise RetrievalGoldenV2Error(f"corpus file missing: {rel_path}")
    text = file_path.read_text(encoding="utf-8")
    observed_sha = sha256_text(text)
    declared_sha = manifest_entry.get("sha256")
    if observed_sha != declared_sha:
        raise RetrievalGoldenV2Error(
            f"corpus sha256 drift for {source}: observed={observed_sha} declared={declared_sha}"
        )
    return text.splitlines()


def _span_text(lines: list[str], line_start: int, line_end: int) -> str:
    if line_start < 1 or line_end < line_start or line_end > len(lines):
        raise RetrievalGoldenV2Error(
            f"invalid line span {line_start}-{line_end} for file with {len(lines)} lines"
        )
    return "\n".join(lines[line_start - 1 : line_end])


def validate_oracle(
    payload: dict[str, Any] | None = None,
    *,
    path: Path | str = ORACLE_PATH,
    check_v1_unchanged: bool = True,
) -> dict[str, Any]:
    """Validate v2 oracle deterministically. Returns a summary dict."""
    if payload is None:
        payload = load_oracle(path)

    _validate_manifest_digest(payload)
    manifest_by_source = _manifest_sources(payload)
    if len(manifest_by_source) != 20:
        raise RetrievalGoldenV2Error(f"expected 20 corpus sources, got {len(manifest_by_source)}")

    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise RetrievalGoldenV2Error("queries must be a list")

    query_ids = [q.get("query_id") for q in queries]
    if len(query_ids) != 35:
        raise RetrievalGoldenV2Error(f"expected 35 queries, got {len(query_ids)}")
    if len(set(query_ids)) != 35:
        raise RetrievalGoldenV2Error("query_id values must be unique")

    v1_golden = load_v1_golden_set() if check_v1_unchanged else []
    if check_v1_unchanged and len(v1_golden) != 35:
        raise RetrievalGoldenV2Error("v1 GOLDEN_SET must contain 35 queries")

    line_cache: dict[str, list[str]] = {}
    invalid_source_refs = 0
    quote_mismatches = 0
    evidence_spans_validated = 0

    answerability_counts: dict[str, int] = {"ANSWERABLE": 0, "PARTIAL": 0, "ABSENT": 0}
    v1_verdict_counts: dict[str, int] = {k: 0 for k in V1_VERDICT_VALUES}
    gating_eligible = 0
    non_gating = 0
    holdout = 0
    development = 0

    for query in queries:
        qid = query["query_id"]
        answerability = query.get("answerability")
        if answerability not in ANSWERABILITY_VALUES:
            raise RetrievalGoldenV2Error(f"{qid}: invalid answerability {answerability!r}")
        answerability_counts[answerability] += 1

        verdict = query.get("v1_label_verdict")
        if verdict not in V1_VERDICT_VALUES:
            raise RetrievalGoldenV2Error(f"{qid}: invalid v1_label_verdict {verdict!r}")
        v1_verdict_counts[verdict] += 1

        split = query.get("split")
        if split == "holdout":
            holdout += 1
            raise RetrievalGoldenV2Error(f"{qid}: holdout split forbidden in v2")
        if split != "development/diagnostic":
            raise RetrievalGoldenV2Error(f"{qid}: split must be development/diagnostic")
        development += 1

        if query.get("gating_eligible") is True:
            gating_eligible += 1
        elif query.get("gating_eligible") is False:
            non_gating += 1
        else:
            raise RetrievalGoldenV2Error(f"{qid}: gating_eligible must be boolean")

        v1_index = query.get("v1_index")
        if not isinstance(v1_index, int) or v1_index < 0 or v1_index > 34:
            raise RetrievalGoldenV2Error(f"{qid}: invalid v1_index {v1_index!r}")

        if check_v1_unchanged:
            expected_text = v1_golden[v1_index]["query"]
            if query.get("query") != expected_text:
                raise RetrievalGoldenV2Error(f"{qid}: query text drift vs v1 GOLDEN_SET")

        if qid in {"Q25", "Q26"}:
            if answerability != "ANSWERABLE":
                raise RetrievalGoldenV2Error(
                    f"{qid}: must have answerability ANSWERABLE (ambiguous != partial)"
                )
            if query.get("gating_eligible") is not False:
                raise RetrievalGoldenV2Error(f"{qid}: must be non-gating")
            if verdict != "AMBIGUOUS":
                raise RetrievalGoldenV2Error(f"{qid}: must have v1_label_verdict AMBIGUOUS")

        if qid in {f"Q{i:02d}" for i in range(31, 36)}:
            if answerability != "PARTIAL":
                raise RetrievalGoldenV2Error(f"{qid}: must have answerability PARTIAL")

        seen_sources: set[str] = set()
        for rel in query.get("relevant_sources", []):
            source = rel.get("source")
            if source not in manifest_by_source:
                invalid_source_refs += 1
                raise RetrievalGoldenV2Error(f"{qid}: unknown source {source!r}")
            if source in seen_sources:
                raise RetrievalGoldenV2Error(f"{qid}: duplicate relevant_sources entry for {source}")
            seen_sources.add(source)

            grade = rel.get("grade")
            if grade not in VALID_GRADES:
                raise RetrievalGoldenV2Error(f"{qid}/{source}: grade must be 1 or 2, got {grade!r}")

            if source not in line_cache:
                line_cache[source] = _read_source_lines(source, manifest_by_source[source])
            lines = line_cache[source]

            for span in rel.get("evidence", []):
                line_start = span.get("line_start")
                line_end = span.get("line_end")
                quote = span.get("quote")
                if not isinstance(line_start, int) or not isinstance(line_end, int):
                    raise RetrievalGoldenV2Error(f"{qid}/{source}: line_start/line_end must be int")
                try:
                    actual = _span_text(lines, line_start, line_end)
                except RetrievalGoldenV2Error:
                    quote_mismatches += 1
                    raise
                if actual != quote:
                    quote_mismatches += 1
                    raise RetrievalGoldenV2Error(
                        f"{qid}/{source}:{line_start}-{line_end}: quote mismatch"
                    )
                evidence_spans_validated += 1

    if answerability_counts != {"ANSWERABLE": 30, "PARTIAL": 5, "ABSENT": 0}:
        raise RetrievalGoldenV2Error(
            f"answerability aggregate must be ANSWERABLE=30 PARTIAL=5 ABSENT=0, "
            f"observed={answerability_counts}"
        )
    if holdout != 0:
        raise RetrievalGoldenV2Error("v2 must contain zero holdout queries")
    if development != 35:
        raise RetrievalGoldenV2Error(f"v2 must contain 35 development/diagnostic queries, got {development}")

    absence_policy = payload.get("absence_policy", {})
    if absence_policy.get("absent_queries", -1) != 0:
        raise RetrievalGoldenV2Error("absence_policy.absent_queries must be 0")

    if payload.get("dataset_status") != "DEVELOPMENT/DIAGNOSTIC NUCLEUS":
        raise RetrievalGoldenV2Error("dataset_status must be DEVELOPMENT/DIAGNOSTIC NUCLEUS")

    summary = payload.get("adjudication_summary", {})
    expected_answerability = summary.get("answerability", {})
    for label, count in expected_answerability.items():
        if answerability_counts.get(label, 0) != count:
            raise RetrievalGoldenV2Error(
                f"answerability count mismatch for {label}: "
                f"observed={answerability_counts.get(label, 0)} expected={count}"
            )
    expected_v1 = summary.get("v1_label_verdict", {})
    for label, count in expected_v1.items():
        if v1_verdict_counts.get(label, 0) != count:
            raise RetrievalGoldenV2Error(
                f"v1_label_verdict count mismatch for {label}: "
                f"observed={v1_verdict_counts.get(label, 0)} expected={count}"
            )
    if summary.get("gating_eligible") != gating_eligible:
        raise RetrievalGoldenV2Error(
            f"gating_eligible summary mismatch: observed={gating_eligible} "
            f"expected={summary.get('gating_eligible')}"
        )
    if summary.get("non_gating") != non_gating:
        raise RetrievalGoldenV2Error(
            f"non_gating summary mismatch: observed={non_gating} "
            f"expected={summary.get('non_gating')}"
        )
    split_summary = summary.get("split", {})
    if split_summary.get("development/diagnostic") != development:
        raise RetrievalGoldenV2Error("split.development/diagnostic must match query count")
    if split_summary.get("holdout", -1) != 0:
        raise RetrievalGoldenV2Error("split.holdout must be 0")

    return {
        "query_count": len(queries),
        "answerability": answerability_counts,
        "v1_label_verdict": v1_verdict_counts,
        "gating_eligible": gating_eligible,
        "non_gating": non_gating,
        "development": development,
        "holdout": holdout,
        "evidence_spans_validated": evidence_spans_validated,
        "invalid_source_refs": invalid_source_refs,
        "quote_span_mismatches": quote_mismatches,
        "corpus_manifest_digest": payload.get("corpus_manifest_digest"),
        "schema_version": payload.get("schema_version"),
        "repo_head": payload.get("repo_head"),
    }


def load_baseline_identity(path: Path | str = BASELINE_IDENTITY_PATH) -> dict[str, Any]:
    identity_path = Path(path)
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "retrieval_baseline_a_identity":
        raise RetrievalGoldenV2Error("invalid baseline identity schema_version")
    return payload


if __name__ == "__main__":
    result = validate_oracle()
    print(json.dumps(result, indent=2))
