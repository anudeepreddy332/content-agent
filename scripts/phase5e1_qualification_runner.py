"""Phase-5E1E raw retrieval qualification harness.

Fixtures contain questions and canonical gold intervals only.  This module calls
both qualified backends and derives coverage, parity, metrics, and disposition.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Callable
from dataclasses import dataclass

from scripts.phase5a0_baseline import (
    graded_ndcg_at_k,
    reciprocal_rank,
    source_recall_at_k,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONTRACT_PATH = (
    REPO_ROOT / "evals/fixtures/phase5e1_qualification_contract.json"
)
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "reports/phase5/phase5e1a/holdout-run-1"
AUTHORITATIVE_HOLDOUT_PATH = REPO_ROOT / "evals/fixtures/retrieval_holdout_v1.json"
AUTHORITATIVE_HOLDOUT_SCHEMA = "retrieval_holdout_v1"
# This is the externally approved byte digest of the designated sealed fixture.
# The fixture is intentionally not checked into this release worktree.
AUTHORITATIVE_HOLDOUT_SHA256 = (
    "9d45c7255886c5dcd3060260a470080c2f1ca5080361c37274dbbab24167874f"
)
EXPECTED_CONTRACT_SHA256 = (
    "7c19cb7b0ae6126be19816615a8976e0685be18b1f8686c31658338afdf38b47"
)
BACKENDS = ("cswp_local", "cswp_qdrant")
RANKING_PARITY_FIELDS = (
    "dense_top20",
    "bm25_rank_order",
    "hybrid_seed_top5",
)
CANONICAL_INTERVAL_FIELDS = (
    "retrieval_seeds",
    "expanded_rows",
    "kb_results",
    *RANKING_PARITY_FIELDS,
    "hybrid_top10",
)
QUALIFIED_UNIT_COUNT = 159
SECONDARY_K_VALUES = (1, 3, 5)
MRR_DEPTH = 10
EVIDENCE_BEARING = frozenset(("ANSWERABLE", "PARTIAL"))
DEVELOPMENT_PASS_DISPOSITION = "PHASE-5E1A-DEVELOPMENT-EVALUATION-PASS"
DEVELOPMENT_FAILURE_DISPOSITION = "PHASE-5E1A-DEVELOPMENT-EVALUATION-FAIL"
REQUIRED_RUNTIME_IDENTITY = {
    "cswp_local": (
        "source_corpus_fingerprint",
        "index_fingerprint",
        "minilm_model_id",
        "minilm_model_revision",
        "qualified_contract",
    ),
    "cswp_qdrant": (
        "source_corpus_fingerprint",
        "index_fingerprint",
        "minilm_model_id",
        "minilm_model_revision",
        "qualified_contract",
        "collection_name",
        "live_collection_fingerprint",
    ),
}
SHARED_RUNTIME_IDENTITY = REQUIRED_RUNTIME_IDENTITY["cswp_local"]


class QualificationHarnessError(ValueError):
    pass


class ArchiveEvidenceConflict(QualificationHarnessError):
    """Archived structured evidence conflicts with its raw execution record."""


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationHarnessError(f"{path}: expected JSON object")
    return value


@dataclass(frozen=True)
class FixtureSnapshot:
    fixture: dict[str, Any]
    sha256: str
    raw_bytes: bytes


def load_fixture_snapshot(path: Path) -> FixtureSnapshot:
    """Read, hash, and parse exactly one immutable fixture-byte snapshot."""

    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise QualificationHarnessError(f"{path}: expected JSON object")
    return FixtureSnapshot(value, hashlib.sha256(raw).hexdigest(), raw)


def load_contract(path: Path | None = None) -> dict[str, Any]:
    contract = load_json(path or DEFAULT_CONTRACT_PATH)
    if contract.get("schema_version") != "phase5e1_qualification_v1":
        raise QualificationHarnessError("unsupported qualification contract")
    if sha256_json(contract) != EXPECTED_CONTRACT_SHA256:
        raise QualificationHarnessError("frozen contract SHA-256 mismatch")
    return contract


def contract_sha256(contract: dict[str, Any] | None = None) -> str:
    return sha256_json(contract or load_contract())


def gold_spans(query: dict[str, Any]) -> list[dict[str, Any]]:
    spans = []
    for relevant in query.get("relevant_sources", []):
        for span in relevant.get("evidence", []):
            if (
                not isinstance(span.get("char_start"), int)
                or not isinstance(span.get("char_end"), int)
                or span["char_start"] >= span["char_end"]
            ):
                raise QualificationHarnessError(
                    f"{query['query_id']}: gold spans need valid canonical intervals"
                )
            spans.append(
                {
                    **span,
                    "source": relevant["source"],
                    "grade": int(span.get("grade", relevant.get("grade", 0))),
                }
            )
    return spans


def validate_fixture(fixture: dict[str, Any]) -> None:
    if fixture.get("schema_version") not in {
        "phase5e1_qualification_oracle_v1",
        "retrieval_golden_v2",
        "retrieval_holdout_v1",
    }:
        raise QualificationHarnessError("unsupported fixture schema")
    ids = set()
    for q in fixture.get("queries", []):
        if not isinstance(q.get("query_id"), str) or q["query_id"] in ids:
            raise QualificationHarnessError("query ids must be unique")
        ids.add(q["query_id"])
        if q.get("answerability") not in {"ANSWERABLE", "PARTIAL", "ABSENT"}:
            raise QualificationHarnessError(f"{q['query_id']}: invalid answerability")
        if q["answerability"] == "ABSENT" and q.get("relevant_sources"):
            raise QualificationHarnessError("ABSENT cannot have evidence")
        if q["answerability"] == "ABSENT":
            labels = q.get("hard_negatives", [])
            if not isinstance(labels, list) or any(
                not isinstance(label, dict)
                or not any(
                    isinstance(label.get(field), str) and label[field]
                    for field in ("source", "chunk_id")
                )
                for label in labels
            ):
                raise QualificationHarnessError(
                    f"{q['query_id']}: hard negatives need a source or chunk_id"
                )
        gold_spans(q)
    if not ids:
        raise QualificationHarnessError("fixture has no queries")


def interval_covered(span: dict[str, Any], rows: list[dict[str, Any]], k: int) -> bool:
    intervals = []
    for row in rows[:k]:
        if row.get("source") != span["source"]:
            continue
        for interval in row.get("source_intervals", []):
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise QualificationHarnessError("raw row invalid source interval")
            start, end = (
                max(int(interval[0]), span["char_start"]),
                min(int(interval[1]), span["char_end"]),
            )
            if start < end:
                intervals.append((start, end))
    cursor = span["char_start"]
    for start, end in sorted(intervals):
        if start > cursor:
            break
        cursor = max(cursor, end)
        if cursor >= span["char_end"]:
            return True
    return False


def coverage(
    spans: list[dict[str, Any]], rows: list[dict[str, Any]], k: int
) -> list[str]:
    return [
        f"{s['source']}::{s['span_id']}" for s in spans if interval_covered(s, rows, k)
    ]


def recall(spans: list[dict[str, Any]], rows: list[dict[str, Any]], k: int) -> float:
    return round(len(coverage(spans, rows, k)) / len(spans), 8) if spans else 1.0


def source_recall(
    spans: list[dict[str, Any]], rows: list[dict[str, Any]], k: int
) -> float:
    sources = {s["source"] for s in spans}
    return (
        round(len(sources & {r.get("source") for r in rows[:k]}) / len(sources), 8)
        if sources
        else 0.0
    )


def mrr(spans: list[dict[str, Any]], rows: list[dict[str, Any]], k: int) -> float:
    sources = {s["source"] for s in spans}
    return next(
        (
            round(1 / rank, 8)
            for rank, row in enumerate(rows[:k], 1)
            if row.get("source") in sources
        ),
        0.0,
    )


def ndcg(spans: list[dict[str, Any]], rows: list[dict[str, Any]], k: int) -> float:
    grades = {s["source"]: max(int(s["grade"]), 0) for s in spans}
    credited = set()
    dcg = 0.0
    for rank, row in enumerate(rows[:k], 1):
        source = row.get("source")
        if source in grades and source not in credited:
            dcg += (2 ** grades[source] - 1) / math.log2(rank + 1)
            credited.add(source)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1)
    )
    return round(dcg / idcg, 8) if idcg else 0.0


def require_rows(raw: dict[str, Any], field: str) -> list[dict[str, Any]]:
    rows = raw.get(field)
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or not row.get("chunk_id")
        or not row.get("source")
        or not isinstance(row.get("source_intervals"), list)
        for row in rows
    ):
        raise QualificationHarnessError(f"raw backend output missing usable {field}")
    return rows


def require_rank_rows(raw: dict[str, Any], field: str) -> list[dict[str, Any]]:
    """Require an already-computed diagnostic ranking without rerunning retrieval."""

    rows = raw.get(field)
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("chunk_id"), str)
        or not row["chunk_id"]
        or not isinstance(row.get("source"), str)
        or not row["source"]
        for row in rows
    ):
        raise QualificationHarnessError(
            f"raw backend output missing usable {field} ranking diagnostic"
        )
    return rows


def _canonical_source_intervals(
    units_list: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the only admissible chunk-to-source-interval mapping.

    Qualification coverage is owned by frozen CSWP source spans, never by an
    adapter's generated text, rank, or caller-supplied interval assertion.
    """

    mapping: dict[str, dict[str, Any]] = {}
    for unit in units_list:
        chunk_id = unit.get("chunk_id")
        source_path = unit.get("source_path")
        spans = unit.get("source_spans")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise QualificationHarnessError("qualified CSWP unit missing chunk_id")
        if chunk_id in mapping:
            raise QualificationHarnessError(
                f"qualified CSWP contains duplicate chunk_id: {chunk_id!r}"
            )
        if not isinstance(source_path, str) or not source_path:
            raise QualificationHarnessError(
                f"qualified CSWP unit {chunk_id!r} missing source_path"
            )
        if not isinstance(spans, list) or not spans:
            raise QualificationHarnessError(
                f"qualified CSWP unit {chunk_id!r} missing source_spans"
            )
        intervals: list[tuple[int, int]] = []
        for span in spans:
            if not isinstance(span, dict):
                raise QualificationHarnessError(
                    f"qualified CSWP unit {chunk_id!r} has invalid source span"
                )
            start, end = span.get("source_char_start"), span.get("source_char_end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start >= end
            ):
                raise QualificationHarnessError(
                    f"qualified CSWP unit {chunk_id!r} has invalid source interval"
                )
            intervals.append((start, end))
        if intervals != sorted(intervals) or len(intervals) != len(set(intervals)):
            raise QualificationHarnessError(
                f"qualified CSWP unit {chunk_id!r} has duplicate or unordered source intervals"
            )
        mapping[chunk_id] = {
            "source": Path(source_path).stem,
            "source_intervals": tuple(intervals),
        }
    return mapping


def qualified_canonical_source_map() -> dict[str, dict[str, Any]]:
    """Load and validate the frozen production CSWP corpus for scoring."""

    from agent.cswp.constants import PRODUCTION_INDEX_DIR, UNITS_FILE
    from agent.cswp.loader import load_production_index, load_units_jsonl

    # The production loader owns the current-corpus representation check and
    # declared manifest/unit-count check.  Read the ordered source separately
    # so duplicate chunk IDs cannot be silently collapsed into the loader map.
    _representation, units, _by_source = load_production_index(PRODUCTION_INDEX_DIR)
    units_list = load_units_jsonl(PRODUCTION_INDEX_DIR / UNITS_FILE)
    if len(units_list) != QUALIFIED_UNIT_COUNT or len(units) != QUALIFIED_UNIT_COUNT:
        raise QualificationHarnessError(
            f"qualified CSWP corpus must contain exactly {QUALIFIED_UNIT_COUNT} units"
        )
    mapping = _canonical_source_intervals(units_list)
    if len(mapping) != QUALIFIED_UNIT_COUNT or set(mapping) != set(units):
        raise QualificationHarnessError("qualified CSWP chunk mapping is incomplete or inconsistent")
    return mapping


def hydrate_canonical_source_intervals(
    raw: dict[str, Any], *, canonical_source_map: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Resolve actual adapter rows to authoritative CSWP source intervals."""

    hydrated = dict(raw)
    for field in CANONICAL_INTERVAL_FIELDS:
        rows = raw.get(field)
        if not isinstance(rows, list):
            # Let the normal raw-output contract name the missing field below.
            continue
        hydrated_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                hydrated_rows.append(row)
                continue
            chunk_id = row.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                hydrated_rows.append(row)
                continue
            if chunk_id in seen:
                raise QualificationHarnessError(
                    f"{field} contains duplicate chunk_id: {chunk_id!r}"
                )
            seen.add(chunk_id)
            canonical = canonical_source_map.get(chunk_id)
            if canonical is None:
                raise QualificationHarnessError(
                    f"{field} contains unknown qualified chunk_id: {chunk_id!r}"
                )
            if row.get("source") not in (None, canonical["source"]):
                raise QualificationHarnessError(
                    f"{field} source disagrees with canonical mapping for {chunk_id!r}"
                )
            supplied = row.get("source_intervals")
            expected = canonical["source_intervals"]
            if supplied is not None:
                if not isinstance(supplied, list):
                    raise QualificationHarnessError(
                        f"{field} source intervals are invalid for {chunk_id!r}"
                    )
                try:
                    observed = tuple((int(interval[0]), int(interval[1])) for interval in supplied)
                except (IndexError, TypeError, ValueError):
                    raise QualificationHarnessError(
                        f"{field} source intervals are invalid for {chunk_id!r}"
                    ) from None
                if observed != expected:
                    raise QualificationHarnessError(
                        f"{field} source intervals disagree with canonical mapping for {chunk_id!r}"
                    )
            hydrated_rows.append(
                {
                    **row,
                    "source": canonical["source"],
                    "source_intervals": [list(interval) for interval in expected],
                }
            )
        hydrated[field] = hydrated_rows
    return hydrated


def canonical_raw(
    raw: dict[str, Any], *, canonical_source_map: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    if canonical_source_map is not None:
        raw = hydrate_canonical_source_intervals(
            raw, canonical_source_map=canonical_source_map
        )
    if (
        not isinstance(raw.get("packed_fingerprint"), str)
        or not raw["packed_fingerprint"]
    ):
        raise QualificationHarnessError("raw backend output missing packed fingerprint")
    return {
        "seeds": require_rows(raw, "retrieval_seeds"),
        "expanded": require_rows(raw, "expanded_rows"),
        "packed": require_rows(raw, "kb_results"),
        **{field: require_rank_rows(raw, field) for field in RANKING_PARITY_FIELDS},
        "hybrid_top10": require_rank_rows(raw, "hybrid_top10"),
        "packed_fingerprint": raw["packed_fingerprint"],
        "raw": raw,
    }


def compare_backends(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    mismatches = []
    for layer in ("seeds", "expanded", "packed"):
        a, b = (
            [r["chunk_id"] for r in left[layer]],
            [r["chunk_id"] for r in right[layer]],
        )
        if a != b:
            mismatches.append({"field": f"{layer}.chunk_id", "local": a, "qdrant": b})
    for field in RANKING_PARITY_FIELDS:
        a, b = [r["chunk_id"] for r in left[field]], [r["chunk_id"] for r in right[field]]
        if a != b:
            mismatches.append({"field": f"{field}.chunk_id", "local": a, "qdrant": b})
    if left["packed_fingerprint"] != right["packed_fingerprint"]:
        mismatches.append(
            {
                "field": "packed_evidence_fingerprint",
                "local": left["packed_fingerprint"],
                "qdrant": right["packed_fingerprint"],
            }
        )
    return mismatches


def runtime_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def expected_runtime_identity(
    fixture: dict[str, Any], contract: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Load the immutable identity that every raw result must prove."""

    explicit = fixture.get("expected_runtime_identity")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise QualificationHarnessError("expected runtime identity must be an object")
        expected = explicit
    else:
        from agent.cswp.loader import load_production_index
        from agent.qualified_rag import _runtime_identity
        from agent.shadow_qdrant.index import load_index_manifest

        _representation, units, _by_source = load_production_index()
        local = _runtime_identity(units)
        qdrant_manifest = load_index_manifest()
        expected = {
            "cswp_local": local,
            "cswp_qdrant": {
                **local,
                "collection_name": qdrant_manifest["collection_name"],
                # This is recomputed from local canonical payloads, not trusted
                # from qdrant_shadow_manifest.json.
                "live_collection_fingerprint": local["index_fingerprint"],
            },
        }

    for backend in BACKENDS:
        if not isinstance(expected.get(backend), dict):
            raise QualificationHarnessError(f"expected runtime identity missing {backend}")
        for field in REQUIRED_RUNTIME_IDENTITY[backend]:
            value = expected[backend].get(field)
            if not isinstance(value, str) or not value:
                raise QualificationHarnessError(
                    f"expected runtime identity missing {backend}.{field}"
                )
    required_contract = contract["provenance_gate"]["required_qualified_contract"]
    required_revision = contract["provenance_gate"]["expected_minilm_revision"]
    for backend in BACKENDS:
        if expected[backend]["qualified_contract"] != required_contract:
            raise QualificationHarnessError("expected qualified contract contradicts frozen contract")
        if expected[backend]["minilm_model_revision"] != required_revision:
            raise QualificationHarnessError("expected MiniLM revision contradicts frozen contract")
    return expected


def validate_real_adapter_preflight(
    expected: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate the qualified corpus and live serving target before query one."""

    from agent.cswp.loader import load_production_index
    from agent.kb_backend.qdrant_serving import validate_startup
    from agent.qualified_rag import _runtime_identity

    canonical_source_map = qualified_canonical_source_map()
    _representation, units, _by_source = load_production_index()
    local_identity = _runtime_identity(units)
    if local_identity != expected["cswp_local"]:
        raise QualificationHarnessError(
            "qualified local corpus/model/index identity disagrees with frozen expectation"
        )

    qdrant = validate_startup()
    if qdrant.get("point_count") != QUALIFIED_UNIT_COUNT:
        raise QualificationHarnessError(
            f"live Qdrant serving collection must contain exactly {QUALIFIED_UNIT_COUNT} units"
        )
    qdrant_identity = {
        **{
            field: qdrant.get(field)
            for field in REQUIRED_RUNTIME_IDENTITY["cswp_qdrant"]
            if field != "qualified_contract"
        },
        "qualified_contract": expected["cswp_qdrant"]["qualified_contract"],
    }
    if qdrant_identity != expected["cswp_qdrant"]:
        raise QualificationHarnessError(
            "live Qdrant alias, fingerprint, or model/index identity disagrees with frozen expectation"
        )
    return (
        canonical_source_map,
        {
            "cswp_local": {
                "unit_count": len(canonical_source_map),
                **local_identity,
            },
            "cswp_qdrant": qdrant,
        },
    )


def validate_runtime_identity(
    *,
    query_id: str,
    outputs: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, str]],
    frozen: dict[str, dict[str, str]] | None,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Fail closed on absent, cross-backend, or run-wide identity drift."""

    observed: dict[str, dict[str, str]] = {}
    failures: list[dict[str, Any]] = []
    for backend in BACKENDS:
        identity: dict[str, str] = {}
        for field in REQUIRED_RUNTIME_IDENTITY[backend]:
            value = outputs[backend].get(field)
            if not isinstance(value, str) or not value:
                failures.append(
                    {
                        "gate": "required_provenance_missing",
                        "detail": {"query_id": query_id, "backend": backend, "field": field},
                    }
                )
            else:
                identity[field] = value
                if value != expected[backend][field]:
                    failures.append(
                        {
                            "gate": "expected_provenance_mismatch",
                            "detail": {
                                "query_id": query_id,
                                "backend": backend,
                                "field": field,
                                "expected": expected[backend][field],
                                "observed": value,
                            },
                        }
                    )
        observed[backend] = identity

    for field in SHARED_RUNTIME_IDENTITY:
        left = observed["cswp_local"].get(field)
        right = observed["cswp_qdrant"].get(field)
        if left is not None and right is not None and left != right:
            failures.append(
                {
                    "gate": "backend_provenance_mismatch",
                    "detail": {"query_id": query_id, "field": field, "local": left, "qdrant": right},
                }
            )
    live = observed["cswp_qdrant"].get("live_collection_fingerprint")
    index = observed["cswp_qdrant"].get("index_fingerprint")
    if live is not None and index is not None and live != index:
        failures.append(
            {
                "gate": "live_content_fingerprint_mismatch",
                "detail": {"query_id": query_id, "live": live, "index": index},
            }
        )

    if frozen is None:
        frozen = observed
    else:
        for backend in BACKENDS:
            for field in REQUIRED_RUNTIME_IDENTITY[backend]:
                prior = frozen[backend].get(field)
                current = observed[backend].get(field)
                if prior is not None and current is not None and prior != current:
                    failures.append(
                        {
                            "gate": "run_wide_provenance_mismatch",
                            "detail": {
                                "query_id": query_id,
                                "backend": backend,
                                "field": field,
                                "frozen": prior,
                                "observed": current,
                            },
                        }
                    )
    return observed, failures, frozen


def provenance(
    fixture: dict[str, Any],
    fixture_sha256: str,
    contract: dict[str, Any],
    expected_runtime_identity: dict[str, dict[str, str]],
    frozen_runtime_identity: dict[str, dict[str, str]] | None,
    runtime_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = fixture.get("expected_provenance", {})
    observed = {
        "execution_git_sha": runtime_git_sha(),
        "fixture_sha256": fixture_sha256,
        "contract_sha256": sha256_json(contract),
        "backend_identity": list(BACKENDS),
        "expected_runtime_identity": expected_runtime_identity,
        "runtime_identity": frozen_runtime_identity,
    }
    bad = [key for key, value in expected.items() if observed.get(key) != value]
    return {
        "expected": expected,
        "observed": observed,
        "passed": not bad and not runtime_failures,
        "mismatches": bad + runtime_failures,
    }


def _diagnostic_available(value: Any, *, basis: str) -> dict[str, Any]:
    return {"status": "AVAILABLE", "value": value, "basis": basis}


def _diagnostic_unavailable(reason: str) -> dict[str, Any]:
    return {"status": "NOT_APPLICABLE", "value": None, "reason": reason}


def _numeric_top_value(
    rows: list[dict[str, Any]], field: str, *, basis: str
) -> dict[str, Any]:
    if not rows:
        return _diagnostic_unavailable(f"{basis} has no rows")
    value = rows[0].get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return _diagnostic_unavailable(f"{basis}[0].{field} is unavailable")
    return _diagnostic_available(round(float(value), 8), basis=f"{basis}[0].{field}")


def _hard_negative_match(label: dict[str, Any], row: dict[str, Any]) -> bool:
    """Match every identifier specified by a fixture-owned hard-negative label."""

    return all(
        label.get(field) is None or row.get(field) == label[field]
        for field in ("source", "chunk_id")
    )


def absent_diagnostics(query: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Report frozen ABSENT diagnostics from existing retrieval outputs only."""

    if query["answerability"] != "ABSENT":
        raise QualificationHarnessError("ABSENT diagnostics requested for evidence-bearing query")
    hybrid = raw["hybrid_top10"]
    dense = raw["dense_top20"]
    bm25 = raw["bm25_rank_order"]
    packed = raw["packed"]
    labels = query.get("hard_negatives", [])
    exposure: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        for layer, rows in (("retrieved_hybrid_top10", hybrid), ("expanded", raw["expanded"]), ("packed", packed)):
            for rank, row in enumerate(rows, start=1):
                if _hard_negative_match(label, row):
                    exposure.append(
                        {
                            "label_index": label_index,
                            "label": label,
                            "layer": layer,
                            "rank_or_output_order": rank,
                            "source": row["source"],
                            "chunk_id": row["chunk_id"],
                        }
                    )
    source_counts: dict[str, int] = {}
    for row in hybrid:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    if source_counts:
        max_count = max(source_counts.values())
        concentration: dict[str, Any] = _diagnostic_available(
            {
                "sequence": "hybrid_top10_ranking",
                "unique_source_count": len(source_counts),
                "max_source_count": max_count,
                "max_source_fraction": round(max_count / len(hybrid), 8),
                "dominant_sources": sorted(
                    source for source, count in source_counts.items() if count == max_count
                ),
            },
            basis="hybrid_top10 source frequency",
        )
    else:
        concentration = _diagnostic_unavailable("hybrid_top10 has no source rows")

    dense_top1_similarity = _numeric_top_value(
        dense, "native_score", basis="dense_top20"
    )
    if dense_top1_similarity["status"] == "AVAILABLE":
        dense_top1_distance = _diagnostic_available(
            round(1.0 - dense_top1_similarity["value"], 8),
            basis="1 - dense_top20[0].native_score (cosine similarity)",
        )
    else:
        dense_top1_distance = _diagnostic_unavailable(
            "dense_top1_similarity is unavailable"
        )
    dense_scores = [
        float(row["native_score"])
        for row in dense
        if isinstance(row.get("native_score"), (int, float))
        and math.isfinite(float(row["native_score"]))
    ]
    max_dense_similarity = (
        _diagnostic_available(
            round(max(dense_scores), 8), basis="max dense_top20.native_score"
        )
        if dense_scores
        else _diagnostic_unavailable("dense_top20.native_score is unavailable")
    )
    return {
        "interpretation": "retrieval_diagnostics_only_no_abstention_or_refusal_claim",
        "hard_negative_exposure": _diagnostic_available(
            {
                "observed": bool(exposure),
                "label_count": len(labels),
                "matches": exposure,
                "match_semantics": "all supplied label source/chunk_id identifiers match",
            },
            basis="fixture hard_negatives against existing retrieved/expanded/packed outputs",
        ),
        "source_concentration": concentration,
        "dense_top1_distance": dense_top1_distance,
        "dense_top1_similarity": dense_top1_similarity,
        "bm25_top_score": _numeric_top_value(
            bm25, "native_score", basis="bm25_rank_order"
        ),
        "hybrid_top1_rrf_score": _numeric_top_value(
            hybrid, "rrf_score", basis="hybrid_top10"
        ),
        "max_dense_similarity": max_dense_similarity,
        "retrieved_chunk_count": _diagnostic_available(
            len(hybrid), basis="hybrid_top10 ranked retrieval sequence"
        ),
        "packed_chunk_count": _diagnostic_available(
            len(packed), basis="seed-first packed output"
        ),
    }


def score_query(query: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    spans = gold_spans(query)
    seeds, expanded, packed = raw["seeds"], raw["expanded"], raw["packed"]
    hybrid_top10 = raw["hybrid_top10"]
    if len(hybrid_top10) < MRR_DEPTH:
        raise QualificationHarnessError("raw backend output missing a true hybrid top-10")
    answerability = query["answerability"]
    evidence_bearing = answerability in EVIDENCE_BEARING
    source_relevance = {
        source["source"]: int(source["grade"])
        for source in query.get("relevant_sources", [])
    }

    def evidence_at(rows: list[dict[str, Any]], k: int) -> float | None:
        return recall(spans, rows, k) if evidence_bearing else None

    def evidence_layer(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not evidence_bearing:
            return {
                "population": "NOT_APPLICABLE_ABSENT",
                "evidence_recall_at": {str(k): None for k in SECONDARY_K_VALUES},
                "evidence_recall_full": None,
                "source_rank_metrics": None,
            }
        values: dict[str, Any] = {
            "population": "evidence_bearing_exact_spans",
            "evidence_recall_at": {
                str(k): evidence_at(rows, k) for k in SECONDARY_K_VALUES
            },
            "evidence_recall_full": recall(spans, rows, len(rows)),
        }
        values["source_rank_metrics"] = {
            "population": "relevant_sources_grade",
            "source_recall_at": {
                str(k): round(source_recall_at_k(rows, source_relevance, k), 8)
                for k in SECONDARY_K_VALUES
            },
            "mrr_at_10": round(
                reciprocal_rank(rows, source_relevance, MRR_DEPTH), 8
            ),
            "graded_ndcg_at": {
                str(k): round(graded_ndcg_at_k(rows, source_relevance, k), 8)
                for k in SECONDARY_K_VALUES
            },
        }
        return values

    retrieved_secondary = evidence_layer(hybrid_top10)
    expanded_secondary = evidence_layer(expanded)
    packed_secondary = evidence_layer(packed)
    return {
        "query_id": query["query_id"],
        "answerability": answerability,
        "coverage": {
            "retrieved": coverage(spans, seeds, 5),
            "expanded": coverage(spans, expanded, len(expanded)),
            "packed": coverage(spans, packed, len(packed)),
        },
        "metrics": {
            "evidence_recall_at_1": retrieved_secondary["evidence_recall_at"]["1"],
            "evidence_recall_at_3": retrieved_secondary["evidence_recall_at"]["3"],
            "evidence_recall_at_5": retrieved_secondary["evidence_recall_at"]["5"],
            "expanded_evidence_recall": expanded_secondary["evidence_recall_full"],
            "packed_evidence_recall": packed_secondary["evidence_recall_full"],
            "source_recall": None
            if not evidence_bearing
            else retrieved_secondary["source_rank_metrics"]["source_recall_at"]["5"],
            "mrr": None
            if not evidence_bearing
            else retrieved_secondary["source_rank_metrics"]["mrr_at_10"],
            "ndcg": None
            if not evidence_bearing
            else retrieved_secondary["source_rank_metrics"]["graded_ndcg_at"]["5"],
        },
        "secondary_metrics": {
            "retrieved": {
                "sequence": "hybrid_top10_ranking",
                "sequence_semantics": "true hybrid retrieval ranking",
                **retrieved_secondary,
            },
            "expanded": {
                "sequence": "seed_first_expansion_output",
                "sequence_semantics": (
                    "deterministic expansion output order; positions are not retrieval ranks"
                ),
                **expanded_secondary,
            },
            "packed": {
                "sequence": "seed_first_packed_output",
                "sequence_semantics": (
                    "deterministic seed-first packing order; positions are not retrieval ranks"
                ),
                **packed_secondary,
            },
        },
        "absent_diagnostics": absent_diagnostics(query, raw)
        if answerability == "ABSENT"
        else None,
        "raw_identities": {
            "seed_chunk_ids": [r["chunk_id"] for r in seeds],
            "expanded_chunk_ids": [r["chunk_id"] for r in expanded],
            "packed_chunk_ids": [r["chunk_id"] for r in packed],
            "dense_top20_chunk_ids": [r["chunk_id"] for r in raw["dense_top20"]],
            "bm25_rank_order_chunk_ids": [
                r["chunk_id"] for r in raw["bm25_rank_order"]
            ],
            "hybrid_seed_top5_chunk_ids": [
                r["chunk_id"] for r in raw["hybrid_seed_top5"]
            ],
            "hybrid_top10_chunk_ids": [r["chunk_id"] for r in hybrid_top10],
            "packed_evidence_fingerprint": raw["packed_fingerprint"],
        },
    }


def deterministic_backend_snapshot(
    raw: dict[str, Any], score: dict[str, Any], backend: str
) -> dict[str, Any]:
    """Capture all archived deterministic evidence for one backend/query."""

    return {
        "rankings": {
            "dense_top20_chunk_ids": [row["chunk_id"] for row in raw["dense_top20"]],
            "bm25_rank_order_chunk_ids": [
                row["chunk_id"] for row in raw["bm25_rank_order"]
            ],
            "hybrid_top5_chunk_ids": [
                row["chunk_id"] for row in raw["hybrid_seed_top5"]
            ],
            "hybrid_top10_chunk_ids": [
                row["chunk_id"] for row in raw["hybrid_top10"]
            ],
            "seed_chunk_ids": [row["chunk_id"] for row in raw["seeds"]],
            "expanded_chunk_ids": [row["chunk_id"] for row in raw["expanded"]],
            "packed_chunk_ids": [row["chunk_id"] for row in raw["packed"]],
        },
        "packed_evidence_fingerprint": raw["packed_fingerprint"],
        "runtime_identity": {
            field: raw["raw"].get(field)
            for field in REQUIRED_RUNTIME_IDENTITY[backend]
        },
        "per_query_scoring": score,
        "absent_diagnostics": score["absent_diagnostics"],
    }


def deterministic_execution_snapshot(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Make query/backend evidence directly comparable across two executions."""

    return {
        row["query_id"]: row["backends"]
        for row in rows
    }


def deterministic_mismatches(
    first: Any, second: Any, *, query_id: str | None = None,
    backend: str | None = None, field: str = "",
) -> list[dict[str, Any]]:
    """Report ordered, backend-scoped deterministic differences without masking them."""

    if isinstance(first, dict) and isinstance(second, dict):
        mismatches = []
        for key in sorted(set(first) | set(second)):
            child_field = f"{field}.{key}" if field else key
            if key not in first or key not in second:
                mismatches.append(
                    {
                        "query_id": query_id,
                        "backend": backend,
                        "field": child_field,
                        "execution_1": first.get(key),
                        "execution_2": second.get(key),
                    }
                )
            else:
                mismatches.extend(
                    deterministic_mismatches(
                        first[key], second[key], query_id=query_id,
                        backend=backend, field=child_field,
                    )
                )
        return mismatches
    if isinstance(first, list) and isinstance(second, list):
        mismatches = []
        if len(first) != len(second):
            mismatches.append(
                {
                    "query_id": query_id,
                    "backend": backend,
                    "field": f"{field}.length",
                    "execution_1": len(first),
                    "execution_2": len(second),
                }
            )
        for index, (left, right) in enumerate(zip(first, second)):
            mismatches.extend(
                deterministic_mismatches(
                    left, right, query_id=query_id, backend=backend,
                    field=f"{field}[{index}]",
                )
            )
        return mismatches
    if first != second:
        return [{
            "query_id": query_id,
            "backend": backend,
            "field": field,
            "execution_1": first,
            "execution_2": second,
        }]
    return []


def compare_deterministic_executions(
    first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare every archived deterministic field per query and backend."""

    mismatches = []
    for query_id in sorted(set(first) | set(second)):
        first_backends = first.get(query_id)
        second_backends = second.get(query_id)
        if not isinstance(first_backends, dict) or not isinstance(second_backends, dict):
            mismatches.append(
                {
                    "query_id": query_id,
                    "backend": None,
                    "field": "backend_snapshot",
                    "execution_1": first_backends,
                    "execution_2": second_backends,
                }
            )
            continue
        for backend in BACKENDS:
            mismatches.extend(
                deterministic_mismatches(
                    first_backends.get(backend), second_backends.get(backend),
                    query_id=query_id, backend=backend,
                )
            )
    return mismatches


def evaluate_evidence_gates(
    queries: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    contract: dict[str, Any],
    parity: list[dict[str, Any]],
    prov: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate neutral retrieval evidence against the frozen metric gates.

    This shared routine deliberately returns no release disposition.  A
    successful evidence assessment is not a holdout qualification: only the
    public authoritative entry point may turn it into a release decision.
    """
    failures = []
    if parity:
        failures.append({"gate": "backend_identity_mismatch", "detail": parity})
    if not prov["passed"]:
        failures.append({"gate": "provenance_mismatch", "detail": prov})
    for query, score in zip(queries, scores):
        if score["answerability"] not in EVIDENCE_BEARING:
            continue
        metrics, cov = score["metrics"], score["coverage"]
        if (
            score["answerability"] == "PARTIAL"
            and metrics["packed_evidence_recall"] != 1.0
        ):
            failures.append(
                {
                    "gate": "partial_available_recall_below_one",
                    "detail": score["query_id"],
                }
            )
        if metrics["packed_evidence_recall"] < metrics["evidence_recall_at_5"]:
            failures.append(
                {"gate": "packed_below_retrieved_recall", "detail": score["query_id"]}
            )
        grade2 = {
            f"{s['source']}::{s['span_id']}"
            for s in gold_spans(query)
            if s["grade"] == 2
        }
        lost = (set(cov["retrieved"]) - set(cov["packed"])) & grade2
        if lost:
            failures.append(
                {
                    "gate": "critical_grade2_lost_at_packed",
                    "detail": {"query_id": score["query_id"], "spans": sorted(lost)},
                }
            )
    evidence = [s for s in scores if s["answerability"] in EVIDENCE_BEARING]
    answerable = [s for s in scores if s["answerability"] == "ANSWERABLE"]
    macro = (
        round(fmean(s["metrics"]["packed_evidence_recall"] for s in evidence), 8)
        if evidence
        else 0.0
    )
    ans = (
        round(fmean(s["metrics"]["packed_evidence_recall"] for s in answerable), 8)
        if answerable
        else 0.0
    )
    if macro < float(contract["primary_metric"]["floor"]):
        failures.append({"gate": "macro_packed_below_floor", "detail": macro})
    if answerable and ans < float(
        contract["stratum_metrics"]["answerable_packed_macro"]["floor"]
    ):
        failures.append({"gate": "answerable_macro_below_floor", "detail": ans})
    return {
        "primary_metric": {
            "observed": macro,
            "floor": contract["primary_metric"]["floor"],
        },
        "answerable_packed_macro": {
            "observed": ans,
            "floor": contract["stratum_metrics"]["answerable_packed_macro"]["floor"],
        },
        "hard_failures": failures,
        "overall_pass": not failures,
    }


def reserve_archive(root: Path) -> None:
    """Atomically claim the immutable run identity before executing retrieval."""

    try:
        root.mkdir(parents=True)
    except FileExistsError as exc:
        raise QualificationHarnessError(
            f"archive already exists at {root}; fail closed before retrieval"
        ) from exc


def append_execution_journal(path: Path, record: dict[str, Any]) -> None:
    """Durably append one non-secret qualification observation."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sanitized_failure(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


REQUIRED_ARCHIVE_ARTIFACTS = frozenset(
    {
        "backend_per_query.json",
        "per_query.json",
        "summary.json",
        "parity.json",
        "raw_backend_outputs.json",
        "execution_2_raw_backend_outputs.json",
        "executions.json",
        "provenance.json",
        "contract.json",
        "determinism.json",
        "disposition.json",
        "execution_journal.jsonl",
        "oracle_fixture.json",
    }
)
REQUIRED_ARCHIVE_RAW_FIELDS = frozenset((*CANONICAL_INTERVAL_FIELDS, "packed_fingerprint"))


def _archive_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_archive_scores(scores: Any, query_ids: list[str]) -> bool:
    if not isinstance(scores, list) or len(scores) != len(query_ids):
        return False
    observed = [score.get("query_id") for score in scores if isinstance(score, dict)]
    if observed != query_ids or len(observed) != len(set(observed)):
        return False
    return all(
        isinstance(score.get("metrics"), dict)
        and isinstance(score.get("coverage"), dict)
        and isinstance(score.get("raw_identities"), dict)
        and isinstance(score.get("answerability"), str)
        for score in scores
    )


def _complete_archive_backend_scores(scores: Any, query_ids: list[str]) -> bool:
    """Require the structured score emitted for each backend and query."""

    if not isinstance(scores, list) or len(scores) != len(query_ids):
        return False
    if [row.get("query_id") for row in scores if isinstance(row, dict)] != query_ids:
        return False
    return all(
        isinstance(row.get("backends"), dict)
        and set(row["backends"]) == set(BACKENDS)
        and all(
            isinstance(row["backends"][backend], dict)
            and row["backends"][backend].get("query_id") == row["query_id"]
            and isinstance(row["backends"][backend].get("metrics"), dict)
            and isinstance(row["backends"][backend].get("coverage"), dict)
            and isinstance(row["backends"][backend].get("raw_identities"), dict)
            and isinstance(row["backends"][backend].get("answerability"), str)
            for backend in BACKENDS
        )
        for row in scores
    )


def _complete_archive_raw(raw: Any, query_ids: list[str]) -> bool:
    if not isinstance(raw, dict) or set(raw) != set(BACKENDS):
        return False
    return all(
        isinstance(rows, list)
        and len(rows) == len(query_ids)
        and all(
            isinstance(row, dict) and REQUIRED_ARCHIVE_RAW_FIELDS <= set(row)
            for row in rows
        )
        for rows in raw.values()
    )


def _complete_archive_parity(parity: Any) -> bool:
    return (
        isinstance(parity, dict)
        and parity.get("passed") is True
        and parity.get("mismatches") == []
    )


def _raw_identity_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    """Derive the score-visible identities from canonical archived raw output."""

    return {
        "seed_chunk_ids": [row["chunk_id"] for row in raw["seeds"]],
        "expanded_chunk_ids": [row["chunk_id"] for row in raw["expanded"]],
        "packed_chunk_ids": [row["chunk_id"] for row in raw["packed"]],
        "dense_top20_chunk_ids": [row["chunk_id"] for row in raw["dense_top20"]],
        "bm25_rank_order_chunk_ids": [
            row["chunk_id"] for row in raw["bm25_rank_order"]
        ],
        "hybrid_seed_top5_chunk_ids": [
            row["chunk_id"] for row in raw["hybrid_seed_top5"]
        ],
        "hybrid_top10_chunk_ids": [
            row["chunk_id"] for row in raw["hybrid_top10"]
        ],
        "packed_evidence_fingerprint": raw["packed_fingerprint"],
    }


def archived_deterministic_execution_snapshot(
    scores: list[dict[str, Any]],
    backend_scores: list[dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
    query_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Rebuild runtime determinism evidence from archived raw/structured output.

    No stored determinism snapshot is consulted here.  The archive supplies
    one raw backend result and one structured backend score for every query;
    the canonical runtime snapshot helper then defines the exact semantics.
    """

    if not (
        _complete_archive_scores(scores, query_ids)
        and _complete_archive_backend_scores(backend_scores, query_ids)
        and _complete_archive_raw(raw, query_ids)
    ):
        raise QualificationHarnessError("malformed archived deterministic evidence")
    rebuilt: dict[str, dict[str, Any]] = {}
    for index, query_id in enumerate(query_ids):
        row = backend_scores[index]
        if scores[index] != row["backends"]["cswp_local"]:
            raise ArchiveEvidenceConflict("local score disagrees with backend score evidence")
        rebuilt[query_id] = {}
        for backend in BACKENDS:
            canonical = canonical_raw(raw[backend][index])
            score = row["backends"][backend]
            if score.get("raw_identities") != _raw_identity_snapshot(canonical):
                raise ArchiveEvidenceConflict(
                    "structured score disagrees with archived raw evidence"
                )
            rebuilt[query_id][backend] = deterministic_backend_snapshot(
                canonical, score, backend
            )
    return rebuilt


def recompute_archived_score_truth(
    fixture: dict[str, Any],
    scores: list[dict[str, Any]],
    backend_scores: list[dict[str, Any]],
    raw: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
    provenance_record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-score one archived execution solely from verified oracle bytes/raw rows.

    This deliberately reuses ``score_query`` and ``evaluate_evidence_gates``:
    the archive verifier is not an independently maintained metric engine.
    """

    recomputed_scores = []
    parity = []
    for index, query in enumerate(fixture["queries"]):
        canonical = {
            backend: canonical_raw(raw[backend][index]) for backend in BACKENDS
        }
        recomputed_backends = {
            backend: score_query(query, canonical[backend]) for backend in BACKENDS
        }
        if scores[index] != recomputed_backends["cswp_local"]:
            raise ArchiveEvidenceConflict("archived local score is not raw/oracle truth")
        if backend_scores[index] != {
            "query_id": query["query_id"], "backends": recomputed_backends
        }:
            raise ArchiveEvidenceConflict("archived backend scores are not raw/oracle truth")
        recomputed_scores.append(recomputed_backends["cswp_local"])
        parity.extend(
            {"query_id": query["query_id"], **m}
            for m in compare_backends(
                canonical["cswp_local"], canonical["cswp_qdrant"]
            )
        )
    return recomputed_scores, parity


def archive_verification_failure(root: Path) -> str | None:
    """Return a bounded reason when an archive cannot prove authoritative PASS.

    The verifier is deliberately read-only.  It treats the terminal manifest as
    an integrity envelope, then proves the archived attempts, score evidence,
    parity, determinism, and approved identities agree with that envelope.
    """

    try:
        actual = {
            path.name
            for path in root.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        if not REQUIRED_ARCHIVE_ARTIFACTS <= actual or not (root / "manifest.json").is_file():
            return "missing_required_artifact"
        archive = {name: _archive_json(root / name) for name in REQUIRED_ARCHIVE_ARTIFACTS if name != "execution_journal.jsonl"}
        manifest = _archive_json(root / "manifest.json")
        journal = [json.loads(line) for line in (root / "execution_journal.jsonl").read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError, TypeError):
        return "unreadable_required_artifact"

    if not isinstance(manifest, dict):
        return "invalid_terminal_manifest"
    inventory = manifest.get("artifacts")
    if not isinstance(inventory, dict) or set(inventory) != actual:
        return "manifest_inventory_mismatch"
    try:
        for name, detail in inventory.items():
            path = root / name
            if (
                not isinstance(detail, dict)
                or detail.get("sha256") != sha256_file(path)
                or detail.get("bytes") != path.stat().st_size
            ):
                return "artifact_digest_mismatch"
    except OSError:
        return "artifact_digest_mismatch"

    summary = archive["summary.json"]
    disposition_record = archive["disposition.json"]
    aggregate = summary.get("aggregate") if isinstance(summary, dict) else None
    disposition_aggregate = (
        disposition_record.get("aggregate") if isinstance(disposition_record, dict) else None
    )
    frozen_pass = load_contract()["disposition_values"][0]
    terminal_values = (manifest, summary, disposition_record)
    if not all(isinstance(value, dict) for value in terminal_values):
        return "invalid_terminal_record"
    if (
        manifest.get("terminal_artifact") != "manifest.json"
        or manifest.get("manifest_inventory_excludes_self") is not True
        or any(value.get("terminal_status") != "COMPLETE" for value in terminal_values)
        or any(value.get("overall_pass") is not True for value in terminal_values)
        or any(value.get("disposition") != frozen_pass for value in terminal_values)
    ):
        return "terminal_outcome_not_authoritative_pass"
    if (
        not isinstance(aggregate, dict)
        or not isinstance(disposition_aggregate, dict)
        or aggregate.get("overall_pass") is not True
        or aggregate.get("disposition") != frozen_pass
        or aggregate.get("hard_failures") != []
        or disposition_aggregate.get("overall_pass") is not True
        or disposition_aggregate.get("disposition") != frozen_pass
    ):
        return "aggregate_disposition_disagreement"

    attempt = manifest.get("attempt_identity")
    if not isinstance(attempt, dict):
        return "missing_attempt_identity"
    query_ids = attempt.get("query_ids")
    if (
        attempt.get("execution_count") != 2
        or attempt.get("backends") != list(BACKENDS)
        or not isinstance(query_ids, list)
        or not query_ids
        or any(not isinstance(query_id, str) for query_id in query_ids)
        or len(query_ids) != len(set(query_ids))
    ):
        return "invalid_attempt_identity"

    contract = archive["contract.json"]
    provenance_record = archive["provenance.json"]
    authorization = summary.get("authorization")
    contract_identity = summary.get("contract_identity")
    if (
        not isinstance(contract, dict)
        or sha256_json(contract) != EXPECTED_CONTRACT_SHA256
        or not isinstance(provenance_record, dict)
        or provenance_record.get("passed") is not True
        or provenance_record.get("mismatches") != []
        or not isinstance(authorization, dict)
        or not all(isinstance(authorization.get(key), str) and authorization[key] for key in (
            "approved_execution_sha", "approved_fixture_sha256"
        ))
        or contract_identity != manifest.get("contract_identity")
        or contract_identity != attempt.get("contract_identity")
        or authorization != attempt.get("authorization")
        or provenance_record.get("observed") != attempt.get("provenance")
    ):
        return "authorization_or_provenance_mismatch"
    observed = provenance_record["observed"]
    if (
        not isinstance(observed, dict)
        or observed.get("execution_git_sha") != authorization["approved_execution_sha"]
        or observed.get("fixture_sha256") != authorization["approved_fixture_sha256"]
        or observed.get("fixture_sha256") != attempt.get("fixture_sha256")
        or observed.get("contract_sha256") != contract_identity.get("canonical_json_sha256")
        or contract_identity.get("canonical_json_sha256") != sha256_json(contract)
    ):
        return "authorization_or_provenance_mismatch"

    try:
        oracle_snapshot = load_fixture_snapshot(root / "oracle_fixture.json")
        validate_fixture(oracle_snapshot.fixture)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, QualificationHarnessError):
        return "oracle_digest_mismatch"
    if (
        oracle_snapshot.sha256 != authorization["approved_fixture_sha256"]
        or oracle_snapshot.sha256 != attempt.get("fixture_sha256")
        or oracle_snapshot.sha256 != observed.get("fixture_sha256")
        or [query.get("query_id") for query in oracle_snapshot.fixture["queries"]]
        != query_ids
    ):
        return "oracle_digest_mismatch"

    per_query = archive["per_query.json"]
    backend_per_query = archive["backend_per_query.json"]
    parity = archive["parity.json"]
    executions = archive["executions.json"]
    raw_by_execution = (
        archive["raw_backend_outputs.json"],
        archive["execution_2_raw_backend_outputs.json"],
    )
    if (
        not isinstance(per_query, dict)
        or not isinstance(parity, dict)
        or not isinstance(executions, list)
        or len(executions) != 2
        or not _complete_archive_parity(parity.get("execution_1"))
        or not _complete_archive_parity(parity.get("execution_2"))
    ):
        return "missing_or_failed_execution_evidence"
    for number, raw in enumerate(raw_by_execution, start=1):
        scores = per_query.get(f"execution_{number}")
        backend_scores = backend_per_query.get(f"execution_{number}") if isinstance(backend_per_query, dict) else None
        execution = executions[number - 1]
        if (
            not _complete_archive_scores(scores, query_ids)
            or not _complete_archive_backend_scores(backend_scores, query_ids)
            or not _complete_archive_raw(raw, query_ids)
            or not isinstance(execution, dict)
            or execution.get("execution") != number
            or execution.get("status") != "COMPLETE"
            or execution.get("failure") is not None
            or execution.get("per_query") != scores
            or execution.get("backend_per_query") != backend_scores
            or execution.get("raw_backend_outputs") != raw
            or execution.get("backend_parity") != parity.get(f"execution_{number}")
            or execution.get("provenance") != provenance_record
        ):
            return "incomplete_execution_evidence"

    determinism = archive["determinism.json"]
    if (
        not isinstance(determinism, dict)
    ):
        return "missing_determinism_evidence"
    if not all(isinstance(determinism.get(key), dict) for key in ("execution_1", "execution_2")):
        return "missing_determinism_evidence"
    try:
        rebuilt = [
            archived_deterministic_execution_snapshot(
                per_query[f"execution_{number}"],
                backend_per_query[f"execution_{number}"],
                raw,
                query_ids,
            )
            for number, raw in enumerate(raw_by_execution, start=1)
        ]
    except ArchiveEvidenceConflict:
        return "execution_snapshot_conflict"
    except (KeyError, TypeError, QualificationHarnessError):
        return "malformed_deterministic_field"
    stored = (determinism.get("execution_1"), determinism.get("execution_2"))
    if any(
        deterministic_mismatches(snapshot, evidence)
        for snapshot, evidence in zip(stored, rebuilt, strict=True)
    ):
        return "execution_snapshot_conflict"
    if compare_deterministic_executions(rebuilt[0], rebuilt[1]):
        return "archived_determinism_mismatch"
    if (
        determinism.get("passed") is not True
        or determinism.get("mismatches") != []
        or parity.get("determinism") != determinism
    ):
        return "determinism_assertion_conflict"
    recomputed_aggregates = []
    for number, raw in enumerate(raw_by_execution, start=1):
        try:
            recomputed_scores, recomputed_parity = recompute_archived_score_truth(
                oracle_snapshot.fixture,
                per_query[f"execution_{number}"],
                backend_per_query[f"execution_{number}"], raw, contract,
                provenance_record,
            )
        except (ArchiveEvidenceConflict, KeyError, TypeError, QualificationHarnessError):
            return "score_truth_mismatch"
        expected_parity = {"passed": not recomputed_parity, "mismatches": recomputed_parity}
        if parity.get(f"execution_{number}") != expected_parity:
            return "qualification_gate_mismatch"
        recomputed_aggregates.append(
            evaluate_evidence_gates(
                oracle_snapshot.fixture["queries"], recomputed_scores, contract,
                recomputed_parity, provenance_record,
            )
        )
    expected_aggregate = {
        **recomputed_aggregates[0],
        "disposition": frozen_pass if recomputed_aggregates[0]["overall_pass"] else None,
    }
    if (
        recomputed_aggregates[0] != recomputed_aggregates[1]
        or aggregate != expected_aggregate
        or disposition_aggregate != {
            "overall_pass": expected_aggregate["overall_pass"],
            "disposition": expected_aggregate["disposition"],
        }
    ):
        return "aggregate_score_mismatch"
    if not isinstance(journal, list) or not all(isinstance(row, dict) for row in journal):
        return "invalid_execution_journal"
    for number in (1, 2):
        completed = [
            row for row in journal
            if row.get("event") == "query_completed" and row.get("execution") == number
        ]
        terminal = [
            row for row in journal
            if row.get("event") == "execution_completed" and row.get("execution") == number
        ]
        if (
            [row.get("query_id") for row in completed] != query_ids
            or len(terminal) != 1
            or terminal[0].get("terminal_status") != "COMPLETE"
        ):
            return "incomplete_execution_journal"
    if any(row.get("event") in {"execution_incomplete", "archive_finalization_incomplete"} for row in journal):
        return "incomplete_execution_journal"
    return None


def archive_is_successful(root: Path) -> bool:
    """Return true only when the archive semantically proves authoritative PASS."""

    return archive_verification_failure(root) is None


def write_qualification_archive(
    payload: dict[str, Any], root: Path, *, fixture_snapshot: FixtureSnapshot,
    already_reserved: bool = False
) -> dict[str, str]:
    """Create a final immutable archive after all retrieval has completed.

    The manifest is created last and is the terminal completion witness.  An
    archive without it (for example, after abrupt process termination) is not
    interpretable as a successful qualification.
    """

    if not already_reserved:
        reserve_archive(root)
    snapshot = fixture_snapshot
    if snapshot.sha256 != payload["provenance"]["observed"]["fixture_sha256"]:
        raise QualificationHarnessError("archive fixture snapshot disagrees with provenance")
    oracle_path = root / "oracle_fixture.json"
    with oracle_path.open("xb") as handle:
        handle.write(snapshot.raw_bytes)
    values = {
        "per_query": {
            "execution_1": payload["per_query"],
            "execution_2": payload.get("execution_2", {}).get("per_query"),
        },
        "backend_per_query": {
            "execution_1": payload["backend_per_query"],
            "execution_2": payload.get("execution_2", {}).get("backend_per_query"),
        },
        "summary": {
            "terminal_status": payload["terminal_status"],
            "overall_pass": payload["overall_pass"],
            "disposition": payload["disposition"],
            "aggregate": payload["aggregate"],
            "authorization": payload.get("authorization"),
            "contract_identity": payload["contract_identity"],
        },
        "parity": {
            "execution_1": payload["backend_parity"],
            "execution_2": payload.get("execution_2", {}).get("backend_parity"),
            "determinism": payload["determinism"],
        },
        "raw_backend_outputs": payload["raw_backend_outputs"],
        "execution_2_raw_backend_outputs": payload.get("execution_2", {}).get(
            "raw_backend_outputs"
        ),
        "executions": payload["executions"],
        "provenance": payload["provenance"],
        "contract": payload["contract"],
        "determinism": payload["determinism"],
        "disposition": {
            "terminal_status": payload["terminal_status"],
            "disposition": payload["disposition"],
            "overall_pass": payload["overall_pass"],
            "aggregate": {
                "overall_pass": payload["aggregate"]["overall_pass"],
                "disposition": payload["aggregate"]["disposition"],
            },
        },
    }
    paths = {"oracle_fixture": str(oracle_path)}
    for name, value in values.items():
        path = root / f"{name}.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json_dumps(value) + "\n")
        paths[name] = str(path)
    inventory = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(root.iterdir())
        if path.is_file()
    }
    manifest = {
        "terminal_status": payload["terminal_status"],
        "overall_pass": payload["overall_pass"],
        "disposition": payload["disposition"],
        "artifacts": inventory,
        "manifest_inventory_excludes_self": True,
        "terminal_artifact": "manifest.json",
        "contract_identity": payload["contract_identity"],
        "attempt_identity": {
            "execution_count": len(payload["executions"]),
            "query_ids": [score["query_id"] for score in payload["per_query"]],
            "backends": list(BACKENDS),
            "fixture_sha256": snapshot.sha256,
            "authorization": payload.get("authorization"),
            "provenance": payload["provenance"]["observed"],
            "contract_identity": payload["contract_identity"],
        },
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json_dumps(manifest) + "\n")
    paths["manifest"] = str(manifest_path)
    return paths


def execute_qualified_backends(query: str) -> dict[str, dict[str, Any]]:
    from agent.kb_backend import local, qdrant_serving

    return {
        "cswp_local": local.retrieve(query, n_seeds=5),
        "cswp_qdrant": qdrant_serving.retrieve(query, n_seeds=5),
    }


def _run_evaluation(
    *,
    fixture_path: Path,
    contract_path: Path | None = None,
    executor: Callable[[str], dict[str, dict[str, Any]]] = execute_qualified_backends,
    determinism_executor: Callable[[str], dict[str, dict[str, Any]]] | None = None,
    canonical_source_map: dict[str, dict[str, Any]] | None = None,
    fixture_snapshot: FixtureSnapshot | None = None,
    _execution_number: int = 1,
) -> dict[str, Any]:
    """Run reusable retrieval evaluation mechanics and return neutral evidence.

    No argument can select release authority, a release archive, preflight
    policy, or a HOLDOUT disposition.  Callers may use injected executors for
    development-only evaluation, but those results are evidence, not release
    decisions.
    """
    contract = load_contract(contract_path)
    resolved_contract_path = contract_path or DEFAULT_CONTRACT_PATH
    contract_identity = {
        "canonical_json_sha256": sha256_json(contract),
        "contract_file_sha256": sha256_file(resolved_contract_path),
        "historical_embedded_digest": contract.get("contract_sha256"),
    }
    snapshot = fixture_snapshot or load_fixture_snapshot(fixture_path)
    fixture = snapshot.fixture
    validate_fixture(fixture)
    expected_identity = expected_runtime_identity(fixture, contract)
    if executor is execute_qualified_backends and canonical_source_map is None:
        canonical_source_map = qualified_canonical_source_map()
    raw = {name: [] for name in BACKENDS}
    scores = []
    backend_per_query: list[dict[str, Any]] = []
    determinism_rows: list[dict[str, Any]] = []
    parity = []
    frozen_runtime_identity = None
    runtime_failures: list[dict[str, Any]] = []
    execution_failure: dict[str, Any] | None = None
    for query in fixture["queries"]:
        outputs: Any = None
        stage = "backend_execution"
        try:
            outputs = executor(query["query"])
            # Preserve what the adapters returned before any validation or
            # canonicalization can reject it.  Successful rows are replaced
            # below with their canonicalized form; failed rows remain raw.
            if isinstance(outputs, dict):
                for name in BACKENDS:
                    if name in outputs:
                        raw[name].append(outputs[name])
            stage = "backend_output_shape"
            if not isinstance(outputs, dict) or set(outputs) != set(BACKENDS):
                raise QualificationHarnessError("both raw backend objects are required")
            stage = "canonicalization"
            canonical = {
                name: canonical_raw(
                    outputs[name], canonical_source_map=canonical_source_map
                )
                for name in BACKENDS
            }
            stage = "runtime_identity"
            _observed, failures, frozen_runtime_identity = validate_runtime_identity(
                query_id=query["query_id"],
                outputs={name: canonical[name]["raw"] for name in BACKENDS},
                expected=expected_identity,
                frozen=frozen_runtime_identity,
            )
            runtime_failures.extend(failures)
            for name in BACKENDS:
                raw[name][-1] = canonical[name]["raw"]
            stage = "backend_parity"
            parity.extend(
                [
                    {"query_id": query["query_id"], **m}
                    for m in compare_backends(
                        canonical["cswp_local"], canonical["cswp_qdrant"]
                    )
                ]
            )
            # Identity is checked before this query can affect scoring.
            stage = "scoring"
            backend_scores = {
                name: score_query(query, canonical[name]) for name in BACKENDS
            }
            score = backend_scores["cswp_local"]
            scores.append(score)
            backend_per_query.append(
                {"query_id": query["query_id"], "backends": backend_scores}
            )
            determinism_rows.append(
                {
                    "query_id": query["query_id"],
                    "backends": {
                        name: deterministic_backend_snapshot(
                            canonical[name], backend_scores[name], name
                        )
                        for name in BACKENDS
                    },
                }
            )
        except Exception as exc:
            execution_failure = {
                "query_id": query["query_id"],
                "stage": stage,
                "failure": sanitized_failure(exc),
            }
            # Evidence collection is neutral even when an execution is
            # incomplete.  The caller receives the partial raw evidence and a
            # failed assessment; only the authoritative owner decides how to
            # persist or dispose of that result.
            break
    prov = provenance(
        fixture,
        snapshot.sha256,
        contract,
        expected_identity,
        frozen_runtime_identity,
        runtime_failures,
    )
    aggregate = evaluate_evidence_gates(fixture["queries"], scores, contract, parity, prov)
    if execution_failure is not None:
        aggregate["hard_failures"].append(
            {"gate": "execution_incomplete", "detail": execution_failure}
        )
    payload = {
        "contract_identity": contract_identity,
        "fixture_sha256": snapshot.sha256,
        "raw_backend_outputs": raw,
        "per_query": scores,
        "backend_per_query": backend_per_query,
        "determinism_evidence": determinism_rows,
        "backend_parity": {"passed": not parity, "mismatches": parity},
        "provenance": prov,
        "aggregate": aggregate,
        **aggregate,
    }
    payload["execution"] = {
        "execution": _execution_number,
        "status": "INCOMPLETE" if execution_failure else "COMPLETE",
        "failure": execution_failure,
        "raw_backend_outputs": raw,
        "per_query": scores,
        "backend_per_query": backend_per_query,
        "backend_parity": payload["backend_parity"],
        "provenance": prov,
    }
    payload["executions"] = [payload["execution"]]
    canonical_payload = {
        "backend_evidence": deterministic_execution_snapshot(determinism_rows),
    }
    payload["determinism"] = {
        "execution_1": canonical_payload["backend_evidence"],
        "run_a_fingerprint": sha256_json(canonical_payload),
    }
    if _execution_number == 1 and execution_failure is None and determinism_executor is not None:
        try:
            repeat = _run_evaluation(
                fixture_path=fixture_path, contract_path=contract_path,
                executor=determinism_executor, fixture_snapshot=snapshot,
                _execution_number=2, canonical_source_map=canonical_source_map,
            )
            payload["execution_2"] = repeat
            payload["executions"].append(repeat["execution"])
            run_b = {
                "backend_evidence": deterministic_execution_snapshot(
                    repeat["determinism_evidence"]
                ),
            }
            mismatches = compare_deterministic_executions(
                canonical_payload["backend_evidence"], run_b["backend_evidence"]
            )
            payload["determinism"]["execution_2"] = run_b["backend_evidence"]
            payload["determinism"]["run_b_fingerprint"] = sha256_json(run_b)
            payload["determinism"]["mismatches"] = mismatches
            payload["determinism"]["passed"] = (
                repeat["terminal_status"] == "COMPLETE" and not mismatches
            )
            if repeat["terminal_status"] == "INCOMPLETE":
                payload["determinism"].update(
                    {"status": "INCOMPLETE", "failure": repeat["execution"]["failure"]}
                )
                payload["aggregate"]["hard_failures"].append(
                    {"gate": "second_execution_incomplete", "detail": repeat["execution"]["failure"]}
                )
            elif not payload["determinism"]["passed"]:
                payload["aggregate"]["hard_failures"].append(
                    {"gate": "determinism_mismatch", "detail": payload["determinism"]}
                )
        except Exception as exc:
            payload["determinism"].update(
                {"passed": False, "status": "INCOMPLETE", "error": sanitized_failure(exc)}
            )
            payload["aggregate"]["hard_failures"].append(
                {"gate": "second_execution_incomplete", "detail": sanitized_failure(exc)}
            )
    elif _execution_number == 1 and execution_failure is not None:
        payload["determinism"].update(
            {"passed": False, "status": "INCOMPLETE", "failure": execution_failure}
        )
    payload["aggregate"]["overall_pass"] = not payload["aggregate"]["hard_failures"]
    payload["overall_pass"] = payload["aggregate"]["overall_pass"]
    payload["terminal_status"] = (
        "INCOMPLETE"
        if execution_failure is not None or payload["determinism"].get("status") == "INCOMPLETE"
        else "COMPLETE"
    )
    return payload


def run_development_evaluation(
    *,
    fixture_path: Path,
    contract_path: Path | None = None,
    executor: Callable[[str], dict[str, dict[str, Any]]] = execute_qualified_backends,
    determinism_executor: Callable[[str], dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run a non-authoritative development evaluation.

    This API intentionally accepts synthetic fixtures and injected executors.
    Its disposition is permanently scoped to development and it cannot reserve
    or write a release archive.
    """

    evidence = _run_evaluation(
        fixture_path=fixture_path,
        contract_path=contract_path,
        executor=executor,
        determinism_executor=determinism_executor,
    )
    evidence["decision_scope"] = "development"
    evidence["aggregate"]["disposition"] = (
        DEVELOPMENT_PASS_DISPOSITION
        if evidence["overall_pass"]
        else DEVELOPMENT_FAILURE_DISPOSITION
    )
    evidence["disposition"] = evidence["aggregate"]["disposition"]
    return evidence


def run_qualification(
    *,
    fixture_path: Path,
    contract_path: Path | None = None,
    executor: Callable[[str], dict[str, dict[str, Any]]] = execute_qualified_backends,
    determinism_executor: Callable[[str], dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Backward-compatible name for :func:`run_development_evaluation`.

    Qualification callers that need a release disposition must use the
    argument-free authoritative API below; this compatibility function cannot
    be elevated through invocation flags or caller-provided authority.
    """

    return run_development_evaluation(
        fixture_path=fixture_path,
        contract_path=contract_path,
        executor=executor,
        determinism_executor=determinism_executor,
    )


def _load_authoritative_holdout_snapshot() -> FixtureSnapshot:
    """Load only the designated fixture and verify its external release identity."""

    if not AUTHORITATIVE_HOLDOUT_PATH.is_file():
        raise QualificationHarnessError(
            "designated authoritative holdout fixture is unavailable; fail closed before retrieval"
        )
    snapshot = load_fixture_snapshot(AUTHORITATIVE_HOLDOUT_PATH)
    fixture = snapshot.fixture
    if fixture.get("schema_version") != AUTHORITATIVE_HOLDOUT_SCHEMA:
        raise QualificationHarnessError("designated authoritative holdout has an invalid schema")
    if fixture.get("holdout") is not True:
        raise QualificationHarnessError("designated authoritative fixture is not marked as holdout")
    if snapshot.sha256 != AUTHORITATIVE_HOLDOUT_SHA256:
        raise QualificationHarnessError(
            "designated authoritative holdout byte SHA-256 does not match release approval"
        )
    validate_fixture(fixture)
    return snapshot


def approved_authoritative_execution_sha(fixture: dict[str, Any]) -> str:
    """Read the execution SHA from a fixture already verified by byte digest.

    The value is release authority only after ``_load_authoritative_holdout_snapshot``
    establishes the exact fixture bytes against the independently approved
    digest.  It therefore cannot be supplied or changed by an invocation
    caller, while avoiding an impossible source-file self-hash pin.
    """

    value = fixture.get("approved_execution_sha")
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise QualificationHarnessError(
            "designated authoritative holdout is missing an approved execution Git SHA"
        )
    return value


def run_authoritative_qualification(
    *, archive_root: Path | None = None
) -> dict[str, Any]:
    """Run the only authoritative Phase-5 qualification path.

    Fixture selection, runtime SHA, adapters, and real preflight are release
    authority, not caller options.  The absence of the sealed fixture is an
    expected fail-closed state until the separately authorized integration.
    """

    snapshot = _load_authoritative_holdout_snapshot()
    approved_execution_sha = approved_authoritative_execution_sha(snapshot.fixture)
    if runtime_git_sha() != approved_execution_sha:
        raise QualificationHarnessError(
            "runtime HEAD does not match the approved authoritative execution SHA"
        )
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    expected_identity = expected_runtime_identity(snapshot.fixture, contract)
    # This must finish before the archive is reserved and before query one.
    canonical_source_map, preflight = validate_real_adapter_preflight(expected_identity)
    resolved_archive = archive_root or DEFAULT_ARCHIVE_ROOT
    reserve_archive(resolved_archive)
    evidence = _run_evaluation(
        fixture_path=AUTHORITATIVE_HOLDOUT_PATH,
        contract_path=DEFAULT_CONTRACT_PATH,
        executor=execute_qualified_backends,
        determinism_executor=execute_qualified_backends,
        fixture_snapshot=snapshot,
        canonical_source_map=canonical_source_map,
    )
    evidence["decision_scope"] = "authoritative_release"
    evidence["preflight"] = preflight
    evidence["authorization"] = {
        "approved_execution_sha": approved_execution_sha,
        "approved_fixture_sha256": AUTHORITATIVE_HOLDOUT_SHA256,
    }
    evidence["contract"] = contract
    evidence["aggregate"]["disposition"] = (
        contract["disposition_values"][0]
        if evidence["overall_pass"]
        else contract["disposition_values"][1]
    )
    evidence["disposition"] = evidence["aggregate"]["disposition"]

    journal = resolved_archive / "execution_journal.jsonl"
    for execution in evidence["executions"]:
        number = execution["execution"]
        append_execution_journal(journal, {"event": "execution_started", "execution": number})
        for index, score in enumerate(execution["per_query"]):
            query = snapshot.fixture["queries"][index]
            append_execution_journal(journal, {
                "event": "query_completed",
                "execution": number,
                "query_id": query["query_id"],
                "raw_backend_outputs": {
                    name: execution["raw_backend_outputs"][name][index]
                    for name in BACKENDS
                },
                "per_query": score,
            })
        if execution["failure"] is not None:
            append_execution_journal(journal, {
                "event": "execution_incomplete",
                "execution": number,
                "raw_backend_outputs": (
                    {
                        name: execution["raw_backend_outputs"][name][-1]
                        for name in BACKENDS
                    }
                    if all(execution["raw_backend_outputs"][name] for name in BACKENDS)
                    else None
                ),
                **execution["failure"],
            })
        append_execution_journal(journal, {
            "event": "execution_completed",
            "execution": number,
            "terminal_status": execution["status"],
            "overall_pass": evidence["overall_pass"],
            "disposition": evidence["disposition"],
        })
    evidence["archive_paths"] = write_qualification_archive(
        evidence, resolved_archive, fixture_snapshot=snapshot, already_reserved=True
    )
    if evidence["overall_pass"] and not archive_is_successful(resolved_archive):
        raise QualificationHarnessError(
            "authoritative PASS requires a completed, verified release archive"
        )
    return evidence


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-root", type=Path)
    a = p.parse_args(argv)
    report = run_authoritative_qualification(archive_root=a.archive_root)
    print(canonical_json_dumps(report))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
