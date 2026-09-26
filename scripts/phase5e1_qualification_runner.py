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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONTRACT_PATH = (
    REPO_ROOT / "evals/fixtures/phase5e1_qualification_contract.json"
)
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "reports/phase5/phase5e1a/holdout-run-1"
EXPECTED_CONTRACT_SHA256 = (
    "7c19cb7b0ae6126be19816615a8976e0685be18b1f8686c31658338afdf38b47"
)
BACKENDS = ("cswp_local", "cswp_qdrant")
EVIDENCE_BEARING = frozenset(("ANSWERABLE", "PARTIAL"))
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


def load_fixture_snapshot(path: Path) -> FixtureSnapshot:
    """Read, hash, and parse exactly one immutable fixture-byte snapshot."""

    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise QualificationHarnessError(f"{path}: expected JSON object")
    return FixtureSnapshot(value, hashlib.sha256(raw).hexdigest())


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


def canonical_raw(raw: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(raw.get("packed_fingerprint"), str)
        or not raw["packed_fingerprint"]
    ):
        raise QualificationHarnessError("raw backend output missing packed fingerprint")
    return {
        "seeds": require_rows(raw, "retrieval_seeds"),
        "expanded": require_rows(raw, "expanded_rows"),
        "packed": require_rows(raw, "kb_results"),
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


def score_query(query: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    spans = gold_spans(query)
    seeds, expanded, packed = raw["seeds"], raw["expanded"], raw["packed"]
    return {
        "query_id": query["query_id"],
        "answerability": query["answerability"],
        "coverage": {
            "retrieved": coverage(spans, seeds, 5),
            "expanded": coverage(spans, expanded, len(expanded)),
            "packed": coverage(spans, packed, len(packed)),
        },
        "metrics": {
            "evidence_recall_at_1": recall(spans, seeds, 1),
            "evidence_recall_at_3": recall(spans, seeds, 3),
            "evidence_recall_at_5": recall(spans, seeds, 5),
            "expanded_evidence_recall": recall(spans, expanded, len(expanded)),
            "packed_evidence_recall": recall(spans, packed, len(packed)),
            "source_recall": source_recall(spans, seeds, 5),
            "mrr": mrr(spans, seeds, 5),
            "ndcg": ndcg(spans, seeds, 5),
        },
        "raw_identities": {
            "seed_chunk_ids": [r["chunk_id"] for r in seeds],
            "expanded_chunk_ids": [r["chunk_id"] for r in expanded],
            "packed_chunk_ids": [r["chunk_id"] for r in packed],
            "packed_evidence_fingerprint": raw["packed_fingerprint"],
        },
    }


def disposition(
    queries: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    contract: dict[str, Any],
    parity: list[dict[str, Any]],
    prov: dict[str, Any],
) -> dict[str, Any]:
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
        "disposition": contract["disposition_values"][0]
        if not failures
        else contract["disposition_values"][1],
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


def write_qualification_archive(
    payload: dict[str, Any], root: Path, *, already_reserved: bool = False
) -> dict[str, str]:
    if not already_reserved:
        reserve_archive(root)
    values = {
        "raw_backend_outputs": payload["raw_backend_outputs"],
        "per_query": payload["per_query"],
        "aggregate": payload["aggregate"],
        "provenance": payload["provenance"],
        "contract": payload["contract"],
        "disposition": {
            "disposition": payload["disposition"],
            "overall_pass": payload["overall_pass"],
        },
    }
    paths = {}
    for name, value in values.items():
        path = root / f"{name}.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json_dumps(value) + "\n")
        paths[name] = str(path)
    return paths


def execute_qualified_backends(query: str) -> dict[str, dict[str, Any]]:
    from agent.kb_backend import local, qdrant_serving

    return {
        "cswp_local": local.retrieve(query, n_seeds=5),
        "cswp_qdrant": qdrant_serving.retrieve(query, n_seeds=5),
    }


def run_qualification(
    *,
    fixture_path: Path,
    contract_path: Path | None = None,
    archive_root: Path | None = None,
    write_archive: bool = False,
    executor: Callable[[str], dict[str, dict[str, Any]]] = execute_qualified_backends,
    determinism_executor: Callable[[str], dict[str, dict[str, Any]]] | None = None,
    authoritative: bool = False,
    approved_execution_sha: str | None = None,
    approved_fixture_sha256: str | None = None,
    _fixture_snapshot: FixtureSnapshot | None = None,
    _journal: Path | None = None,
    _execution_number: int = 1,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if write_archive and not authoritative:
        raise QualificationHarnessError("archive qualification requires authoritative authorization")
    if authoritative:
        if not approved_execution_sha:
            raise QualificationHarnessError("authoritative qualification requires approved execution SHA")
        if approved_execution_sha != runtime_git_sha():
            raise QualificationHarnessError("approved execution SHA does not match runtime HEAD")
        if not approved_fixture_sha256:
            raise QualificationHarnessError("authoritative qualification requires approved fixture SHA-256")
    snapshot = _fixture_snapshot or load_fixture_snapshot(fixture_path)
    fixture = snapshot.fixture
    if authoritative and approved_fixture_sha256 != snapshot.sha256:
        raise QualificationHarnessError("approved fixture SHA-256 does not match fixture bytes")
    validate_fixture(fixture)
    expected_identity = expected_runtime_identity(fixture, contract)
    resolved_archive = archive_root or DEFAULT_ARCHIVE_ROOT
    if write_archive:
        reserve_archive(resolved_archive)
        _journal = resolved_archive / "execution_journal.jsonl"
    if _journal is not None:
        append_execution_journal(_journal, {"event": "execution_started", "execution": _execution_number})
    raw = {name: [] for name in BACKENDS}
    scores = []
    parity = []
    frozen_runtime_identity = None
    runtime_failures: list[dict[str, Any]] = []
    for query in fixture["queries"]:
        try:
            outputs = executor(query["query"])
        except Exception as exc:
            if _journal is not None:
                append_execution_journal(_journal, {
                    "event": "execution_incomplete", "execution": _execution_number,
                    "query_id": query["query_id"], "failure": sanitized_failure(exc),
                })
            raise
        if set(outputs) != set(BACKENDS):
            raise QualificationHarnessError("both raw backend objects are required")
        canonical = {name: canonical_raw(outputs[name]) for name in BACKENDS}
        _observed, failures, frozen_runtime_identity = validate_runtime_identity(
            query_id=query["query_id"],
            outputs={name: canonical[name]["raw"] for name in BACKENDS},
            expected=expected_identity,
            frozen=frozen_runtime_identity,
        )
        runtime_failures.extend(failures)
        for name in BACKENDS:
            raw[name].append(canonical[name]["raw"])
        parity.extend(
            [
                {"query_id": query["query_id"], **m}
                for m in compare_backends(
                    canonical["cswp_local"], canonical["cswp_qdrant"]
                )
            ]
        )
        # Identity is checked before this query can affect scoring.
        score = score_query(query, canonical["cswp_local"])
        scores.append(score)
        if _journal is not None:
            append_execution_journal(_journal, {
                "event": "query_completed", "execution": _execution_number,
                "query_id": query["query_id"],
                "raw_backend_outputs": {name: canonical[name]["raw"] for name in BACKENDS},
                "per_query": score,
            })
    prov = provenance(
        fixture,
        snapshot.sha256,
        contract,
        expected_identity,
        frozen_runtime_identity,
        runtime_failures,
    )
    aggregate = disposition(fixture["queries"], scores, contract, parity, prov)
    payload = {
        "contract": contract,
        "fixture_sha256": snapshot.sha256,
        "raw_backend_outputs": raw,
        "per_query": scores,
        "backend_parity": {"passed": not parity, "mismatches": parity},
        "provenance": prov,
        "aggregate": aggregate,
        **aggregate,
    }
    # Authoritative qualification always performs two complete executions.
    if authoritative and determinism_executor is None:
        determinism_executor = executor
    canonical_payload = {
        "per_query": payload["per_query"],
        "backend_parity": payload["backend_parity"],
        "provenance": payload["provenance"],
        "runtime_identity": payload["provenance"]["observed"]["runtime_identity"],
    }
    payload["determinism"] = {"run_a_fingerprint": sha256_json(canonical_payload)}
    if _journal is not None and _execution_number == 1:
        append_execution_journal(
            _journal,
            {"event": "execution_queries_completed", "execution": _execution_number},
        )
    if determinism_executor is not None:
        try:
            repeat = run_qualification(
                fixture_path=fixture_path, contract_path=contract_path,
                executor=determinism_executor, _fixture_snapshot=snapshot,
                _journal=_journal, _execution_number=2,
            )
            run_b = {
                "per_query": repeat["per_query"],
                "backend_parity": repeat["backend_parity"],
                "provenance": repeat["provenance"],
                "runtime_identity": repeat["provenance"]["observed"]["runtime_identity"],
            }
            payload["determinism"]["run_b_fingerprint"] = sha256_json(run_b)
            payload["determinism"]["passed"] = (
                payload["determinism"]["run_a_fingerprint"]
                == payload["determinism"]["run_b_fingerprint"]
            )
            if not payload["determinism"]["passed"]:
                payload["aggregate"]["hard_failures"].append(
                    {"gate": "determinism_mismatch", "detail": payload["determinism"]}
                )
        except Exception as exc:
            payload["determinism"].update({"passed": False, "status": "INCOMPLETE", "error": str(exc)})
            payload["aggregate"]["hard_failures"].append(
                {"gate": "second_execution_incomplete", "detail": str(exc)}
            )
    if authoritative and not payload["determinism"].get("passed"):
        payload["aggregate"]["hard_failures"].append(
            {"gate": "second_execution_missing", "detail": payload["determinism"]}
        )
    payload["aggregate"]["overall_pass"] = not payload["aggregate"]["hard_failures"]
    payload["aggregate"]["disposition"] = contract["disposition_values"][0] if payload["aggregate"]["overall_pass"] else contract["disposition_values"][1]
    payload["overall_pass"] = payload["aggregate"]["overall_pass"]
    payload["disposition"] = payload["aggregate"]["disposition"]
    if _journal is not None:
        append_execution_journal(_journal, {
            "event": "execution_completed", "execution": _execution_number,
            "overall_pass": payload["overall_pass"], "disposition": payload["disposition"],
        })
    if write_archive:
        payload["archive_paths"] = write_qualification_archive(
            payload, resolved_archive, already_reserved=True
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    p.add_argument("--write-archive", action="store_true")
    p.add_argument("--archive-root", type=Path)
    p.add_argument("--approved-execution-sha", required=True)
    p.add_argument("--approved-fixture-sha256", required=True)
    a = p.parse_args(argv)
    report = run_qualification(
        fixture_path=a.fixture,
        contract_path=a.contract,
        archive_root=a.archive_root,
        write_archive=a.write_archive,
        authoritative=True,
        approved_execution_sha=a.approved_execution_sha,
        approved_fixture_sha256=a.approved_fixture_sha256,
    )
    print(canonical_json_dumps(report))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
