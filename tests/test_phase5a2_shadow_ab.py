"""Phase 5A2 causal controls, metrics, artifacts, and offline boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from scripts import phase5a2_shadow_ab as experiment


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports/phase5/phase5a2"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def test_manifest_freezes_only_representation_as_intended_variable():
    manifest = load("experiment_manifest.json")
    held = manifest["held_constant"]
    assert manifest["required_parent"] == experiment.STARTING_HEAD
    assert manifest["causal_variable"] == "document representation"
    assert held["embedding_model_revision"] == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert held["fusion_config"] == {
        "algorithm": "reciprocal rank fusion over chunk_id",
        "candidate_k": 20,
        "final_k": 10,
        "rank_origin": 0,
        "rrf_constant": 60,
        "tie_break": "rrf_score descending, chunk_id ascending",
    }
    assert manifest["arms"]["A"]["vector_count"] == 73
    assert manifest["arms"]["B"]["vector_count"] == 510
    assert manifest["arms"]["B"]["representation_fingerprint"] == (
        "a9402fea0c4fbbae6613346370c0759b0239d3d870a386526980e1c40815e852"
    )
    assert manifest["candidate_b_retrieval_text_grammar"].startswith(
        "title + LF + join(heading_path"
    )
    assert manifest["production_retrieval_changed"] is False
    assert manifest["production_qdrant_reads"] == 0
    assert manifest["production_qdrant_writes"] == 0
    assert manifest["provider_calls"] == 0
    assert manifest["external_network_calls"] == 0
    declared = manifest.pop("manifest_sha256")
    assert experiment.sha256_json(manifest) == declared


def test_aggregate_report_has_all_required_metrics_and_no_composite_winner():
    report = load("aggregate_comparison.json")
    for arm in ("A", "B"):
        for channel in ("dense", "bm25", "hybrid"):
            metrics = report["arms"][arm][channel][
                "gating_33_excludes_q25_q26"
            ]
            for k in experiment.K_VALUES:
                assert f"recall@{k}" in metrics
                assert f"precision@{k}" in metrics
                assert f"ndcg@{k}" in metrics
                assert f"source_recall@{k}" in metrics
                assert f"evidence_span_recall@{k}" in metrics
            assert "mrr@10" in metrics
    assert report["single_composite_score"] is None
    assert report["absence_metrics"] is None
    a = report["arms"]["A"]["hybrid"]["gating_33_excludes_q25_q26"]
    b = report["arms"]["B"]["hybrid"]["gating_33_excludes_q25_q26"]
    assert a["evidence_span_recall@5"] == 0.84848485
    assert b["evidence_span_recall@5"] == 0.51515152
    assert a["mrr@10"] == 0.97979798
    assert b["mrr@10"] == 0.96969697
    assert a["ndcg@5"] == 0.97094024
    assert b["ndcg@5"] == 0.95807764


def test_every_gating_query_has_A_B_results_and_conservative_classification():
    report = load("per_query_comparison.json")
    gating = [row for row in report["queries"] if row["gating_eligible"]]
    assert len(gating) == 33
    assert {row["classification"] for row in gating} <= {
        "IMPROVED",
        "SAME",
        "REGRESSED",
    }
    for row in gating:
        for channel in ("dense", "bm25", "hybrid"):
            comparison = row["channels"][channel]
            assert set(comparison) == {
                "A",
                "B",
                "metric_deltas_B_minus_A",
                "classification",
            }
    by_id = {row["query_id"]: row for row in report["queries"]}
    assert by_id["Q15"]["classification"] == "SAME"
    assert by_id["Q19"]["classification"] == "REGRESSED"
    assert by_id["Q21"]["classification"] == "REGRESSED"
    assert by_id["Q22"]["classification"] == "REGRESSED"
    assert by_id["Q07"]["classification"] == "REGRESSED"
    assert by_id["Q23"]["classification"] == "REGRESSED"


def test_two_repetitions_have_identical_deterministic_hashes():
    first = load("run-1/determinism_report.json")
    second = load("run-2/determinism_report.json")
    runtime = load("runtime_summary.json")
    assert first["deterministic_result_sha256"] == second[
        "deterministic_result_sha256"
    ]
    assert first["component_sha256"] == second["component_sha256"]
    assert runtime["deterministic_rerun_equal"] is True
    assert runtime["deterministic_result_sha256"] == first[
        "deterministic_result_sha256"
    ]


def test_corpus_growth_and_crowding_are_quantified_not_used_as_sole_rejection():
    effect = load("aggregate_comparison.json")["corpus_size_effect"]
    assert effect["vector_count_ratio"] == 6.98630137
    assert effect["embedding_work"]["delta"] == 437
    assert effect["float32_vector_bytes"]["delta"] == 671232
    assert effect["small_child_crowding_query_count"] > 0
    assert effect["hybrid_top10_average"]["B"]["unique_sources"] < effect[
        "hybrid_top10_average"
    ]["A"]["unique_sources"]
    assert effect["hybrid_top10_average"]["B"]["exact_duplicate_text_pairs"] == 0


def test_every_meaningful_regression_has_evidence_or_explicit_unknown():
    report = load("failure_analysis.json")
    aggregate = load("aggregate_comparison.json")
    assert report["regression_count"] == len(
        aggregate["gating_queries"]["regressed"]
    )
    for row in report["regressions"]:
        assert row["likely_causes_supported"]
        assert row["support"]
        if row["likely_causes_supported"] == ["unknown"]:
            assert "does not isolate a narrower cause" in row["support"][0]


@dataclass(frozen=True)
class Span:
    source: str
    char_start: int
    char_end: int


def test_exact_span_metric_uses_source_interval_union_and_preserves_gaps():
    chunks = [
        experiment.EvalChunk(0, "a", "s", "s.md", "one", ((0, 5),), 1),
        experiment.EvalChunk(1, "b", "s", "s.md", "two", ((5, 10),), 1),
    ]
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    assert experiment._span_covered(Span("s", 2, 8), rows, by_id, 2)
    gapped = [
        experiment.EvalChunk(0, "a", "s", "s.md", "one", ((0, 4),), 1),
        experiment.EvalChunk(1, "b", "s", "s.md", "two", ((5, 10),), 1),
    ]
    assert not experiment._span_covered(
        Span("s", 2, 8),
        rows,
        {chunk.chunk_id: chunk for chunk in gapped},
        2,
    )


def test_regression_precedence_never_hides_a_mixed_query():
    baseline = {key: 1.0 for key in experiment.QUALITY_KEYS}
    candidate = dict(baseline)
    candidate["evidence_span_recall@5"] = 0.5
    candidate["mrr@10"] = 1.1
    assert experiment._classification(baseline, candidate) == "REGRESSED"


def test_script_has_no_production_or_qdrant_imports():
    source = (ROOT / "scripts/phase5a2_shadow_ab.py").read_text(encoding="utf-8")
    assert "from agent" not in source
    assert "import agent" not in source
    assert "qdrant_client" not in source


def test_cli_offline_guard_fails_closed_before_external_socket_use():
    code = (
        "from scripts.phase5a2_shadow_ab import install_offline_guard;"
        "install_offline_guard();import socket;"
        'socket.create_connection(("example.com",443))'
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "OFFLINE_NETWORK_FORBIDDEN" in result.stderr
