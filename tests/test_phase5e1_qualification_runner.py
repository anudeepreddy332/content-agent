"""Phase-5E1E raw authority tests; no sealed fixture, network, or providers."""

from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
import scripts.phase5e1_qualification_runner as runner
from scripts.phase5e1_qualification_runner import (
    QualificationHarnessError,
    archive_is_successful,
    contract_sha256,
    load_contract,
    reserve_archive,
    run_qualification,
    score_query,
    write_qualification_archive,
)
from scripts.phase5a0_baseline import graded_ndcg_at_k

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "evals/fixtures/phase5e1_qualification_synthetic_oracle.json"
CONTRACT = ROOT / "evals/fixtures/phase5e1_qualification_contract.json"


@pytest.fixture(autouse=True)
def isolated_default_release_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner, "DEFAULT_ARCHIVE_ROOT", tmp_path / "default-authoritative-archive"
    )


def row(source, chunk, start=0, end=20):
    return {"source": source, "chunk_id": chunk, "source_intervals": [[start, end]]}


def executor(*, packed_mismatch=False, packed_loss=False):
    def call(query):
        source = {
            "synthetic answerable one": "doc-a",
            "synthetic answerable two": "doc-b",
            "synthetic answerable three": "doc-c",
            "synthetic partial one": "doc-p",
            "synthetic absent diagnostic": "doc-hn",
        }[query]
        rows = [row(source, source + "__0001")]
        hybrid_top10 = rows + [
            row(f"background-{rank}", f"background-{rank}__0001")
            for rank in range(2, 11)
        ]
        for rank, ranked in enumerate(hybrid_top10, start=1):
            ranked["rrf_score"] = round(1 / (60 + rank - 1), 8)
        rows[0]["native_score"] = 0.9
        packed = [] if packed_loss and source == "doc-a" else rows
        base = {
            "retrieval_seeds": rows,
            "dense_top20": rows,
            "bm25_rank_order": rows,
            "hybrid_seed_top5": rows,
            "hybrid_top10": hybrid_top10,
            "expanded_rows": rows,
            "packed_rows": rows,
            "kb_results": packed,
            "packed_fingerprint": "fp-" + source,
            "source_corpus_fingerprint": "corpus",
            "index_fingerprint": "index",
            "minilm_model_id": "all-MiniLM-L6-v2",
            "minilm_model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "qualified_contract": "DRAFTER_PACKED_EVIDENCE_V1",
            "collection_name": "qualified",
            "live_collection_fingerprint": "index",
        }
        other = dict(base)
        if packed_mismatch:
            other = {
                **other,
                "kb_results": [row(source, source + "__DIFF")],
                "packed_fingerprint": "different",
            }
        return {"cswp_local": base, "cswp_qdrant": other}

    return call


def test_contract_is_fixed():
    assert contract_sha256(load_contract(CONTRACT)) == contract_sha256()


def test_raw_coverage_and_metrics_are_executable():
    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=executor()
    )
    assert report["overall_pass"] is True
    assert report["per_query"][0]["metrics"]["mrr"] == 1.0
    assert report["per_query"][0]["metrics"]["ndcg"] == 1.0
    assert report["per_query"][0]["metrics"]["source_recall"] == 1.0
    assert report["decision_scope"] == "development"
    assert report["disposition"] == runner.DEVELOPMENT_PASS_DISPOSITION
    assert report["disposition"] != load_contract(CONTRACT)["disposition_values"][0]


def _adapter_shape_without_intervals(chunk_id="chunk-a", source="doc-a"):
    row = {"chunk_id": chunk_id, "source": source}
    return {
        "retrieval_seeds": [row],
        "expanded_rows": [row],
        "kb_results": [row],
        "dense_top20": [row],
        "bm25_rank_order": [row],
        "hybrid_seed_top5": [row],
        "hybrid_top10": [row],
        "packed_fingerprint": "adapter-shape",
    }


def test_actual_adapter_shape_hydrates_exact_canonical_intervals():
    raw = _adapter_shape_without_intervals()
    canonical = runner.canonical_raw(
        raw,
        canonical_source_map={
            "chunk-a": {"source": "doc-a", "source_intervals": ((7, 21),)}
        },
    )
    assert raw["retrieval_seeds"][0].get("source_intervals") is None
    assert canonical["seeds"] == [
        {"chunk_id": "chunk-a", "source": "doc-a", "source_intervals": [[7, 21]]}
    ]
    assert canonical["hybrid_top10"][0]["source_intervals"] == [[7, 21]]


@pytest.mark.parametrize(
    ("field", "row", "message"),
    [
        ("retrieval_seeds", {"chunk_id": "missing", "source": "doc-a"}, "unknown qualified"),
        ("retrieval_seeds", {"chunk_id": "chunk-a", "source": "wrong"}, "source disagrees"),
        (
            "hybrid_top10",
            {"chunk_id": "chunk-a", "source": "doc-a", "source_intervals": [[8, 21]]},
            "intervals disagree",
        ),
    ],
)
def test_canonical_adapter_mapping_fails_closed(field, row, message):
    raw = _adapter_shape_without_intervals()
    raw[field] = [row]
    with pytest.raises(QualificationHarnessError, match=message):
        runner.canonical_raw(
            raw,
            canonical_source_map={
                "chunk-a": {"source": "doc-a", "source_intervals": ((7, 21),)}
            },
        )


def test_158_unit_corpus_rejects_before_mapping(monkeypatch):
    monkeypatch.setattr(
        "agent.cswp.loader.load_production_index",
        lambda *_args, **_kwargs: ({}, {str(i): {} for i in range(158)}, {}),
    )
    monkeypatch.setattr(
        "agent.cswp.loader.load_units_jsonl", lambda _path: [{} for _ in range(158)]
    )
    with pytest.raises(QualificationHarnessError, match="exactly 159"):
        runner.qualified_canonical_source_map()


def _metric_raw(hybrid_top10):
    return {
        "seeds": hybrid_top10[:5],
        "expanded": hybrid_top10,
        "packed": hybrid_top10,
        "dense_top20": hybrid_top10,
        "bm25_rank_order": hybrid_top10,
        "hybrid_seed_top5": hybrid_top10[:5],
        "hybrid_top10": hybrid_top10,
        "packed_fingerprint": "metric-fixture",
    }


def test_secondary_metrics_use_source_grade_and_credit_duplicate_source_once():
    query = {
        "query_id": "metric-grades",
        "answerability": "ANSWERABLE",
        "relevant_sources": [
            {
                "source": "high-source",
                "grade": 2,
                "evidence": [{"span_id": "high:1", "grade": 1, "char_start": 0, "char_end": 20}],
            },
            {
                "source": "low-source",
                "grade": 1,
                "evidence": [{"span_id": "low:1", "grade": 2, "char_start": 0, "char_end": 20}],
            },
        ],
    }
    hybrid_top10 = [
        row("high-source", "high-1"),
        row("high-source", "high-2"),
        row("low-source", "low-1"),
        *[row(f"noise-{rank}", f"noise-{rank}") for rank in range(4, 11)],
    ]
    score = score_query(query, _metric_raw(hybrid_top10))
    ranked = score["secondary_metrics"]["retrieved"]["source_rank_metrics"]
    expected = round(
        graded_ndcg_at_k(hybrid_top10, {"high-source": 2, "low-source": 1}, 3), 8
    )
    assert ranked["source_recall_at"] == {"1": 0.5, "3": 1.0, "5": 1.0}
    assert ranked["graded_ndcg_at"]["3"] == expected
    assert ranked["graded_ndcg_at"]["3"] < 1.0


def test_mrr_at_10_uses_existing_hybrid_rank_seven_not_five_seeds():
    query = {
        "query_id": "metric-rank-seven",
        "answerability": "ANSWERABLE",
        "relevant_sources": [
            {
                "source": "relevant",
                "grade": 2,
                "evidence": [{"span_id": "relevant:1", "grade": 2, "char_start": 0, "char_end": 20}],
            }
        ],
    }
    hybrid_top10 = [
        *[row(f"noise-{rank}", f"noise-{rank}") for rank in range(1, 7)],
        row("relevant", "relevant-7"),
        *[row(f"noise-{rank}", f"noise-{rank}") for rank in range(8, 11)],
    ]
    score = score_query(query, _metric_raw(hybrid_top10))
    retrieved = score["secondary_metrics"]["retrieved"]
    assert retrieved["source_rank_metrics"]["mrr_at_10"] == pytest.approx(1 / 7)
    assert score["metrics"]["mrr"] == pytest.approx(1 / 7)


def test_layer_metrics_and_absent_na_preserve_primary_population():
    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=executor()
    )
    answerable = report["per_query"][0]
    assert answerable["metrics"]["evidence_recall_at_5"] == answerable[
        "secondary_metrics"
    ]["retrieved"]["evidence_recall_at"]["5"]
    assert answerable["metrics"]["expanded_evidence_recall"] == answerable[
        "secondary_metrics"
    ]["expanded"]["evidence_recall_full"]
    assert answerable["metrics"]["packed_evidence_recall"] == answerable[
        "secondary_metrics"
    ]["packed"]["evidence_recall_full"]
    assert answerable["secondary_metrics"]["expanded"]["source_rank_metrics"][
        "status"
    ] == "NOT_APPLICABLE"
    absent = next(row for row in report["per_query"] if row["answerability"] == "ABSENT")
    assert absent["metrics"]["evidence_recall_at_1"] is None
    assert absent["metrics"]["packed_evidence_recall"] is None
    assert absent["secondary_metrics"]["retrieved"]["evidence_recall_at"] == {
        "1": None,
        "3": None,
        "5": None,
    }
    assert report["aggregate"]["primary_metric"]["observed"] == 1.0


def test_absent_diagnostics_use_fixture_labels_and_existing_raw_scores():
    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=executor()
    )
    absent = next(row for row in report["per_query"] if row["answerability"] == "ABSENT")
    diagnostics = absent["absent_diagnostics"]
    assert set(diagnostics) == {
        "interpretation",
        "hard_negative_exposure",
        "source_concentration",
        "dense_top1_distance",
        "dense_top1_similarity",
        "bm25_top_score",
        "hybrid_top1_rrf_score",
        "max_dense_similarity",
        "retrieved_chunk_count",
        "packed_chunk_count",
    }
    exposure = diagnostics["hard_negative_exposure"]["value"]
    assert exposure["observed"] is True
    assert exposure["matches"][0]["chunk_id"] == "doc-hn__0001"
    assert exposure["matches"][0]["source"] == "doc-hn"
    concentration = diagnostics["source_concentration"]["value"]
    assert concentration["unique_source_count"] == 10
    assert concentration["max_source_count"] == 1
    assert concentration["max_source_fraction"] == 0.1
    assert diagnostics["dense_top1_similarity"]["value"] == 0.9
    assert diagnostics["dense_top1_distance"]["value"] == 0.1
    assert diagnostics["max_dense_similarity"]["value"] == 0.9
    assert diagnostics["bm25_top_score"]["value"] == 0.9
    assert diagnostics["hybrid_top1_rrf_score"]["value"] == round(1 / 60, 8)
    assert diagnostics["retrieved_chunk_count"]["value"] == 10
    assert diagnostics["packed_chunk_count"]["value"] == 1
    assert absent["metrics"]["packed_evidence_recall"] is None
    assert "refusal_safety" not in diagnostics


def test_absent_diagnostics_report_no_exposure_and_explicit_na_for_missing_scores():
    query = {
        "query_id": "absent-no-exposure",
        "answerability": "ABSENT",
        "hard_negatives": [{"source": "hard-negative", "chunk_id": "hard-negative-1"}],
    }
    hybrid_top10 = [
        row(f"noise-{rank}", f"noise-{rank}") for rank in range(1, 11)
    ]
    score = score_query(query, _metric_raw(hybrid_top10))
    diagnostics = score["absent_diagnostics"]
    assert diagnostics["hard_negative_exposure"]["value"]["observed"] is False
    for field in (
        "dense_top1_distance",
        "dense_top1_similarity",
        "bm25_top_score",
        "hybrid_top1_rrf_score",
        "max_dense_similarity",
    ):
        assert diagnostics[field]["status"] == "NOT_APPLICABLE"
        assert diagnostics[field]["value"] is None
        assert diagnostics[field]["reason"]


def test_absent_diagnostic_score_drift_fails_determinism(monkeypatch):
    calls = 0

    def changed_second_execution(query):
        nonlocal calls
        calls += 1
        outputs = executor()(query)
        if calls > 5 and query == "synthetic absent diagnostic":
            for output in outputs.values():
                output["hybrid_top10"][0]["rrf_score"] = 0.5
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=changed_second_execution,
        **_authorized(monkeypatch),
    )
    assert report["determinism"]["passed"] is False
    assert any(
        failure["gate"] == "determinism_mismatch"
        for failure in report["aggregate"]["hard_failures"]
    )


def test_forged_coverage_and_packed_recall_cannot_pass():
    call = executor(packed_loss=True)

    def forged(query):
        result = call(query)
        for output in result.values():
            output.update({"covered_spans": ["anything"], "packed_recall": 1.0})
        return result

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=forged
    )
    assert report["overall_pass"] is False
    assert any(
        f["gate"] == "critical_grade2_lost_at_packed" for f in report["hard_failures"]
    )


def test_missing_backend_object_fails_closed():
    with pytest.raises(QualificationHarnessError, match="both raw backend"):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=lambda _: {"cswp_local": {}},
        )


def test_direct_backend_identity_comparison_ignores_forged_mismatch_count():
    call = executor(packed_mismatch=True)

    def forged(query):
        outputs = call(query)
        for output in outputs.values():
            output["mismatch_count"] = 0
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=forged
    )
    assert report["overall_pass"] is False
    assert report["backend_parity"]["mismatches"]


@pytest.mark.parametrize("field", ["dense_top20", "bm25_rank_order", "hybrid_seed_top5"])
def test_missing_required_rank_diagnostic_fails_closed(field):
    def missing(query):
        outputs = executor()(query)
        del outputs["cswp_qdrant"][field]
        return outputs

    with pytest.raises(QualificationHarnessError, match=field):
        run_qualification(fixture_path=FIXTURE, contract_path=CONTRACT, executor=missing)


@pytest.mark.parametrize("field", ["dense_top20", "bm25_rank_order", "hybrid_seed_top5"])
def test_ordered_rank_diagnostic_mismatch_fails_backend_identity_gate(field):
    def mismatched(query):
        outputs = executor()(query)
        outputs["cswp_qdrant"][field] = [row("doc-other", "doc-other__0001")]
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=mismatched
    )
    assert report["overall_pass"] is False
    assert any(
        issue["field"] == f"{field}.chunk_id"
        for issue in report["backend_parity"]["mismatches"]
    )


def test_rank_diagnostic_repeat_run_drift_fails_determinism(monkeypatch):
    calls = 0

    def changed_second_execution(query):
        nonlocal calls
        calls += 1
        outputs = executor()(query)
        if calls > 5:
            for output in outputs.values():
                output["dense_top20"] = [row("doc-other", "doc-other__0001")]
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=changed_second_execution,
        **_authorized(monkeypatch),
    )
    assert report["determinism"]["passed"] is False
    assert any(
        failure["gate"] == "determinism_mismatch"
        for failure in report["aggregate"]["hard_failures"]
    )


def test_wrong_contract_sha_fails(tmp_path):
    contract = json.loads(CONTRACT.read_text())
    contract["primary_metric"]["floor"] = 0.1
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(contract))
    with pytest.raises(QualificationHarnessError, match="contract SHA"):
        run_qualification(
            fixture_path=FIXTURE, contract_path=changed, executor=executor()
        )


def test_wrong_fixture_or_corpus_identity_fails(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    fixture["expected_provenance"] = {
        "fixture_sha256": "wrong",
        "source_corpus_fingerprint": "wrong",
    }
    changed = tmp_path / "fixture.json"
    changed.write_text(json.dumps(fixture))
    report = run_qualification(
        fixture_path=changed, contract_path=CONTRACT, executor=executor()
    )
    assert any(f["gate"] == "provenance_mismatch" for f in report["hard_failures"])


@pytest.mark.parametrize("field, value", [("qualified_contract", None), ("index_fingerprint", "")])
def test_missing_or_null_required_provenance_cannot_pass(field, value):
    call = executor()

    def missing(query):
        outputs = call(query)
        outputs["cswp_local"][field] = value
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=missing
    )
    assert report["overall_pass"] is False
    assert any(f["gate"] == "provenance_mismatch" for f in report["hard_failures"])


def test_query_one_provenance_drift_is_not_hidden_by_later_queries():
    call = executor()
    calls = 0

    def inconsistent(query):
        nonlocal calls
        calls += 1
        outputs = call(query)
        if calls == 1:
            for output in outputs.values():
                output["source_corpus_fingerprint"] = "wrong-corpus"
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=inconsistent
    )
    assert report["overall_pass"] is False
    assert any(
        issue["gate"] == "run_wide_provenance_mismatch"
        for issue in report["provenance"]["mismatches"]
    )


def test_backend_provenance_mismatch_cannot_pass():
    call = executor()

    def inconsistent(query):
        outputs = call(query)
        outputs["cswp_qdrant"]["index_fingerprint"] = "other-index"
        outputs["cswp_qdrant"]["live_collection_fingerprint"] = "other-index"
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=inconsistent
    )
    assert report["overall_pass"] is False
    assert any(
        issue["gate"] == "backend_provenance_mismatch"
        for issue in report["provenance"]["mismatches"]
    )


def test_stable_forged_runtime_identity_cannot_become_the_baseline():
    call = executor()

    def forged(query):
        outputs = call(query)
        for output in outputs.values():
            output.update(
                {
                    "source_corpus_fingerprint": "forged-corpus",
                    "index_fingerprint": "forged-index",
                    "minilm_model_id": "forged-model",
                    "minilm_model_revision": "forged-revision",
                    "qualified_contract": "FORGED_CONTRACT",
                }
            )
        outputs["cswp_qdrant"].update(
            {
                "collection_name": "forged-collection",
                "live_collection_fingerprint": "forged-index",
            }
        )
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=forged
    )
    assert report["overall_pass"] is False
    assert any(
        issue["gate"] == "expected_provenance_mismatch"
        for issue in report["provenance"]["mismatches"]
    )


def test_legacy_fingerprint_name_cannot_satisfy_canonical_contract():
    call = executor()

    def legacy_only(query):
        outputs = call(query)
        qdrant = outputs["cswp_qdrant"]
        qdrant["qdrant_live_content_fingerprint"] = qdrant.pop(
            "live_collection_fingerprint"
        )
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=legacy_only
    )
    assert report["overall_pass"] is False
    assert any(
        issue["detail"]["field"] == "live_collection_fingerprint"
        for issue in report["provenance"]["mismatches"]
        if issue["gate"] == "required_provenance_missing"
    )


def _authorized(monkeypatch):
    monkeypatch.setattr(runner, "runtime_git_sha", lambda: "authorized")
    return {
        "authoritative": True,
        "write_archive": True,
        "approved_execution_sha": "authorized",
        "approved_fixture_sha256": runner.sha256_file(FIXTURE),
    }


def test_authoritative_mode_without_archive_rejects_before_retrieval(monkeypatch):
    monkeypatch.setattr(runner, "runtime_git_sha", lambda: "authorized")
    calls = 0

    def counted(query):
        nonlocal calls
        calls += 1
        return executor()(query)

    with pytest.raises(QualificationHarnessError, match="reserved and written archive"):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=counted,
            authoritative=True,
            approved_execution_sha="authorized",
            approved_fixture_sha256=runner.sha256_file(FIXTURE),
        )
    assert calls == 0


def test_cli_forces_authoritative_archive_and_rejects_optional_flag(monkeypatch, capsys):
    recorded = {}

    def invoked(**kwargs):
        recorded.update(kwargs)
        return {"overall_pass": True}

    monkeypatch.setattr(runner, "run_qualification", invoked)
    assert runner.main(
        [
            "--fixture", str(FIXTURE),
            "--approved-execution-sha", "approved",
            "--approved-fixture-sha256", runner.sha256_file(FIXTURE),
        ]
    ) == 0
    assert recorded["authoritative"] is True
    assert recorded["write_archive"] is True
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--fixture", str(FIXTURE),
                "--approved-execution-sha", "approved",
                "--approved-fixture-sha256", runner.sha256_file(FIXTURE),
                "--write-archive",
            ]
        )
    capsys.readouterr()


def test_archive_reservation_precedes_first_authoritative_query(tmp_path, monkeypatch):
    archive = tmp_path / "reserved-before-query"
    observed = []

    def counted(query):
        observed.append(
            archive.is_dir() and (archive / "execution_journal.jsonl").is_file()
        )
        return executor()(query)

    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=counted,
        archive_root=archive,
        **_authorized(monkeypatch),
    )
    assert all(observed)
    assert report["disposition"] == load_contract(CONTRACT)["disposition_values"][0]
    assert archive_is_successful(archive) is True


def test_failed_archive_finalization_cannot_issue_authoritative_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "write_qualification_archive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("archive write failed")),
    )
    with pytest.raises(OSError, match="archive write failed"):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=executor(),
            archive_root=tmp_path / "failed-archive",
            **_authorized(monkeypatch),
        )


def test_unverified_archive_cannot_issue_authoritative_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "archive_is_successful", lambda _archive: False)
    with pytest.raises(QualificationHarnessError, match="completed, verified release archive"):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=executor(),
            archive_root=tmp_path / "unverified-archive",
            **_authorized(monkeypatch),
        )


def _assert_incomplete_archive(report, archive, *, stage, execution, query_id="SYN-A2"):
    assert report["terminal_status"] == "INCOMPLETE"
    assert report["overall_pass"] is False
    assert report["disposition"] == load_contract(CONTRACT)["disposition_values"][1]
    record = report["executions"][execution - 1]
    assert record["failure"]["query_id"] == query_id
    assert record["failure"]["stage"] == stage
    assert len(record["per_query"]) == 1
    assert len(record["raw_backend_outputs"]["cswp_local"]) == 2
    assert len(record["raw_backend_outputs"]["cswp_qdrant"]) == 2
    journal = [
        json.loads(line)
        for line in (archive / "execution_journal.jsonl").read_text().splitlines()
    ]
    incomplete = [
        item
        for item in journal
        if item["event"] == "execution_incomplete" and item["execution"] == execution
    ]
    assert incomplete[-1]["query_id"] == query_id
    assert incomplete[-1]["stage"] == stage
    assert incomplete[-1]["raw_backend_outputs"] is not None
    assert journal[-1]["event"] == "execution_completed"
    assert journal[-1]["terminal_status"] == "INCOMPLETE"
    assert archive_is_successful(archive) is False


def test_query_two_malformed_ranking_finalizes_incomplete_with_raw_outputs(tmp_path, monkeypatch):
    archive = tmp_path / "malformed-ranking"
    calls = 0

    def malformed(query):
        nonlocal calls
        calls += 1
        outputs = executor()(query)
        if calls == 2:
            outputs["cswp_local"]["dense_top20"] = None
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=malformed,
        archive_root=archive,
        **_authorized(monkeypatch),
    )
    _assert_incomplete_archive(report, archive, stage="canonicalization", execution=1)
    assert report["executions"][0]["raw_backend_outputs"]["cswp_local"][1][
        "dense_top20"
    ] is None


def test_query_two_validation_exception_finalizes_incomplete(tmp_path, monkeypatch):
    archive = tmp_path / "validation-failure"
    original = runner.validate_runtime_identity

    def invalid(*, query_id, **kwargs):
        if query_id == "SYN-A2":
            raise QualificationHarnessError("synthetic validation failure")
        return original(query_id=query_id, **kwargs)

    monkeypatch.setattr(runner, "validate_runtime_identity", invalid)
    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=executor(),
        archive_root=archive,
        **_authorized(monkeypatch),
    )
    _assert_incomplete_archive(report, archive, stage="runtime_identity", execution=1)


def test_query_two_scoring_exception_finalizes_incomplete(tmp_path, monkeypatch):
    archive = tmp_path / "scoring-failure"
    original = runner.score_query

    def unscorable(query, raw):
        if query["query_id"] == "SYN-A2":
            raise QualificationHarnessError("synthetic scoring failure")
        return original(query, raw)

    monkeypatch.setattr(runner, "score_query", unscorable)
    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=executor(),
        archive_root=archive,
        **_authorized(monkeypatch),
    )
    _assert_incomplete_archive(report, archive, stage="scoring", execution=1)


def test_execution_two_partial_results_survive_archive_and_execution_artifact(tmp_path, monkeypatch):
    archive = tmp_path / "execution-two-failure"
    calls = 0

    def malformed_second_execution(query):
        nonlocal calls
        calls += 1
        outputs = executor()(query)
        if calls == 7:
            outputs["cswp_qdrant"]["hybrid_top10"] = None
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=malformed_second_execution,
        archive_root=archive,
        **_authorized(monkeypatch),
    )
    assert calls == 7
    assert len(report["executions"]) == 2
    _assert_incomplete_archive(report, archive, stage="canonicalization", execution=2)
    execution_artifact = json.loads((archive / "executions.json").read_text())
    assert execution_artifact[1]["failure"]["query_id"] == "SYN-A2"
    assert len(execution_artifact[1]["per_query"]) == 1
    second_raw = json.loads((archive / "execution_2_raw_backend_outputs.json").read_text())
    assert len(second_raw["cswp_local"]) == len(second_raw["cswp_qdrant"]) == 2
    assert second_raw["cswp_qdrant"][1]["hybrid_top10"] is None


@pytest.mark.parametrize(
    "failure",
    [
        "qualified CSWP corpus must contain exactly 159 units",
        "invalid Qdrant startup",
        "live collection fingerprint mismatch",
    ],
)
def test_real_preflight_rejection_makes_zero_executor_calls(monkeypatch, failure):
    monkeypatch.setattr(runner, "runtime_git_sha", lambda: "authorized")
    monkeypatch.setattr(
        runner,
        "validate_real_adapter_preflight",
        lambda _expected: (_ for _ in ()).throw(QualificationHarnessError(failure)),
    )
    calls = 0

    def counted(query):
        nonlocal calls
        calls += 1
        return executor()(query)

    with pytest.raises(QualificationHarnessError, match=failure):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=counted,
            real_adapter_mode=True,
            **_authorized(monkeypatch),
        )
    assert calls == 0


def test_archive_is_create_once(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=executor(),
        archive_root=archive,
        **_authorized(monkeypatch),
    )
    with pytest.raises(QualificationHarnessError, match="archive already exists"):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=executor(),
            archive_root=archive,
            **_authorized(monkeypatch),
        )


def test_existing_archive_rejects_before_executor_calls(tmp_path, monkeypatch):
    archive = tmp_path / "reserved"
    archive.mkdir()
    calls = 0

    def counted(query):
        nonlocal calls
        calls += 1
        return executor()(query)

    with pytest.raises(QualificationHarnessError, match="before retrieval"):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=counted,
            archive_root=archive,
            **_authorized(monkeypatch),
        )
    assert calls == 0


def test_concurrent_archive_reservation_has_exactly_one_claimant(tmp_path):
    archive = tmp_path / "race"

    def claimant():
        try:
            reserve_archive(archive)
        except QualificationHarnessError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: claimant(), range(2)))
    assert claims.count(True) == 1


def test_second_archive_write_cannot_overwrite(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=executor(),
        archive_root=archive,
        **_authorized(monkeypatch),
    )
    original_summary = (archive / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_qualification_archive(report, archive, already_reserved=True)
    assert (archive / "summary.json").read_bytes() == original_summary


def test_partial_denominator_excludes_missing_obligation():
    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=executor()
    )
    partial = next(row for row in report["per_query"] if row["query_id"] == "SYN-P1")
    assert partial["metrics"]["packed_evidence_recall"] == 1.0


def test_deterministic_second_raw_run_matches():
    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=executor(),
        determinism_executor=executor(),
    )
    assert report["determinism"]["passed"] is True


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "approved execution SHA"),
        ({"approved_execution_sha": "wrong"}, "runtime HEAD"),
        ({"approved_execution_sha": "authorized"}, "approved fixture SHA-256"),
        (
            {
                "approved_execution_sha": "authorized",
                "approved_fixture_sha256": "wrong",
            },
            "fixture bytes",
        ),
    ],
)
def test_authorization_rejects_before_any_retrieval(kwargs, message, monkeypatch):
    monkeypatch.setattr(runner, "runtime_git_sha", lambda: "authorized")
    calls = 0

    def counted(query):
        nonlocal calls
        calls += 1
        return executor()(query)

    with pytest.raises(QualificationHarnessError, match=message):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=counted,
            authoritative=True,
            write_archive=True,
            **kwargs,
        )
    assert calls == 0


def test_authorized_fixture_snapshot_is_scored_after_file_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "runtime_git_sha", lambda: "authorized")
    fixture = tmp_path / "fixture.json"
    original = FIXTURE.read_bytes()
    fixture.write_bytes(original)
    calls = 0

    def mutate_after_snapshot(query):
        nonlocal calls
        calls += 1
        if calls == 1:
            fixture.write_text('{"schema_version":"retrieval_holdout_v1","queries":[]}')
        return executor()(query)

    report = run_qualification(
        fixture_path=fixture,
        contract_path=CONTRACT,
        executor=mutate_after_snapshot,
        authoritative=True,
        write_archive=True,
        approved_execution_sha="authorized",
        approved_fixture_sha256=__import__("hashlib").sha256(original).hexdigest(),
    )
    assert report["fixture_sha256"] == __import__("hashlib").sha256(original).hexdigest()
    assert len(report["per_query"]) == 5


def test_authorized_matching_sha_and_digest_permit_execution(monkeypatch):
    monkeypatch.setattr(runner, "runtime_git_sha", lambda: "authorized")
    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=executor(),
        authoritative=True,
        write_archive=True,
        approved_execution_sha="authorized",
        approved_fixture_sha256=runner.sha256_file(FIXTURE),
    )
    assert report["overall_pass"] is True


def test_authorization_rejection_does_not_reserve_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "runtime_git_sha", lambda: "authorized")
    archive = tmp_path / "authoritative-archive"
    with pytest.raises(QualificationHarnessError, match="fixture SHA-256"):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            archive_root=archive,
            executor=lambda _: pytest.fail("retrieval must not execute"),
            authoritative=True,
            write_archive=True,
            approved_execution_sha="authorized",
            approved_fixture_sha256="wrong",
        )
    assert not archive.exists()


def test_authoritative_attempt_runs_two_complete_matching_executions(monkeypatch):
    calls = 0

    def counted(query):
        nonlocal calls
        calls += 1
        return executor()(query)

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=counted, **_authorized(monkeypatch)
    )
    assert calls == 10
    assert report["determinism"]["passed"] is True
    assert report["overall_pass"] is report["aggregate"]["overall_pass"] is True
    assert report["disposition"] == report["aggregate"]["disposition"]


@pytest.mark.parametrize("mode", ["packed", "metric", "identity"])
def test_authoritative_second_run_mismatch_fails_consistently(mode, monkeypatch):
    calls = 0

    def changed(query):
        nonlocal calls
        calls += 1
        result = executor()(query)
        if calls > 5:
            for output in result.values():
                if mode == "packed":
                    output["kb_results"] = [row("doc-z", "doc-z__0001")]
                    output["packed_fingerprint"] = "changed"
                elif mode == "metric":
                    output["kb_results"] = []
                else:
                    output["index_fingerprint"] = "changed-index"
                    output["live_collection_fingerprint"] = "changed-index"
        return result

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=changed, **_authorized(monkeypatch)
    )
    assert report["determinism"]["passed"] is False
    assert report["overall_pass"] is report["aggregate"]["overall_pass"] is False
    assert report["disposition"] == report["aggregate"]["disposition"]


def test_authoritative_second_run_exception_is_incomplete(monkeypatch):
    calls = 0

    def fail_second(query):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("second execution exploded")
        return executor()(query)

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=fail_second, **_authorized(monkeypatch)
    )
    assert report["determinism"]["status"] == "INCOMPLETE"
    assert report["overall_pass"] is report["aggregate"]["overall_pass"] is False
    assert report["disposition"] == report["aggregate"]["disposition"]


def test_authoritative_archive_journals_every_query_of_both_executions(tmp_path, monkeypatch):
    archive = tmp_path / "journal"
    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=executor(),
        archive_root=archive, **_authorized(monkeypatch)
    )
    records = [json.loads(line) for line in (archive / "execution_journal.jsonl").read_text().splitlines()]
    completed = [r for r in records if r["event"] == "query_completed"]
    assert [r["execution"] for r in completed].count(1) == 5
    assert [r["execution"] for r in completed].count(2) == 5
    assert all(set(r["raw_backend_outputs"]) == {"cswp_local", "cswp_qdrant"} for r in completed)
    assert {path.name for path in archive.iterdir()} >= {
        "per_query.json", "summary.json", "parity.json", "manifest.json",
        "disposition.json", "execution_journal.jsonl", "raw_backend_outputs.json",
        "execution_2_raw_backend_outputs.json", "provenance.json", "contract.json",
        "determinism.json", "executions.json",
    }
    per_query = json.loads((archive / "per_query.json").read_text())
    summary = json.loads((archive / "summary.json").read_text())
    disposition_record = json.loads((archive / "disposition.json").read_text())
    manifest = json.loads((archive / "manifest.json").read_text())
    assert len(per_query["execution_1"]) == len(per_query["execution_2"]) == 5
    assert summary["overall_pass"] is report["overall_pass"] is True
    assert summary["disposition"] == disposition_record["disposition"] == report["disposition"]
    assert summary["aggregate"] == report["aggregate"]
    assert manifest["artifacts"]["execution_journal.jsonl"]["bytes"] > 0
    assert manifest["manifest_inventory_excludes_self"] is True
    assert set(manifest["artifacts"]) == {
        path.name for path in archive.iterdir() if path.name != "manifest.json"
    }
    archived_raw = json.loads((archive / "raw_backend_outputs.json").read_text())
    archived_repeat_raw = json.loads(
        (archive / "execution_2_raw_backend_outputs.json").read_text()
    )
    archived_metrics = json.loads((archive / "per_query.json").read_text())
    for raw_execution in (archived_raw, archived_repeat_raw):
        for backend_rows in raw_execution.values():
            assert {"dense_top20", "bm25_rank_order", "hybrid_seed_top5"} <= set(
                backend_rows[0]
            )
            assert len(backend_rows[0]["hybrid_top10"]) == 10
    for execution in ("execution_1", "execution_2"):
        assert "secondary_metrics" in archived_metrics[execution][0]
        assert archived_metrics[execution][0]["secondary_metrics"]["retrieved"][
            "source_rank_metrics"
        ]["mrr_at_10"] == 1.0
        archived_absent = next(
            row for row in archived_metrics[execution] if row["answerability"] == "ABSENT"
        )
        assert archived_absent["absent_diagnostics"]["hard_negative_exposure"][
            "value"
        ]["observed"] is True
    assert archive_is_successful(archive) is True


def test_first_execution_failure_retains_prior_journal_evidence(tmp_path, monkeypatch):
    archive = tmp_path / "journal"
    calls = 0

    def fail_query_two(query):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("query two failed")
        return executor()(query)

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=fail_query_two,
        archive_root=archive, **_authorized(monkeypatch)
    )
    records = [json.loads(line) for line in (archive / "execution_journal.jsonl").read_text().splitlines()]
    assert [r["event"] for r in records].count("query_completed") == 1
    incomplete = [r for r in records if r["event"] == "execution_incomplete"]
    assert incomplete[-1]["execution"] == 1
    assert records[-1]["event"] == "execution_completed"
    assert records[-1]["terminal_status"] == "INCOMPLETE"
    assert report["terminal_status"] == "INCOMPLETE"
    assert report["overall_pass"] is False
    assert json.loads((archive / "summary.json").read_text())["terminal_status"] == "INCOMPLETE"
    assert json.loads((archive / "disposition.json").read_text())["overall_pass"] is False
    assert archive_is_successful(archive) is False


def test_second_execution_failure_retains_both_execution_evidence(tmp_path, monkeypatch):
    archive = tmp_path / "journal"
    calls = 0

    def fail_second_after_two(query):
        nonlocal calls
        calls += 1
        if calls == 8:
            raise RuntimeError("second query-three failed")
        return executor()(query)

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=fail_second_after_two,
        archive_root=archive, **_authorized(monkeypatch)
    )
    records = [json.loads(line) for line in (archive / "execution_journal.jsonl").read_text().splitlines()]
    completed = [r for r in records if r["event"] == "query_completed"]
    assert [r["execution"] for r in completed].count(1) == 5
    assert [r["execution"] for r in completed].count(2) == 2
    assert any(r["event"] == "execution_incomplete" and r["execution"] == 2 for r in records)
    assert report["overall_pass"] is False
    assert report["terminal_status"] == "INCOMPLETE"
    assert len(report["executions"]) == 2
    assert len(report["executions"][0]["per_query"]) == 5
    assert len(report["executions"][1]["per_query"]) == 2
    assert archive_is_successful(archive) is False


def test_determinism_mismatch_finalizes_as_complete_consistent_fail(tmp_path, monkeypatch):
    archive = tmp_path / "mismatch"
    calls = 0

    def changed_second_execution(query):
        nonlocal calls
        calls += 1
        outputs = executor()(query)
        if calls > 5:
            for output in outputs.values():
                output["kb_results"] = [row("doc-z", "doc-z__0001")]
                output["packed_fingerprint"] = "changed"
        return outputs

    report = run_qualification(
        fixture_path=FIXTURE, contract_path=CONTRACT, executor=changed_second_execution,
        archive_root=archive, **_authorized(monkeypatch)
    )
    summary = json.loads((archive / "summary.json").read_text())
    disposition_record = json.loads((archive / "disposition.json").read_text())
    assert report["terminal_status"] == summary["terminal_status"] == "COMPLETE"
    assert report["overall_pass"] is summary["overall_pass"] is disposition_record["overall_pass"] is False
    assert report["disposition"] == summary["disposition"] == disposition_record["disposition"]
    assert archive_is_successful(archive) is False


def test_missing_or_incomplete_terminal_artifacts_never_count_as_success(tmp_path):
    archive = tmp_path / "missing-terminal"
    archive.mkdir()
    (archive / "summary.json").write_text(
        json.dumps({"terminal_status": "COMPLETE", "overall_pass": True, "disposition": "PASS"})
    )
    (archive / "disposition.json").write_text(
        json.dumps({"terminal_status": "COMPLETE", "overall_pass": True, "disposition": "PASS"})
    )
    assert archive_is_successful(archive) is False
