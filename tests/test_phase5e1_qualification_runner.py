"""Phase-5E1E raw authority tests; no sealed fixture, network, or providers."""

from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from scripts.phase5e1_qualification_runner import (
    QualificationHarnessError,
    contract_sha256,
    load_contract,
    reserve_archive,
    run_qualification,
    write_qualification_archive,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "evals/fixtures/phase5e1_qualification_synthetic_oracle.json"
CONTRACT = ROOT / "evals/fixtures/phase5e1_qualification_contract.json"


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
        packed = [] if packed_loss and source == "doc-a" else rows
        base = {
            "retrieval_seeds": rows,
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


def test_archive_is_create_once(tmp_path):
    archive = tmp_path / "archive"
    run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=executor(),
        archive_root=archive,
        write_archive=True,
    )
    with pytest.raises(QualificationHarnessError, match="archive already exists"):
        run_qualification(
            fixture_path=FIXTURE,
            contract_path=CONTRACT,
            executor=executor(),
            archive_root=archive,
            write_archive=True,
        )


def test_existing_archive_rejects_before_executor_calls(tmp_path):
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
            write_archive=True,
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


def test_second_archive_write_cannot_overwrite(tmp_path):
    archive = tmp_path / "archive"
    report = run_qualification(
        fixture_path=FIXTURE,
        contract_path=CONTRACT,
        executor=executor(),
        archive_root=archive,
        write_archive=True,
    )
    with pytest.raises(FileExistsError):
        write_qualification_archive(report, archive, already_reserved=True)


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
