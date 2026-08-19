"""Regression coverage for evaluation paths that must fail closed."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from config import PROMPT_HASHES, PROMPT_VERSION
from main import _write_telemetry as write_run_telemetry
from scripts import benchmark, check_telemetry_fields


ROOT = Path(__file__).resolve().parents[1]
TOPICS = json.loads((ROOT / "evals" / "topics.json").read_text())
CONTRACT = json.loads((ROOT / "evals" / "benchmark_release_contract.json").read_text())
EXPECTED_SHA = "ca29d32b4869269daa47142615d298580a577a77"


def _telemetry(
    run_id: str,
    topic: str,
    *,
    verification_status: str | None = "completed",
    verified: int = 0,
    weak: int = 0,
    unverified: int = 0,
    prompt_version: str = PROMPT_VERSION,
    prompt_hashes: dict | None = None,
) -> dict:
    record = {
        "run_id": run_id,
        "topic": topic,
        "slug": "topic",
        "timestamp": "2026-08-12T00:00:00+00:00",
        "prompt_version": prompt_version,
        "prompt_hashes": dict(PROMPT_HASHES if prompt_hashes is None else prompt_hashes),
        "iteration_metrics": [],
        "experiment_flags": {},
        "claims_verified": verified,
        "claims_weak": weak,
        "claims_unverified": unverified,
        "grounding_score": 0.0,
        "grounding_breakdown": {},
        "grounding_report": [],
        "reflection_score": 7,
        "reflection_notes": "",
        "web_sources_count": 0,
        "kb_results_count": 0,
        "web_sources": [],
        "kb_results": [],
        "attribution": {},
        "total_cost_usd": 0.0,
        "total_tokens": 0,
        "latency_ms": {},
        "error_log": [],
        "hitl_status": "approved",
        "git_status": "dry_run",
    }
    if verification_status is not None:
        record["verification_status"] = verification_status
    return record


def _write_telemetry(tmp_path: Path, run_id: str, topic: str, **kwargs) -> Path:
    path = tmp_path / "outputs" / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_telemetry(run_id, topic, **kwargs)), encoding="utf-8")
    return path


def _install_v1_manifest(tmp_path: Path) -> None:
    (tmp_path / "evals").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "evals/topics.json", tmp_path / "evals/topics.json")
    shutil.copy(
        ROOT / "evals/benchmark_release_contract.json",
        tmp_path / "evals/benchmark_release_contract.json",
    )


def _topic_by_id(topic_id: int = 1) -> dict:
    return next(topic for topic in TOPICS if topic["id"] == topic_id)


def _completed_proc(run_id: str) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=f"RUN_ID={run_id}\n", stderr="")


def _failed_proc() -> SimpleNamespace:
    return SimpleNamespace(returncode=7, stdout="", stderr="injected failure")


def _aggregate(results_dir: Path) -> dict:
    return json.loads(next(results_dir.glob("benchmark_*.json")).read_text())


def _mock_git_identity(monkeypatch, sha: str = EXPECTED_SHA, *, clean: bool = True) -> None:
    monkeypatch.setattr(benchmark, "_git_rev_parse_head", lambda: sha)
    monkeypatch.setattr(benchmark, "_git_diff_clean", lambda staged=False: clean)


def _mock_release_github(monkeypatch, sha: str = EXPECTED_SHA) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", sha)
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", "owner/repo/.github/workflows/eval.yml@refs/heads/main")


def _run_benchmark(
    tmp_path: Path,
    monkeypatch,
    *,
    topic_id: int = 1,
    telemetry_kwargs: dict | None = None,
    run_id: str = "new",
    mode: str = "smoke",
    limit: int | None = None,
    gate: bool = True,
    expected_code_sha: str | None = None,
    proc_factory=_completed_proc,
) -> Path:
    _install_v1_manifest(tmp_path)
    topic = _topic_by_id(topic_id)
    telemetry_kwargs = telemetry_kwargs or {}
    path = _write_telemetry(tmp_path, run_id, topic["topic"], **telemetry_kwargs)
    path.write_text(json.dumps(_telemetry(run_id, topic["topic"], **telemetry_kwargs)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: proc_factory(run_id),
    )
    benchmark.run_benchmark.callback(
        mode=mode,
        limit=limit,
        topic_id=None if limit is not None else topic_id,
        gate=gate,
        expected_code_sha=expected_code_sha,
    )
    return tmp_path / "outputs" / "benchmark_results"


def _expect_preflight_failure(callback) -> None:
    with pytest.raises((click.ClickException, SystemExit)) as raised:
        callback()
    if isinstance(raised.value, SystemExit):
        assert raised.value.code == 2


def _assert_zero_subprocess_calls(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []

    def sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return _completed_proc("sentinel")

    monkeypatch.setattr(benchmark.subprocess, "run", sentinel)
    return calls


# --- Existing telemetry checker protections ---


def test_telemetry_checker_rejects_nonzero_run_with_stale_valid_telemetry(tmp_path, monkeypatch):
    _write_telemetry(tmp_path, "stale", "Gradient Descent")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="", stderr="injected failure"),
    )

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(ROOT / "scripts" / "check_telemetry_fields.py"), run_name="__main__")

    assert raised.value.code == 1


def test_telemetry_checker_rejects_missing_exact_run_even_with_stale_file(tmp_path, monkeypatch):
    _write_telemetry(tmp_path, "old", "Gradient Descent")
    monkeypatch.setattr(check_telemetry_fields.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    assert check_telemetry_fields.run_check(runs_dir=tmp_path / "outputs" / "runs") == 1


def test_telemetry_checker_accepts_matching_exact_run(tmp_path, monkeypatch):
    _write_telemetry(tmp_path, "new", "Gradient Descent")
    monkeypatch.setattr(check_telemetry_fields.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    assert check_telemetry_fields.run_check(runs_dir=tmp_path / "outputs" / "runs") == 0


def test_telemetry_checker_rejects_run_id_mismatch(tmp_path, monkeypatch):
    path = _write_telemetry(tmp_path, "new", "Gradient Descent")
    data = json.loads(path.read_text())
    data["run_id"] = "other"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(check_telemetry_fields.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    assert check_telemetry_fields.run_check(runs_dir=tmp_path / "outputs" / "runs") == 1


def test_telemetry_checker_rejects_topic_mismatch(tmp_path, monkeypatch):
    _write_telemetry(tmp_path, "new", "Other Topic")
    monkeypatch.setattr(check_telemetry_fields.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    assert check_telemetry_fields.run_check(runs_dir=tmp_path / "outputs" / "runs") == 1


def test_telemetry_checker_handles_missing_file_without_index_error(tmp_path, monkeypatch):
    monkeypatch.setattr(check_telemetry_fields.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    assert check_telemetry_fields.run_check(runs_dir=tmp_path / "outputs" / "runs") == 1


# --- Smoke benchmark behavior ---


def test_benchmark_rejects_zero_verdict_parse_failure(tmp_path, monkeypatch, capsys):
    _install_v1_manifest(tmp_path)
    topic = _topic_by_id(1)
    _write_telemetry(tmp_path, "new", topic["topic"], verification_status="parse_failed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    with pytest.raises(SystemExit) as raised:
        benchmark.run_benchmark.callback(mode="smoke", limit=None, topic_id=1, gate=True, expected_code_sha=None)

    assert raised.value.code == 1
    assert "uvr=N/A" in capsys.readouterr().out
    result = _aggregate(tmp_path / "outputs" / "benchmark_results")["ordered_unit_results"][0]
    assert result["uvr"] is None
    assert result["verification_status"] == "parse_failed"


def test_benchmark_allows_explicit_completed_zero_claim_case_and_excludes_it_from_uvr(tmp_path, monkeypatch):
    _install_v1_manifest(tmp_path)
    topic = dict(_topic_by_id(1))
    topic["allow_zero_claims"] = True
    topics = [dict(item) for item in TOPICS]
    topics[0] = topic
    manifest_identity = {
        "manifest_id": CONTRACT["manifest_id"],
        "manifest_path": CONTRACT["manifest_path"],
        "manifest_sha256": CONTRACT["manifest_sha256"],
        "expected_topic_count": CONTRACT["expected_topic_count"],
        "ordered_topic_ids": CONTRACT["ordered_topic_ids"],
        "actual_topic_count": len(topics),
        "actual_topic_ids": [item["id"] for item in topics],
    }
    monkeypatch.setattr(
        benchmark,
        "load_validated_manifest",
        lambda contract: (topics, manifest_identity),
    )
    _write_telemetry(tmp_path, "new", topic["topic"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    benchmark.run_benchmark.callback(mode="smoke", limit=None, topic_id=1, gate=True, expected_code_sha=None)

    aggregate = _aggregate(tmp_path / "outputs" / "benchmark_results")
    assert aggregate["aggregate_metrics"]["mean_unverified_rate"] is None
    assert aggregate["release_qualification"] == "NON_RELEASE"


def test_benchmark_rejects_completed_zero_claim_case_without_opt_in(tmp_path, monkeypatch):
    _install_v1_manifest(tmp_path)
    topic = _topic_by_id(1)
    _write_telemetry(tmp_path, "new", topic["topic"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    with pytest.raises(SystemExit) as raised:
        benchmark.run_benchmark.callback(mode="smoke", limit=None, topic_id=1, gate=True, expected_code_sha=None)

    assert raised.value.code == 1
    result = _aggregate(tmp_path / "outputs" / "benchmark_results")["ordered_unit_results"][0]
    assert result["evaluation_status"] == "unscorable_incomplete"


@pytest.mark.parametrize(
    ("verified", "unverified", "should_fail"),
    [(1, 0, False), (17, 3, False), (16, 4, True)],
)
def test_benchmark_preserves_uvr_gate_boundaries(tmp_path, monkeypatch, verified, unverified, should_fail):
    _install_v1_manifest(tmp_path)
    topic = _topic_by_id(1)
    _write_telemetry(
        tmp_path,
        "new",
        topic["topic"],
        verified=verified,
        unverified=unverified,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    if should_fail:
        with pytest.raises(SystemExit) as raised:
            benchmark.run_benchmark.callback(mode="smoke", limit=None, topic_id=1, gate=True, expected_code_sha=None)
        assert raised.value.code == 1
    else:
        benchmark.run_benchmark.callback(mode="smoke", limit=None, topic_id=1, gate=True, expected_code_sha=None)


def test_historical_telemetry_without_status_is_unknown_and_unscorable():
    topic = _topic_by_id(1)
    outcome = benchmark.verification_outcome(
        _telemetry("old", topic["topic"], verification_status=None),
        topic,
    )

    assert outcome["verification_status"] == "unknown"
    assert outcome["uvr"] is None
    assert outcome["validation_error"] == "verification_status=unknown"


def test_telemetry_persists_verification_status_and_na_uvr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_run_telemetry({
        "run_id": "parse-failed",
        "topic": "Gradient Descent",
        "grounding_report": [],
        "verification_status": "parse_failed",
    })
    record = json.loads(path.read_text())

    assert record["verification_status"] == "parse_failed"
    assert record["grounded_depth"]["N"] == 0
    assert record["grounded_depth"]["unverified_fraction"] is None


# --- P0-2a smoke selection and preflight ---


@pytest.mark.parametrize("topic_id", [999])
def test_unknown_topic_id_fails_before_subprocess(tmp_path, monkeypatch, topic_id):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises((click.ClickException, SystemExit)) as raised:
        benchmark.run_benchmark.callback(
            mode="smoke", limit=None, topic_id=topic_id, gate=True, expected_code_sha=None,
        )

    if isinstance(raised.value, SystemExit):
        assert raised.value.code == 2
    assert calls == []


@pytest.mark.parametrize("limit", [0, -1, 21, 999])
def test_invalid_smoke_limits_fail_before_subprocess(tmp_path, monkeypatch, limit):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=limit, topic_id=None, gate=True, expected_code_sha=None,
        ),
    )
    assert calls == []


def test_smoke_requires_exactly_one_selector(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=1, gate=True, expected_code_sha=None,
        ),
    )
    assert calls == []


def test_smoke_gate_pass_is_non_release(tmp_path, monkeypatch, capsys):
    _install_v1_manifest(tmp_path)
    topic = _topic_by_id(1)
    _write_telemetry(tmp_path, "new", topic["topic"], verified=10, unverified=1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    benchmark.run_benchmark.callback(mode="smoke", limit=None, topic_id=1, gate=True, expected_code_sha=None)

    report = _aggregate(tmp_path / "outputs" / "benchmark_results")
    assert report["release_qualification"] == "NON_RELEASE"
    assert "SMOKE GATE: PASS — NON-RELEASE" in capsys.readouterr().out


def test_preflight_failure_writes_no_report(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=None, topic_id=999, gate=True, expected_code_sha=None,
        ),
    )
    assert calls == []
    assert not (tmp_path / "outputs" / "benchmark_results").exists()


# --- Contract and manifest validation ---


def test_missing_release_contract_fails(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    (tmp_path / "evals").mkdir(parents=True)
    shutil.copy(ROOT / "evals/topics.json", tmp_path / "evals/topics.json")
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


def test_malformed_release_contract_fails(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    (tmp_path / "evals/benchmark_release_contract.json").write_text("{not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


def test_manifest_digest_mismatch_fails(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    topics_path = tmp_path / "evals/topics.json"
    topics = json.loads(topics_path.read_text())
    topics[0]["topic"] = "Mutated Topic Title"
    topics_path.write_text(json.dumps(topics), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


def test_manifest_duplicate_topic_id_fails():
    topics = copy.deepcopy(TOPICS)
    topics[1]["id"] = topics[0]["id"]
    failures = benchmark._validate_manifest_topics(topics)
    assert any("duplicate manifest topic id" in failure for failure in failures)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_topic_count", 19),
        ("ordered_topic_ids", list(range(1, 21))[:-1]),
        ("manifest_sha256", "0" * 64),
    ],
)
def test_v1_contract_immutability_rejects_mutated_identity(tmp_path, monkeypatch, field, value):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    contract = copy.deepcopy(CONTRACT)
    contract[field] = value
    (tmp_path / "evals/benchmark_release_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


def _write_contract(tmp_path: Path, contract: dict) -> None:
    (tmp_path / "evals/benchmark_release_contract.json").write_text(
        json.dumps(contract), encoding="utf-8",
    )


def test_v1_coordinated_count_and_ids_mutation_fails_before_subprocess(tmp_path, monkeypatch):
    """Coherent 19-topic contract still fails because frozen V1 requires 20."""
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    topics_path = tmp_path / "evals/topics.json"
    topics = json.loads(topics_path.read_text())[:19]
    topics_path.write_text(json.dumps(topics), encoding="utf-8")
    contract = copy.deepcopy(CONTRACT)
    contract["expected_topic_count"] = 19
    contract["ordered_topic_ids"] = list(range(1, 20))
    contract["manifest_sha256"] = hashlib.sha256(topics_path.read_bytes()).hexdigest()
    _write_contract(tmp_path, contract)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


def test_v1_coordinated_manifest_bytes_and_digest_mutation_fails(tmp_path, monkeypatch):
    """Updated digest matching mutated manifest bytes still fails frozen V1 SHA."""
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    topics_path = tmp_path / "evals/topics.json"
    topics = json.loads(topics_path.read_text())
    topics[0]["topic"] = "Mutated Topic Title"
    topics_path.write_text(json.dumps(topics), encoding="utf-8")
    contract = copy.deepcopy(CONTRACT)
    contract["manifest_sha256"] = hashlib.sha256(topics_path.read_bytes()).hexdigest()
    _write_contract(tmp_path, contract)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


def test_v1_smaller_coherent_manifest_still_fails_v1_identity(tmp_path, monkeypatch):
    """10-topic coherent manifest + matching digest/count/IDs cannot redefine V1."""
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    topics_path = tmp_path / "evals/topics.json"
    topics = json.loads(topics_path.read_text())[:10]
    topics_path.write_text(json.dumps(topics), encoding="utf-8")
    contract = copy.deepcopy(CONTRACT)
    contract["expected_topic_count"] = 10
    contract["ordered_topic_ids"] = list(range(1, 11))
    contract["manifest_sha256"] = hashlib.sha256(topics_path.read_bytes()).hexdigest()
    _write_contract(tmp_path, contract)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


@pytest.mark.parametrize(
    "manifest_id",
    ["content-agent-release-topics-v2", "wrong-manifest-id"],
)
def test_v1_rejects_differing_manifest_id(tmp_path, monkeypatch, manifest_id):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    contract = copy.deepcopy(CONTRACT)
    contract["manifest_id"] = manifest_id
    _write_contract(tmp_path, contract)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


def test_v1_rejects_differing_manifest_path(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    contract = copy.deepcopy(CONTRACT)
    contract["manifest_path"] = "evals/other_topics.json"
    _write_contract(tmp_path, contract)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


@pytest.mark.parametrize("schema_version", [0, 2])
def test_v1_rejects_unsupported_schema_version(tmp_path, monkeypatch, schema_version):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    contract = copy.deepcopy(CONTRACT)
    contract["schema_version"] = schema_version
    _write_contract(tmp_path, contract)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="smoke", limit=1, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


# --- Manifest topic schema validation ---


def _malformed_canonical_topics() -> list[dict]:
    return copy.deepcopy(TOPICS)


def test_manifest_missing_category_fails():
    topics = _malformed_canonical_topics()
    del topics[0]["category"]
    failures = benchmark._validate_manifest_topics(topics)
    assert any(
        "missing fields" in failure and "category" in failure
        for failure in failures
    )


def test_manifest_blank_category_fails():
    topics = _malformed_canonical_topics()
    topics[0]["category"] = "   "
    failures = benchmark._validate_manifest_topics(topics)
    assert any("field 'category' must be a nonempty string" in failure for failure in failures)


@pytest.mark.parametrize("field", ["topic", "slug", "card_id", "series"])
def test_manifest_blank_string_fields_fail(field):
    topics = _malformed_canonical_topics()
    topics[0][field] = ""
    failures = benchmark._validate_manifest_topics(topics)
    assert any(
        f"field '{field}' must be a nonempty string" in failure
        for failure in failures
    )


@pytest.mark.parametrize("topic_id", ["1", 1.5, True, 0, -1])
def test_manifest_invalid_topic_ids_fail(topic_id):
    topics = _malformed_canonical_topics()
    topics[0]["id"] = topic_id
    failures = benchmark._validate_manifest_topics(topics)
    assert any("id must be a positive integer" in failure for failure in failures)


def test_manifest_value_not_a_list_fails():
    failures = benchmark._validate_manifest_document({"id": 1}, benchmark.V1_RELEASE_CONTRACT)
    assert failures == ["manifest must be a nonempty JSON array"]


def test_manifest_empty_list_fails():
    failures = benchmark._validate_manifest_document([], benchmark.V1_RELEASE_CONTRACT)
    assert failures == ["manifest must be a nonempty JSON array"]


def test_manifest_topic_entry_not_an_object_fails():
    topics = _malformed_canonical_topics()
    topics[0] = "not-an-object"
    failures = benchmark._validate_manifest_topics(topics)
    assert any("is not an object" in failure for failure in failures)


def test_manifest_duplicate_topic_name_fails():
    topics = _malformed_canonical_topics()
    topics[1]["topic"] = topics[0]["topic"]
    failures = benchmark._validate_manifest_topics(topics)
    assert any("duplicate manifest topic name" in failure for failure in failures)


def test_manifest_duplicate_slug_fails():
    topics = _malformed_canonical_topics()
    topics[1]["slug"] = topics[0]["slug"]
    failures = benchmark._validate_manifest_topics(topics)
    assert any("duplicate manifest slug" in failure for failure in failures)


def test_manifest_topic_count_mismatch_fails():
    topics = _malformed_canonical_topics()[:19]
    failures = benchmark._validate_manifest_document(topics, benchmark.V1_RELEASE_CONTRACT)
    assert any("manifest topic count mismatch" in failure for failure in failures)


def test_manifest_topic_order_mismatch_fails():
    topics = _malformed_canonical_topics()
    topics[0], topics[1] = topics[1], topics[0]
    failures = benchmark._validate_manifest_document(topics, benchmark.V1_RELEASE_CONTRACT)
    assert any("do not match release contract order" in failure for failure in failures)


# --- Release preflight ---


def test_release_forbids_selectors(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    _mock_release_github(monkeypatch)
    _mock_git_identity(monkeypatch)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="release", limit=1, topic_id=None, gate=True, expected_code_sha=EXPECTED_SHA,
        ),
    )
    assert calls == []


def test_release_requires_gate_and_expected_sha(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="release", limit=None, topic_id=None, gate=False, expected_code_sha=None,
        ),
    )
    assert calls == []


@pytest.mark.parametrize("github_ref", ["refs/heads/feature", "refs/pull/1/merge"])
def test_release_rejects_non_main_github_ref(tmp_path, monkeypatch, github_ref):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    _mock_git_identity(monkeypatch)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF", github_ref)
    monkeypatch.setenv("GITHUB_SHA", EXPECTED_SHA)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", "wf")
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="release", limit=None, topic_id=None, gate=True, expected_code_sha=EXPECTED_SHA,
        ),
    )
    assert calls == []


def test_release_rejects_dirty_preflight_before_subprocess(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    _mock_release_github(monkeypatch)
    _mock_git_identity(monkeypatch, clean=False)
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="release", limit=None, topic_id=None, gate=True, expected_code_sha=EXPECTED_SHA,
        ),
    )
    assert calls == []


def test_release_rejects_wrong_expected_sha_before_subprocess(tmp_path, monkeypatch):
    calls = _assert_zero_subprocess_calls(monkeypatch)
    _install_v1_manifest(tmp_path)
    _mock_release_github(monkeypatch, sha=EXPECTED_SHA)
    _mock_git_identity(monkeypatch, sha="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    monkeypatch.chdir(tmp_path)

    _expect_preflight_failure(
        lambda: benchmark.run_benchmark.callback(
            mode="release", limit=None, topic_id=None, gate=True, expected_code_sha=EXPECTED_SHA,
        ),
    )
    assert calls == []


# --- Release unit validation ---


def _release_result(topic: dict, run_id: str, **telemetry_kwargs) -> dict:
    telemetry = _telemetry(run_id, topic["topic"], verified=10, unverified=1, **telemetry_kwargs)
    outcome = benchmark.verification_outcome(telemetry, topic)
    return {
        "id": topic["id"],
        "topic": topic["topic"],
        "status": "success",
        "run_id": run_id,
        "wall_time_s": 1.0,
        "telemetry": telemetry,
        "verification_status": outcome["verification_status"],
        "evaluation_status": outcome["evaluation_status"],
        "uvr": outcome["uvr"],
        "validation_error": outcome["validation_error"],
        "stderr": None,
        "subprocess_exit_code": 0,
    }


def test_release_validation_rejects_19_and_21_units():
    config, _ = benchmark.resolve_evaluation_config()
    expected_ids = CONTRACT["ordered_topic_ids"]
    nineteen = [_release_result(_topic_by_id(i), f"run-{i}") for i in expected_ids[:19]]
    failures = benchmark._validate_release_units(
        nineteen, expected_topic_ids=expected_ids, evaluation_config=config,
    )
    assert any("expected 20 release units" in failure for failure in failures)

    twenty_one = nineteen + [
        _release_result({"id": 99, "topic": "Extra"}, "run-extra"),
        _release_result(_topic_by_id(20), "run-20"),
    ]
    failures = benchmark._validate_release_units(
        twenty_one, expected_topic_ids=expected_ids, evaluation_config=config,
    )
    assert failures


def test_release_validation_rejects_out_of_order_and_duplicate_run_ids():
    config, _ = benchmark.resolve_evaluation_config()
    expected_ids = CONTRACT["ordered_topic_ids"]
    shuffled = [_release_result(_topic_by_id(i), f"run-{i}") for i in reversed(expected_ids)]
    failures = benchmark._validate_release_units(
        shuffled, expected_topic_ids=expected_ids, evaluation_config=config,
    )
    assert any("out of order" in failure for failure in failures)

    ordered = [_release_result(_topic_by_id(i), "same-run") for i in expected_ids]
    failures = benchmark._validate_release_units(
        ordered, expected_topic_ids=expected_ids, evaluation_config=config,
    )
    assert any("unique" in failure for failure in failures)


@pytest.mark.parametrize(
    "verification_status",
    ["parse_failed", "skipped_cost_gate", "upstream_failed", "unknown"],
)
def test_release_validation_rejects_incomplete_verification_statuses(verification_status):
    config, _ = benchmark.resolve_evaluation_config()
    expected_ids = CONTRACT["ordered_topic_ids"]
    results = [_release_result(_topic_by_id(i), f"run-{i}") for i in expected_ids]
    results[0] = _release_result(
        _topic_by_id(1),
        "run-1",
        verification_status=verification_status,
    )
    results[0]["verification_status"] = verification_status
    results[0]["evaluation_status"] = "verification_incomplete"
    results[0]["uvr"] = None
    failures = benchmark._validate_release_units(
        results, expected_topic_ids=expected_ids, evaluation_config=config,
    )
    assert failures


def test_release_validation_rejects_zero_verdict_and_mixed_prompt_identity():
    config, _ = benchmark.resolve_evaluation_config()
    expected_ids = CONTRACT["ordered_topic_ids"]
    results = [_release_result(_topic_by_id(i), f"run-{i}") for i in expected_ids]
    results[0]["telemetry"] = _telemetry("run-1", _topic_by_id(1)["topic"])
    results[0]["evaluation_status"] = "unscorable_incomplete"
    results[0]["uvr"] = None
    failures = benchmark._validate_release_units(
        results, expected_topic_ids=expected_ids, evaluation_config=config,
    )
    assert any("zero-verdict" in failure or "unscorable" in failure for failure in failures)

    mixed = [_release_result(_topic_by_id(i), f"run-{i}") for i in expected_ids]
    mixed[1]["telemetry"]["prompt_version"] = "sha-deadbeefdead"
    failures = benchmark._validate_release_units(
        mixed, expected_topic_ids=expected_ids, evaluation_config=config,
    )
    assert any("prompt_version mismatch" in failure for failure in failures)


# --- Synthetic full release pass ---


def test_synthetic_full_release_passes_once(tmp_path, monkeypatch, capsys):
    _install_v1_manifest(tmp_path)
    _mock_release_github(monkeypatch, sha=EXPECTED_SHA)
    _mock_git_identity(monkeypatch, sha=EXPECTED_SHA)
    monkeypatch.chdir(tmp_path)

    run_ids = {topic_id: f"run-{topic_id:02d}" for topic_id in CONTRACT["ordered_topic_ids"]}
    for topic_id, run_id in run_ids.items():
        topic = _topic_by_id(topic_id)
        _write_telemetry(tmp_path, run_id, topic["topic"], verified=17, unverified=3)

    def mock_run(cmd, **kwargs):
        if cmd[0:3] == ["uv", "run", "python"]:
            topic_name = cmd[cmd.index("--topic") + 1]
            topic = next(item for item in TOPICS if item["topic"] == topic_name)
            return _completed_proc(run_ids[topic["id"]])
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(benchmark.subprocess, "run", mock_run)

    benchmark.run_benchmark.callback(
        mode="release", limit=None, topic_id=None, gate=True, expected_code_sha=EXPECTED_SHA,
    )

    report = _aggregate(tmp_path / "outputs" / "benchmark_results")
    assert report["release_qualification"] == "PASS"
    assert report["mode"] == "release"
    assert len(report["ordered_unit_results"]) == 20
    benchmark._validate_evidence_payload(report)
    assert "RELEASE GATE: PASS" in capsys.readouterr().out


def test_release_subprocess_failure_writes_fail_not_pass(tmp_path, monkeypatch, capsys):
    _install_v1_manifest(tmp_path)
    _mock_release_github(monkeypatch, sha=EXPECTED_SHA)
    _mock_git_identity(monkeypatch, sha=EXPECTED_SHA)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: _failed_proc())

    with pytest.raises(SystemExit) as raised:
        benchmark.run_benchmark.callback(
            mode="release", limit=None, topic_id=None, gate=True, expected_code_sha=EXPECTED_SHA,
        )

    assert raised.value.code == 1
    report = _aggregate(tmp_path / "outputs" / "benchmark_results")
    assert report["release_qualification"] == "FAIL"
    assert "RELEASE GATE: PASS" not in capsys.readouterr().out


@pytest.mark.parametrize("drift_field", ["head_sha", "staged_clean", "unstaged_clean"])
def test_post_run_code_identity_drift_fails_release(tmp_path, monkeypatch, drift_field, capsys):
    _install_v1_manifest(tmp_path)
    _mock_release_github(monkeypatch, sha=EXPECTED_SHA)
    monkeypatch.chdir(tmp_path)

    for topic_id in CONTRACT["ordered_topic_ids"]:
        topic = _topic_by_id(topic_id)
        _write_telemetry(tmp_path, f"run-{topic_id}", topic["topic"], verified=10, unverified=1)

    rev_calls = {"n": 0}
    diff_calls = {"n": 0}

    def rev_parse():
        rev_calls["n"] += 1
        if drift_field == "head_sha" and rev_calls["n"] > 1:
            return "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        return EXPECTED_SHA

    def diff_clean(*, staged: bool = False) -> bool:
        diff_calls["n"] += 1
        if drift_field == "staged_clean" and staged and diff_calls["n"] > 2:
            return False
        if drift_field == "unstaged_clean" and not staged and diff_calls["n"] > 2:
            return False
        return True

    monkeypatch.setattr(benchmark, "_git_rev_parse_head", rev_parse)
    monkeypatch.setattr(benchmark, "_git_diff_clean", diff_clean)

    def mock_run(cmd, **kwargs):
        topic_name = cmd[cmd.index("--topic") + 1]
        topic = next(item for item in TOPICS if item["topic"] == topic_name)
        return _completed_proc(f"run-{topic['id']}")

    monkeypatch.setattr(benchmark.subprocess, "run", mock_run)

    with pytest.raises(SystemExit) as raised:
        benchmark.run_benchmark.callback(
            mode="release", limit=None, topic_id=None, gate=True, expected_code_sha=EXPECTED_SHA,
        )

    assert raised.value.code == 1
    report = _aggregate(tmp_path / "outputs" / "benchmark_results")
    assert report["release_qualification"] == "FAIL"
    assert "RELEASE GATE: PASS" not in capsys.readouterr().out


# --- Evidence digest and secret safety ---


def _valid_evidence_payload(**overrides) -> dict:
    evaluation_config, evaluation_config_sha256 = benchmark.resolve_evaluation_config()
    payload = {
        "schema_version": benchmark.EVIDENCE_SCHEMA_VERSION,
        "timestamp_utc": "2026-08-19T12:00:00+00:00",
        "mode": "smoke",
        "release_qualification": "NON_RELEASE",
        "gate_requested": True,
        "github_actions_identity": {
            "github_actions": None,
            "github_ref": None,
            "github_sha": None,
            "github_run_id": None,
            "github_run_attempt": None,
            "github_workflow_ref": None,
        },
        "expected_code_sha": None,
        "preflight_code_identity": None,
        "final_code_identity": None,
        "release_contract_identity": dict(CONTRACT),
        "selected_manifest_identity": {
            "manifest_id": CONTRACT["manifest_id"],
            "manifest_path": CONTRACT["manifest_path"],
            "manifest_sha256": CONTRACT["manifest_sha256"],
            "selected_topic_count": 1,
            "selected_topic_ids": [1],
        },
        "evaluation_configuration": evaluation_config,
        "evaluation_config_sha256": evaluation_config_sha256,
        "ordered_unit_results": [{"id": 1, "status": "success"}],
        "aggregate_metrics": {"total_runs": 1, "successful": 1, "failed": 0},
        "gate_failures": [],
    }
    payload.update(overrides)
    body = dict(payload)
    body.pop("evidence_sha256", None)
    payload["evidence_sha256"] = benchmark._compute_evidence_digest(body)
    return payload


def test_evidence_digest_rejects_mutation_and_missing_digest():
    signed = _valid_evidence_payload()
    benchmark._validate_evidence_payload(signed)

    tampered = dict(signed)
    tampered["mode"] = "release"
    with pytest.raises(ValueError):
        benchmark._validate_evidence_payload(tampered)

    missing = dict(signed)
    missing["evidence_sha256"] = ""
    with pytest.raises(ValueError):
        benchmark._validate_evidence_payload(missing)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "timestamp_utc",
        "mode",
        "release_qualification",
        "gate_requested",
        "github_actions_identity",
        "expected_code_sha",
        "preflight_code_identity",
        "final_code_identity",
        "release_contract_identity",
        "selected_manifest_identity",
        "evaluation_configuration",
        "evaluation_config_sha256",
        "ordered_unit_results",
        "aggregate_metrics",
        "gate_failures",
    ],
)
def test_evidence_rejects_missing_critical_field_even_with_recomputed_digest(field):
    payload = _valid_evidence_payload()
    body = dict(payload)
    body.pop("evidence_sha256", None)
    body.pop(field, None)
    malformed = dict(body)
    malformed["evidence_sha256"] = benchmark._compute_evidence_digest(body)
    with pytest.raises(ValueError):
        benchmark._validate_evidence_payload(malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "invalid"),
        ("release_qualification", "MAYBE"),
        ("expected_code_sha", "not-a-sha"),
        ("evaluation_config_sha256", "0" * 64),
    ],
)
def test_evidence_rejects_invalid_field_values_even_with_recomputed_digest(field, value):
    payload = _valid_evidence_payload(**{field: value})
    with pytest.raises(ValueError):
        benchmark._validate_evidence_payload(payload)


def test_evidence_rejects_tampered_configuration_identity_even_with_valid_digest():
    payload = _valid_evidence_payload()
    tampered_config = dict(payload["evaluation_configuration"])
    tampered_config["PROMPT_VERSION"] = "sha-deadbeefdead"
    body = dict(payload)
    body.pop("evidence_sha256", None)
    body["evaluation_configuration"] = tampered_config
    # Recompute digest over mismatched config + stale evaluation_config_sha256.
    body["evidence_sha256"] = benchmark._compute_evidence_digest(body)
    with pytest.raises(ValueError, match="evaluation_config_sha256 mismatch"):
        benchmark._validate_evidence_payload(body)


def _valid_release_pass_evidence_payload() -> dict:
    evaluation_config, evaluation_config_sha256 = benchmark.resolve_evaluation_config()
    code_identity = {
        "head_sha": EXPECTED_SHA,
        "staged_clean": True,
        "unstaged_clean": True,
    }
    ordered_ids = list(CONTRACT["ordered_topic_ids"])
    payload = {
        "schema_version": benchmark.EVIDENCE_SCHEMA_VERSION,
        "timestamp_utc": "2026-08-19T12:00:00+00:00",
        "mode": "release",
        "release_qualification": "PASS",
        "gate_requested": True,
        "github_actions_identity": {
            "github_actions": "true",
            "github_ref": "refs/heads/main",
            "github_sha": EXPECTED_SHA,
            "github_run_id": "12345",
            "github_run_attempt": "1",
            "github_workflow_ref": "owner/repo/.github/workflows/eval.yml@refs/heads/main",
        },
        "expected_code_sha": EXPECTED_SHA,
        "preflight_code_identity": copy.deepcopy(code_identity),
        "final_code_identity": copy.deepcopy(code_identity),
        "release_contract_identity": copy.deepcopy(CONTRACT),
        "selected_manifest_identity": {
            "manifest_id": CONTRACT["manifest_id"],
            "manifest_path": CONTRACT["manifest_path"],
            "manifest_sha256": CONTRACT["manifest_sha256"],
            "expected_topic_count": CONTRACT["expected_topic_count"],
            "ordered_topic_ids": list(ordered_ids),
            "actual_topic_count": CONTRACT["expected_topic_count"],
            "actual_topic_ids": list(ordered_ids),
        },
        "evaluation_configuration": evaluation_config,
        "evaluation_config_sha256": evaluation_config_sha256,
        "ordered_unit_results": [
            _release_result(_topic_by_id(topic_id), f"run-{topic_id:02d}")
            for topic_id in ordered_ids
        ],
        "aggregate_metrics": {"total_runs": 20, "successful": 20, "failed": 0},
        "gate_failures": [],
    }
    payload["aggregate_metrics"] = benchmark._aggregate_metrics(payload["ordered_unit_results"])
    body = dict(payload)
    payload["evidence_sha256"] = benchmark._compute_evidence_digest(body)
    return payload


def _recompute_evidence_digest(payload: dict) -> dict:
    body = dict(payload)
    body.pop("evidence_sha256", None)
    payload["evidence_sha256"] = benchmark._compute_evidence_digest(body)
    return payload


def _tamper_remove_github_run_id(payload: dict) -> None:
    identity = dict(payload["github_actions_identity"])
    identity.pop("github_run_id")
    payload["github_actions_identity"] = identity


def _tamper_change_github_ref(payload: dict) -> None:
    identity = dict(payload["github_actions_identity"])
    identity["github_ref"] = "refs/heads/feature"
    payload["github_actions_identity"] = identity


def _tamper_change_github_sha(payload: dict) -> None:
    identity = dict(payload["github_actions_identity"])
    identity["github_sha"] = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    payload["github_actions_identity"] = identity


def _tamper_preflight_staged_dirty(payload: dict) -> None:
    identity = dict(payload["preflight_code_identity"])
    identity["staged_clean"] = False
    payload["preflight_code_identity"] = identity


def _tamper_change_final_head(payload: dict) -> None:
    identity = dict(payload["final_code_identity"])
    identity["head_sha"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    payload["final_code_identity"] = identity


def _tamper_contract_manifest_sha(payload: dict) -> None:
    identity = dict(payload["release_contract_identity"])
    identity["manifest_sha256"] = "0" * 64
    payload["release_contract_identity"] = identity


def _tamper_contract_expected_count(payload: dict) -> None:
    identity = dict(payload["release_contract_identity"])
    identity["expected_topic_count"] = 19
    payload["release_contract_identity"] = identity


def _tamper_selected_manifest_sha(payload: dict) -> None:
    identity = dict(payload["selected_manifest_identity"])
    identity["manifest_sha256"] = "0" * 64
    payload["selected_manifest_identity"] = identity


def _tamper_selected_topic_ids_order(payload: dict) -> None:
    identity = dict(payload["selected_manifest_identity"])
    identity["actual_topic_ids"] = list(reversed(identity["actual_topic_ids"]))
    payload["selected_manifest_identity"] = identity


def _tamper_remove_one_result(payload: dict) -> None:
    payload["ordered_unit_results"] = list(payload["ordered_unit_results"][:-1])


def _tamper_duplicate_result_id(payload: dict) -> None:
    results = [dict(result) for result in payload["ordered_unit_results"]]
    results[1]["id"] = results[0]["id"]
    payload["ordered_unit_results"] = results


def _tamper_duplicate_run_id(payload: dict) -> None:
    results = [dict(result) for result in payload["ordered_unit_results"]]
    results[1]["run_id"] = results[0]["run_id"]
    payload["ordered_unit_results"] = results


def _tamper_nonempty_gate_failures(payload: dict) -> None:
    payload["gate_failures"] = ["injected gate failure"]


def _tamper_gate_requested_false(payload: dict) -> None:
    payload["gate_requested"] = False


def _copy_first_result(payload: dict) -> dict:
    results = [dict(result) for result in payload["ordered_unit_results"]]
    results[0] = dict(results[0])
    telemetry = results[0].get("telemetry")
    if isinstance(telemetry, dict):
        results[0]["telemetry"] = dict(telemetry)
    payload["ordered_unit_results"] = results
    return results[0]


def _tamper_status_failed(payload: dict) -> None:
    _copy_first_result(payload)["status"] = "failed"


def _tamper_subprocess_exit_code(payload: dict) -> None:
    _copy_first_result(payload)["subprocess_exit_code"] = 7


def _tamper_uvr_one(payload: dict) -> None:
    _copy_first_result(payload)["uvr"] = 1.0


def _tamper_evaluation_unscorable(payload: dict) -> None:
    _copy_first_result(payload)["evaluation_status"] = "unscorable"


def _tamper_incorrect_result_topic(payload: dict) -> None:
    _copy_first_result(payload)["topic"] = "Not A Frozen V1 Topic"


def _tamper_telemetry_run_id_mismatch(payload: dict) -> None:
    _copy_first_result(payload)["telemetry"]["run_id"] = "other-run"


def _tamper_telemetry_topic_mismatch(payload: dict) -> None:
    _copy_first_result(payload)["telemetry"]["topic"] = "Other Topic"


def _tamper_telemetry_parse_failed(payload: dict) -> None:
    _copy_first_result(payload)["telemetry"]["verification_status"] = "parse_failed"


def _tamper_result_verification_not_completed(payload: dict) -> None:
    _copy_first_result(payload)["verification_status"] = "parse_failed"


def _tamper_zero_verdict_counts(payload: dict) -> None:
    result = _copy_first_result(payload)
    result["telemetry"]["claims_verified"] = 0
    result["telemetry"]["claims_weak"] = 0
    result["telemetry"]["claims_unverified"] = 0


def _tamper_telemetry_prompt_version(payload: dict) -> None:
    _copy_first_result(payload)["telemetry"]["prompt_version"] = "sha-deadbeefdead"


def _tamper_telemetry_prompt_hashes(payload: dict) -> None:
    _copy_first_result(payload)["telemetry"]["prompt_hashes"] = {"draft_system": "deadbeef"}


def _tamper_aggregate_successful(payload: dict) -> None:
    payload["aggregate_metrics"] = dict(payload["aggregate_metrics"])
    payload["aggregate_metrics"]["successful"] = 19


def _tamper_aggregate_failed(payload: dict) -> None:
    payload["aggregate_metrics"] = dict(payload["aggregate_metrics"])
    payload["aggregate_metrics"]["failed"] = 1


def _tamper_aggregate_unscorable(payload: dict) -> None:
    payload["aggregate_metrics"] = dict(payload["aggregate_metrics"])
    payload["aggregate_metrics"]["unscorable"] = 1


def _tamper_aggregate_total_runs(payload: dict) -> None:
    payload["aggregate_metrics"] = dict(payload["aggregate_metrics"])
    payload["aggregate_metrics"]["total_runs"] = 19


def test_valid_release_pass_evidence_validates():
    payload = _valid_release_pass_evidence_payload()
    benchmark._validate_evidence_payload(payload)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (_tamper_remove_github_run_id, "github_run_id"),
        (_tamper_change_github_ref, "github_ref"),
        (_tamper_change_github_sha, "github_sha"),
        (_tamper_preflight_staged_dirty, "preflight staged_clean"),
        (_tamper_change_final_head, "stable code identity|final HEAD"),
        (_tamper_contract_manifest_sha, "manifest_sha256"),
        (_tamper_contract_expected_count, "expected_topic_count"),
        (_tamper_selected_manifest_sha, "manifest SHA"),
        (_tamper_selected_topic_ids_order, "actual_topic_ids"),
        (_tamper_remove_one_result, "exactly 20 ordered unit results"),
        (_tamper_duplicate_result_id, "1..20 in order"),
        (_tamper_duplicate_run_id, "run IDs must be unique"),
        (_tamper_nonempty_gate_failures, "empty gate_failures"),
        (_tamper_gate_requested_false, "gate_requested"),
    ],
    ids=[
        "remove_github_run_id",
        "change_github_ref",
        "change_github_sha",
        "preflight_staged_dirty",
        "change_final_head",
        "contract_manifest_sha",
        "contract_expected_count",
        "selected_manifest_sha",
        "selected_topic_ids_order",
        "remove_one_result",
        "duplicate_result_id",
        "duplicate_run_id",
        "nonempty_gate_failures",
        "gate_requested_false",
    ],
)
def test_release_pass_rejects_nested_tamper_even_with_recomputed_digest(mutator, match):
    payload = _valid_release_pass_evidence_payload()
    benchmark._validate_evidence_payload(payload)
    mutator(payload)
    _recompute_evidence_digest(payload)
    with pytest.raises(ValueError, match=match):
        benchmark._validate_evidence_payload(payload)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (_tamper_status_failed, "CLI failed"),
        (_tamper_subprocess_exit_code, "subprocess exit code 7"),
        (_tamper_uvr_one, "UVR"),
        (_tamper_evaluation_unscorable, "evaluation_status=unscorable"),
        (_tamper_incorrect_result_topic, "frozen V1 manifest"),
        (_tamper_telemetry_run_id_mismatch, "telemetry run_id mismatch"),
        (_tamper_telemetry_topic_mismatch, "telemetry topic mismatch"),
        (_tamper_telemetry_parse_failed, "telemetry verification_status"),
        (_tamper_result_verification_not_completed, "verification_status=parse_failed"),
        (_tamper_zero_verdict_counts, "zero-verdict"),
        (_tamper_telemetry_prompt_version, "prompt_version mismatch"),
        (_tamper_telemetry_prompt_hashes, "prompt_hashes mismatch"),
        (_tamper_aggregate_successful, "aggregate successful"),
        (_tamper_aggregate_failed, "aggregate failed"),
        (_tamper_aggregate_unscorable, "aggregate unscorable"),
        (_tamper_aggregate_total_runs, "aggregate total_runs"),
    ],
    ids=[
        "status_failed",
        "subprocess_exit_code",
        "uvr_one",
        "evaluation_unscorable",
        "incorrect_result_topic",
        "telemetry_run_id",
        "telemetry_topic",
        "telemetry_parse_failed",
        "result_verification_not_completed",
        "zero_verdicts",
        "prompt_version",
        "prompt_hashes",
        "aggregate_successful",
        "aggregate_failed",
        "aggregate_unscorable",
        "aggregate_total_runs",
    ],
)
def test_release_pass_rejects_semantic_tamper_even_with_recomputed_digest(mutator, match):
    payload = _valid_release_pass_evidence_payload()
    benchmark._validate_evidence_payload(payload)
    mutator(payload)
    _recompute_evidence_digest(payload)
    with pytest.raises(ValueError, match=match):
        benchmark._validate_evidence_payload(payload)


def test_tampered_configuration_identity_cannot_produce_release_pass(tmp_path, monkeypatch, capsys):
    _install_v1_manifest(tmp_path)
    _mock_release_github(monkeypatch, sha=EXPECTED_SHA)
    _mock_git_identity(monkeypatch, sha=EXPECTED_SHA)
    monkeypatch.chdir(tmp_path)

    run_ids = {topic_id: f"run-{topic_id:02d}" for topic_id in CONTRACT["ordered_topic_ids"]}
    for topic_id, run_id in run_ids.items():
        topic = _topic_by_id(topic_id)
        _write_telemetry(tmp_path, run_id, topic["topic"], verified=17, unverified=3)

    def mock_run(cmd, **kwargs):
        if cmd[0:3] == ["uv", "run", "python"]:
            topic_name = cmd[cmd.index("--topic") + 1]
            topic = next(item for item in TOPICS if item["topic"] == topic_name)
            return _completed_proc(run_ids[topic["id"]])
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(benchmark.subprocess, "run", mock_run)

    benchmark.run_benchmark.callback(
        mode="release", limit=None, topic_id=None, gate=True, expected_code_sha=EXPECTED_SHA,
    )

    report = _aggregate(tmp_path / "outputs" / "benchmark_results")
    assert report["release_qualification"] == "PASS"
    tampered = dict(report)
    tampered["evaluation_config_sha256"] = "0" * 64
    body = dict(tampered)
    body.pop("evidence_sha256", None)
    tampered["evidence_sha256"] = benchmark._compute_evidence_digest(body)
    with pytest.raises(ValueError, match="evaluation_config_sha256 mismatch"):
        benchmark._validate_evidence_payload(tampered)


def test_evaluation_config_excludes_raw_secret_urls():
    config, digest = benchmark.resolve_evaluation_config()
    serialized = json.dumps(config)

    assert "api_key" not in serialized.lower()
    assert "DEEPSEEK_BASE_URL_SHA256" in config
    assert "QDRANT_URL_SHA256" in config
    assert digest == hashlib.sha256(benchmark._canonical_json(config).encode("utf-8")).hexdigest()


def test_report_path_and_config_contain_no_injected_credentials(tmp_path, monkeypatch):
    _install_v1_manifest(tmp_path)
    topic = _topic_by_id(1)
    secret = "sk-live-super-secret-key"
    _write_telemetry(tmp_path, "new", topic["topic"], verified=5, unverified=0)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", f"https://{secret}@api.deepseek.com/v1")
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: _completed_proc("new"))

    benchmark.run_benchmark.callback(mode="smoke", limit=None, topic_id=1, gate=True, expected_code_sha=None)

    report_text = next((tmp_path / "outputs" / "benchmark_results").glob("benchmark_*.json")).read_text()
    assert secret not in report_text


# --- Workflow structure ---


def test_eval_workflow_release_guard_precedes_provider_steps():
    workflow = (ROOT / ".github/workflows/eval.yml").read_text()
    guard_index = workflow.index("Reject release mode off main branch")
    checkout_index = workflow.index("actions/checkout@v4")
    sync_index = workflow.index("uv sync")
    ingest_index = workflow.index("Ingest KB")
    release_index = workflow.index("--expected-code-sha ${{ github.sha }}")

    assert guard_index < checkout_index < sync_index < ingest_index
    assert release_index > guard_index
    assert 'default: smoke' in workflow
    assert 'mode release --gate --expected-code-sha ${{ github.sha }}' in workflow.replace("\n", " ")


def test_cli_requires_mode():
    with pytest.raises(click.ClickException):
        benchmark.run_benchmark.main(["--gate"], standalone_mode=False)
