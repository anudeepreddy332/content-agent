"""Regression coverage for evaluation paths that must fail closed."""
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from main import _write_telemetry as write_run_telemetry
from scripts import benchmark, check_telemetry_fields


ROOT = Path(__file__).resolve().parents[1]


def _telemetry(
    run_id: str,
    topic: str,
    *,
    verification_status: str | None = "completed",
    verified: int = 0,
    weak: int = 0,
    unverified: int = 0,
) -> dict:
    record = {
        "run_id": run_id,
        "topic": topic,
        "slug": "topic",
        "timestamp": "2026-08-12T00:00:00+00:00",
        "prompt_version": "test",
        "prompt_hashes": {},
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
        "kb_context_stats": {
            "candidate_children": 0,
            "candidate_unique_sources": 0,
            "draft": {},
            "verifier": {},
        },
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
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_telemetry(run_id, topic, **kwargs)), encoding="utf-8")
    return path


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


def _completed_proc(run_id: str) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=f"RUN_ID={run_id}\n", stderr="")


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


def _run_benchmark(tmp_path, monkeypatch, topic, telemetry):
    topics_path = tmp_path / "evals" / "topics.json"
    topics_path.parent.mkdir(parents=True)
    topics_path.write_text(json.dumps([topic]), encoding="utf-8")
    path = _write_telemetry(tmp_path, "new", topic["topic"])
    path.write_text(json.dumps(telemetry), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: _completed_proc("new"),
    )
    return tmp_path / "outputs" / "benchmark_results"


def _topic(*, allow_zero_claims: bool = False) -> dict:
    topic = {"id": 1, "topic": "Gradient Descent", "card_id": "standalone", "series": "Test"}
    if allow_zero_claims:
        topic["allow_zero_claims"] = True
    return topic


def _aggregate(results_dir: Path) -> dict:
    return json.loads(next(results_dir.glob("benchmark_*.json")).read_text())


def test_benchmark_rejects_zero_verdict_parse_failure(tmp_path, monkeypatch, capsys):
    topic = _topic()
    results_dir = _run_benchmark(
        tmp_path,
        monkeypatch,
        topic,
        _telemetry("new", topic["topic"], verification_status="parse_failed"),
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.run_benchmark.callback(limit=None, topic_id=None, gate=True)

    assert raised.value.code == 1
    assert "uvr=N/A" in capsys.readouterr().out
    result = _aggregate(results_dir)["runs"][0]
    assert result["uvr"] is None
    assert result["verification_status"] == "parse_failed"


def test_benchmark_allows_explicit_completed_zero_claim_case_and_excludes_it_from_uvr(tmp_path, monkeypatch):
    topic = _topic(allow_zero_claims=True)
    results_dir = _run_benchmark(tmp_path, monkeypatch, topic, _telemetry("new", topic["topic"]))

    benchmark.run_benchmark.callback(limit=None, topic_id=None, gate=True)

    aggregate = _aggregate(results_dir)
    assert aggregate["mean_unverified_rate"] is None
    assert aggregate["unscorable"][0]["evaluation_status"] == "allowed_zero_claims"


def test_benchmark_rejects_completed_zero_claim_case_without_opt_in(tmp_path, monkeypatch):
    topic = _topic()
    results_dir = _run_benchmark(tmp_path, monkeypatch, topic, _telemetry("new", topic["topic"]))

    with pytest.raises(SystemExit) as raised:
        benchmark.run_benchmark.callback(limit=None, topic_id=None, gate=True)

    assert raised.value.code == 1
    assert _aggregate(results_dir)["runs"][0]["evaluation_status"] == "unscorable_incomplete"


@pytest.mark.parametrize(
    ("verified", "unverified", "should_fail"),
    [(1, 0, False), (17, 3, False), (16, 4, True)],
)
def test_benchmark_preserves_uvr_gate_boundaries(tmp_path, monkeypatch, verified, unverified, should_fail):
    topic = _topic()
    _run_benchmark(
        tmp_path,
        monkeypatch,
        topic,
        _telemetry("new", topic["topic"], verified=verified, unverified=unverified),
    )

    if should_fail:
        with pytest.raises(SystemExit) as raised:
            benchmark.run_benchmark.callback(limit=None, topic_id=None, gate=True)
        assert raised.value.code == 1
    else:
        benchmark.run_benchmark.callback(limit=None, topic_id=None, gate=True)


def test_historical_telemetry_without_status_is_unknown_and_unscorable():
    outcome = benchmark.verification_outcome(_telemetry("old", "Gradient Descent", verification_status=None), _topic())

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
