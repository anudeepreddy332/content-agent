"""Frozen protocol validation using public fixtures and isolated operator state."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

import scripts.phase5e1_qualification_runner as runner

_spec = importlib.util.spec_from_file_location(
    "public_protocol_support", Path(__file__).with_name("test_phase5e1_qualification_runner.py")
)
support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(support)
_real_control_validation = runner.validate_approved_execution_controls


def test_initial_preflight_uses_uncached_live_validator(monkeypatch):
    from agent.kb_backend import qdrant_serving

    observed = []
    monkeypatch.setattr(runner, "_validate_qualified_preflight", lambda expected, validator: observed.append(validator))
    runner.validate_real_adapter_preflight({})
    assert observed == [qdrant_serving.validate_fresh_release_preflight]


@pytest.fixture
def production(tmp_path, monkeypatch):
    fixture = json.loads(support.FIXTURE.read_text())
    fixture["holdout"] = True  # Public mechanism eligibility only.
    approval_sha = "a" * 40
    monkeypatch.setattr(runner, "runtime_git_sha", lambda: approval_sha)
    monkeypatch.setattr(runner, "validate_approved_execution_controls", lambda _approval: None)
    support._trusted_approval(tmp_path, monkeypatch, fixture=fixture, execution_sha=approval_sha)
    identity = fixture["expected_runtime_identity"]
    monkeypatch.setattr(runner, "derive_trusted_qualified_runtime_identity", lambda _contract: identity)
    adapter = support.executor()
    source_map = {}
    for query in fixture["queries"]:
        output = adapter(query["query"])["cswp_local"]
        for field in runner.CANONICAL_INTERVAL_FIELDS:
            for row in output[field]:
                source_map[row["chunk_id"]] = {
                    "source": row["source"],
                    "source_intervals": tuple(tuple(span) for span in row["source_intervals"]),
                }
    calls = []

    def execute(backend, query):
        calls.append((backend, query))
        return adapter(query)[backend]

    monkeypatch.setattr(runner, "_execute_authoritative_backend", execute)
    from agent.kb_backend.qdrant_serving import SERVING_ALIAS

    preflight = {
        "cswp_local": {**identity["cswp_local"], "unit_count": 159},
        "cswp_qdrant": {**identity["cswp_qdrant"], "point_count": 159, "startup_validated": True, "serving_alias": SERVING_ALIAS},
    }
    monkeypatch.setattr(runner, "validate_real_adapter_preflight", lambda _identity: (source_map, preflight))
    monkeypatch.setattr(runner, "validate_fresh_release_qdrant_preflight", lambda _identity: (source_map, preflight))
    root = tmp_path / "production-attempt"
    monkeypatch.setattr(runner, "DEFAULT_ARCHIVE_ROOT", root)
    return root, calls


def journal(root):
    path = root / "execution_journal.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_public_production_path_uses_real_state_machine(production):
    root, calls = production
    report = runner.run_authoritative_qualification()
    assert report["state"] == "PASS"
    assert len(calls) == 20
    assert runner.verify_authoritative_release(root) == "PHASE-5E1A-HOLDOUT-PASS"
    assert set(path.name for path in root.iterdir()) == set(runner.AUTH_PROTOCOL_ARTIFACTS) | {"manifest.json"}


def test_completed_verifier_rejects_missing_invocation_grammar(production):
    root, _calls = production
    assert runner.run_authoritative_qualification()["state"] == "PASS"
    records = [record for record in journal(root) if record["event"] != "backend_invocation_intent"]
    (root / "execution_journal.jsonl").write_text("".join(runner.canonical_json_dumps(row) + "\n" for row in records))
    support._refresh_protocol_manifest(root)
    assert runner.verify_authoritative_release(root) != "PHASE-5E1A-HOLDOUT-PASS"


def test_truthful_gate_fail_is_distinct_from_incomplete(production, monkeypatch):
    root, _calls = production
    original = runner.evaluate_evidence_gates

    def fail_gate(*args, **kwargs):
        result = original(*args, **kwargs)
        result["hard_failures"].append({"gate": "public_frozen_gate_failure"})
        result["overall_pass"] = False
        return result

    monkeypatch.setattr(runner, "evaluate_evidence_gates", fail_gate)
    assert runner.run_authoritative_qualification()["state"] == "FAIL"
    assert json.loads((root / "disposition.json").read_text())["disposition"] == "FAIL"


def test_local_validation_precedes_qdrant_invocation(production, monkeypatch):
    root, calls = production
    original = runner._execute_authoritative_backend

    def bad_local(backend, query):
        result = original(backend, query)
        if backend == "cswp_local":
            result["index_fingerprint"] = "invalid-local-index"
        return result

    monkeypatch.setattr(runner, "_execute_authoritative_backend", bad_local)
    assert runner.run_authoritative_qualification()["state"] in {"FAIL", "INCOMPLETE"}
    assert [backend for backend, _query in calls] == ["cswp_local"]
    assert journal(root)[-1]["event"] == "execution_completed"


def test_final_preflight_exception_after_manifest_never_repairs_archive(production, monkeypatch):
    root, _calls = production
    original = runner.validate_fresh_release_qdrant_preflight
    calls = 0

    def failure(identity):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("public completed-live-check failure")
        return original(identity)

    monkeypatch.setattr(runner, "validate_fresh_release_qdrant_preflight", failure)
    report = runner.run_authoritative_qualification()
    assert report["overall_pass"] is False
    assert (root / "manifest.json").exists()


TRANSITIONS = (
    ("approval", "load_trusted_release_approval", 1),
    ("execution_control", "runtime_git_sha", 1),
    ("control_bytes", "validate_approved_execution_controls", 1),
    ("oracle", "load_approved_fixture_snapshot", 1),
    ("runtime", "derive_trusted_qualified_runtime_identity", 1),
    ("initial_preflight", "validate_real_adapter_preflight", 1),
    ("reservation", "reserve_protocol_archive", 1),
    ("execution_transition", "_protocol_execution", 2),
    ("materialization", "write_durable_protocol_json", 5),
    ("determinism", "compare_deterministic_executions", 1),
    ("score_truth", "score_query", 1),
    ("gates", "evaluate_evidence_gates", 1),
    ("final_preflight", "validate_fresh_release_qdrant_preflight", 1),
    ("readiness", "_verify_protocol_archive", 1),
    ("terminal_verification", "write_durable_protocol_json", 16),
    ("terminal_state", "write_durable_protocol_json", 17),
    ("disposition", "write_durable_protocol_json", 18),
    ("manifest", "_protocol_manifest", 1),
    ("completed_verification", "verify_completed_protocol_archive", 1),
)


@pytest.mark.parametrize("category,function,ordinal", TRANSITIONS, ids=[row[0] for row in TRANSITIONS])
def test_transition_failure_never_creates_verified_pass(production, monkeypatch, category, function, ordinal):
    root, calls = production
    original = getattr(runner, function)
    observed = 0

    def failure(*args, **kwargs):
        nonlocal observed
        observed += 1
        if observed == ordinal:
            raise OSError(f"public transition failure: {category}")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, function, failure)
    try:
        report = runner.run_authoritative_qualification()
    except (OSError, runner.QualificationHarnessError):
        report = {"overall_pass": False}
    assert report["overall_pass"] is False
    assert observed >= ordinal
    assert len(calls) <= 20
    completed_counts = {
        "execution_transition": 5, "materialization": 10, "determinism": 10,
        "score_truth": 0, "gates": 5, "final_preflight": 10, "readiness": 10,
        "terminal_verification": 10, "terminal_state": 10, "disposition": 10,
        "manifest": 10, "completed_verification": 10,
    }
    if category in completed_counts:
        assert len([row for row in journal(root) if row["event"] == "query_completed"]) == completed_counts[category]
        for name in ("attempt.json", "oracle_identity.json", "contract.json", "pre_retrieval_preflight.json"):
            assert json.loads((root / name).read_text())
        for name in ("terminal_state.json", "disposition.json"):
            if (root / name).exists():
                assert json.loads((root / name).read_text())["pass"] is False
    else:
        assert not root.exists()
    if not (root / "manifest.json").exists():
        assert not any(row.get("state") == "PASS" for row in journal(root))


@pytest.mark.parametrize("ordinal", range(1, 21))
def test_every_backend_invocation_is_one_shot(production, monkeypatch, ordinal):
    root, calls = production
    original = runner._execute_authoritative_backend
    count = 0

    def failure(backend, query):
        nonlocal count
        count += 1
        if count == ordinal:
            raise OSError("uncertain one-shot backend call")
        return original(backend, query)

    monkeypatch.setattr(runner, "_execute_authoritative_backend", failure)
    report = runner.run_authoritative_qualification()
    assert report["state"] == "INCOMPLETE"
    assert count == ordinal
    assert len(calls) == ordinal - 1
    assert len([row for row in journal(root) if row["event"] == "backend_invocation_intent"]) == ordinal
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize("ordinal", range(1, 11))
@pytest.mark.parametrize("boundary", ("before", "after"))
def test_every_query_completion_abrupt_death_preserves_predecessors(production, monkeypatch, ordinal, boundary):
    root, _calls = production
    original = runner.append_durable_protocol_journal
    count = 0

    def death(archive, record):
        nonlocal count
        if record["event"] == "query_completed":
            count += 1
            if count == ordinal and boundary == "before":
                os._exit(73)
        original(archive, record)
        if record["event"] == "query_completed" and count == ordinal and boundary == "after":
            os._exit(73)

    monkeypatch.setattr(runner, "append_durable_protocol_journal", death)
    pid = os.fork()
    if pid == 0:
        runner.run_authoritative_qualification()
        os._exit(74)
    _pid, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 73
    completed = [row for row in journal(root) if row["event"] == "query_completed"]
    assert len(completed) == ordinal - (boundary == "before")
    assert not (root / "manifest.json").exists()


DURABLE_CLASSES = ("attempt.json", "execution_journal.jsonl", "executions.json", "authoritative_verification.json", "terminal_state.json", "manifest.json")
DURABLE_MODES = ("pre_open", "collision", "partial_write", "file_fsync", "directory_fsync", "death_before", "death_after")


@pytest.mark.parametrize("artifact,mode", [(artifact, mode) for artifact in DURABLE_CLASSES for mode in DURABLE_MODES if (artifact, mode) != ("execution_journal.jsonl", "collision")])
def test_durable_io_failure_classes(production, monkeypatch, artifact, mode):
    root, _calls = production
    original_open, original_write, original_fsync = os.open, os.write, os.fsync
    fd_paths = {}
    fired = False

    def open_fault(path, flags, *args, **kwargs):
        nonlocal fired
        target = Path(path) == root / artifact
        if target and not fired and mode == "pre_open":
            fired = True
            raise OSError("injected pre-open failure")
        if target and not fired and mode == "collision" and flags & os.O_EXCL:
            fired = True
            fd = original_open(path, flags, *args, **kwargs)
            os.close(fd)
        fd = original_open(path, flags, *args, **kwargs)
        fd_paths[fd] = Path(path)
        return fd

    def write_fault(fd, data):
        nonlocal fired
        if fd_paths.get(fd) == root / artifact and not fired and mode == "partial_write":
            fired = True
            original_write(fd, data[:max(1, len(data) // 2)])
            raise OSError("injected partial write")
        return original_write(fd, data)

    def fsync_fault(fd):
        nonlocal fired
        file_target = fd_paths.get(fd) == root / artifact
        directory_target = fd_paths.get(fd) == root and (root / artifact).exists()
        if not fired and ((file_target and mode in {"file_fsync", "death_before", "death_after"}) or (directory_target and mode == "directory_fsync")):
            fired = True
            if mode == "death_before":
                os._exit(73)
            if mode == "death_after":
                original_fsync(fd)
                os._exit(73)
            raise OSError("injected sync failure")
        return original_fsync(fd)

    monkeypatch.setattr(os, "open", open_fault)
    monkeypatch.setattr(os, "write", write_fault)
    monkeypatch.setattr(os, "fsync", fsync_fault)
    if mode.startswith("death"):
        pid = os.fork()
        if pid == 0:
            runner.run_authoritative_qualification()
            os._exit(74)
        _pid, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 73
    else:
        try:
            report = runner.run_authoritative_qualification()
        except (OSError, runner.QualificationHarnessError):
            report = {"overall_pass": False}
        assert report["overall_pass"] is False
        assert fired
        if artifact == "manifest.json":
            assert runner.verify_authoritative_release(root) != "PHASE-5E1A-HOLDOUT-PASS"
    # A partial last append is uncertain, never a completed query. All earlier
    # complete records remain usable without overwriting the failed attempt.
    journal_path = root / "execution_journal.jsonl"
    records = [json.loads(line) for line in journal_path.read_text().splitlines(keepends=True) if line.endswith("\n")] if journal_path.exists() else []
    expected_completed = 0 if artifact in {"attempt.json", "execution_journal.jsonl"} else 10
    assert len([row for row in records if row["event"] == "query_completed"]) == expected_completed


def test_readiness_verification_prevents_corrupt_materialization_commit(production, monkeypatch):
    root, _calls = production
    original = runner.write_durable_protocol_json

    def damage(archive, name, value):
        if name == "per_query.json":
            value = {"execution_1": [], "execution_2": []}
        return original(archive, name, value)

    monkeypatch.setattr(runner, "write_durable_protocol_json", damage)
    try:
        report = runner.run_authoritative_qualification()
    except runner.QualificationHarnessError:
        report = {"overall_pass": False}
    assert report["overall_pass"] is False
    assert not (root / "manifest.json").exists()


def test_completed_verifier_recomputes_provenance(production):
    root, _calls = production
    assert runner.run_authoritative_qualification()["state"] == "PASS"
    path = root / "provenance.json"
    value = json.loads(path.read_text())
    value["execution_1"] = {"passed": True, "mismatches": [], "forged": True}
    path.write_text(runner.canonical_json_dumps(value) + "\n")
    support._refresh_protocol_manifest(root)
    assert runner.verify_authoritative_release(root) != "PHASE-5E1A-HOLDOUT-PASS"


@pytest.mark.parametrize("primitive", ("reserve", "immutable", "journal"))
def test_durable_writes_sync_file_and_directory(production, monkeypatch, primitive):
    root, _calls = production
    observed = []
    original_open, original_fsync = os.open, os.fsync
    paths = {}

    def observe_open(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        paths[fd] = Path(path)
        return fd

    def observe_fsync(fd):
        observed.append(paths.get(fd))
        return original_fsync(fd)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "fsync", observe_fsync)
    report = runner.run_authoritative_qualification()
    assert report["state"] == "PASS"
    assert root.parent in observed
    assert root in observed
    if primitive != "reserve":
        artifact = root / ("attempt.json" if primitive == "immutable" else "execution_journal.jsonl")
        index = observed.index(artifact)
        assert observed[index + 1] == root


def test_live_preflight_timing_variation_preserves_identity(production, monkeypatch):
    root, _calls = production
    count = 0
    original = runner.validate_fresh_release_qdrant_preflight

    def timed(identity):
        nonlocal count
        count += 1
        source_map, evidence = original(identity)
        return source_map, {**evidence, "cswp_qdrant": {**evidence["cswp_qdrant"], "hydrate_ms": count}}

    monkeypatch.setattr(runner, "validate_fresh_release_qdrant_preflight", timed)
    assert runner.run_authoritative_qualification()["state"] == "PASS"
    assert (root / "manifest.json").exists()


def test_protocol_path_custody_rejects_unsafe_files(production):
    root, _calls = production
    assert runner.run_authoritative_qualification()["state"] == "PASS"
    (root / "summary.json").chmod(0o644)
    assert runner.verify_authoritative_release(root) != "PHASE-5E1A-HOLDOUT-PASS"


def test_production_archive_root_is_not_caller_selectable(production):
    with pytest.raises(TypeError):
        runner.run_authoritative_qualification(archive_root=production[0])


def test_archive_never_copies_oracle_bytes(production):
    root, _calls = production
    runner.run_authoritative_qualification()
    oracle = json.loads((root / "oracle_identity.json").read_text())
    assert "queries" not in oracle and "raw_bytes" not in oracle
    assert not (root / "oracle_fixture.json").exists()
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in root.iterdir())


def test_production_vs_development_authority_separation(production):
    root, _calls = production
    result = runner.run_development_evaluation(fixture_path=support.FIXTURE, executor=support.executor())
    assert result["decision_scope"] == "development"
    assert result["disposition"] != "PHASE-5E1A-HOLDOUT-PASS"
    assert not root.exists()


@pytest.mark.parametrize("case", ("missing", "corrupt", "self_inventory", "wrong_protocol", "wrong_terminal", "extra_inventory", "wrong_digest"))
def test_seven_manifest_cases(production, case):
    root, _calls = production
    runner.run_authoritative_qualification()
    path = root / "manifest.json"
    value = json.loads(path.read_text())
    if case == "missing":
        path.unlink()
    elif case == "corrupt":
        path.write_text("not-json")
    else:
        if case == "self_inventory":
            value["artifacts"]["manifest.json"] = {}
        elif case == "wrong_protocol":
            value["protocol"] = "other"
        elif case == "wrong_terminal":
            value["terminal_artifact"] = "summary.json"
        elif case == "extra_inventory":
            value["artifacts"]["extra.json"] = {}
        else:
            value["artifacts"]["attempt.json"]["sha256"] = "0" * 64
        path.write_text(runner.canonical_json_dumps(value) + "\n")
    assert runner.verify_authoritative_release(root) != "PHASE-5E1A-HOLDOUT-PASS"


def test_no_post_manifest_repair_into_pass(production):
    root, _calls = production
    runner.run_authoritative_qualification()
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(runner.QualificationHarnessError):
        runner.write_durable_protocol_json(root, "repair.json", {"passed": True})
    with pytest.raises(runner.QualificationHarnessError):
        runner.append_durable_protocol_journal(root, {"event": "repair"})
    assert (root / "manifest.json").read_bytes() == before


def test_only_completed_verifier_owns_pass(production, monkeypatch):
    root, _calls = production
    monkeypatch.setattr(runner, "verify_completed_protocol_archive", lambda *_args, **_kwargs: "public_verification_rejected")
    assert runner.run_authoritative_qualification() == {
        "state": "COMMITTED_UNVERIFIED", "overall_pass": False, "verdict": "public_verification_rejected",
    }
    assert json.loads((root / "terminal_state.json").read_text())["pass"] is False


def test_completed_verifier_recomputes_determinism_and_duplicates():
    from agent.drafter_packed_evidence import serialize_drafter_packed_evidence_v1

    unit = {"retrieval_text": "trusted public text"}
    row = {
        "chunk_id": "public__0001", "source": "public", "text": "trusted public text",
        "source_intervals": [[0, 19]], "seed_rank": 1, "seed_chunk_id": "public__0001",
        "relation": "seed", "source_path": "public.md", "document_id": "public",
        "document_version": "1", "source_sha256": "public-digest",
    }
    raw = {field: [dict(row)] for field in runner.CANONICAL_INTERVAL_FIELDS}
    raw["packed_rows"] = [dict(row)]
    raw["serialized_groups"] = serialize_drafter_packed_evidence_v1(raw["packed_rows"], {row["chunk_id"]: unit})
    raw["packed_fingerprint"] = "fabricated-fingerprint"
    mapping = {row["chunk_id"]: {"source": "public", "source_intervals": ((0, 19),), "qualified_unit": unit}}
    with pytest.raises(runner.QualificationHarnessError):
        runner.canonical_raw(raw, canonical_source_map=mapping)


def test_authorization_and_control_checks_precede_oracle_parse(production, monkeypatch):
    _root, calls = production
    import subprocess

    monkeypatch.setattr(runner, "validate_approved_execution_controls", _real_control_validation)

    original = runner.subprocess.run

    def drift(command, *args, **kwargs):
        if command[:3] == ["git", "diff", "--quiet"]:
            return subprocess.CompletedProcess(command, 1)
        return original(command, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", drift)
    parsed = []
    original_parse = runner._parse_verified_fixture_bytes

    def parse(raw):
        parsed.append(True)
        return original_parse(raw)

    monkeypatch.setattr(runner, "_parse_verified_fixture_bytes", parse)
    with pytest.raises(runner.QualificationHarnessError):
        runner.run_authoritative_qualification()
    assert parsed == [] and calls == []


@pytest.mark.parametrize("artifact", ("pre_retrieval_preflight.json", "final_live_preflight.json"))
def test_archived_preflight_cannot_substitute_pass_flags(production, artifact):
    root, _calls = production
    assert runner.run_authoritative_qualification()["state"] == "PASS"
    path = root / artifact
    value = json.loads(path.read_text())
    value["evidence"]["cswp_qdrant"]["index_fingerprint"] = "wrong-trusted-index"
    path.write_text(runner.canonical_json_dumps(value) + "\n")
    support._refresh_protocol_manifest(root)
    assert runner.verify_authoritative_release(root) != "PHASE-5E1A-HOLDOUT-PASS"


@pytest.mark.parametrize("artifact", runner.AUTH_PROTOCOL_ARTIFACTS + ("manifest.json",))
def test_all_protocol_files_require_private_custody(production, artifact):
    root, _calls = production
    runner.run_authoritative_qualification()
    (root / artifact).chmod(0o644)
    assert runner.verify_authoritative_release(root) != "PHASE-5E1A-HOLDOUT-PASS"


def test_manifest_is_last_and_completed_verifier_is_independent(production, monkeypatch):
    root, _calls = production
    original_write = runner._durable_exclusive_bytes
    original_verify = runner.verify_completed_protocol_archive
    events = []

    def observe_write(archive, name, data):
        result = original_write(archive, name, data)
        events.append(name)
        return result

    def observe_verify(*args, **kwargs):
        assert (root / "manifest.json").exists()
        events.append("completed_verification")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(runner, "_durable_exclusive_bytes", observe_write)
    monkeypatch.setattr(runner, "verify_completed_protocol_archive", observe_verify)
    assert runner.run_authoritative_qualification()["state"] == "PASS"
    assert events[-6:] == ["final_live_preflight.json", "authoritative_verification.json", "terminal_state.json", "disposition.json", "manifest.json", "completed_verification"]


def test_initial_artifacts_are_durable_before_query_one(production, monkeypatch):
    root, _calls = production
    original = runner._execute_authoritative_backend

    def inspect_before_call(backend, query):
        assert all((root / name).is_file() for name in ("attempt.json", "oracle_identity.json", "contract.json", "pre_retrieval_preflight.json"))
        assert journal(root)[-1]["event"] == "backend_invocation_intent"
        return original(backend, query)

    monkeypatch.setattr(runner, "_execute_authoritative_backend", inspect_before_call)
    assert runner.run_authoritative_qualification()["state"] == "PASS"


def test_execution_one_completes_before_execution_two(production):
    root, _calls = production
    runner.run_authoritative_qualification()
    records = journal(root)
    second = records.index({"event": "execution_started", "execution": 2})
    assert records[second - 1] == {"event": "execution_completed", "execution": 1, "terminal_status": "COMPLETE", "overall_pass": True}


@pytest.mark.parametrize("mode", ("pre_create", "collision", "directory_fsync", "death_before", "death_after"))
def test_reservation_durability_failures(production, monkeypatch, mode):
    root, calls = production
    if mode == "collision":
        root.mkdir(mode=0o700)
    original_mkdir, original_sync = os.mkdir, runner._fsync_directory

    def mkdir(path, *args, **kwargs):
        if Path(path) == root and mode == "pre_create":
            raise OSError("public pre-reservation failure")
        return original_mkdir(path, *args, **kwargs)

    def sync(path):
        if Path(path) == root.parent:
            if mode == "directory_fsync":
                raise OSError("public reservation directory sync failure")
            if mode == "death_before":
                os._exit(73)
            original_sync(path)
            if mode == "death_after":
                os._exit(73)
        else:
            original_sync(path)

    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(runner, "_fsync_directory", sync)
    if mode.startswith("death"):
        pid = os.fork()
        if pid == 0:
            runner.run_authoritative_qualification()
            os._exit(74)
        _pid, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 73
    else:
        with pytest.raises((OSError, runner.QualificationHarnessError)):
            runner.run_authoritative_qualification()
    assert calls == []
    assert not (root / "manifest.json").exists()


def test_fixture_eligibility_is_after_authorization(production, monkeypatch):
    _root, calls = production
    approval = runner.load_trusted_release_approval()
    approval.fixture_path.chmod(0o644)
    with pytest.raises(runner.QualificationHarnessError):
        runner.run_authoritative_qualification()
    assert calls == []
