"""Deterministic coverage for the resumable paired benchmark harness."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_runtime, paired_benchmark as paired


TOPIC = {"id": 1, "topic": "Gradient Descent", "card_id": "standalone", "series": "Test"}
TEST_DIGEST = "test-experiment-digest"
TEST_SNAPSHOT_HASH = "test-snapshot-hash"


def _snapshot(path: Path, topics=None) -> Path:
    topics = topics or [TOPIC]
    entries = {
        query: {"max_results": 5, "results": [{"title": query, "url": f"https://{index}.example", "content": "x", "score": 0.9}]}
        for index, query in enumerate(paired.benchmark_web_queries(topics))
    }
    paired.atomic_write_json(path, {"schema_version": 1, "queries": entries})
    return path


def _telemetry(run_id: str, topic: dict, *, verified=17, unverified=3, status="completed", sv=4) -> dict:
    total = verified + unverified
    return {
        "run_id": run_id,
        "topic": topic["topic"],
        "verification_status": status,
        "claims_verified": verified,
        "claims_weak": 0,
        "claims_unverified": unverified,
        "grounded_depth": {"SV": sv, "S": sv, "V": verified, "N": total},
        "total_cost_usd": 0.01,
        "error_log": [],
        "frozen_web": {
            "snapshot_hash": TEST_SNAPSHOT_HASH,
            "queries": sorted(paired.benchmark_web_queries([topic])),
        },
    }


def _execution_shas() -> dict[str, str]:
    return {spec.arm: spec.architecture_sha for spec in paired.ARMS}


def _write_unit(root: Path, spec: paired.ArmSpec, topic: dict, *, uvr_verified=17, uvr_unverified=3, sv=4):
    record = paired.unit_from_telemetry(
        spec, topic, spec.architecture_sha, TEST_DIGEST, TEST_SNAPSHOT_HASH, f"{spec.arm}-{topic['id']}",
        _telemetry(f"{spec.arm}-{topic['id']}", topic, verified=uvr_verified, unverified=uvr_unverified, sv=sv),
    )
    paired.atomic_write_json(paired.unit_path(root, topic, spec.arm), record)


def _canonical_topics() -> list[dict]:
    return paired.load_topics(paired.CANONICAL_TOPICS_PATH)


def _write_aggregate_manifest(root: Path, digest: str = TEST_DIGEST) -> None:
    paired.atomic_write_json(paired.experiment_path(root), {
        "experiment_digest": digest,
        "frozen_web_snapshot_hash": TEST_SNAPSHOT_HASH,
        "execution_shas": _execution_shas(),
    })


def _write_complete_primary(root: Path, *, candidate_verified=17, candidate_unverified=3, candidate_sv=5):
    topics = _canonical_topics()
    _write_aggregate_manifest(root)
    for topic in topics:
        _write_unit(root, paired.ARMS[0], topic, sv=4)
        _write_unit(
            root, paired.ARMS[1], topic,
            uvr_verified=candidate_verified, uvr_unverified=candidate_unverified, sv=candidate_sv,
        )
    return topics


def _bound_experiment(root: Path, deepseek_config: dict[str, str]) -> dict:
    topics = _canonical_topics()
    snapshot_path = _snapshot(root / "frozen_web.json", topics)
    manifest = paired.initialize_experiment(root, topics, snapshot_path)
    runtime = paired.install_runtime(root)
    source_inventory = paired.expected_benchmark_sources()
    collections = {
        spec.arm: {
            "schema_version": 1,
            "arm": spec.arm,
            "collection_name": paired.collection_name(root.name, spec.arm),
            "architecture_sha": spec.architecture_sha,
            "execution_sha": spec.architecture_sha,
            "chunk_contract": spec.chunk_contract,
            "embedding_model": paired.EMBEDDING_MODEL,
            "source_count": 20,
            "point_count": spec.expected_point_count,
            "vector_size": 384,
            "distance": "cosine",
            "source_inventory_digest": paired._sha256(sorted(source_inventory)),
        }
        for spec in paired.ARMS
    }
    return paired.bind_experiment_digest(root, manifest, _execution_shas(), collections, runtime, deepseek_config)


def test_capture_once_per_exact_query_and_snapshot_validates(tmp_path):
    calls = []

    def search(query, max_results, force_refresh):
        calls.append((query, max_results, force_refresh))
        return [{"title": query, "url": "https://example.test", "content": "evidence", "score": 0.8}]

    snapshot_path = tmp_path / "frozen_web.json"
    snapshot = paired.capture_frozen_web_snapshot([TOPIC], snapshot_path, search)

    assert len(calls) == 3
    assert all(max_results == 5 and force_refresh is True for _, max_results, force_refresh in calls)
    assert paired.load_frozen_web_snapshot(snapshot_path, [TOPIC]) == snapshot
    with pytest.raises(paired.BenchmarkStateError, match="refusing to overwrite"):
        paired.capture_frozen_web_snapshot([TOPIC], snapshot_path, search)


def test_experiment_provenance_binds_architecture_topics_and_snapshot(tmp_path):
    topics = paired.load_topics(paired.CANONICAL_TOPICS_PATH)
    snapshot_path = _snapshot(tmp_path / "experiment" / "frozen_web.json", topics)
    manifest = paired.initialize_experiment(tmp_path / "experiment", topics, snapshot_path)

    assert manifest["baseline_architecture_sha"] == paired.BASELINE_SHA
    assert manifest["candidate_architecture_sha"] == paired.CANDIDATE_SHA
    assert manifest["topic_set_hash"] == paired._sha256(topics)
    assert manifest["frozen_web_snapshot_hash"] == paired._sha256(paired.load_json(snapshot_path, "snapshot"))
    assert manifest["arms"]["baseline"]["collection_name"] != manifest["arms"]["candidate"]["collection_name"]
    assert paired.initialize_experiment(tmp_path / "experiment", topics, snapshot_path) == manifest


@pytest.mark.parametrize("changed", [
    {"DEEPSEEK_MODEL": "deepseek-reasoner", "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1"},
    {"DEEPSEEK_MODEL": "deepseek-chat", "DEEPSEEK_BASE_URL": "https://alternate.deepseek.example/v1"},
])
def test_changed_effective_deepseek_config_invalidates_saved_unit(tmp_path, changed):
    initial = {"DEEPSEEK_MODEL": "deepseek-chat", "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1"}
    bound = _bound_experiment(tmp_path, initial)
    topic = _canonical_topics()[0]
    record = paired.unit_from_telemetry(
        paired.ARMS[0], topic, paired.ARMS[0].architecture_sha,
        bound["experiment_digest"], TEST_SNAPSHOT_HASH, "baseline-1", _telemetry("baseline-1", topic),
    )
    paired.atomic_write_json(paired.unit_path(tmp_path, topic, "baseline"), record)
    rebound = _bound_experiment(tmp_path, changed)

    assert rebound["deepseek"] == changed
    assert rebound["experiment_digest"] != bound["experiment_digest"]
    with pytest.raises(paired.BenchmarkStateError, match="experiment_digest"):
        paired.load_valid_unit(
            tmp_path, paired.ARMS[0], topic, paired.ARMS[0].architecture_sha,
            rebound["experiment_digest"], TEST_SNAPSHOT_HASH,
        )


def test_missing_frozen_query_cannot_call_live_tavily(tmp_path, monkeypatch):
    snapshot_path = _snapshot(tmp_path / "frozen_web.json")
    calls = []
    from tools import web_search

    monkeypatch.setenv(benchmark_runtime.SNAPSHOT_ENV, str(snapshot_path))
    monkeypatch.setattr(web_search, "_get_client", lambda: calls.append("live") or None)
    benchmark_runtime._snapshot = None
    benchmark_runtime._snapshot_path = None
    benchmark_runtime.install_frozen_web_search()

    with pytest.raises(benchmark_runtime.FrozenWebEvidenceError, match="missing exact query"):
        web_search.web_search("not captured", force_refresh=True)
    assert calls == []


@pytest.mark.parametrize("result", [
    {"title": "", "url": "https://example.test", "content": "evidence", "score": 0.8},
    {"title": "title", "url": "https://example.test", "content": "evidence", "score": float("nan")},
])
def test_malformed_frozen_web_evidence_fails_closed(tmp_path, result):
    snapshot_path = _snapshot(tmp_path / "frozen_web.json")
    snapshot = paired.load_json(snapshot_path, "snapshot")
    query = paired.benchmark_web_queries([TOPIC])[0]
    snapshot["queries"][query]["results"] = [result]
    paired.atomic_write_json(snapshot_path, snapshot)

    with pytest.raises(paired.BenchmarkStateError, match="frozen web evidence"):
        paired.load_frozen_web_snapshot(snapshot_path, [TOPIC])


def test_consumed_snapshot_identity_must_match_completed_unit(tmp_path):
    spec = paired.ARMS[0]
    _write_unit(tmp_path, spec, TOPIC)
    path = paired.unit_path(tmp_path, TOPIC, spec.arm)
    record = paired.load_json(path, "unit")
    record["frozen_web"]["snapshot_hash"] = "unexpected-snapshot"
    paired.atomic_write_json(path, record)

    with pytest.raises(paired.BenchmarkStateError, match="frozen snapshot identity"):
        paired.load_valid_unit(
            tmp_path, spec, TOPIC, spec.architecture_sha, TEST_DIGEST, TEST_SNAPSHOT_HASH,
        )


def test_force_refresh_uses_same_frozen_evidence_for_both_arms(tmp_path, monkeypatch):
    snapshot_path = _snapshot(tmp_path / "frozen_web.json")
    consumption_path = tmp_path / "consumption.json"
    monkeypatch.setenv(benchmark_runtime.SNAPSHOT_ENV, str(snapshot_path))
    monkeypatch.setenv(benchmark_runtime.CONSUMPTION_ENV, str(consumption_path))
    benchmark_runtime._snapshot = None
    benchmark_runtime._snapshot_path = None
    query = paired.benchmark_web_queries([TOPIC])[0]
    from tools import web_search

    original = web_search.web_search
    monkeypatch.setattr(web_search, "web_search", original)
    benchmark_runtime.install_frozen_web_search()

    baseline = web_search.web_search(query, force_refresh=False)
    candidate = web_search.web_search(query, force_refresh=True)

    assert baseline == candidate
    baseline[0]["content"] = "mutated"
    assert benchmark_runtime.frozen_web_search(query)[0]["content"] == "x"
    assert paired.load_json(consumption_path, "consumption") == {
        "snapshot_hash": paired._sha256(paired.load_json(snapshot_path, "snapshot")),
        "queries": [query],
    }
    envs = [paired.arm_environment(tmp_path, snapshot_path, f"paired_exp_{spec.arm}", "http://qdrant") for spec in paired.ARMS]
    assert {env[benchmark_runtime.SNAPSHOT_ENV] for env in envs} == {str(snapshot_path.resolve())}


def test_snapshot_adapter_does_not_change_production_search_when_disabled(monkeypatch):
    from tools import web_search

    def sentinel(*args, **kwargs):
        return []

    monkeypatch.delenv(benchmark_runtime.SNAPSHOT_ENV, raising=False)
    monkeypatch.setattr(web_search, "web_search", sentinel)
    benchmark_runtime.install_frozen_web_search()

    assert web_search.web_search is sentinel


def test_child_bootstrap_imports_arm_before_sitecustomize_and_cannot_use_live_tavily(tmp_path):
    snapshot_path = _snapshot(tmp_path / "frozen_web.json")
    runtime = paired.install_runtime(tmp_path / "runtime_parent")
    arm = tmp_path / "arm"
    (arm / "tools").mkdir(parents=True)
    (arm / "agent").mkdir()
    (arm / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (arm / "agent" / "__init__.py").write_text("", encoding="utf-8")
    (arm / "tools" / "web_search.py").write_text(
        "def web_search(*args, **kwargs):\n"
        "    raise AssertionError('live Tavily call attempted')\n",
        encoding="utf-8",
    )
    (arm / "agent" / "nodes.py").write_text(
        "from tools.web_search import web_search\n"
        "def retrieve_node(state):\n"
        "    return {'error_log': []}\n",
        encoding="utf-8",
    )
    (arm / "agent" / "graph.py").write_text(
        "from agent.nodes import retrieve_node\n",
        encoding="utf-8",
    )
    consumption_path = tmp_path / "consumption.json"
    env = paired.arm_environment(
        runtime,
        snapshot_path,
        "paired_test",
        "http://qdrant.test",
        consumption_path,
        arm_checkout=arm,
    )
    query = paired.benchmark_web_queries([TOPIC])[0]
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from benchmark_runtime import FrozenWebEvidenceError\n"
            "from tools import web_search\n"
            f"assert web_search.web_search({query!r}, force_refresh=True)[0]['content'] == 'x'\n"
            "try:\n"
            "    web_search.web_search('missing frozen query', force_refresh=True)\n"
            "except FrozenWebEvidenceError:\n"
            "    print('missing-fails-closed')\n"
            "else:\n"
            "    raise AssertionError('missing frozen evidence did not fail closed')",
        ],
        cwd=arm,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "missing-fails-closed"
    assert paired.load_json(consumption_path, "consumption") == {
        "snapshot_hash": paired._sha256(paired.load_json(snapshot_path, "snapshot")),
        "queries": [query],
    }


def test_interruption_after_one_unit_resumes_from_valid_record(tmp_path):
    calls = []

    def interrupted(spec, topic):
        calls.append((spec.arm, topic["id"]))
        if len(calls) == 2:
            raise KeyboardInterrupt()
        run_id = f"{spec.arm}-{topic['id']}"
        return run_id, _telemetry(run_id, topic)

    with pytest.raises(KeyboardInterrupt):
        paired.execute_units(tmp_path, [TOPIC], _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH, interrupted)
    assert paired.unit_path(tmp_path, TOPIC, "baseline").exists()
    assert not paired.unit_path(tmp_path, TOPIC, "candidate").exists()

    resumed = []

    def run_remaining(spec, topic):
        resumed.append((spec.arm, topic["id"]))
        run_id = f"{spec.arm}-{topic['id']}"
        return run_id, _telemetry(run_id, topic)

    results = paired.execute_units(
        tmp_path, [TOPIC], _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH, run_remaining,
    )
    assert results[0][1] is True
    assert resumed == [("candidate", 1)]


def test_completed_valid_unit_is_skipped(tmp_path):
    spec = paired.ARMS[0]
    called = []

    def invoke(_spec, _topic):
        called.append(True)
        return "run", _telemetry("run", TOPIC)

    paired.execute_unit(tmp_path, spec, TOPIC, spec.architecture_sha, TEST_DIGEST, TEST_SNAPSHOT_HASH, invoke)
    _, skipped = paired.execute_unit(
        tmp_path, spec, TOPIC, spec.architecture_sha, TEST_DIGEST, TEST_SNAPSHOT_HASH, invoke,
    )
    assert skipped is True
    assert called == [True]


def test_corrupt_saved_unit_fails_closed(tmp_path):
    spec = paired.ARMS[0]
    path = paired.unit_path(tmp_path, TOPIC, spec.arm)
    _write_unit(tmp_path, spec, TOPIC)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(paired.BenchmarkStateError):
        paired.load_valid_unit(tmp_path, spec, TOPIC, spec.architecture_sha, TEST_DIGEST, TEST_SNAPSHOT_HASH)


def test_wrong_arm_fails_closed(tmp_path):
    spec = paired.ARMS[0]
    _write_unit(tmp_path, spec, TOPIC)
    path = paired.unit_path(tmp_path, TOPIC, spec.arm)
    record = json.loads(path.read_text())
    record["arm"] = "candidate"
    paired.atomic_write_json(path, record)

    with pytest.raises(paired.BenchmarkStateError, match="arm"):
        paired.load_valid_unit(tmp_path, spec, TOPIC, spec.architecture_sha, TEST_DIGEST, TEST_SNAPSHOT_HASH)


def test_wrong_execution_sha_fails_closed(tmp_path):
    spec = paired.ARMS[0]
    _write_unit(tmp_path, spec, TOPIC)

    with pytest.raises(paired.BenchmarkStateError, match="execution_sha"):
        paired.load_valid_unit(tmp_path, spec, TOPIC, "wrong-sha", TEST_DIGEST, TEST_SNAPSHOT_HASH)


def test_changed_experiment_digest_prevents_stale_unit_reuse(tmp_path):
    spec = paired.ARMS[0]
    _write_unit(tmp_path, spec, TOPIC)

    with pytest.raises(paired.BenchmarkStateError, match="experiment_digest"):
        paired.load_valid_unit(tmp_path, spec, TOPIC, spec.architecture_sha, "changed-digest", TEST_SNAPSHOT_HASH)


def test_failed_unit_is_saved_separately_and_reruns_without_manual_deletion(tmp_path):
    spec = paired.ARMS[0]

    with pytest.raises(paired.BenchmarkStateError, match="benchmark unit failed"):
        paired.execute_unit(
            tmp_path, spec, TOPIC, spec.architecture_sha, TEST_DIGEST, TEST_SNAPSHOT_HASH,
            lambda *_args: (_ for _ in ()).throw(RuntimeError("injected failure")),
        )
    assert not paired.unit_path(tmp_path, TOPIC, spec.arm).exists()
    assert list((tmp_path / "attempts").glob("*.json"))

    record, skipped = paired.execute_unit(
        tmp_path, spec, TOPIC, spec.architecture_sha, TEST_DIGEST, TEST_SNAPSHOT_HASH,
        lambda _spec, topic: ("recovered", _telemetry("recovered", topic)),
    )
    assert skipped is False
    assert record["run_id"] == "recovered"


class _FakeQdrant:
    def __init__(self, points, *, vector_size=384, distance="cosine", sources=None):
        self.points = points
        self.vector_size = vector_size
        self.distance = distance
        self.sources = sources or {}

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.points])

    def get_collection(self, collection_name):
        return SimpleNamespace(
            points_count=self.points[collection_name],
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(
                        size=self.vector_size,
                        distance=SimpleNamespace(value=self.distance),
                    )
                )
            ),
        )

    def scroll(self, collection_name, offset=None, limit=256, with_payload=None, with_vectors=False):
        source_values = self.sources.get(collection_name, sorted(paired.expected_benchmark_sources()))
        start = int(offset or 0)
        end = min(start + limit, self.points[collection_name])
        records = [
            SimpleNamespace(payload={"source": source_values[index % len(source_values)]})
            for index in range(start, end)
        ]
        return records, (end if end < self.points[collection_name] else None)


def test_unproven_nonempty_collection_and_manifest_count_mismatch_fail(tmp_path):
    spec = paired.ARMS[0]
    name = paired.collection_name(tmp_path.name, spec.arm)
    client = _FakeQdrant({name: 10})

    with pytest.raises(paired.BenchmarkStateError, match="point count"):
        paired.prepare_collection(tmp_path, spec, spec.architecture_sha, 20, client, lambda _name: None)

    manifest = {
        "schema_version": 1,
        "arm": spec.arm,
        "collection_name": name,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": spec.architecture_sha,
        "chunk_contract": spec.chunk_contract,
        "embedding_model": paired.EMBEDDING_MODEL,
        "source_count": 20,
        "point_count": 9,
        "vector_size": 384,
        "distance": "cosine",
        "source_inventory_digest": paired._sha256(sorted(paired.expected_benchmark_sources())),
    }
    with pytest.raises(paired.BenchmarkStateError, match="does not prove"):
        paired.validate_collection_manifest(
            manifest, spec, name, 20, 10, 384, "cosine", paired.expected_benchmark_sources(), spec.architecture_sha,
        )


@pytest.mark.parametrize("source_count,vector_size,distance", [
    (19, 384, "cosine"),
    (20, 385, "cosine"),
    (20, 384, "euclid"),
])
def test_collection_source_and_config_contracts_fail_closed(tmp_path, source_count, vector_size, distance):
    spec = paired.ARMS[0]
    client = _FakeQdrant({}, vector_size=vector_size, distance=distance)

    def ingest(name):
        client.points[name] = spec.expected_point_count

    with pytest.raises(paired.BenchmarkStateError):
        paired.prepare_collection(tmp_path, spec, spec.architecture_sha, source_count, client, ingest)


def test_collection_preparation_uses_distinct_disposable_arm_names(tmp_path):
    client = _FakeQdrant({})
    ingested = []

    def ingest(name):
        ingested.append(name)
        spec = paired.ARMS[len(ingested) - 1]
        client.points[name] = spec.expected_point_count

    baseline = paired.prepare_collection(
        tmp_path, paired.ARMS[0], paired.ARMS[0].architecture_sha, 20, client, ingest,
    )
    candidate = paired.prepare_collection(
        tmp_path, paired.ARMS[1], paired.ARMS[1].architecture_sha, 20, client, ingest,
    )

    assert baseline["collection_name"] != candidate["collection_name"]
    assert all(name.startswith("paired_") for name in ingested)
    assert baseline["point_count"] == 73
    assert candidate["point_count"] == 139


@pytest.mark.parametrize("source_values", [
    sorted(paired.expected_benchmark_sources())[:-1],
    sorted(paired.expected_benchmark_sources())[:-1] + ["gradient-descent"],
])
def test_partial_or_duplicate_indexed_source_inventory_is_rejected(tmp_path, source_values):
    spec = paired.ARMS[0]
    name = paired.collection_name(tmp_path.name, spec.arm)
    client = _FakeQdrant({name: spec.expected_point_count}, sources={name: source_values})

    with pytest.raises(paired.BenchmarkStateError, match="source inventory"):
        paired.prepare_collection(tmp_path, spec, spec.architecture_sha, 20, client, lambda _name: None)


def test_exact_indexed_source_inventory_and_collection_contract_pass(tmp_path):
    spec = paired.ARMS[0]
    client = _FakeQdrant({})

    def ingest(name):
        client.points[name] = spec.expected_point_count

    manifest = paired.prepare_collection(tmp_path, spec, spec.architecture_sha, 20, client, ingest)
    assert manifest["source_inventory_digest"] == paired._sha256(sorted(paired.expected_benchmark_sources()))
    assert paired.prepare_collection(tmp_path, spec, spec.architecture_sha, 20, client, ingest) == manifest


@pytest.mark.parametrize("candidate_verified,candidate_unverified,expected", [
    (17, 3, True),   # exactly 0.15 passes
    (16, 4, False),  # 0.20 fails
])
def test_candidate_uvr_gate_boundary(tmp_path, candidate_verified, candidate_unverified, expected):
    topics = _write_complete_primary(
        tmp_path, candidate_verified=candidate_verified, candidate_unverified=candidate_unverified, candidate_sv=4,
    )

    aggregate = paired.paired_aggregate(
        tmp_path, topics, _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH,
    )
    assert aggregate["gate_pass"] is expected


@pytest.mark.parametrize("topics", [
    lambda canonical: canonical[:19],
    lambda canonical: canonical[:1],
])
def test_truncated_primary_topic_set_cannot_pass(topics):
    canonical = _canonical_topics()
    with pytest.raises(paired.BenchmarkStateError, match="exactly 20"):
        paired.paired_aggregate(
            experiment_dir=Path("unused"), topics=topics(canonical), execution_shas={},
            experiment_digest="x", snapshot_hash="x",
        )


def test_invalid_verification_status_cannot_green_light_pair(tmp_path):
    topics = _write_complete_primary(tmp_path)
    path = paired.unit_path(tmp_path, topics[0], "candidate")
    bad = paired.load_json(path, "candidate unit")
    bad["verification_status"] = "parse_failed"
    paired.atomic_write_json(path, bad)

    aggregate = paired.paired_aggregate(
        tmp_path, topics, _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH,
    )
    assert aggregate["gate_pass"] is False
    assert any("verification_status" in failure for failure in aggregate["gate_failures"])


def test_na_uvr_cannot_green_light_pair(tmp_path):
    topics = _write_complete_primary(tmp_path)
    path = paired.unit_path(tmp_path, topics[0], "candidate")
    record = json.loads(path.read_text())
    record["uvr"] = None
    paired.atomic_write_json(path, record)

    aggregate = paired.paired_aggregate(
        tmp_path, topics, _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH,
    )
    assert aggregate["gate_pass"] is False
    assert any("invalid UVR" in failure for failure in aggregate["gate_failures"])


def test_nan_sv_cannot_green_light_pair(tmp_path):
    topics = _write_complete_primary(tmp_path)
    path = paired.unit_path(tmp_path, topics[0], "candidate")
    record = json.loads(path.read_text())
    record["sv"] = float("nan")
    paired.atomic_write_json(path, record)

    aggregate = paired.paired_aggregate(
        tmp_path, topics, _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH,
    )
    assert aggregate["gate_pass"] is False
    assert any("sv" in failure.lower() for failure in aggregate["gate_failures"])


@pytest.mark.parametrize("field,value", [
    ("claims_verified", float("nan")),
    ("claims_unverified", -1),
    ("claims_weak", 0.5),
])
def test_invalid_claim_counts_cannot_create_unit(field, value):
    telemetry = _telemetry("invalid-count", TOPIC)
    telemetry[field] = value

    with pytest.raises(paired.BenchmarkStateError, match="nonnegative integer"):
        paired.unit_from_telemetry(
            paired.ARMS[0], TOPIC, paired.ARMS[0].architecture_sha,
            TEST_DIGEST, TEST_SNAPSHOT_HASH, "invalid-count", telemetry,
        )


def test_paired_sv_aggregation_and_per_topic_deltas(tmp_path):
    topics = _write_complete_primary(tmp_path)

    aggregate = paired.paired_aggregate(
        tmp_path, topics, _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH,
    )

    assert aggregate["gate_pass"] is True
    assert aggregate["experiment_digest"] == TEST_DIGEST
    assert aggregate["total_primary_units"] == 40
    assert aggregate["baseline_aggregate_sv"] == 4
    assert aggregate["candidate_aggregate_sv"] == 5
    assert len(aggregate["topic_pairs"]) == 20
    assert [row["delta_sv"] for row in aggregate["topic_pairs"]] == [1] * 20
    assert all("delta_uvr" in row for row in aggregate["topic_pairs"])


def test_nineteen_pairs_cannot_aggregate(tmp_path):
    topics = _write_complete_primary(tmp_path)
    paired.unit_path(tmp_path, topics[-1], "candidate").unlink()

    aggregate = paired.paired_aggregate(
        tmp_path, topics, _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH,
    )

    assert aggregate["gate_pass"] is False
    assert any("19/20" in failure for failure in aggregate["gate_failures"])


def test_mixed_experiment_digest_cannot_aggregate(tmp_path):
    topics = _write_complete_primary(tmp_path)
    path = paired.unit_path(tmp_path, topics[0], "candidate")
    record = paired.load_json(path, "candidate unit")
    record["experiment_digest"] = "other-experiment"
    paired.atomic_write_json(path, record)

    aggregate = paired.paired_aggregate(
        tmp_path, topics, _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH,
    )

    assert aggregate["gate_pass"] is False
    assert any("experiment_digest" in failure for failure in aggregate["gate_failures"])


def test_arm_order_alternates_baseline_candidate_then_candidate_baseline(tmp_path):
    topics = [TOPIC, {**TOPIC, "id": 2, "topic": "Backpropagation"}]
    observed = []

    def invoke(spec, topic):
        observed.append((topic["id"], spec.arm))
        run_id = f"{spec.arm}-{topic['id']}"
        return run_id, _telemetry(run_id, topic)

    paired.execute_units(tmp_path, topics, _execution_shas(), TEST_DIGEST, TEST_SNAPSHOT_HASH, invoke)

    assert observed == [(1, "baseline"), (1, "candidate"), (2, "candidate"), (2, "baseline")]
