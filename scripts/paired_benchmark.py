"""Resumable, paired benchmark infrastructure for the baseline and PR #3 arms.

This runner intentionally does not execute at import time.  ``capture-web`` is
the only operation that calls Tavily, and ``run`` requires a previously frozen
snapshot.  Each arm is evaluated from a detached worktree at its frozen
architecture SHA and writes only to a uniquely named disposable collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Callable


BASELINE_SHA = "794851dded770ce87d111e73735d000e23597eb1"
CANDIDATE_SHA = "470e7834cd781d85c4c447b81bb1651106ff766d"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UVR_THRESHOLD = 0.15
UVR_REGRESSION_REVIEW = 0.05
SV_REGRESSION_REVIEW = 1.0


class BenchmarkStateError(RuntimeError):
    """Persisted benchmark evidence is absent, invalid, or non-reproducible."""


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    architecture_sha: str
    chunk_contract: dict
    expected_source_count: int
    expected_point_count: int


ARMS = (
    ArmSpec(
        arm="baseline",
        architecture_sha=BASELINE_SHA,
        chunk_contract={"tokenizer": "cl100k_base", "content_tokens": 400, "overlap_tokens": 50},
        expected_source_count=20,
        expected_point_count=73,
    ),
    ArmSpec(
        arm="candidate",
        architecture_sha=CANDIDATE_SHA,
        chunk_contract={"tokenizer": "all-MiniLM-L6-v2", "content_tokens": 224, "overlap_tokens": 32},
        expected_source_count=20,
        expected_point_count=139,
    ),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    """Persist one evidence record atomically in the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_bytes(_canonical_json(value))
    os.replace(temp, path)


def load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkStateError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkStateError(f"{label} must be a JSON object: {path}")
    return value


def load_topics(path: Path) -> list[dict]:
    try:
        topics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkStateError(f"topics are unreadable: {path}") from error
    if not isinstance(topics, list) or not topics:
        raise BenchmarkStateError("topics must be a non-empty JSON list")
    ids = [topic.get("id") for topic in topics if isinstance(topic, dict)]
    if len(ids) != len(topics) or len(set(ids)) != len(ids):
        raise BenchmarkStateError("topics must have unique ids")
    return topics


CANONICAL_TOPICS_PATH = Path(__file__).resolve().parents[1] / "evals" / "topics.json"


def canonical_topic_hash() -> str:
    return _sha256(load_topics(CANONICAL_TOPICS_PATH))


def require_canonical_topics(topics: list[dict]) -> None:
    if len(topics) != 20:
        raise BenchmarkStateError(f"primary paired benchmark requires exactly 20 topics, got {len(topics)}")
    if _sha256(topics) != canonical_topic_hash():
        raise BenchmarkStateError("primary paired benchmark topics do not match the canonical topic set")


def _usable_frozen_results(query: str, results: object) -> list[dict]:
    if not isinstance(results, list) or not results:
        raise BenchmarkStateError(f"frozen web evidence has no usable results for query: {query!r}")
    for result in results:
        if not isinstance(result, dict):
            raise BenchmarkStateError(f"frozen web evidence has a non-object result for query: {query!r}")
        if not all(isinstance(result.get(field), str) and result[field].strip()
                   for field in ("title", "url", "content")):
            raise BenchmarkStateError(f"frozen web evidence has an unusable result for query: {query!r}")
        score = result.get("score")
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
            raise BenchmarkStateError(f"frozen web evidence has a non-finite score for query: {query!r}")
    return results


def benchmark_web_queries(topics: list[dict]) -> list[str]:
    """Mirror the fixed query templates used by ``retrieve_node`` exactly."""
    queries = []
    for topic in topics:
        name = topic.get("topic")
        if not isinstance(name, str) or not name:
            raise BenchmarkStateError(f"invalid topic for frozen web capture: {topic!r}")
        queries.extend((
            f"{name} explained technical",
            f"{name} failure modes limitations production",
            f"{name} implementation Python example",
        ))
    if len(set(queries)) != len(queries):
        raise BenchmarkStateError("benchmark query generation produced duplicates")
    return queries


def capture_frozen_web_snapshot(
    topics: list[dict],
    output_path: Path,
    search: Callable[..., list[dict]],
) -> dict:
    """Capture exactly one fresh Tavily response for every benchmark query."""
    if output_path.exists():
        raise BenchmarkStateError(f"refusing to overwrite existing frozen snapshot: {output_path}")
    queries = benchmark_web_queries(topics)
    captured = {}
    for query in queries:
        results = search(query, max_results=5, force_refresh=True)
        captured[query] = {"max_results": 5, "results": _usable_frozen_results(query, results)}
    snapshot = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "queries": captured,
    }
    atomic_write_json(output_path, snapshot)
    return snapshot


def load_frozen_web_snapshot(path: Path, topics: list[dict]) -> dict:
    snapshot = load_json(path, "frozen web snapshot")
    if snapshot.get("schema_version") != 1 or not isinstance(snapshot.get("queries"), dict):
        raise BenchmarkStateError("frozen web snapshot has an invalid schema")
    missing = sorted(set(benchmark_web_queries(topics)) - set(snapshot["queries"]))
    if missing:
        raise BenchmarkStateError(f"frozen web snapshot is missing {len(missing)} required query result(s)")
    for query in benchmark_web_queries(topics):
        entry = snapshot["queries"].get(query)
        if not isinstance(entry, dict) or entry.get("max_results") != 5:
            raise BenchmarkStateError(f"frozen web snapshot is malformed for query: {query!r}")
        _usable_frozen_results(query, entry.get("results"))
    return snapshot


def collection_name(experiment_id: str, arm: str) -> str:
    safe_id = re.sub(r"[^a-z0-9_]+", "_", experiment_id.lower()).strip("_")
    if not safe_id:
        raise BenchmarkStateError("experiment id must contain letters or digits")
    return f"paired_{safe_id}_{arm}"


def experiment_path(experiment_dir: Path) -> Path:
    return experiment_dir / "experiment.json"


def _experiment_core(experiment_dir: Path, topics: list[dict], snapshot_path: Path) -> dict:
    snapshot = load_frozen_web_snapshot(snapshot_path, topics)
    experiment_id = experiment_dir.name
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "baseline_architecture_sha": BASELINE_SHA,
        "candidate_architecture_sha": CANDIDATE_SHA,
        "topic_set_hash": _sha256(topics),
        "frozen_web_snapshot": str(snapshot_path.resolve()),
        "frozen_web_snapshot_hash": _sha256(snapshot),
        "embedding_model": EMBEDDING_MODEL,
        "arms": {
            spec.arm: {
                "collection_name": collection_name(experiment_id, spec.arm),
                "architecture_sha": spec.architecture_sha,
                "chunk_contract": spec.chunk_contract,
            }
            for spec in ARMS
        },
    }


def initialize_experiment(experiment_dir: Path, topics: list[dict], snapshot_path: Path) -> dict:
    """Create or verify the minimal immutable provenance record."""
    require_canonical_topics(topics)
    manifest = _experiment_core(experiment_dir, topics, snapshot_path)
    path = experiment_path(experiment_dir)
    if path.exists():
        existing = load_json(path, "experiment manifest")
        if any(existing.get(field) != value for field, value in manifest.items()):
            raise BenchmarkStateError("existing experiment provenance does not match requested benchmark")
        return existing
    atomic_write_json(path, manifest)
    return manifest


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise BenchmarkStateError(f"benchmark runtime file is unreadable: {path}") from error


def bind_experiment_digest(
    experiment_dir: Path,
    manifest: dict,
    execution_shas: dict[str, str],
    collection_manifests: dict[str, dict],
    runtime: Path,
) -> dict:
    """Bind completed units to one digest of the decision-critical run inputs."""
    if set(execution_shas) != {spec.arm for spec in ARMS}:
        raise BenchmarkStateError("experiment execution SHA set is incomplete")
    if set(collection_manifests) != {spec.arm for spec in ARMS}:
        raise BenchmarkStateError("experiment collection manifest set is incomplete")
    runtime_hashes = {
        "harness": _file_hash(Path(__file__)),
        "benchmark_runtime": _file_hash(runtime / "benchmark_runtime.py"),
        "sitecustomize": _file_hash(runtime / "sitecustomize.py"),
    }
    digest_inputs = {
        "baseline_execution_sha": execution_shas["baseline"],
        "candidate_execution_sha": execution_shas["candidate"],
        "harness_hashes": runtime_hashes,
        "embedding_model": EMBEDDING_MODEL,
        "canonical_topic_hash": canonical_topic_hash(),
        "frozen_web_snapshot_hash": manifest["frozen_web_snapshot_hash"],
        "collection_manifest_hashes": {
            arm: _sha256(collection_manifests[arm]) for arm in sorted(collection_manifests)
        },
    }
    bound = {
        **_experiment_core(experiment_dir, load_topics(CANONICAL_TOPICS_PATH), Path(manifest["frozen_web_snapshot"])),
        "execution_shas": execution_shas,
        "harness_hashes": runtime_hashes,
        "collection_manifest_hashes": digest_inputs["collection_manifest_hashes"],
        "experiment_digest": _sha256(digest_inputs),
    }
    path = experiment_path(experiment_dir)
    existing = load_json(path, "experiment manifest")
    if existing != manifest and existing != bound:
        raise BenchmarkStateError("existing experiment provenance has incompatible bound inputs")
    if existing != bound:
        atomic_write_json(path, bound)
    return bound


def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise BenchmarkStateError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def ensure_arm_worktree(repo_root: Path, experiment_dir: Path, spec: ArmSpec) -> tuple[Path, str]:
    """Create/reuse a detached, clean worktree at exactly the arm's base SHA."""
    worktree = experiment_dir / "worktrees" / spec.arm
    if worktree.exists():
        actual_sha = _git(worktree, "rev-parse", "HEAD")
        if actual_sha != spec.architecture_sha:
            raise BenchmarkStateError(f"{spec.arm} worktree SHA mismatch: {actual_sha}")
        if _git(worktree, "status", "--short"):
            raise BenchmarkStateError(f"{spec.arm} worktree is dirty")
        return worktree, actual_sha
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "worktree", "add", "--detach", str(worktree), spec.architecture_sha)
    actual_sha = _git(worktree, "rev-parse", "HEAD")
    if actual_sha != spec.architecture_sha:
        raise BenchmarkStateError(f"created {spec.arm} worktree at unexpected SHA: {actual_sha}")
    return worktree, actual_sha


def _source_signature(seed_docs: Path) -> list[tuple[str, str]]:
    files = sorted(path for path in seed_docs.iterdir() if path.is_file() and path.suffix in {".md", ".txt"})
    if not files:
        raise BenchmarkStateError(f"seed corpus is empty: {seed_docs}")
    return [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]


def verify_same_seed_corpus(baseline_worktree: Path, candidate_worktree: Path) -> int:
    baseline = _source_signature(baseline_worktree / "kb" / "seed_docs")
    candidate = _source_signature(candidate_worktree / "kb" / "seed_docs")
    if baseline != candidate:
        raise BenchmarkStateError("baseline and candidate seed corpora differ")
    return len(baseline)


def collection_manifest_path(experiment_dir: Path, arm: str) -> Path:
    return experiment_dir / "collections" / f"{arm}.json"


def _collection_properties(client, name: str) -> tuple[int, int, str]:
    try:
        info = client.get_collection(collection_name=name)
        vectors = info.config.params.vectors
        size = int(vectors.size)
        distance = getattr(vectors.distance, "value", vectors.distance)
        return int(info.points_count), size, str(distance).lower()
    except Exception as error:
        raise BenchmarkStateError(f"cannot inspect disposable collection {name!r}") from error


def _collection_names(client) -> set[str]:
    try:
        return {item.name for item in client.get_collections().collections}
    except Exception as error:
        raise BenchmarkStateError("cannot list Qdrant collections") from error


def validate_collection_manifest(
    manifest: dict,
    spec: ArmSpec,
    name: str,
    source_count: int,
    point_count: int,
    vector_size: int,
    distance: str,
    execution_sha: str,
) -> None:
    expected = {
        "schema_version": 1,
        "arm": spec.arm,
        "collection_name": name,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": execution_sha,
        "chunk_contract": spec.chunk_contract,
        "embedding_model": EMBEDDING_MODEL,
        "source_count": source_count,
        "point_count": point_count,
        "vector_size": vector_size,
        "distance": distance,
    }
    if manifest != expected:
        raise BenchmarkStateError(f"collection manifest does not prove {spec.arm} collection identity")


def prepare_collection(
    experiment_dir: Path,
    spec: ArmSpec,
    execution_sha: str,
    source_count: int,
    client,
    ingest: Callable[[str], None],
) -> dict:
    """Reuse only a manifest-proven collection, otherwise ingest once into a disposable name."""
    name = collection_name(experiment_dir.name, spec.arm)
    if not name.startswith("paired_"):
        raise BenchmarkStateError("benchmark collection name is not disposable")
    if source_count != spec.expected_source_count:
        raise BenchmarkStateError(
            f"{spec.arm} seed corpus count {source_count} != expected {spec.expected_source_count}"
        )
    path = collection_manifest_path(experiment_dir, spec.arm)
    names = _collection_names(client)
    if name in names:
        point_count, vector_size, distance = _collection_properties(client, name)
        if point_count != spec.expected_point_count:
            raise BenchmarkStateError(
                f"{spec.arm} collection point count {point_count} != expected {spec.expected_point_count}"
            )
        if vector_size != 384 or distance != "cosine":
            raise BenchmarkStateError(f"{spec.arm} collection vector config is not MiniLM 384 cosine")
        if not path.exists():
            raise BenchmarkStateError(f"refusing unexpected nonempty collection: {name}")
        manifest = load_json(path, "collection manifest")
        validate_collection_manifest(
            manifest, spec, name, source_count, point_count, vector_size, distance, execution_sha,
        )
        return manifest

    ingest(name)
    point_count, vector_size, distance = _collection_properties(client, name)
    if point_count != spec.expected_point_count:
        raise BenchmarkStateError(
            f"{spec.arm} ingest point count {point_count} != expected {spec.expected_point_count}"
        )
    if vector_size != 384 or distance != "cosine":
        raise BenchmarkStateError(f"{spec.arm} ingest collection vector config is not MiniLM 384 cosine")
    manifest = {
        "schema_version": 1,
        "arm": spec.arm,
        "collection_name": name,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": execution_sha,
        "chunk_contract": spec.chunk_contract,
        "embedding_model": EMBEDDING_MODEL,
        "source_count": source_count,
        "point_count": point_count,
        "vector_size": vector_size,
        "distance": distance,
    }
    atomic_write_json(path, manifest)
    return manifest


def install_runtime(experiment_dir: Path) -> Path:
    """Install the benchmark-only import hook used by both detached arm processes."""
    runtime = experiment_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("benchmark_runtime.py")
    shutil.copyfile(source, runtime / "benchmark_runtime.py")
    (runtime / "sitecustomize.py").write_text(
        "from benchmark_runtime import (\n"
        "    install_benchmark_guards,\n"
        "    install_frozen_web_search,\n"
        ")\n"
        "install_frozen_web_search()\n"
        "install_benchmark_guards()\n",
        encoding="utf-8",
    )
    return runtime


def arm_environment(
    runtime: Path,
    snapshot_path: Path,
    collection: str,
    qdrant_url: str,
    consumption_path: Path | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    env["CONTENT_AGENT_FROZEN_WEB_SNAPSHOT"] = str(snapshot_path.resolve())
    env["CONTENT_AGENT_FROZEN_WEB_SNAPSHOT_HASH"] = _sha256(load_json(snapshot_path, "frozen web snapshot"))
    env["QDRANT_COLLECTION"] = collection
    env["QDRANT_URL"] = qdrant_url
    if consumption_path is not None:
        env["CONTENT_AGENT_FROZEN_WEB_CONSUMPTION"] = str(consumption_path.resolve())
    env["PYTHONPATH"] = str(runtime) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def unit_path(experiment_dir: Path, topic: dict, arm: str) -> Path:
    return experiment_dir / "units" / f"topic-{int(topic['id']):02d}-{arm}.json"


def failed_attempt_path(experiment_dir: Path, topic: dict, arm: str) -> Path:
    attempt_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    return experiment_dir / "attempts" / f"topic-{int(topic['id']):02d}-{arm}-{attempt_id}.json"


def _nonnegative_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkStateError(f"{label} must be a nonnegative integer")
    return value


def _finite_nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise BenchmarkStateError(f"{label} must be a finite nonnegative number")
    return float(value)


def _claim_counts(telemetry: dict) -> dict[str, int]:
    counts = {
        "verified": _nonnegative_count(telemetry.get("claims_verified"), "claims_verified"),
        "weak": _nonnegative_count(telemetry.get("claims_weak"), "claims_weak"),
        "unverified": _nonnegative_count(telemetry.get("claims_unverified"), "claims_unverified"),
    }
    counts["total"] = sum(counts.values())
    return counts


def _validate_frozen_consumption(telemetry: dict, topic: dict, snapshot_hash: str) -> dict:
    frozen = telemetry.get("frozen_web")
    if not isinstance(frozen, dict) or frozen.get("snapshot_hash") != snapshot_hash:
        raise BenchmarkStateError("telemetry did not consume the expected frozen web snapshot")
    queries = frozen.get("queries")
    expected_queries = sorted(benchmark_web_queries([topic]))
    if not isinstance(queries, list) or queries != expected_queries:
        raise BenchmarkStateError("telemetry did not consume the exact frozen web query set")
    return {"snapshot_hash": snapshot_hash, "queries": queries}


def unit_from_telemetry(
    spec: ArmSpec,
    topic: dict,
    execution_sha: str,
    experiment_digest: str,
    snapshot_hash: str,
    run_id: str,
    telemetry: dict,
) -> dict:
    if telemetry.get("run_id") != run_id or telemetry.get("topic") != topic.get("topic"):
        raise BenchmarkStateError("telemetry identity does not match the invoked benchmark unit")
    required = {"verification_status", "grounded_depth", "total_cost_usd", "error_log"}
    if not required <= set(telemetry):
        raise BenchmarkStateError("telemetry lacks evaluation-integrity fields")
    if telemetry["verification_status"] != "completed":
        raise BenchmarkStateError(f"verification_status={telemetry['verification_status']}")
    counts = _claim_counts(telemetry)
    total = counts["total"]
    if total <= 0:
        raise BenchmarkStateError("completed benchmark unit has no scorable verifier verdicts")
    uvr = counts["unverified"] / total
    grounded_depth = telemetry.get("grounded_depth")
    if not isinstance(grounded_depth, dict):
        raise BenchmarkStateError("telemetry lacks grounded-depth claim counts")
    sv = _nonnegative_count(grounded_depth.get("SV"), "grounded_depth.SV")
    substantive = _nonnegative_count(grounded_depth.get("S"), "grounded_depth.S")
    grounded_verified = _nonnegative_count(grounded_depth.get("V"), "grounded_depth.V")
    grounded_total = _nonnegative_count(grounded_depth.get("N"), "grounded_depth.N")
    if grounded_total != total or grounded_verified != counts["verified"] or sv > substantive or sv > grounded_verified:
        raise BenchmarkStateError("grounded-depth claim counts do not match verifier counts")
    cost_usd = _finite_nonnegative_number(telemetry["total_cost_usd"], "total_cost_usd")
    errors = telemetry["error_log"]
    if not isinstance(errors, list):
        raise BenchmarkStateError("telemetry error_log is invalid")
    frozen_web = _validate_frozen_consumption(telemetry, topic, snapshot_hash)
    return {
        "schema_version": 1,
        "status": "completed",
        "topic_id": topic["id"],
        "topic": topic["topic"],
        "arm": spec.arm,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": execution_sha,
        "experiment_digest": experiment_digest,
        "run_id": run_id,
        "verification_status": telemetry["verification_status"],
        "uvr": uvr,
        "sv": sv,
        "claims": counts,
        "cost_usd": cost_usd,
        "errors": errors,
        "frozen_web": frozen_web,
    }


def validate_unit(
    record: dict,
    spec: ArmSpec,
    topic: dict,
    execution_sha: str,
    experiment_digest: str,
    snapshot_hash: str,
) -> dict:
    expected = {
        "schema_version": 1,
        "status": "completed",
        "topic_id": topic["id"],
        "topic": topic["topic"],
        "arm": spec.arm,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": execution_sha,
        "experiment_digest": experiment_digest,
        "verification_status": "completed",
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise BenchmarkStateError(f"unit result {field} does not match expected {spec.arm} evidence")
    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        raise BenchmarkStateError("unit result is missing RUN_ID")
    if isinstance(record.get("uvr"), bool) or not isinstance(record.get("uvr"), Real) or not math.isfinite(record["uvr"]) or not 0 <= record["uvr"] <= 1:
        raise BenchmarkStateError("unit result has invalid UVR")
    counts = record.get("claims")
    if not isinstance(counts, dict):
        raise BenchmarkStateError("unit result has invalid claim counts")
    validated_counts = {
        key: _nonnegative_count(counts.get(key), f"unit claims.{key}")
        for key in ("verified", "weak", "unverified", "total")
    }
    if validated_counts["total"] != sum(validated_counts[key] for key in ("verified", "weak", "unverified")):
        raise BenchmarkStateError("unit result claim total is inconsistent")
    if validated_counts["total"] <= 0:
        raise BenchmarkStateError("unit result claim total is unscorable")
    if record["uvr"] != validated_counts["unverified"] / validated_counts["total"]:
        raise BenchmarkStateError("unit result UVR does not match claim counts")
    if _nonnegative_count(record.get("sv"), "unit sv") > validated_counts["verified"]:
        raise BenchmarkStateError("unit result SV exceeds verified claim count")
    _finite_nonnegative_number(record.get("cost_usd"), "unit cost_usd")
    if not isinstance(record.get("errors"), list):
        raise BenchmarkStateError("unit result has invalid errors")
    frozen = record.get("frozen_web")
    if not isinstance(frozen, dict) or frozen.get("snapshot_hash") != snapshot_hash:
        raise BenchmarkStateError("unit result frozen snapshot identity does not match")
    if frozen.get("queries") != sorted(benchmark_web_queries([topic])):
        raise BenchmarkStateError("unit result frozen query consumption does not match")
    return record


def load_valid_unit(
    experiment_dir: Path,
    spec: ArmSpec,
    topic: dict,
    execution_sha: str,
    experiment_digest: str,
    snapshot_hash: str,
) -> dict | None:
    path = unit_path(experiment_dir, topic, spec.arm)
    if not path.exists():
        return None
    return validate_unit(
        load_json(path, "unit result"), spec, topic, execution_sha, experiment_digest, snapshot_hash,
    )


def _failure_record(
    spec: ArmSpec, topic: dict, execution_sha: str, experiment_digest: str, error: Exception,
) -> dict:
    return {
        "schema_version": 1,
        "status": "failed",
        "topic_id": topic["id"],
        "topic": topic["topic"],
        "arm": spec.arm,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": execution_sha,
        "experiment_digest": experiment_digest,
        "run_id": None,
        "verification_status": "unknown",
        "uvr": None,
        "sv": None,
        "cost_usd": None,
        "errors": [str(error)],
    }


def execute_unit(
    experiment_dir: Path,
    spec: ArmSpec,
    topic: dict,
    execution_sha: str,
    experiment_digest: str,
    snapshot_hash: str,
    invoke: Callable[[ArmSpec, dict], tuple[str, dict]],
) -> tuple[dict, bool]:
    """Run exactly one arm/topic or return the one valid matching saved result."""
    existing = load_valid_unit(experiment_dir, spec, topic, execution_sha, experiment_digest, snapshot_hash)
    if existing is not None:
        return existing, True
    path = unit_path(experiment_dir, topic, spec.arm)
    try:
        run_id, telemetry = invoke(spec, topic)
        record = unit_from_telemetry(
            spec, topic, execution_sha, experiment_digest, snapshot_hash, run_id, telemetry,
        )
    except Exception as error:
        atomic_write_json(
            failed_attempt_path(experiment_dir, topic, spec.arm),
            _failure_record(spec, topic, execution_sha, experiment_digest, error),
        )
        raise BenchmarkStateError(f"benchmark unit failed for topic {topic['id']} {spec.arm}: {error}") from error
    atomic_write_json(path, record)
    return record, False


def execute_units(
    experiment_dir: Path,
    topics: list[dict],
    execution_shas: dict[str, str],
    experiment_digest: str,
    snapshot_hash: str,
    invoke: Callable[[ArmSpec, dict], tuple[str, dict]],
) -> list[tuple[dict, bool]]:
    """Execute topic × arm cells in a deterministic order, stopping on the first failure."""
    completed = []
    for index, topic in enumerate(topics):
        arms = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
        for spec in arms:
            completed.append(execute_unit(
                experiment_dir, spec, topic, execution_shas[spec.arm], experiment_digest, snapshot_hash, invoke,
            ))
    return completed


def paired_aggregate(
    experiment_dir: Path,
    topics: list[dict],
    execution_shas: dict[str, str],
    experiment_digest: str,
    snapshot_hash: str,
) -> dict:
    """Compute the primary paired gates without turning model variance into retries."""
    rows = []
    failures = []
    baseline_sv = []
    candidate_sv = []
    for topic in topics:
        pair = {}
        for spec in ARMS:
            try:
                pair[spec.arm] = load_valid_unit(
                    experiment_dir, spec, topic, execution_shas[spec.arm], experiment_digest, snapshot_hash,
                )
                if pair[spec.arm] is None:
                    raise BenchmarkStateError("unit result is missing")
            except BenchmarkStateError as error:
                failures.append(f"topic {topic['id']} {spec.arm}: {error}")
        if set(pair) != {"baseline", "candidate"}:
            continue
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        baseline_sv.append(baseline["sv"])
        candidate_sv.append(candidate["sv"])
        delta_uvr = candidate["uvr"] - baseline["uvr"]
        delta_sv = candidate["sv"] - baseline["sv"]
        flags = []
        if delta_uvr > UVR_REGRESSION_REVIEW:
            flags.append("uvr_regression_review")
        if delta_sv < -SV_REGRESSION_REVIEW:
            flags.append("sv_regression_review")
        if candidate["uvr"] > UVR_THRESHOLD:
            failures.append(f"topic {topic['id']} candidate UVR {candidate['uvr']:.3f} > {UVR_THRESHOLD:.2f}")
        rows.append({
            "topic_id": topic["id"],
            "topic": topic["topic"],
            "baseline_uvr": baseline["uvr"],
            "candidate_uvr": candidate["uvr"],
            "delta_uvr": delta_uvr,
            "baseline_sv": baseline["sv"],
            "candidate_sv": candidate["sv"],
            "delta_sv": delta_sv,
            "review_flags": flags,
        })
    baseline_mean_sv = sum(baseline_sv) / len(baseline_sv) if baseline_sv else None
    candidate_mean_sv = sum(candidate_sv) / len(candidate_sv) if candidate_sv else None
    if len(rows) != len(topics):
        failures.append(f"only {len(rows)}/{len(topics)} topic pairs are valid and scorable")
    if baseline_mean_sv is None or candidate_mean_sv is None:
        failures.append("paired aggregate SV is unavailable")
    elif candidate_mean_sv < baseline_mean_sv:
        failures.append(f"candidate aggregate SV {candidate_mean_sv:.3f} < baseline {baseline_mean_sv:.3f}")
    return {
        "schema_version": 1,
        "total_primary_units": len(topics) * len(ARMS),
        "topic_pairs": rows,
        "baseline_aggregate_sv": baseline_mean_sv,
        "candidate_aggregate_sv": candidate_mean_sv,
        "gate_failures": failures,
        "gate_pass": not failures,
    }


def _extract_run_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            if run_id:
                return run_id
    raise BenchmarkStateError("successful arm CLI output did not emit RUN_ID")


def invoke_arm(
    spec: ArmSpec,
    topic: dict,
    worktrees: dict[str, Path],
    runtime: Path,
    snapshot_path: Path,
    collections: dict[str, str],
    qdrant_url: str,
) -> tuple[str, dict]:
    """Run one arm CLI and read only telemetry named by that CLI's RUN_ID."""
    worktree = worktrees[spec.arm]
    consumption_path = runtime / "consumption" / f"{spec.arm}-{topic['id']}-{uuid.uuid4().hex}.json"
    env = arm_environment(runtime, snapshot_path, collections[spec.arm], qdrant_url, consumption_path)
    proc = subprocess.run(
        ["uv", "run", "python", "main.py", "run", "--topic", topic["topic"],
         "--card-id", topic["card_id"], "--series", topic["series"], "--auto"],
        cwd=worktree, env=env, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise BenchmarkStateError(f"arm CLI exited with status {proc.returncode}: {proc.stderr[-500:]}")
    run_id = _extract_run_id(proc.stdout)
    path = worktree / "outputs" / "runs" / f"{run_id}.json"
    telemetry = load_json(path, "exact arm telemetry")
    telemetry["frozen_web"] = load_json(consumption_path, "frozen web consumption")
    return run_id, telemetry


def default_ingest(worktree: Path, collection: str, qdrant_url: str) -> None:
    env = dict(os.environ, QDRANT_COLLECTION=collection, QDRANT_URL=qdrant_url)
    proc = subprocess.run(
        ["uv", "run", "python", "scripts/ingest.py", "--source", "kb/seed_docs"],
        cwd=worktree, env=env, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise BenchmarkStateError(f"ingest failed for {collection}: {proc.stderr[-500:]}")


def run_benchmark(experiment_dir: Path, topics_path: Path, repo_root: Path, qdrant_url: str) -> dict:
    """Prepare isolated arms and execute the paired benchmark when separately authorized."""
    topics = load_topics(topics_path)
    require_canonical_topics(topics)
    snapshot_path = experiment_dir / "frozen_web.json"
    manifest = initialize_experiment(experiment_dir, topics, snapshot_path)
    worktree_info = {spec.arm: ensure_arm_worktree(repo_root, experiment_dir, spec) for spec in ARMS}
    worktrees = {arm: item[0] for arm, item in worktree_info.items()}
    execution_shas = {arm: item[1] for arm, item in worktree_info.items()}
    source_count = verify_same_seed_corpus(worktrees["baseline"], worktrees["candidate"])
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    collections = {}
    for spec in ARMS:
        collection_manifest = prepare_collection(
            experiment_dir, spec, execution_shas[spec.arm], source_count, client,
            lambda name, arm=spec.arm: default_ingest(worktrees[arm], name, qdrant_url),
        )
        collections[spec.arm] = collection_manifest["collection_name"]
    runtime = install_runtime(experiment_dir)
    manifest = bind_experiment_digest(experiment_dir, manifest, execution_shas, {
        spec.arm: load_json(collection_manifest_path(experiment_dir, spec.arm), "collection manifest")
        for spec in ARMS
    }, runtime)
    experiment_digest = manifest["experiment_digest"]
    snapshot_hash = manifest["frozen_web_snapshot_hash"]
    execute_units(
        experiment_dir,
        topics,
        execution_shas,
        experiment_digest,
        snapshot_hash,
        lambda spec, topic: invoke_arm(spec, topic, worktrees, runtime, snapshot_path, collections, qdrant_url),
    )
    aggregate = paired_aggregate(experiment_dir, topics, execution_shas, experiment_digest, snapshot_hash)
    aggregate["experiment_id"] = manifest["experiment_id"]
    atomic_write_json(experiment_dir / "paired_aggregate.json", aggregate)
    if not aggregate["gate_pass"]:
        raise BenchmarkStateError("paired benchmark primary gates failed")
    return aggregate


def _main() -> None:
    parser = argparse.ArgumentParser(description="Paired baseline/candidate benchmark harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture-web", help="capture the frozen Tavily snapshot once")
    capture.add_argument("--experiment-dir", type=Path, required=True)
    capture.add_argument("--topics", type=Path, default=Path("evals/topics.json"))
    run = subparsers.add_parser("run", help="run the prepared paired benchmark")
    run.add_argument("--experiment-dir", type=Path, required=True)
    run.add_argument("--topics", type=Path, default=Path("evals/topics.json"))
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    args = parser.parse_args()
    if args.command == "capture-web":
        from tools.web_search import web_search

        snapshot = capture_frozen_web_snapshot(load_topics(args.topics), args.experiment_dir / "frozen_web.json", web_search)
        print(f"captured {len(snapshot['queries'])} frozen web query results")
        return
    result = run_benchmark(args.experiment_dir, args.topics, args.repo_root, args.qdrant_url)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
