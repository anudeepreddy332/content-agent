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
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


ARMS = (
    ArmSpec(
        arm="baseline",
        architecture_sha=BASELINE_SHA,
        chunk_contract={"tokenizer": "cl100k_base", "content_tokens": 400, "overlap_tokens": 50},
    ),
    ArmSpec(
        arm="candidate",
        architecture_sha=CANDIDATE_SHA,
        chunk_contract={"tokenizer": "all-MiniLM-L6-v2", "content_tokens": 224, "overlap_tokens": 32},
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
        if not isinstance(results, list):
            raise BenchmarkStateError(f"web capture returned invalid results for query: {query!r}")
        captured[query] = {"max_results": 5, "results": results}
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
        if not isinstance(entry, dict) or entry.get("max_results") != 5 or not isinstance(entry.get("results"), list):
            raise BenchmarkStateError(f"frozen web snapshot is malformed for query: {query!r}")
    return snapshot


def collection_name(experiment_id: str, arm: str) -> str:
    safe_id = re.sub(r"[^a-z0-9_]+", "_", experiment_id.lower()).strip("_")
    if not safe_id:
        raise BenchmarkStateError("experiment id must contain letters or digits")
    return f"paired_{safe_id}_{arm}"


def experiment_path(experiment_dir: Path) -> Path:
    return experiment_dir / "experiment.json"


def initialize_experiment(experiment_dir: Path, topics: list[dict], snapshot_path: Path) -> dict:
    """Create or verify the minimal immutable provenance record."""
    snapshot = load_frozen_web_snapshot(snapshot_path, topics)
    experiment_id = experiment_dir.name
    manifest = {
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
    path = experiment_path(experiment_dir)
    if path.exists():
        existing = load_json(path, "experiment manifest")
        if existing != manifest:
            raise BenchmarkStateError("existing experiment provenance does not match requested benchmark")
        return existing
    atomic_write_json(path, manifest)
    return manifest


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


def _collection_point_count(client, name: str) -> int:
    try:
        return int(client.get_collection(collection_name=name).points_count)
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
    path = collection_manifest_path(experiment_dir, spec.arm)
    names = _collection_names(client)
    if name in names:
        point_count = _collection_point_count(client, name)
        if point_count > 0:
            if not path.exists():
                raise BenchmarkStateError(f"refusing unexpected nonempty collection: {name}")
            manifest = load_json(path, "collection manifest")
            validate_collection_manifest(manifest, spec, name, source_count, point_count, execution_sha)
            return manifest
        if path.exists():
            raise BenchmarkStateError(f"refusing manifest-proven collection with zero points: {name}")

    ingest(name)
    point_count = _collection_point_count(client, name)
    if point_count <= 0:
        raise BenchmarkStateError(f"ingest did not create a nonempty disposable collection: {name}")
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
        "from benchmark_runtime import install_frozen_web_search\ninstall_frozen_web_search()\n",
        encoding="utf-8",
    )
    return runtime


def arm_environment(runtime: Path, snapshot_path: Path, collection: str, qdrant_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CONTENT_AGENT_FROZEN_WEB_SNAPSHOT"] = str(snapshot_path.resolve())
    env["QDRANT_COLLECTION"] = collection
    env["QDRANT_URL"] = qdrant_url
    env["PYTHONPATH"] = str(runtime) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def unit_path(experiment_dir: Path, topic: dict, arm: str) -> Path:
    return experiment_dir / "units" / f"topic-{int(topic['id']):02d}-{arm}.json"


def _claim_total(telemetry: dict) -> int:
    return sum(telemetry.get(key, 0) for key in ("claims_verified", "claims_weak", "claims_unverified"))


def unit_from_telemetry(
    spec: ArmSpec,
    topic: dict,
    execution_sha: str,
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
    total = _claim_total(telemetry)
    if total <= 0:
        raise BenchmarkStateError("completed benchmark unit has no scorable verifier verdicts")
    uvr = telemetry.get("claims_unverified", 0) / total
    grounded_depth = telemetry.get("grounded_depth")
    if not isinstance(grounded_depth, dict) or not isinstance(grounded_depth.get("SV"), (int, float)):
        raise BenchmarkStateError("telemetry lacks a numeric grounded-depth SV")
    errors = telemetry["error_log"]
    if not isinstance(errors, list):
        raise BenchmarkStateError("telemetry error_log is invalid")
    return {
        "schema_version": 1,
        "status": "completed",
        "topic_id": topic["id"],
        "topic": topic["topic"],
        "arm": spec.arm,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": execution_sha,
        "run_id": run_id,
        "verification_status": telemetry["verification_status"],
        "uvr": uvr,
        "sv": grounded_depth["SV"],
        "cost_usd": telemetry["total_cost_usd"],
        "errors": errors,
    }


def validate_unit(record: dict, spec: ArmSpec, topic: dict, execution_sha: str) -> dict:
    expected = {
        "schema_version": 1,
        "status": "completed",
        "topic_id": topic["id"],
        "topic": topic["topic"],
        "arm": spec.arm,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": execution_sha,
        "verification_status": "completed",
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise BenchmarkStateError(f"unit result {field} does not match expected {spec.arm} evidence")
    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        raise BenchmarkStateError("unit result is missing RUN_ID")
    if not isinstance(record.get("uvr"), (int, float)) or not 0 <= record["uvr"] <= 1:
        raise BenchmarkStateError("unit result has invalid UVR")
    if not isinstance(record.get("sv"), (int, float)):
        raise BenchmarkStateError("unit result has invalid SV")
    if not isinstance(record.get("errors"), list):
        raise BenchmarkStateError("unit result has invalid errors")
    return record


def load_valid_unit(experiment_dir: Path, spec: ArmSpec, topic: dict, execution_sha: str) -> dict | None:
    path = unit_path(experiment_dir, topic, spec.arm)
    if not path.exists():
        return None
    return validate_unit(load_json(path, "unit result"), spec, topic, execution_sha)


def _failure_record(spec: ArmSpec, topic: dict, execution_sha: str, error: Exception) -> dict:
    return {
        "schema_version": 1,
        "status": "failed",
        "topic_id": topic["id"],
        "topic": topic["topic"],
        "arm": spec.arm,
        "architecture_sha": spec.architecture_sha,
        "execution_sha": execution_sha,
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
    invoke: Callable[[ArmSpec, dict], tuple[str, dict]],
) -> tuple[dict, bool]:
    """Run exactly one arm/topic or return the one valid matching saved result."""
    existing = load_valid_unit(experiment_dir, spec, topic, execution_sha)
    if existing is not None:
        return existing, True
    path = unit_path(experiment_dir, topic, spec.arm)
    try:
        run_id, telemetry = invoke(spec, topic)
        record = unit_from_telemetry(spec, topic, execution_sha, run_id, telemetry)
    except Exception as error:
        atomic_write_json(path, _failure_record(spec, topic, execution_sha, error))
        raise BenchmarkStateError(f"benchmark unit failed for topic {topic['id']} {spec.arm}: {error}") from error
    atomic_write_json(path, record)
    return record, False


def execute_units(
    experiment_dir: Path,
    topics: list[dict],
    execution_shas: dict[str, str],
    invoke: Callable[[ArmSpec, dict], tuple[str, dict]],
) -> list[tuple[dict, bool]]:
    """Execute topic × arm cells in a deterministic order, stopping on the first failure."""
    completed = []
    for topic in topics:
        for spec in ARMS:
            completed.append(execute_unit(experiment_dir, spec, topic, execution_shas[spec.arm], invoke))
    return completed


def paired_aggregate(experiment_dir: Path, topics: list[dict], execution_shas: dict[str, str]) -> dict:
    """Compute the primary paired gates without turning model variance into retries."""
    rows = []
    failures = []
    baseline_sv = []
    candidate_sv = []
    for topic in topics:
        pair = {}
        for spec in ARMS:
            try:
                pair[spec.arm] = load_valid_unit(experiment_dir, spec, topic, execution_shas[spec.arm])
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
    env = arm_environment(runtime, snapshot_path, collections[spec.arm], qdrant_url)
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
    execute_units(
        experiment_dir,
        topics,
        execution_shas,
        lambda spec, topic: invoke_arm(spec, topic, worktrees, runtime, snapshot_path, collections, qdrant_url),
    )
    aggregate = paired_aggregate(experiment_dir, topics, execution_shas)
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
