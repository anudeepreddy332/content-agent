"""Frozen, offline same-parent packing ablation; A and B remain unchanged."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import phase5a1_shadow as shadow  # noqa: E402
from scripts import phase5a2_shadow_ab as ab  # noqa: E402

CONTRACT = ROOT / "evals/fixtures/phase5a2b_packing_contract.json"
OUTPUT = ROOT / "reports/phase5/phase5a2b"
STARTING_HEAD = "a37b402227c8095cca2f3558c3cf60bbc52e2813"


class PackingError(RuntimeError):
    """A packing, provenance, or frozen-control invariant failed."""


def content_interval(child):
    spans = child["source_spans"]
    if not spans or any(s["role"] != "content" for s in spans):
        return None
    if any(
        a["source_char_end"] != b["source_char_start"] for a, b in zip(spans, spans[1:])
    ):
        return None
    return spans[0]["source_char_start"], spans[-1]["source_char_end"]


def compatible(left, right, blocks, raw, contract):
    for key in ("document_version", "source_path", "parent_block_id", "heading_path"):
        if left[key] != right[key]:
            return False
    a, b = blocks[left["origin_block_id"]], blocks[right["origin_block_id"]]
    if any(x["block_type"] not in contract["compatible_types"] for x in (a, b)):
        return False
    if b["ordinal"] - a["ordinal"] not in (0, 1):
        return False
    x, y = content_interval(left), content_interval(right)
    if x is None or y is None or x[1] > y[0]:
        return False
    return not raw[x[1] : y[0]].strip()


def make_unit(group, documents, sources, tokenizer, contract):
    first = group[0]
    source = sources[first["source_path"]]
    if len(group) == 1:
        spans = deepcopy(first["source_spans"])
        content, retrieval = first["canonical_content"], first["retrieval_text"]
    else:
        start = content_interval(first)[0]
        end = content_interval(group[-1])[1]
        spans = [source.span(start, end)]
        content = source.text[start:end]
        retrieval = shadow.serialize(
            documents[first["document_version"]]["title"],
            first["heading_path"],
            content,
        )
    payload = {
        "candidate": "B-PACKED",
        "packing_version": contract["packing_version"],
        "contract_sha256": ab.sha256_json(contract),
        "document_version": first["document_version"],
        "source_path": first["source_path"],
        "source_sha256": first["source_sha256"],
        "parent_block_id": first["parent_block_id"],
        "heading_path": first["heading_path"],
        "constituent_child_ids": [c["chunk_id"] for c in group],
        "ordered_block_ids": list(dict.fromkeys(c["origin_block_id"] for c in group)),
        "source_spans": spans,
        "retrieval_text_sha256": shadow.sha(retrieval),
        "embedding_tokenizer_id": tokenizer.tokenizer_id,
    }
    return {
        **payload,
        "chunk_id": "ca:b-packed:" + ab.sha256_json(payload),
        "canonical_content": content,
        "retrieval_text": retrieval,
        "embedding_content_token_count": tokenizer.count(retrieval),
        "embedding_total_token_count": tokenizer.count(retrieval, specials=True),
    }


def validate_packed(manifest_b, units, sources, tokenizer, contract):
    """Reject loss/duplication, structural crossings, unsafe tokens and false provenance."""
    original = manifest_b["children"]
    children = {c["chunk_id"]: c for c in original}
    blocks = {b["block_id"]: b for b in manifest_b["blocks"]}
    documents = {d["document_version"]: d for d in manifest_b["documents"]}
    flattened = [cid for u in units for cid in u["constituent_child_ids"]]
    if flattened != [c["chunk_id"] for c in original]:
        raise PackingError("child membership/order/loss/duplication")
    if len({u["chunk_id"] for u in units}) != len(units):
        raise PackingError("duplicate logical IDs")
    for unit in units:
        group = [children[cid] for cid in unit["constituent_child_ids"]]
        source = sources[unit["source_path"]]
        if shadow.sha(source.raw) != unit["source_sha256"]:
            raise PackingError("source hash drift")
        for a, b in zip(group, group[1:]):
            if not compatible(a, b, blocks, source.text, contract):
                raise PackingError("structural boundary violation")
        extracted = ""
        for span in unit["source_spans"]:
            expected_span = source.span(
                span["source_char_start"], span["source_char_end"], span["role"]
            )
            if span != expected_span:
                raise PackingError("byte/character/line provenance mismatch")
            extracted += source.extract(span)
        if extracted != unit["canonical_content"]:
            raise PackingError("source content mismatch")
        if unit != make_unit(group, documents, sources, tokenizer, contract):
            raise PackingError("serialization/identity/provenance mismatch")
        count = tokenizer.count(unit["retrieval_text"])
        total = tokenizer.count(unit["retrieval_text"], specials=True)
        if count > 254 or total != count + 2 or total > 256:
            raise PackingError("full serialization overflow")


def pack(manifest_b, tokenizer, contract, root=ROOT):
    """Pure representation transform. Never mutate the frozen B manifest."""
    blocks = {b["block_id"]: b for b in manifest_b["blocks"]}
    documents = {d["document_version"]: d for d in manifest_b["documents"]}
    sources = {
        d["source_path"]: shadow.Source((root / d["source_path"]).read_bytes())
        for d in manifest_b["documents"]
    }
    units, pending = [], []
    for child in manifest_b["children"]:
        if pending:
            raw = sources[pending[0]["source_path"]].text
            if compatible(pending[-1], child, blocks, raw, contract):
                candidate = make_unit(
                    pending + [child], documents, sources, tokenizer, contract
                )
                if candidate["embedding_content_token_count"] <= 254:
                    pending.append(child)
                    continue
            units.append(make_unit(pending, documents, sources, tokenizer, contract))
        pending = [child]
    if pending:
        units.append(make_unit(pending, documents, sources, tokenizer, contract))
    validate_packed(manifest_b, units, sources, tokenizer, contract)
    manifest = {
        "candidate": "B-PACKED",
        "schema_version": "phase5a2b_representation_v1",
        "contract_sha256": ab.sha256_json(contract),
        "frozen_B_fingerprint": shadow.sha(shadow.canonical_json(manifest_b)),
        "children": units,
    }
    counts = sorted(u["embedding_content_token_count"] for u in units)
    report = {
        "retrieval_chunks": len(units),
        "token_distribution": {
            "min": min(counts),
            "median": statistics.median(counts),
            "p95": counts[math.ceil(0.95 * len(counts)) - 1],
            "max": max(counts),
        },
        "children_above_254": sum(x > 254 for x in counts),
        "packing_groups": sum(len(u["constituent_child_ids"]) > 1 for u in units),
        "group_size_histogram": dict(
            sorted(Counter(len(u["constituent_child_ids"]) for u in units).items())
        ),
        "average_children_per_packed_unit": len(manifest_b["children"]) / len(units),
        "provenance_failures": 0,
        "duplicate_logical_ids": 0,
        "structural_boundary_violations": 0,
        "deterministic_fingerprint": ab.sha256_json(manifest),
    }
    return manifest, report


def eval_chunks(packed, oracle):
    source_by_path = {d["path"]: d["source"] for d in oracle["corpus_manifest"]}
    offsets = {
        path: ab._source_offsets((ROOT / path).read_text(encoding="utf-8"))
        for path in source_by_path
    }
    return [
        ab.EvalChunk(
            ordinal=i,
            chunk_id=u["chunk_id"],
            source=source_by_path[u["source_path"]],
            source_path=u["source_path"],
            retrieval_text=u["retrieval_text"],
            source_intervals=tuple(
                interval
                for s in u["source_spans"]
                if (
                    interval := ab._normalize_interval(
                        s["source_char_start"],
                        s["source_char_end"],
                        *offsets[u["source_path"]],
                    )
                )
                is not None
            ),
            embedding_content_token_count=u["embedding_content_token_count"],
        )
        for i, u in enumerate(packed["children"])
    ]


def freeze(output):
    """Durably preregister rules and executable hashes before any retrieval metrics."""
    contract = json.loads(CONTRACT.read_text())
    old = json.loads(
        (ROOT / "reports/phase5/phase5a2/experiment_manifest.json").read_text()
    )
    files = [
        CONTRACT,
        Path(__file__),
        ROOT / "scripts/phase5a0_baseline.py",
        ROOT / "scripts/phase5a1_shadow.py",
        ROOT / "scripts/phase5a2_shadow_ab.py",
        ROOT / "evals/fixtures/retrieval_golden_v2.json",
        ab.BASELINE_MANIFEST,
        ab.CANDIDATE_B_MANIFEST,
    ]
    manifest = {
        "required_parent": STARTING_HEAD,
        "candidate": "B-PACKED",
        "packing_contract": contract,
        "held_constant": old["held_constant"],
        "control_arms": old["arms"],
        "file_sha256": {
            str(p.relative_to(ROOT)): ab.sha256_bytes(p.read_bytes()) for p in files
        },
        "provider_calls": 0,
        "external_network_calls": 0,
    }
    path = output / "experiment_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["manifest_sha256"] != ab.sha256_json(
            {k: v for k, v in existing.items() if k != "manifest_sha256"}
        ):
            raise PackingError("preregistration hash mismatch")
        observed = {
            k: v
            for k, v in existing.items()
            if k not in ("frozen_at_utc", "manifest_sha256")
        }
        if observed != manifest:
            correction_path = output / "control_check_correction.json"
            if not correction_path.exists():
                raise PackingError(
                    "preregistered implementation changed; do not retune"
                )
            correction = json.loads(correction_path.read_text())
            expected = deepcopy(observed)
            script_path = str(Path(__file__).relative_to(ROOT))
            if (
                correction["original_manifest_sha256"] != existing["manifest_sha256"]
                or correction["before_runner_sha256"]
                != expected["file_sha256"][script_path]
            ):
                raise PackingError("invalid historical control-check correction")
            expected["file_sha256"][script_path] = correction["after_runner_sha256"]
            if expected != manifest:
                raise PackingError("change beyond recorded control-check correction")
            manifest["original_manifest_sha256"] = existing["manifest_sha256"]
            manifest["correction_sha256"] = ab.sha256_json(correction)
            manifest["frozen_at_utc"] = existing["frozen_at_utc"]
            manifest["manifest_sha256"] = ab.sha256_json(manifest)
            ab.write_json(output / "execution_manifest.json", manifest)
            return manifest
        return existing
    manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["manifest_sha256"] = ab.sha256_json(manifest)
    ab.write_json(path, manifest)
    return manifest


def diversity(arm):
    rows = [r for r in arm["per_query"] if r["gating_eligible"]]
    return {
        channel: {
            str(k): {
                "unique_sources": statistics.fmean(
                    len({x["source"] for x in r["channels"][channel]["top10"][:k]})
                    for r in rows
                ),
                "repeated_source_slots": statistics.fmean(
                    len(r["channels"][channel]["top10"][:k])
                    - len({x["source"] for x in r["channels"][channel]["top10"][:k]})
                    for r in rows
                ),
                "scope": "33 gating queries; repeated slots = returned slots - unique sources",
            }
            for k in ab.K_VALUES
        }
        for channel in ("dense", "bm25", "hybrid")
    }


def comparisons(arms):
    rows = []
    for i, p in enumerate(arms["B-PACKED"]["per_query"]):
        channels = {}
        for channel in ("dense", "bm25", "hybrid"):
            metrics = {
                name: arm["per_query"][i]["channels"][channel]["metrics"]
                for name, arm in arms.items()
            }
            channels[channel] = {
                "metrics": metrics,
                **{
                    "vs_" + name: {
                        "classification": ab._classification(
                            metrics[name], metrics["B-PACKED"]
                        ),
                        "deltas": {
                            key: ab._delta(metrics["B-PACKED"][key], metrics[name][key])
                            for key in metrics[name]
                        },
                    }
                    for name in ("A", "B")
                },
            }
        rows.append(
            {
                "query_id": p["query_id"],
                "gating_eligible": p["gating_eligible"],
                "channels": channels,
                "vs_A": channels["hybrid"]["vs_A"]["classification"],
                "vs_B": channels["hybrid"]["vs_B"]["classification"],
            }
        )
    return rows


def historical_control_check(observed, historical):
    """Require exact historical ranks/metrics; quantify native-score drift separately."""
    differences = []

    def walk(left, right, path=""):
        if type(left) is not type(right):
            raise PackingError("historical control type drift: " + path)
        if isinstance(left, dict):
            if left.keys() != right.keys():
                raise PackingError("historical control schema drift: " + path)
            for key in left:
                walk(left[key], right[key], path + "/" + key)
        elif isinstance(left, list):
            if len(left) != len(right):
                raise PackingError("historical control length drift: " + path)
            for i, (a, b) in enumerate(zip(left, right)):
                walk(a, b, path + "/" + str(i))
        elif left != right:
            if path.endswith("/native_score"):
                differences.append(abs(left - right))
            else:
                raise PackingError("historical control ranking/metric drift: " + path)

    walk(observed, historical)
    return {
        "rankings_and_metrics_equal": True,
        "native_score_differences": len(differences),
        "max_absolute_native_score_difference": max(differences, default=0),
    }


def evaluate(output, manifest):
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from scripts import phase5a0_baseline as baseline

    oracle, legacy, chunks_b, evidence, _, context = ab.load_representations()
    contract = manifest["packing_contract"]
    tokenizer = shadow.MiniLMTokenizer()
    builds = []
    # Rebuild B independently each time, and validate it against its immutable file.
    for run in (1, 2):
        b, _ = shadow.build_corpus(tokenizer)
        if b != context["candidate_b_manifest"]:
            raise PackingError("B drift")
        packed, report = pack(b, tokenizer, contract)
        builds.append(packed)
        ab.write_json(output / f"run-{run}/build_report.json", report)
    if builds[0] != builds[1]:
        raise PackingError("packing fingerprint mismatch")
    ab.write_json(output / "candidate_b_packed_manifest.json", builds[0])
    ab.write_json(output / "build_report.json", report)
    chunks = {
        "A": ab._a_eval_chunks(legacy),
        "B": chunks_b,
        "B-PACKED": eval_chunks(builds[0], oracle),
    }
    encoder = SentenceTransformer(str(context["snapshot"]), local_files_only=True)
    if (
        encoder.max_seq_length != baseline.MODEL_MAX_SEQUENCE_LENGTH
        or encoder.get_embedding_dimension() != 384
    ):
        raise PackingError("MiniLM identity drift")
    results, runtimes = [], []
    for run in (1, 2):
        embeddings = {}
        embedding_ms = {}
        for name, units in chunks.items():
            embeddings[name], embedding_ms[name] = ab._encode(
                encoder, [u.retrieval_text for u in units]
            )
        query_vectors, query_ms = ab._encode(
            encoder, [q["query"] for q in oracle["queries"]]
        )
        arms, runtime, storage = {}, {}, {}
        for name, units in chunks.items():
            arms[name], runtime[name] = ab.evaluate_arm(
                name, units, embeddings[name], query_vectors, oracle, evidence
            )
            bm25 = BM25Okapi([u.retrieval_text.lower().split() for u in units])
            storage[name] = {
                "vectors": len(units),
                "vector_multiplier_vs_A": len(units) / len(chunks["A"]),
                "float32_embedding_bytes": embeddings[name].nbytes,
                "retrieval_text_utf8_bytes": sum(
                    len(u.retrieval_text.encode()) for u in units
                ),
                "bm25_pickle_protocol_4_bytes": len(pickle.dumps(bm25, protocol=4)),
                "index_scope": "in-memory exact dense vectors and BM25; pickle size is serialized index size, not resident memory",
            }
        # Reproduction of original controls is a gate, not an assumed property.
        previous = json.loads(
            (ROOT / "reports/phase5/phase5a2/aggregate_comparison.json").read_text()
        )
        for name in ("A", "B"):
            if arms[name]["aggregate_metrics"] != previous["arms"][name]:
                raise PackingError("A/B metric reproduction failure")
        previous_queries = json.loads(
            (ROOT / "reports/phase5/phase5a2/per_query_comparison.json").read_text()
        )["queries"]
        control_checks = []
        for i, previous_query in enumerate(previous_queries):
            for name in ("A", "B"):
                if arms[name]["per_query"][i]["query_id"] != previous_query["query_id"]:
                    raise PackingError("A/B query order drift")
                for channel in ("dense", "bm25", "hybrid"):
                    control_checks.append(
                        {
                            "arm": name,
                            "query_id": previous_query["query_id"],
                            "channel": channel,
                            **historical_control_check(
                                arms[name]["per_query"][i]["channels"][channel],
                                previous_query["channels"][channel][name],
                            ),
                        }
                    )
        rows = comparisons(arms)
        gating = {
            name: {
                label: [
                    r["query_id"]
                    for r in rows
                    if r["gating_eligible"] and r["vs_" + name] == label
                ]
                for label in ("IMPROVED", "SAME", "REGRESSED")
            }
            for name in ("A", "B")
        }
        recall = {
            name: arm["aggregate_metrics"]["hybrid"]["gating_33_excludes_q25_q26"][
                "evidence_span_recall@5"
            ]
            for name, arm in arms.items()
        }
        advance = (
            recall["B-PACKED"] >= recall["A"]
            and recall["B-PACKED"] - recall["B"]
            >= contract["advance_gate"][
                "minimum_absolute_hybrid_evidence_recall_at_5_gain_over_B"
            ]
            and not gating["A"]["REGRESSED"]
        )
        result = {
            "experiment_manifest_sha256": manifest["manifest_sha256"],
            "arms": arms,
            "per_query_comparison": rows,
            "gating_queries": gating,
            "diversity": {name: diversity(arm) for name, arm in arms.items()},
            "cost": storage,
            "B_PACKED_ready_to_advance": advance,
            "absence_metrics": None,
            "provider_calls": 0,
            "external_network_calls": 0,
            "historical_control_checks": control_checks,
        }
        runtime_row = {
            "repetition": run,
            "embedding_ms": embedding_ms,
            "query_embedding_ms": query_ms,
            "retrieval": runtime,
            "deterministic_result_sha256": ab.sha256_json(result),
        }
        results.append(result)
        runtimes.append(runtime_row)
        ab.write_json(output / f"run-{run}/runtime_observation.json", runtime_row)
        ab.write_json(
            output / f"run-{run}/determinism_report.json",
            {
                "result_sha256": ab.sha256_json(result),
                "packed_fingerprint": report["deterministic_fingerprint"],
                "arm_sha256": {name: ab.sha256_json(arm) for name, arm in arms.items()},
                "embedding_sha256": {
                    name: ab.sha256_bytes(v.tobytes()) for name, v in embeddings.items()
                },
                "query_embedding_sha256": ab.sha256_bytes(query_vectors.tobytes()),
            },
        )
    if results[0] != results[1]:
        raise PackingError("retrieval rerun mismatch")
    for name, value in (
        ("results", results[0]),
        (
            "runtime_summary",
            {"repetitions": runtimes, "deterministic_rerun_equal": True},
        ),
        ("per_query_comparison", rows),
    ):
        ab.write_json(output / f"{name}.json", value)
    print(
        json.dumps(
            {
                "build": report,
                "gating": gating,
                "hybrid_evidence_recall@5": recall,
                "ready_to_advance": advance,
                "deterministic_rerun_equal": True,
            },
            indent=2,
        )
    )


def main():
    ab.install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    manifest = freeze(OUTPUT)
    if args.freeze_only:
        print(manifest["manifest_sha256"])
    else:
        evaluate(OUTPUT, manifest)


if __name__ == "__main__":
    main()
