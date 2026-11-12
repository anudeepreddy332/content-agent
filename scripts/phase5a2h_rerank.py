"""One frozen offline cross-encoder intervention over archived CSWP union pools."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import phase5a2_shadow_ab as ab  # noqa: E402

STARTING_HEAD = "2a87e8e3e756872667c4468d4a52c34ea8917416"
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
OUTPUT = ROOT / "reports/phase5/phase5a2h"
CONTRACT = ROOT / "evals/fixtures/phase5a2h_reranker_contract.json"
TOKENIZER_FIXTURE = ROOT / "evals/fixtures/phase5a2h_reranker_tokenizer"
CSWP = ROOT / "reports/phase5/phase5a2e/candidate_cswp_manifest.json"
CONTROL = ROOT / "reports/phase5/phase5a2e/results.json"
FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
FOCUS = ("Q09", "Q13", "Q16", "Q17", "Q21", "Q22", "Q19", "Q30")
BATCH_SIZE = 8
MAX_INPUT = 512


class RerankError(RuntimeError):
    """A frozen identity, candidate, token, score or provenance gate failed."""


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return ab.sha256_bytes(Path(path).read_bytes())


def snapshot_path():
    return (
        Path.home()
        / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots"
        / REVISION
    )


def library_versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "tokenizers",
            "sentence-transformers",
            "safetensors",
            "numpy",
        )
    }


def model_identity(snapshot):
    """Inspect local artifacts only; never resolve a hub name or download."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    if snapshot.name != REVISION or any(
        not (snapshot / name).is_file() for name in FILES
    ):
        raise RerankError(
            "ACQUISITION_DEPENDENCY: pinned complete local reranker snapshot missing"
        )
    config = read(snapshot / "config.json")
    if (
        config.get("architectures") != ["BertForSequenceClassification"]
        or config.get("num_hidden_layers") != 6
        or config.get("max_position_embeddings") != MAX_INPUT
        or len(config.get("id2label", {})) != 1
    ):
        raise RerankError("reranker architecture or capacity drift")
    if (
        config.get("sbert_ce_default_activation_function")
        != "torch.nn.modules.linear.Identity"
    ):
        raise RerankError("reranker score activation drift")
    with safe_open(
        snapshot / "model.safetensors", framework="pt", device="cpu"
    ) as weights:
        layers = sorted(
            {
                int(key.split(".")[3])
                for key in weights.keys()
                if key.startswith("bert.encoder.layer.")
            }
        )
        classifier_shape = weights.get_slice("classifier.weight").get_shape()
    if layers != list(range(6)) or classifier_shape != [1, 384]:
        raise RerankError("reranker weight architecture mismatch")
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "file_sha256": {name: file_hash(snapshot / name) for name in FILES},
        "weights_bytes": (snapshot / "model.safetensors").stat().st_size,
        "architecture": "BertForSequenceClassification",
        "encoder_layers": layers,
        "classifier_shape": classifier_shape,
        "max_input_tokens": MAX_INPUT,
        "score": "raw single float32 relevance logit; Identity activation; higher is better; no score fusion",
        "config_name_or_path": config.get("_name_or_path"),
        "config_name_note": "Historical L-12 string retained as metadata; snapshot revision, six weight layers and classifier shape are authoritative. No substitution.",
    }


def load_tokenizer(path, model):
    from transformers import AutoTokenizer

    path = Path(path)
    for name in FILES:
        if name != "model.safetensors" and (
            not (path / name).is_file()
            or file_hash(path / name) != model["file_sha256"][name]
        ):
            raise RerankError("tokenizer identity mismatch: " + name)
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if (
        not tokenizer.is_fast
        or tokenizer.model_max_length != MAX_INPUT
        or tokenizer.num_special_tokens_to_add(pair=True) != 3
    ):
        raise RerankError("unexpected reranker pair tokenizer")
    return tokenizer


def union_pool(query, units):
    """Exact dense-top20 union BM25-top20; no hybrid cutoff or root filtering."""
    combined = {}
    for channel in ("dense", "bm25"):
        rows = query["channels"][channel]["candidate_top20"]
        if (
            not isinstance(rows, list)
            or len(rows) > 20
            or (channel == "dense" and len(rows) != 20)
        ):
            raise RerankError("missing or invalid native top-20 candidate list")
        seen = set()
        for rank, row in enumerate(rows, 1):
            cid = row["chunk_id"]
            if (
                cid in seen
                or cid not in units
                or row["rank"] != rank
                or not math.isfinite(row["native_score"])
            ):
                raise RerankError("invalid candidate ID, rank or native score")
            seen.add(cid)
            unit = units[cid]
            if Path(unit["source_path"]).stem != row["source"]:
                raise RerankError("candidate source mismatch")
            if cid not in combined:
                combined[cid] = {
                    "chunk_id": cid,
                    "source": row["source"],
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "provenance": {
                        key: unit[key]
                        for key in (
                            "document_id",
                            "document_version",
                            "source_path",
                            "source_sha256",
                            "source_spans",
                            "retrieval_text_sha256",
                        )
                    },
                    "h1_only": is_h1(unit),
                }
            combined[cid][channel + "_rank"] = rank
            combined[cid][channel + "_score"] = row["native_score"]
    return [combined[cid] for cid in sorted(combined)]


def is_h1(unit):
    structural = [s for s in unit["structural_segments"] if s["kind"] != "seam"]
    return bool(structural) and all(
        s["kind"] == "heading" and s.get("heading_level") == 1 for s in structural
    )


def pair_encoding(tokenizer, query, candidate, max_input=MAX_INPUT):
    q = tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"]
    c = tokenizer(candidate, add_special_tokens=False, truncation=False)["input_ids"]
    encoded = tokenizer(
        query, candidate, add_special_tokens=True, truncation=False, padding=False
    )
    ids = encoded["input_ids"]
    specials = tokenizer.num_special_tokens_to_add(pair=True)
    # Prove pairing keeps both token sequences; no truncation-policy fallback.
    expected = (
        [tokenizer.cls_token_id]
        + q
        + [tokenizer.sep_token_id]
        + c
        + [tokenizer.sep_token_id]
    )
    if ids != expected or len(ids) != len(q) + len(c) + specials:
        raise RerankError("pair tokenizer changed or silently truncated content")
    if len(ids) > max_input:
        raise RerankError(
            f"RERANKER_PAIR_OVERFLOW: {len(ids)} > {max_input}; STOP without scoring"
        )
    counts = {
        "query_tokens": len(q),
        "candidate_tokens": len(c),
        "special_tokens": specials,
        "total_pair_tokens": len(ids),
        "truncated_tokens": 0,
        "truncation": False,
        "pair_input_sha256": ab.sha256_json(dict(encoded)),
    }
    return encoded, counts


def preflight_pairs(queries, units, tokenizer):
    rows = []
    for query in queries:
        pool = union_pool(query, units)
        for candidate in pool:
            _, counts = pair_encoding(
                tokenizer,
                query["query"],
                units[candidate["chunk_id"]]["retrieval_text"],
            )
            candidate["pair_tokens"] = counts
        rows.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "gating_eligible": query["gating_eligible"],
                "pool_size": len(pool),
                "candidates": pool,
            }
        )
    return rows


def input_data():
    from scripts.retrieval_golden_v2 import load_oracle, validate_oracle
    from scripts import phase5a0_baseline as baseline
    from scripts.phase5a2b_packed import eval_chunks

    oracle = load_oracle()
    validate_oracle(oracle)
    _, _, evidence = baseline.load_sources_and_evidence(oracle)
    manifest = read(CSWP)
    expected = read(ROOT / "reports/phase5/phase5a2e/build_report.json")[
        "deterministic_fingerprint"
    ]
    if (
        ab.sha256_json(manifest) != expected
        or expected
        != "439ccdf81e2aff1bc0bf3448734cb71f774bc8d3e7619dd1e4b51b2685b79bb3"
    ):
        raise RerankError("CSWP representation drift")
    units = {u["chunk_id"]: u for u in manifest["children"]}
    titles = {
        d["source_path"]: d["title"]
        for d in read(ROOT / "reports/phase5/phase5a1/candidate_b_manifest.json")[
            "documents"
        ]
    }
    if len(units) != 159:
        raise RerankError("CSWP logical ID/count drift")
    for unit in units.values():
        raw = (ROOT / unit["source_path"]).read_bytes()
        content = "".join(
            raw[s["source_byte_start"] : s["source_byte_end"]].decode("utf-8")
            for s in unit["source_spans"]
        )
        core = "".join(
            raw[s["source_byte_start"] : s["source_byte_end"]].decode("utf-8")
            for s in unit["source_spans"]
            if s["role"] == "core"
        )
        reconstructed = (
            titles[unit["source_path"]]
            + "\n"
            + " > ".join(unit["heading_path"])
            + "\n\n"
            + content
        )
        if (
            file_hash(ROOT / unit["source_path"]) != unit["source_sha256"]
            or core != unit["canonical_content"]
            or reconstructed != unit["retrieval_text"]
            or ab.sha256_bytes(unit["retrieval_text"].encode())
            != unit["retrieval_text_sha256"]
        ):
            raise RerankError("CSWP provenance/text drift")
    control = read(CONTROL)
    queries = control["arms"]["CSWP"]["per_query"]
    if [(q["query_id"], q["query"], q["gating_eligible"]) for q in queries] != [
        (q["query_id"], q["query"], q["gating_eligible"]) for q in oracle["queries"]
    ]:
        raise RerankError("query text/order/gating drift")
    return (
        oracle,
        evidence,
        units,
        control,
        {c.chunk_id: c for c in eval_chunks(manifest, oracle)},
    )


def freeze():
    """Freeze identity, pair ledger, classification and gates before inference."""

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if (
        git("rev-parse", "HEAD") != STARTING_HEAD
        or git("rev-parse", "refs/remotes/origin/feature/phase5-rag-hardening")
        != STARTING_HEAD
        or git("branch", "--show-current") != "feature/phase5-rag-hardening"
    ):
        raise RerankError("starting checkout/ref mismatch")
    if CONTRACT.exists():
        raise RerankError("contract already frozen; refusing overwrite")
    model = model_identity(snapshot_path())
    tokenizer = load_tokenizer(snapshot_path(), model)
    oracle, _, units, control, _ = input_data()
    pairs = preflight_pairs(control["arms"]["CSWP"]["per_query"], units, tokenizer)
    paths = [
        "scripts/phase5a0_baseline.py",
        "scripts/phase5a1_shadow.py",
        "scripts/phase5a2_shadow_ab.py",
        "scripts/phase5a2b_packed.py",
        "scripts/phase5a2e_cswp.py",
        "evals/fixtures/retrieval_golden_v2.json",
        "evals/fixtures/phase5a2e_cswp_contract.json",
        "reports/phase5/phase5a0/baseline_a_manifest.json",
        "reports/phase5/phase5a2e/experiment_manifest.json",
        "reports/phase5/phase5a2e/candidate_cswp_manifest.json",
        "reports/phase5/phase5a2e/results.json",
    ] + [row["path"] for row in oracle["corpus_manifest"]]
    contract = {
        "schema_version": "phase5a2h_cross_encoder_contract_v1",
        "required_parent": STARTING_HEAD,
        "remote_check": "local origin/feature/phase5-rag-hardening ref matches required HEAD; no network fetch",
        "model": model,
        "library_versions": library_versions(),
        "input_file_sha256": {name: file_hash(ROOT / name) for name in paths},
        "cswp_fingerprint": ab.sha256_json(read(CSWP)),
        "candidate_pool_sha256": ab.sha256_json(pairs),
        "candidate_generation": "Replay exact archived CSWP dense-top-20 union BM25-top-20 by canonical chunk_id; preserve native ranks/scores; no new embeddings or candidate filtering",
        "determinism": {
            "device": "cpu",
            "dtype": "float32",
            "seed": 0,
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "deterministic_algorithms": True,
            "mkldnn_enabled": False,
            "attention_implementation": "eager",
            "eval_mode": True,
            "inference_mode": True,
            "batch_size": BATCH_SIZE,
            "candidate_order": "chunk_id ascending",
            "final_order": "raw relevance logit descending then chunk_id ascending",
        },
        "tokenization": {
            "input": "query and exact CSWP retrieval text as separate pair arguments, without stripping",
            "max_input_tokens": MAX_INPUT,
            "special_tokens": 3,
            "truncation": False,
            "padding": "longest in each fixed batch after length preflight",
            "overflow": "STOP before any experiment scoring; never invent a truncation policy",
        },
        "metric_implementation": "unchanged phase5a2_shadow_ab._rank_metrics and _span_covered",
        "classification": {
            "evidence": "top-5 exact gold-span coverage sets: any lost span => REGRESSED, else any gained => IMPROVED, else SAME; report gains/losses and recall separately",
            "ranking": "source recall, precision, graded nDCG at 1/3/5/10 and MRR@10; any component decrease => REGRESSED, else any increase => IMPROVED, else SAME; report mixed component changes",
            "ranking_only_regression": "EVIDENCE SAME and RANKING-METRIC REGRESSED",
        },
        "advance_gate": {
            "material_improvement": "reranked gating evidence Recall@5 > CSWP control and >= frozen A; conservative sufficient interpretation of approaches/exceeds A, with no tuned numeric threshold",
            "critical_regressions": "zero grade-2 spans lost at top-5 versus either CSWP control or A on gating queries",
            "determinism": "two fresh process runs have byte-identical scores, rankings and metrics",
            "token_safety": "zero pairs truncated or above 512",
            "resource_bound": "one fixed six-layer CPU float32 model, at most 40 candidates/query, batch size 8, no retries or parameter sweep; measured p50/p95 and peak RSS; no production latency SLO claimed",
        },
        "absence_metrics": None,
        "diagnostic_non_gating": ["Q25", "Q26"],
        "production_retrieval_changed": False,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
        "provider_calls": 0,
        "external_network_calls": 0,
        "mmr": False,
        "candidate_c": False,
    }
    contract["contract_sha256"] = ab.sha256_json(contract)
    ab.write_json(CONTRACT, contract)
    ab.write_json(OUTPUT / "candidate_pools.json", pairs)
    print(
        json.dumps(
            {
                "contract_sha256": contract["contract_sha256"],
                "pairs": sum(q["pool_size"] for q in pairs),
                "max_pair_tokens": max(
                    c["pair_tokens"]["total_pair_tokens"]
                    for q in pairs
                    for c in q["candidates"]
                ),
            },
            indent=2,
        )
    )


def verify_contract(contract, check_model=True):
    payload = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if ab.sha256_json(payload) != contract["contract_sha256"]:
        raise RerankError("frozen contract digest mismatch")
    for name, expected in contract["input_file_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise RerankError("frozen input drift: " + name)
    # The parent experiment binds the full baseline/representation dependency chain.
    for name, expected in read(
        ROOT / "reports/phase5/phase5a2e/experiment_manifest.json"
    )["file_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise RerankError("historical dependency drift: " + name)
    if check_model and (
        model_identity(snapshot_path()) != contract["model"]
        or library_versions() != contract["library_versions"]
    ):
        raise RerankError("reranker/runtime identity drift")


def configure_torch():
    import random
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def load_model(snapshot):
    from transformers import AutoModelForSequenceClassification
    import torch

    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RerankError("incomplete or incompatible reranker weights")
    model.to("cpu").eval()
    if model.config.num_labels != 1 or model.config.num_hidden_layers != 6:
        raise RerankError("loaded architecture mismatch")
    return model


def sort_scores(pool, scores):
    if (
        len(scores) != len(pool)
        or len({r["chunk_id"] for r in pool}) != len(pool)
        or not all(math.isfinite(s) for s in scores)
    ):
        raise RerankError("score roster mismatch or nonfinite relevance score")
    rows = [
        {
            "chunk_id": r["chunk_id"],
            "source": r["source"],
            "cross_encoder_score": float(score),
        }
        for r, score in zip(pool, scores, strict=True)
    ]
    rows.sort(key=lambda row: (-row["cross_encoder_score"], row["chunk_id"]))
    return [{**row, "rank": rank} for rank, row in enumerate(rows, 1)]


def score_pool(model, tokenizer, query, pool, units):
    import torch

    scores = []
    for start in range(0, len(pool), BATCH_SIZE):
        batch = pool[start : start + BATCH_SIZE]
        encodings = []
        for candidate in batch:
            encoded, counts = pair_encoding(
                tokenizer, query, units[candidate["chunk_id"]]["retrieval_text"]
            )
            if counts != candidate["pair_tokens"]:
                raise RerankError("scored pair differs from preflight")
            encodings.append(dict(encoded))
        padded = tokenizer.pad(encodings, padding=True, return_tensors="pt")
        if padded["input_ids"].shape[1] > MAX_INPUT or padded["attention_mask"].sum(
            dim=1
        ).tolist() != [c["pair_tokens"]["total_pair_tokens"] for c in batch]:
            raise RerankError("padded pair length/truncation mismatch")
        with torch.inference_mode():
            logits = model(**padded).logits
        if tuple(logits.shape) != (len(batch), 1):
            raise RerankError("unexpected score shape")
        scores.extend(logits[:, 0].tolist())
    return sort_scores(pool, scores)


def span_key(span):
    return span.source + "::" + span.span_id


def span_coverage(spans, rows, chunks, k):
    return {span_key(s) for s in spans if ab._span_covered(s, rows, chunks, k)}


def classify_evidence(before, after):
    lost, gained = sorted(before - after), sorted(after - before)
    return {
        "classification": "EVIDENCE REGRESSED"
        if lost
        else "EVIDENCE IMPROVED"
        if gained
        else "EVIDENCE SAME",
        "lost_spans": lost,
        "gained_spans": gained,
    }


def classify_ranking(before, after):
    keys = [
        f"{metric}@{k}"
        for metric in ("source_recall", "precision", "ndcg")
        for k in ab.K_VALUES
    ] + ["mrr@10"]
    decreased = [key for key in keys if after[key] < before[key]]
    increased = [key for key in keys if after[key] > before[key]]
    return {
        "classification": "RANKING-METRIC REGRESSED"
        if decreased
        else "RANKING-METRIC IMPROVED"
        if increased
        else "RANKING-METRIC SAME",
        "decreased_metrics": decreased,
        "increased_metrics": increased,
        "mixed": bool(decreased and increased),
    }


def occupancy(rows):
    result = {}
    for k in ab.K_VALUES:
        counts = Counter(r["source"] for r in rows[:k])
        returned = min(k, len(rows))
        largest = max(counts.values(), default=0)
        result[str(k)] = {
            "distinct_source_count": len(counts),
            "repeated_source_slots": returned - len(counts),
            "max_same_document_slots": largest,
            "max_same_document_fraction": largest / returned if returned else 0.0,
        }
    return result


def evaluate_rows(oracle, evidence, chunks, units, control, pools, ranked):
    from scripts import phase5a0_baseline as baseline

    cswp_queries = {q["query_id"]: q for q in control["arms"]["CSWP"]["per_query"]}
    a_queries = {q["query_id"]: q for q in control["arms"]["A"]["per_query"]}
    # A is historical only. Its exact span IDs/coverage are recovered from frozen legacy spans.
    a_manifest = read(ROOT / "reports/phase5/phase5a0/baseline_a_manifest.json")
    source_paths = {
        entry["source"]: entry["path"] for entry in oracle["corpus_manifest"]
    }
    a_chunks = {
        c["chunk_id"]: ab.EvalChunk(
            c["ordinal"],
            c["chunk_id"],
            c["source"],
            source_paths[c["source"]],
            "",
            ((c["source_char_start"], c["source_char_end"]),),
            0,
        )
        for c in a_manifest["chunking"]["ordered_chunks"]
    }
    spans_by_query = defaultdict(list)
    for span in evidence:
        spans_by_query[span.query_id].append(span)
    rows = []
    for query, pool in zip(oracle["queries"], pools, strict=True):
        qid = query["query_id"]
        ordered = ranked[qid]
        if {r["chunk_id"] for r in ordered} != {
            r["chunk_id"] for r in pool["candidates"]
        } or len(ordered) != pool["pool_size"]:
            raise RerankError("reranked result is not the complete frozen pool")
        spans = spans_by_query[qid]
        control_rows = cswp_queries[qid]["channels"]["hybrid"]["top10"]
        a_rows = a_queries[qid]["channels"]["hybrid"]["top10"]
        cm = ab._rank_metrics(query, spans, control_rows, chunks)
        am = ab._rank_metrics(query, spans, a_rows, a_chunks)
        if (
            cm != cswp_queries[qid]["channels"]["hybrid"]["metrics"]
            or am != a_queries[qid]["channels"]["hybrid"]["metrics"]
        ):
            raise RerankError("frozen control metric reproduction failed")
        rm = ab._rank_metrics(query, spans, ordered, chunks)
        coverage = {
            name: span_coverage(spans, result, mapping, 5)
            for name, result, mapping in [
                ("control", control_rows, chunks),
                ("reranked", ordered, chunks),
                ("A", a_rows, a_chunks),
            ]
        }
        pool_coverage = span_coverage(
            spans, pool["candidates"], chunks, pool["pool_size"]
        )
        evidence_delta = classify_evidence(coverage["control"], coverage["reranked"])
        ranking_delta = classify_ranking(cm, rm)
        critical = {span_key(s) for s in spans if s.grade == 2}
        h1 = []
        rank_by_id = {r["chunk_id"]: r["rank"] for r in ordered}
        control_rank = {r["chunk_id"]: r["rank"] for r in control_rows}
        for c in pool["candidates"]:
            if c["h1_only"]:
                h1.append(
                    {
                        "chunk_id": c["chunk_id"],
                        "source": c["source"],
                        "dense_rank": c["dense_rank"],
                        "bm25_rank": c["bm25_rank"],
                        "control_hybrid_rank_at_10": control_rank.get(c["chunk_id"]),
                        "reranked_rank": rank_by_id[c["chunk_id"]],
                        "demoted_from_control_top5": c["chunk_id"]
                        in {r["chunk_id"] for r in control_rows[:5]}
                        and rank_by_id[c["chunk_id"]] > 5,
                    }
                )
        rows.append(
            {
                "query_id": qid,
                "query": query["query"],
                "gating_eligible": query["gating_eligible"],
                "channels": {
                    "control": {"metrics": cm, "top10": control_rows},
                    "reranked": {
                        "metrics": rm,
                        "top10": ordered[:10],
                        "top5": ordered[:5],
                    },
                    "A": {"metrics": am, "top10": a_rows},
                },
                "evidence_comparison": evidence_delta,
                "ranking_comparison": ranking_delta,
                "ranking_only_regression": evidence_delta["classification"]
                == "EVIDENCE SAME"
                and ranking_delta["classification"] == "RANKING-METRIC REGRESSED",
                "evidence_vs_A": classify_evidence(coverage["A"], coverage["reranked"]),
                "critical_spans_lost_vs_control": sorted(
                    (coverage["control"] - coverage["reranked"]) & critical
                ),
                "critical_spans_lost_vs_A": sorted(
                    (coverage["A"] - coverage["reranked"]) & critical
                ),
                "span_coverage_at_5": {
                    name: sorted(value) for name, value in coverage.items()
                },
                "candidate_pool": {
                    "size": pool["pool_size"],
                    "evidence_recall": len(pool_coverage) / len(spans)
                    if spans
                    else 0.0,
                    "source_recall": baseline.source_recall_at_k(
                        pool["candidates"],
                        {r["source"]: r["grade"] for r in query["relevant_sources"]},
                        pool["pool_size"],
                    ),
                    "covered_spans": sorted(pool_coverage),
                    "missing_spans": sorted(
                        {span_key(s) for s in spans} - pool_coverage
                    ),
                },
                "ranked_pool": ordered,
                "h1_observations": h1,
                "occupancy": {
                    "control": occupancy(control_rows),
                    "reranked": occupancy(ordered),
                    "A": occupancy(a_rows),
                },
            }
        )
    aggregate = {
        channel: {
            scope: ab._aggregate(rows, channel, gating_only=gating)
            for scope, gating in [
                ("gating_33_excludes_q25_q26", True),
                ("all_35_diagnostic", False),
            ]
        }
        for channel in ("control", "reranked", "A")
    }
    for channel, name in [("control", "CSWP"), ("A", "A")]:
        if aggregate[channel] != control["arms"][name]["aggregate_metrics"]["hybrid"]:
            raise RerankError("frozen control aggregate mismatch")
    gating = [r for r in rows if r["gating_eligible"]]
    lists = {
        label: [r["query_id"] for r in gating if r[field]["classification"] == label]
        for field, prefix in [
            ("evidence_comparison", "EVIDENCE"),
            ("ranking_comparison", "RANKING-METRIC"),
        ]
        for label in [prefix + " IMPROVED", prefix + " SAME", prefix + " REGRESSED"]
    }
    lists["ranking_only_regressions"] = [
        r["query_id"] for r in gating if r["ranking_only_regression"]
    ]
    pool_summary = {
        scope: {
            "query_count": len(selected),
            "mean_candidates": statistics.fmean(
                r["candidate_pool"]["size"] for r in selected
            ),
            "min_candidates": min(r["candidate_pool"]["size"] for r in selected),
            "max_candidates": max(r["candidate_pool"]["size"] for r in selected),
            "evidence_recall": statistics.fmean(
                r["candidate_pool"]["evidence_recall"] for r in selected
            ),
            "source_recall": statistics.fmean(
                r["candidate_pool"]["source_recall"] for r in selected
            ),
        }
        for scope, selected in [("gating_33", gating), ("all_35", rows)]
    }
    diversity = {
        channel: {
            str(k): {
                metric: statistics.fmean(
                    r["occupancy"][channel][str(k)][metric] for r in gating
                )
                for metric in gating[0]["occupancy"][channel][str(k)]
            }
            for k in ab.K_VALUES
        }
        for channel in ("control", "reranked", "A")
    }
    recalls = {
        name: aggregate[name]["gating_33_excludes_q25_q26"]["evidence_span_recall@5"]
        for name in aggregate
    }
    critical_losses = [
        r["query_id"]
        for r in gating
        if r["critical_spans_lost_vs_control"] or r["critical_spans_lost_vs_A"]
    ]
    return {
        "aggregate_metrics": aggregate,
        "candidate_pool": pool_summary,
        "gating_classification": lists,
        "per_query": rows,
        "focus_query_ids": list(FOCUS),
        "occupancy_gating": diversity,
        "quality_gate": {
            "material_evidence_improvement": recalls["reranked"] > recalls["control"]
            and recalls["reranked"] >= recalls["A"],
            "critical_evidence_regressions": critical_losses,
            "quality_pass": recalls["reranked"] > recalls["control"]
            and recalls["reranked"] >= recalls["A"]
            and not critical_losses,
        },
        "absence_metrics": None,
        "absence_reason": "golden-v2 has zero ABSENT queries; all 35 are development/diagnostic",
        "provider_calls": 0,
        "external_network_calls": 0,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
        "embeddings_generated": 0,
    }


def rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def run(output):
    contract = read(CONTRACT)
    verify_contract(contract)
    oracle, evidence, units, control, chunks = input_data()
    tokenizer = load_tokenizer(snapshot_path(), contract["model"])
    pairs = preflight_pairs(control["arms"]["CSWP"]["per_query"], units, tokenizer)
    if ab.sha256_json(pairs) != contract["candidate_pool_sha256"] or pairs != read(
        OUTPUT / "candidate_pools.json"
    ):
        raise RerankError("frozen candidate/pair ledger drift")
    configure_torch()
    before = rss_bytes()
    start = time.perf_counter_ns()
    model = load_model(snapshot_path())
    load_ms = (time.perf_counter_ns() - start) / 1e6
    parameters = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    ranked, timings = {}, []
    for pool in pairs:
        start = time.perf_counter_ns()
        ranked[pool["query_id"]] = score_pool(
            model, tokenizer, pool["query"], pool["candidates"], units
        )
        timings.append(
            {
                "query_id": pool["query_id"],
                "candidates": pool["pool_size"],
                "elapsed_ms": (time.perf_counter_ns() - start) / 1e6,
            }
        )
    result = evaluate_rows(oracle, evidence, chunks, units, control, pairs, ranked)
    result["contract_sha256"] = contract["contract_sha256"]
    result["token_safety"] = {
        "candidate_pairs": sum(q["pool_size"] for q in pairs),
        "max_pair_tokens": max(
            c["pair_tokens"]["total_pair_tokens"]
            for q in pairs
            for c in q["candidates"]
        ),
        "pairs_truncated": 0,
        "pairs_exceeding_max_input": 0,
    }
    result["deterministic_score_sha256"] = ab.sha256_json(ranked)
    result["scoring_runner_sha256"] = file_hash(Path(__file__))
    runtime = {
        "result_sha256": ab.sha256_json(result),
        "model_load_ms": load_ms,
        "latency": ab._latency_summary([t["elapsed_ms"] for t in timings]),
        "per_query": timings,
        "latency_scope": "one pass over all 35 queries; includes pair tokenization, batching, inference and sorting, excludes model loading; first query cold; no warmup or retries",
        "resources": {
            "parameters": parameters,
            "float32_parameter_bytes": parameter_bytes,
            "weights_file_bytes": contract["model"]["weights_bytes"],
            "process_peak_rss_before_model_bytes": before,
            "process_peak_rss_after_evaluation_bytes": rss_bytes(),
            "rss_scope": "process lifetime peak, includes Python/torch/transformers/evaluator and model, not isolated model allocation",
            "device": "cpu",
            "threads": 1,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "provider_calls": 0,
        "external_network_calls": 0,
    }
    ab.write_json(output / "results.json", result, canonical=True)
    ab.write_json(output / "runtime.json", runtime)
    print(
        json.dumps(
            {
                "result_sha256": runtime["result_sha256"],
                "token_safety": result["token_safety"],
                "quality_gate": result["quality_gate"],
                "latency": runtime["latency"],
            },
            indent=2,
        )
    )


def main():
    ab.install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        if (
            args.output is None
            or not args.output.resolve().is_relative_to(OUTPUT.resolve())
            or args.output.resolve() == OUTPUT.resolve()
        ):
            raise RerankError("run output must be an isolated phase5a2h subdirectory")
        if (args.output / "results.json").exists():
            raise RerankError("refusing to overwrite an experiment result")
        run(args.output)


if __name__ == "__main__":
    main()
