"""Phase 5B1: Candidate C v1 bounded adjacent-neighbor expansion.

Retrieval ranks stay frozen at CSWP hybrid/RRF top-5 from Phase 5A2E.
The only new variable is post-retrieval ±1 neighbor expansion plus a
fixed 2000 cl100k pack budget. This is not full parent-child retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import phase5a2_shadow_ab as ab  # noqa: E402
from scripts.phase5a2b_packed import eval_chunks  # noqa: E402
from agent.retrieval import expansion as _expansion  # noqa: E402

PACK_BUDGET_CL100K = _expansion.PACK_BUDGET_CL100K
cl100k = _expansion.cl100k
cl100k_count = _expansion.cl100k_count
interval_key = _expansion.interval_key
flatten = _expansion.flatten

STARTING_HEAD = "f67b07054a8823ffbc6b3ebc23c3233f9e5e7496"
CSWP_FINGERPRINT = (
    "439ccdf81e2aff1bc0bf3448734cb71f774bc8d3e7619dd1e4b51b2685b79bb3"
)
CONTRACT = ROOT / "evals/fixtures/phase5b1_candidate_c_contract.json"
OUTPUT = ROOT / "reports/phase5/phase5b1"
CSWP = ROOT / "reports/phase5/phase5a2e/candidate_cswp_manifest.json"
CONTROL = ROOT / "reports/phase5/phase5a2e/results.json"
A_MANIFEST = ROOT / "reports/phase5/phase5a0/baseline_a_manifest.json"
DRAFTER_K = 3
DRAFTER_CHAR_LIMIT = 2000
VERIFIER_K = 5
FOCUS = ("Q09", "Q13", "Q19", "Q21", "Q22", "Q30")
RELATIONS = ("PREVIOUS", "SEED", "NEXT")


class CandidateCError(RuntimeError):
    """A Candidate C identity, expansion, pack, or provenance invariant failed."""


def _wrap_expansion_error(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _expansion.ExpansionError as exc:
            raise CandidateCError(str(exc)) from exc

    return wrapped


expand_seeds = _wrap_expansion_error(_expansion.expand_seeds)
dedupe_groups = _wrap_expansion_error(_expansion.dedupe_groups)
pack_units_seed_first = _wrap_expansion_error(_expansion.pack_units_seed_first)
assert_seed_preservation_invariant = _wrap_expansion_error(
    _expansion.assert_seed_preservation_invariant
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return ab.sha256_bytes(Path(path).read_bytes())


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1)]


def token_summary(values):
    ordered = [int(v) for v in values]
    return {
        "min": min(ordered, default=0),
        "median": int(statistics.median(ordered)) if ordered else 0,
        "p95": int(percentile(ordered, 95)),
        "max": max(ordered, default=0),
        "mean": round(statistics.fmean(ordered), 6) if ordered else 0.0,
    }


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_starting_head():
    if (
        git("rev-parse", "HEAD") != STARTING_HEAD
        or git("rev-parse", "refs/remotes/origin/feature/phase5-rag-hardening")
        != STARTING_HEAD
        or git("branch", "--show-current") != "feature/phase5-rag-hardening"
    ):
        raise CandidateCError("starting checkout/ref mismatch")


def input_files():
    from scripts.retrieval_golden_v2 import load_oracle, validate_oracle

    oracle = load_oracle()
    validate_oracle(oracle)
    names = [
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
    ]
    names.extend(row["path"] for row in oracle["corpus_manifest"])
    return names, oracle


def load_units():
    manifest = read(CSWP)
    if ab.sha256_json(manifest) != CSWP_FINGERPRINT:
        raise CandidateCError("CSWP representation drift")
    units = {u["chunk_id"]: u for u in manifest["children"]}
    if len(units) != 159:
        raise CandidateCError("CSWP logical ID/count drift")
    by_source = defaultdict(list)
    for unit in manifest["children"]:
        by_source[unit["source_path"]].append(unit)
    for rows in by_source.values():
        starts = [u["retrieval_char_start"] for u in rows]
        if starts != sorted(starts):
            raise CandidateCError("CSWP source order is not deterministic")
    return manifest, units, dict(by_source)


def frozen_seeds(control):
    rows = []
    for query in control["arms"]["CSWP"]["per_query"]:
        seeds = []
        for rank, item in enumerate(query["channels"]["hybrid"]["top10"][:5], 1):
            if item["rank"] != rank:
                raise CandidateCError("frozen hybrid top-5 rank drift")
            seeds.append(
                {
                    "rank": rank,
                    "chunk_id": item["chunk_id"],
                    "source": item["source"],
                    "rrf_score": item["rrf_score"],
                    "dense_rank": item.get("dense_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                }
            )
        rows.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "gating_eligible": query["gating_eligible"],
                "seeds": seeds,
            }
        )
    return rows


def pack_units(rows, units, budget, encoding):
    """Legacy frozen-order packer (pre-5B2C baseline for counterfactual comparison)."""

    packed = []
    skipped = []
    used = 0
    exhausted = False
    for row in rows:
        cost = cl100k_count(units[row["chunk_id"]]["retrieval_text"], encoding)
        if used + cost > budget:
            exhausted = True
            skipped.append({**row, "cl100k_tokens": cost, "skip_reason": "PACK_BUDGET"})
            continue
        packed.append({**row, "cl100k_tokens": cost})
        used += cost
    return packed, skipped, used, exhausted


def content_prefix_len(unit):
    slice_len = unit["retrieval_char_end"] - unit["retrieval_char_start"]
    prefix = len(unit["retrieval_text"]) - slice_len
    if prefix < 0 or unit["retrieval_text"][prefix:] != unit["retrieval_text"][-slice_len:]:
        raise CandidateCError("retrieval_text/source slice mismatch")
    return prefix


def clipped_source_intervals(unit, chunks, char_limit):
    text = unit["retrieval_text"]
    if char_limit is None or len(text) <= char_limit:
        return chunks[unit["chunk_id"]].source_intervals
    prefix = content_prefix_len(unit)
    kept = max(0, char_limit - prefix)
    raw_start = unit["retrieval_char_start"]
    raw_end = min(unit["retrieval_char_end"], raw_start + kept)
    offsets = ab._source_offsets(
        (ROOT / unit["source_path"]).read_text(encoding="utf-8")
    )
    intervals = []
    for span in unit["source_spans"]:
        start = max(span["source_char_start"], raw_start)
        end = min(span["source_char_end"], raw_end)
        normalized = ab._normalize_interval(start, end, *offsets)
        if normalized is not None:
            intervals.append(normalized)
    return tuple(intervals)


def layer_chunks(rows, units, chunks, char_limit=None):
    mapping = {}
    for row in rows:
        unit = units[row["chunk_id"]]
        base = chunks[row["chunk_id"]]
        mapping[row["chunk_id"]] = ab.EvalChunk(
            base.ordinal,
            base.chunk_id,
            base.source,
            base.source_path,
            unit["retrieval_text"]
            if char_limit is None
            else unit["retrieval_text"][:char_limit],
            clipped_source_intervals(unit, chunks, char_limit),
            base.embedding_content_token_count,
        )
    return mapping


def overlap_chars(rows, units):
    by_source = defaultdict(list)
    for row in rows:
        unit = units[row["chunk_id"]]
        for start, end in (
            (s["source_char_start"], s["source_char_end"]) for s in unit["source_spans"]
        ):
            by_source[unit["source_path"]].append((start, end))
    extra = 0
    for intervals in by_source.values():
        total = sum(end - start for start, end in intervals)
        merged = []
        for start, end in sorted(intervals):
            if not merged or start >= merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        union = sum(end - start for start, end in merged)
        extra += total - union
    return extra


def unique_interval_count(rows, units):
    keys = {
        (
            units[row["chunk_id"]]["source_path"],
            tuple(
                (s["source_char_start"], s["source_char_end"])
                for s in units[row["chunk_id"]]["source_spans"]
            ),
        )
        for row in rows
    }
    return len(keys)


def occupancy(rows, units):
    sources = [Path(units[row["chunk_id"]]["source_path"]).stem for row in rows]
    counts = Counter(sources)
    n = len(sources)
    largest = max(counts.values(), default=0)
    return {
        "units": n,
        "distinct_sources": len(counts),
        "repeated_source_slots": n - len(counts),
        "max_same_document_slots": largest,
        "max_same_document_fraction": round(largest / n, 8) if n else 0.0,
    }


def source_recall(query, rows, units):
    relevant = {row["source"] for row in query["relevant_sources"]}
    if not relevant:
        return 0.0
    exposed = {Path(units[row["chunk_id"]]["source_path"]).stem for row in rows}
    return round(len(relevant & exposed) / len(relevant), 8)


def evidence_recall(spans, rows, mapping):
    if not spans:
        return 0.0
    return round(
        sum(ab._span_covered(span, rows, mapping, len(rows)) for span in spans)
        / len(spans),
        8,
    )


def covered_keys(spans, rows, mapping):
    return {
        span.source + "::" + span.span_id
        for span in spans
        if ab._span_covered(span, rows, mapping, len(rows))
    }


def layer_metrics(query, spans, rows, units, chunks, char_limit=None):
    mapping = layer_chunks(rows, units, chunks, char_limit)
    encoding = cl100k()
    texts = []
    for row in rows:
        text = units[row["chunk_id"]]["retrieval_text"]
        texts.append(text if char_limit is None else text[:char_limit])
    return {
        "evidence_span_recall": evidence_recall(spans, rows, mapping),
        "source_recall": source_recall(query, rows, units),
        "units": len(rows),
        "unique_source_intervals": unique_interval_count(rows, units),
        "cl100k_tokens": sum(cl100k_count(text, encoding) for text in texts),
        "minilm_content_tokens": sum(
            units[row["chunk_id"]]["embedding_content_token_count"] for row in rows
        ),
        "occupancy": occupancy(rows, units),
        "covered_spans": sorted(covered_keys(spans, rows, mapping)),
        "chunk_ids": [row["chunk_id"] for row in rows],
    }


def seed_group_layer(query, spans, packed, units, chunks, max_rank, char_limit):
    """Apply the production K/clip contract to frozen retrieved seed ranks."""

    by_rank = defaultdict(list)
    for row in packed:
        if row["seed_rank"] <= max_rank:
            by_rank[row["seed_rank"]].append(row)
    exposed = []
    mapping = {}
    encoding = cl100k()
    texts = []
    for rank in range(1, max_rank + 1):
        remaining = char_limit
        for row in by_rank.get(rank, []):
            unit = units[row["chunk_id"]]
            text = unit["retrieval_text"]
            if remaining is not None:
                if remaining <= 0:
                    break
                text = text[:remaining]
                remaining -= len(text)
            clip = None if char_limit is None else len(text)
            if char_limit is None:
                intervals = chunks[row["chunk_id"]].source_intervals
            else:
                intervals = clipped_source_intervals(unit, chunks, clip)
            base = chunks[row["chunk_id"]]
            mapping[row["chunk_id"]] = ab.EvalChunk(
                base.ordinal,
                base.chunk_id,
                base.source,
                base.source_path,
                text,
                intervals,
                base.embedding_content_token_count,
            )
            exposed.append(row)
            texts.append(text)
    return {
        "evidence_span_recall": evidence_recall(spans, exposed, mapping),
        "source_recall": source_recall(query, exposed, units),
        "units": len(exposed),
        "unique_source_intervals": unique_interval_count(exposed, units),
        "cl100k_tokens": sum(cl100k_count(text, encoding) for text in texts),
        "minilm_content_tokens": sum(
            units[row["chunk_id"]]["embedding_content_token_count"] for row in exposed
        ),
        "occupancy": occupancy(exposed, units),
        "covered_spans": sorted(covered_keys(spans, exposed, mapping)),
        "chunk_ids": [row["chunk_id"] for row in exposed],
        "seed_ranks": list(range(1, max_rank + 1)),
    }


def a_top5_cl100k(query_id, control, a_chunks, source_texts, encoding):
    rows = next(
        q for q in control["arms"]["A"]["per_query"] if q["query_id"] == query_id
    )["channels"]["hybrid"]["top10"][:5]
    total = 0
    for row in rows:
        chunk = a_chunks[row["chunk_id"]]
        text = source_texts[chunk["source"]][
            chunk["source_char_start"] : chunk["source_char_end"]
        ]
        total += cl100k_count(text, encoding)
    return total


def classify(before, after):
    lost = sorted(set(before) - set(after))
    gained = sorted(set(after) - set(before))
    return {
        "classification": (
            "EVIDENCE REGRESSED"
            if lost
            else "EVIDENCE IMPROVED"
            if gained
            else "EVIDENCE SAME"
        ),
        "lost_spans": lost,
        "gained_spans": gained,
    }


def freeze():
    require_starting_head()
    if CONTRACT.exists():
        raise CandidateCError("contract already frozen; refusing overwrite")
    names, _oracle = input_files()
    manifest, _units, _by_source = load_units()
    control = read(CONTROL)
    seeds = frozen_seeds(control)
    if [(row["query_id"], row["query"], row["gating_eligible"]) for row in seeds] != [
        (q["query_id"], q["query"], q["gating_eligible"])
        for q in control["arms"]["CSWP"]["per_query"]
    ]:
        raise CandidateCError("query text/order/gating drift")
    contract = {
        "schema_version": "phase5b1_candidate_c_v1_contract",
        "name": "Candidate C v1 — Bounded Adjacent-Neighbor Expansion",
        "parent_child_retrieval": False,
        "required_parent": STARTING_HEAD,
        "remote_check": (
            "local origin/feature/phase5-rag-hardening ref matches required HEAD; "
            "no network fetch"
        ),
        "pack_budget_cl100k": PACK_BUDGET_CL100K,
        "pack_tokenizer": "cl100k_base",
        "pack_policy": (
            "whole CSWP retrieval_text units only; if the next unit exceeds the "
            "remaining 2000-token ceiling, skip it and continue in frozen order; "
            "record PACK_BUDGET_EXHAUSTED; never split a unit; never use A's "
            "per-query token count as execution logic"
        ),
        "expansion": {
            "seeds": "frozen CSWP hybrid/RRF top-5 from reports/phase5/phase5a2e/results.json",
            "distance": 1,
            "members": ["PREVIOUS", "SEED", "NEXT"],
            "same_document_id": True,
            "same_document_version": True,
            "same_source_path_and_sha": True,
            "contiguous_source_order": True,
            "cross_document": False,
            "distance_2": False,
            "full_h1_parent_dump": False,
            "semantic_similarity": False,
            "query_specific": False,
            "gold_labels": False,
            "reranker_seeds": False,
        },
        "dedup": [
            "canonical chunk_id first-seen in seed-rank then source order",
            "exact source interval identity; non-identical intervals are kept",
        ],
        "ordering": [
            "seed rank ascending",
            "within a seed group source reading order",
            "chunk_id ascending tie-break",
            "earliest seed-rank provenance is primary",
        ],
        "exposure_layers": [
            "retrieved",
            "expanded",
            "packed",
            "drafter_exposed",
            "verifier_exposed",
        ],
        "drafter_contract": {
            "k": DRAFTER_K,
            "character_prefix_per_item": DRAFTER_CHAR_LIMIT,
            "items": (
                "the frozen retrieved top-3 seeds remain kb_results[:3]; each "
                "result exposes its packed PREVIOUS/SEED/NEXT retrieval_text "
                "in source order, clipped independently to 2000 characters"
            ),
        },
        "verifier_contract": {
            "k": VERIFIER_K,
            "character_clip": None,
            "items": (
                "the frozen retrieved top-5 seeds remain kb_results[:5]; each "
                "result exposes its packed PREVIOUS/SEED/NEXT retrieval_text "
                "in source order with no character clip"
            ),
        },
        "cswp_fingerprint": CSWP_FINGERPRINT,
        "seed_sha256": ab.sha256_json(seeds),
        "input_file_sha256": {name: file_hash(ROOT / name) for name in names},
        "production_retrieval_changed": False,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
        "provider_calls": 0,
        "external_network_calls": 0,
        "mmr": False,
        "reranker": False,
        "diagnostic_non_gating": ["Q25", "Q26"],
        "absence_metrics": None,
        "advance_gate": {
            "expanded_improves_gating_evidence": (
                "expanded exact-span recall > retrieved on the 33 gating queries"
            ),
            "useful_gains_survive_pack_and_downstream": (
                "focus gains that exist at expanded must be reported at packed, "
                "drafter-exposed and verifier-exposed; success is not expanded-only"
            ),
            "no_critical_retrieved_grade2_loss": (
                "no gating grade-2 span covered at retrieved may be lost at packed"
            ),
            "bounded_tokens": "every packed query uses at most 2000 cl100k tokens",
            "determinism": "two runs have byte-identical results",
        },
    }
    contract["contract_sha256"] = ab.sha256_json(contract)
    ab.write_json(CONTRACT, contract)
    ab.write_json(OUTPUT / "frozen_seeds.json", seeds)
    print(
        ab.canonical_json(
            {
                "contract_sha256": contract["contract_sha256"],
                "seed_sha256": contract["seed_sha256"],
                "pack_budget_cl100k": PACK_BUDGET_CL100K,
            }
        )
    )
    return contract


def verify_contract(contract):
    payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if ab.sha256_json(payload) != contract["contract_sha256"]:
        raise CandidateCError("frozen contract digest mismatch")
    for name, expected in contract["input_file_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise CandidateCError("frozen input drift: " + name)
    if contract["pack_budget_cl100k"] != PACK_BUDGET_CL100K:
        raise CandidateCError("pack budget drift")
    if ab.sha256_json(read(CSWP)) != contract["cswp_fingerprint"]:
        raise CandidateCError("CSWP fingerprint drift")


def prove_unit_text(units):
    titles = {
        d["source_path"]: d["title"]
        for d in read(ROOT / "reports/phase5/phase5a1/candidate_b_manifest.json")[
            "documents"
        ]
    }
    for unit in units.values():
        raw = (ROOT / unit["source_path"]).read_bytes()
        content = "".join(
            raw[s["source_byte_start"] : s["source_byte_end"]].decode("utf-8")
            for s in unit["source_spans"]
        )
        reconstructed = (
            titles[unit["source_path"]]
            + "\n"
            + " > ".join(unit["heading_path"])
            + "\n\n"
            + content
        )
        if reconstructed != unit["retrieval_text"]:
            raise CandidateCError("CSWP retrieval_text mutation")
        if file_hash(ROOT / unit["source_path"]) != unit["source_sha256"]:
            raise CandidateCError("source SHA drift")


def evaluate_once(oracle, evidence, control, units, by_source, chunks, seeds, encoding):

    a_chunks = {c["chunk_id"]: c for c in read(A_MANIFEST)["chunking"]["ordered_chunks"]}
    source_texts = {
        Path(path).stem: (ROOT / path).read_text(encoding="utf-8").strip()
        for path in {u["source_path"] for u in units.values()}
    }
    spans_by = defaultdict(list)
    for span in evidence:
        spans_by[span.query_id].append(span)
    queries = {q["query_id"]: q for q in oracle["queries"]}
    per_query = []
    for seed_row, cswp_q in zip(seeds, control["arms"]["CSWP"]["per_query"], strict=True):
        qid = seed_row["query_id"]
        query = queries[qid]
        spans = spans_by[qid]
        retrieved = [
            {
                "chunk_id": seed["chunk_id"],
                "relation": "SEED",
                "seed_rank": seed["rank"],
                "seed_chunk_id": seed["chunk_id"],
            }
            for seed in seed_row["seeds"]
        ]
        control_ids = [r["chunk_id"] for r in cswp_q["channels"]["hybrid"]["top10"][:5]]
        if [r["chunk_id"] for r in retrieved] != control_ids:
            raise CandidateCError("frozen CSWP seed ranks drifted")
        groups = expand_seeds(seed_row["seeds"], units, by_source)
        expanded_groups, dup_ids, additional = dedupe_groups(groups)
        expanded = flatten(expanded_groups)
        packed, skipped, used, exhausted, pack_stats = pack_units_seed_first(
            expanded, units, PACK_BUDGET_CL100K, encoding
        )
        assert_seed_preservation_invariant(
            expanded, units, PACK_BUDGET_CL100K, encoding
        )
        if used > PACK_BUDGET_CL100K:
            raise CandidateCError("pack exceeded frozen 2000 cl100k ceiling")
        drafter = seed_group_layer(
            query, spans, packed, units, chunks, DRAFTER_K, DRAFTER_CHAR_LIMIT
        )
        verifier = seed_group_layer(
            query, spans, packed, units, chunks, VERIFIER_K, None
        )
        layers = {
            "retrieved": layer_metrics(query, spans, retrieved, units, chunks),
            "expanded": layer_metrics(query, spans, expanded, units, chunks),
            "packed": layer_metrics(query, spans, packed, units, chunks),
            "drafter_exposed": drafter,
            "verifier_exposed": verifier,
        }
        retrieved_cov = layers["retrieved"]["covered_spans"]
        comparisons = {
            name: classify(retrieved_cov, layers[name]["covered_spans"])
            for name in ("expanded", "packed", "drafter_exposed", "verifier_exposed")
        }
        grade2 = {span.source + "::" + span.span_id for span in spans if span.grade == 2}
        lost_grade2 = sorted(
            (set(retrieved_cov) & grade2) - set(layers["packed"]["covered_spans"])
        )
        per_query.append(
            {
                "query_id": qid,
                "query": query["query"],
                "gating_eligible": query["gating_eligible"],
                "layers": layers,
                "comparisons_vs_retrieved": comparisons,
                "pack": {
                    "policy": pack_stats["policy"],
                    "used_cl100k": used,
                    "budget_cl100k": PACK_BUDGET_CL100K,
                    "utilization": round(used / PACK_BUDGET_CL100K, 8),
                    "exhausted": exhausted,
                    "skipped_units": skipped,
                    "seeds_retained": pack_stats["seeds_retained"],
                    "neighbors_retained": pack_stats["neighbors_retained"],
                    "seed_only_cl100k": pack_stats["seed_only_cl100k"],
                    "status": "PACK_BUDGET_EXHAUSTED" if exhausted else "PACK_WITHIN_BUDGET",
                },
                "duplicate_chunk_ids_removed": dup_ids,
                "additional_seed_relationships": additional,
                "overlapping_source_chars_expanded": overlap_chars(expanded, units),
                "overlapping_source_chars_packed": overlap_chars(packed, units),
                "legacy_a_top5_cl100k": a_top5_cl100k(
                    qid, control, a_chunks, source_texts, encoding
                ),
                "critical_grade2_lost_at_packed": lost_grade2,
                "provenance": [
                    {
                        "chunk_id": row["chunk_id"],
                        "originating_seed_rank": row["seed_rank"],
                        "seed_chunk_id": row["seed_chunk_id"],
                        "relation": row["relation"],
                        "source_intervals": row["source_intervals"],
                        "additional_seed_relationships": additional.get(row["chunk_id"], []),
                    }
                    for row in expanded
                ],
            }
        )
        control_metrics = cswp_q["channels"]["hybrid"]["metrics"]
        if layers["retrieved"]["evidence_span_recall"] != control_metrics[
            "evidence_span_recall@5"
        ]:
            raise CandidateCError("retrieved evidence recall failed to reproduce CSWP hybrid")
    gating = [row for row in per_query if row["gating_eligible"]]

    def mean_layer(name, field):
        return round(
            sum(row["layers"][name][field] for row in gating) / len(gating), 8
        )

    recalls = {
        name: mean_layer(name, "evidence_span_recall")
        for name in (
            "retrieved",
            "expanded",
            "packed",
            "drafter_exposed",
            "verifier_exposed",
        )
    }
    source_recalls = {
        name: mean_layer(name, "source_recall")
        for name in recalls
    }
    tokens = {
        name: token_summary(row["layers"][name]["cl100k_tokens"] for row in gating)
        for name in recalls
    }
    multipliers = [
        row["layers"]["expanded"]["cl100k_tokens"]
        / row["layers"]["retrieved"]["cl100k_tokens"]
        for row in gating
        if row["layers"]["retrieved"]["cl100k_tokens"]
    ]
    critical = [
        row["query_id"]
        for row in gating
        if row["critical_grade2_lost_at_packed"]
    ]
    focus = {qid: next(row for row in per_query if row["query_id"] == qid) for qid in FOCUS}
    packed_ok = all(row["pack"]["used_cl100k"] <= PACK_BUDGET_CL100K for row in per_query)
    useful = (
        focus["Q13"]["layers"]["packed"]["evidence_span_recall"]
        > focus["Q13"]["layers"]["retrieved"]["evidence_span_recall"]
        and focus["Q13"]["layers"]["drafter_exposed"]["evidence_span_recall"]
        > focus["Q13"]["layers"]["retrieved"]["evidence_span_recall"]
        and focus["Q13"]["layers"]["verifier_exposed"]["evidence_span_recall"]
        > focus["Q13"]["layers"]["retrieved"]["evidence_span_recall"]
    )
    advance = (
        recalls["expanded"] > recalls["retrieved"]
        and useful
        and not critical
        and packed_ok
        and recalls["packed"] >= recalls["retrieved"]
        and recalls["drafter_exposed"] >= recalls["retrieved"]
        and recalls["verifier_exposed"] >= recalls["retrieved"]
    )
    return {
        "schema_version": "phase5b1_candidate_c_v1_results",
        "name": "Candidate C v1 — Bounded Adjacent-Neighbor Expansion",
        "parent_child_retrieval": False,
        "aggregate_gating_33": {
            "evidence_span_recall": recalls,
            "source_recall": source_recalls,
            "tokens_cl100k": tokens,
            "expansion_multiplier": {
                "mean": round(statistics.fmean(multipliers), 8),
                "median": round(statistics.median(multipliers), 8),
                "min": round(min(multipliers), 8),
                "max": round(max(multipliers), 8),
            },
            "duplicate_chunk_ids_removed_mean": round(
                statistics.fmean(row["duplicate_chunk_ids_removed"] for row in gating),
                8,
            ),
            "overlapping_source_chars_expanded_mean": round(
                statistics.fmean(
                    row["overlapping_source_chars_expanded"] for row in gating
                ),
                8,
            ),
            "pack_utilization_mean": round(
                statistics.fmean(row["pack"]["utilization"] for row in gating), 8
            ),
            "pack_exhausted_queries": [
                row["query_id"] for row in gating if row["pack"]["exhausted"]
            ],
            "legacy_a_top5_cl100k": token_summary(
                row["legacy_a_top5_cl100k"] for row in gating
            ),
            "distinct_sources_packed_mean": round(
                statistics.fmean(
                    row["layers"]["packed"]["occupancy"]["distinct_sources"]
                    for row in gating
                ),
                8,
            ),
            "same_document_fraction_packed_mean": round(
                statistics.fmean(
                    row["layers"]["packed"]["occupancy"]["max_same_document_fraction"]
                    for row in gating
                ),
                8,
            ),
        },
        "focus": {
            qid: {
                "retrieved": row["layers"]["retrieved"]["evidence_span_recall"],
                "expanded": row["layers"]["expanded"]["evidence_span_recall"],
                "packed": row["layers"]["packed"]["evidence_span_recall"],
                "drafter_exposed": row["layers"]["drafter_exposed"]["evidence_span_recall"],
                "verifier_exposed": row["layers"]["verifier_exposed"][
                    "evidence_span_recall"
                ],
                "comparisons_vs_retrieved": row["comparisons_vs_retrieved"],
            }
            for qid, row in focus.items()
        },
        "quality_gate": {
            "expanded_improves_gating_evidence": recalls["expanded"]
            > recalls["retrieved"],
            "useful_q13_survives_pack_drafter_verifier": useful,
            "critical_grade2_packed_losses": critical,
            "packed_within_budget": packed_ok,
            "no_retrieved_evidence_regression_at_packed": recalls["packed"]
            >= recalls["retrieved"],
            "candidate_c_ready_to_advance": advance,
        },
        "per_query": per_query,
        "production_retrieval_changed": False,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
        "provider_calls": 0,
        "external_network_calls": 0,
        "absence_metrics": None,
    }


def run(output):
    contract = read(CONTRACT)
    verify_contract(contract)
    from scripts import phase5a0_baseline as baseline
    from scripts.retrieval_golden_v2 import load_oracle, validate_oracle

    oracle = load_oracle()
    validate_oracle(oracle)
    _, _, evidence = baseline.load_sources_and_evidence(oracle)
    control = read(CONTROL)
    manifest, units, by_source = load_units()
    prove_unit_text(units)
    chunks = {c.chunk_id: c for c in eval_chunks(manifest, oracle)}
    seeds = frozen_seeds(control)
    if ab.sha256_json(seeds) != contract["seed_sha256"]:
        raise CandidateCError("frozen seed ledger drift")
    if seeds != read(OUTPUT / "frozen_seeds.json"):
        raise CandidateCError("frozen_seeds.json drift")
    encoding = cl100k()
    result = evaluate_once(
        oracle, evidence, control, units, by_source, chunks, seeds, encoding
    )
    result["contract_sha256"] = contract["contract_sha256"]
    return result


def evaluate():
    results = []
    for index in (1, 2):
        directory = OUTPUT / f"run-{index}"
        if (directory / "results.json").exists():
            raise CandidateCError("refusing to overwrite an experiment result")
        result = run(directory)
        fingerprint = ab.sha256_json(result)
        ab.write_json(directory / "results.json", result, canonical=True)
        ab.write_json(
            directory / "runtime.json",
            {
                "result_sha256": fingerprint,
                "provider_calls": 0,
                "external_network_calls": 0,
            },
        )
        results.append((result, fingerprint))
    if results[0][1] != results[1][1] or results[0][0] != results[1][0]:
        raise CandidateCError("deterministic rerun mismatch")
    result, fingerprint = results[0]
    summary = {
        "result_sha256": fingerprint,
        "deterministic_rerun": True,
        "aggregate_gating_33": result["aggregate_gating_33"],
        "focus": result["focus"],
        "quality_gate": result["quality_gate"],
        "provider_calls": 0,
        "external_network_calls": 0,
        "production_retrieval_changed": False,
        "production_qdrant_reads": 0,
        "production_qdrant_writes": 0,
    }
    ab.write_json(OUTPUT / "evaluation_summary.json", summary)
    ab.write_json(OUTPUT / "results.json", result, canonical=True)
    print(ab.canonical_json(summary))
    return summary


def main():
    ab.install_offline_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        evaluate()


if __name__ == "__main__":
    main()
