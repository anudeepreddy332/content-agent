"""Candidate C v1 bounded-neighbor expansion gates."""

from copy import deepcopy

import pytest

from scripts import phase5a2_shadow_ab as ab
from scripts import phase5b1_candidate_c as c


def synthetic_unit(chunk_id, source, start, end, text="x"):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-" + source,
        "document_version": "ver-" + source,
        "source_path": "kb/seed_docs/" + source + ".md",
        "source_sha256": "sha-" + source,
        "retrieval_char_start": start,
        "retrieval_char_end": end,
        "heading_path": [],
        "retrieval_text": "T\n\n" + text,
        "canonical_content": text,
        "source_spans": [
            {
                "source_char_start": start,
                "source_char_end": end,
                "role": "core",
            }
        ],
        "embedding_content_token_count": 1,
    }


def test_neighbor_is_only_immediate_same_document():
    a = synthetic_unit("a", "svm", 0, 10)
    b = synthetic_unit("b", "svm", 10, 20)
    d = synthetic_unit("d", "svm", 20, 30)
    other = synthetic_unit("z", "rag", 0, 10)
    by_source = {
        a["source_path"]: [a, b, d],
        other["source_path"]: [other],
    }
    units = {u["chunk_id"]: u for u in (a, b, d, other)}
    groups = c.expand_seeds(
        [{"rank": 1, "chunk_id": "b", "source": "svm", "rrf_score": 1}],
        units,
        by_source,
    )
    ids = [m["chunk_id"] for m in groups[0]["members"]]
    rels = [m["relation"] for m in groups[0]["members"]]
    assert ids == ["a", "b", "d"]
    assert rels == ["PREVIOUS", "SEED", "NEXT"]
    edge = c.expand_seeds(
        [{"rank": 1, "chunk_id": "a", "source": "svm", "rrf_score": 1}],
        units,
        by_source,
    )
    assert [m["chunk_id"] for m in edge[0]["members"]] == ["a", "b"]
    assert all(m["chunk_id"] != "z" for g in groups + edge for m in g["members"])


def test_dedup_keeps_earliest_seed_and_nonidentical_intervals():
    a = synthetic_unit("a", "svm", 0, 10, "aa")
    b = synthetic_unit("b", "svm", 8, 18, "bb")
    d = synthetic_unit("d", "svm", 18, 28, "dd")
    by_source = {a["source_path"]: [a, b, d]}
    units = {u["chunk_id"]: u for u in (a, b, d)}
    seeds = [
        {"rank": 1, "chunk_id": "b", "source": "svm", "rrf_score": 1},
        {"rank": 2, "chunk_id": "d", "source": "svm", "rrf_score": 1},
    ]
    groups = c.expand_seeds(seeds, units, by_source)
    kept, dropped, extra = c.dedupe_groups(groups)
    ids = [m["chunk_id"] for g in kept for m in g["members"]]
    assert ids == ["a", "b", "d"]
    assert dropped == 2
    assert extra
    assert c.interval_key(a) != c.interval_key(b)


def test_pack_skips_and_continues_without_splitting():
    units = {
        "big": {"retrieval_text": "word " * 400},
        "small": {"retrieval_text": "ok"},
        "mid": {"retrieval_text": "word " * 50},
    }
    rows = [
        {"chunk_id": "big", "relation": "SEED", "seed_rank": 1, "seed_chunk_id": "big"},
        {
            "chunk_id": "small",
            "relation": "NEXT",
            "seed_rank": 1,
            "seed_chunk_id": "big",
        },
        {"chunk_id": "mid", "relation": "SEED", "seed_rank": 2, "seed_chunk_id": "mid"},
    ]
    encoding = c.cl100k()
    packed, skipped, used, exhausted = c.pack_units(rows, units, 80, encoding)
    assert exhausted is True
    assert [r["chunk_id"] for r in packed] == ["small", "mid"]
    assert [r["chunk_id"] for r in skipped] == ["big"]
    assert used <= 80


def test_frozen_seeds_match_phase5a2e_hybrid_top5():
    contract = c.read(c.CONTRACT)
    c.verify_contract(contract)
    seeds = c.read(c.OUTPUT / "frozen_seeds.json")
    control = c.read(c.CONTROL)
    assert ab.sha256_json(seeds) == contract["seed_sha256"]
    for row, query in zip(seeds, control["arms"]["CSWP"]["per_query"], strict=True):
        assert row["query_id"] == query["query_id"]
        assert [s["chunk_id"] for s in row["seeds"]] == [
            item["chunk_id"] for item in query["channels"]["hybrid"]["top10"][:5]
        ]


def test_two_runs_are_byte_identical():
    first = (c.OUTPUT / "run-1/results.json").read_bytes()
    second = (c.OUTPUT / "run-2/results.json").read_bytes()
    assert first == second
    result = c.read(c.OUTPUT / "run-1/results.json")
    runtime = c.read(c.OUTPUT / "run-1/runtime.json")
    assert runtime["result_sha256"] == ab.sha256_json(result)
    assert result["provider_calls"] == result["external_network_calls"] == 0
    assert result["production_qdrant_reads"] == result["production_qdrant_writes"] == 0
    assert result["production_retrieval_changed"] is False
    assert result["parent_child_retrieval"] is False


def test_exposure_layers_and_focus_hypotheses():
    result = c.read(c.OUTPUT / "results.json")
    gating = [row for row in result["per_query"] if row["gating_eligible"]]
    assert len(gating) == 33
    agg = result["aggregate_gating_33"]["evidence_span_recall"]
    assert agg["retrieved"] == 0.78787879
    assert agg["expanded"] > agg["retrieved"]
    for row in gating:
        assert row["pack"]["used_cl100k"] <= 2000
        assert set(row["layers"]) == {
            "retrieved",
            "expanded",
            "packed",
            "drafter_exposed",
            "verifier_exposed",
        }
        for item in row["provenance"]:
            assert item["relation"] in c.RELATIONS
            assert item["originating_seed_rank"] >= 1
            assert item["source_intervals"]
    focus = result["focus"]
    assert focus["Q09"]["retrieved"] == 0.0
    assert focus["Q09"]["expanded"] == 1.0
    assert focus["Q13"]["retrieved"] == 0.0
    assert focus["Q13"]["expanded"] == focus["Q13"]["packed"] == 1.0
    assert focus["Q13"]["drafter_exposed"] == focus["Q13"]["verifier_exposed"] == 1.0
    assert focus["Q19"]["expanded"] == focus["Q19"]["retrieved"] == 0.5
    assert focus["Q21"]["expanded"] == focus["Q21"]["retrieved"] == 0.5
    assert focus["Q22"]["retrieved"] == 0.5
    assert focus["Q22"]["expanded"] == 1.0
    assert focus["Q22"]["packed"] == focus["Q22"]["drafter_exposed"] == 0.5
    assert focus["Q30"]["expanded"] == focus["Q30"]["retrieved"] == 0.5
    assert result["quality_gate"]["candidate_c_ready_to_advance"] is False
    assert result["quality_gate"]["useful_q13_survives_pack_drafter_verifier"] is True


def test_contract_drift_fails_closed():
    contract = c.read(c.CONTRACT)
    bad = deepcopy(contract)
    bad["pack_budget_cl100k"] = 1999
    with pytest.raises(c.CandidateCError, match="digest mismatch"):
        c.verify_contract(bad)
