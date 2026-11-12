"""CSWP-v1 lossless ledger, H1 barrier, overlap, and golden-span validation."""

from collections import defaultdict
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from scripts import phase5a0_baseline as baseline
from scripts import phase5a1_shadow as shadow
from scripts import phase5a2_shadow_ab as ab
from scripts import phase5a2e_cswp as cswp
from scripts.retrieval_golden_v2 import load_oracle, validate_oracle


@pytest.fixture
def setup():
    return shadow.MiniLMTokenizer(), json.loads(cswp.CONTRACT.read_text())


def fixture_manifest(tmp_path, text, tokenizer):
    (tmp_path / "fixture.md").write_bytes(text.encode())
    doc, blocks = shadow.parse_document(text.encode(), "fixture.md")
    children, errors = shadow.chunk_document(text.encode(), doc, blocks, tokenizer)
    shadow.validate_document(text.encode(), doc, blocks, children, errors, tokenizer)
    assert not errors
    return {"documents": [doc], "blocks": blocks, "children": children}


def gold_spans():
    oracle = load_oracle()
    validate_oracle(oracle)
    _, _, evidence = baseline.load_sources_and_evidence(oracle)
    by_query = defaultdict(list)
    for span in evidence:
        by_query[span.query_id].append(span)
    return by_query


def packed_covers(units, span, oracle):
    entry = next(row for row in oracle["corpus_manifest"] if row["source"] == span.source)
    raw_start, _ = ab._source_offsets(
        (cswp.ROOT / entry["path"]).read_text(encoding="utf-8")
    )
    span_start = span.char_start + raw_start
    span_end = span.char_end + raw_start
    intervals = []
    for unit in units:
        if unit["source_path"] != entry["path"]:
            continue
        start = max(span_start, unit["core_char_start"])
        end = min(span_end, unit["core_char_end"])
        if start < end:
            intervals.append((start, end))
    covered = span_start
    for start, end in sorted(intervals):
        if start > covered:
            return False
        covered = max(covered, end)
    return covered >= span_end


def test_paragraph_h2_paragraph_is_one_contiguous_core(tmp_path, setup):
    tokenizer, contract = setup
    text = "# Title\n\nParagraph A.\n\n## Heading\n\nParagraph B.\n"
    original = fixture_manifest(tmp_path, text, tokenizer)
    frozen = deepcopy(original)
    packed, _ = cswp.pack(original, tokenizer, contract, tmp_path)
    assert original == frozen
    body = next(
        unit for unit in packed["children"] if "Paragraph A." in unit["canonical_content"]
    )
    assert "## Heading" in body["canonical_content"]
    assert "Paragraph B." in body["canonical_content"]
    assert "# Title" not in body["canonical_content"]
    assert "Heading" not in body["heading_path"]


def test_h1_is_a_hard_barrier(tmp_path, setup):
    tokenizer, contract = setup
    text = "# Title\n\nBefore.\n\n# Other root\n\nAfter.\n"
    packed, _ = cswp.pack(
        fixture_manifest(tmp_path, text, tokenizer), tokenizer, contract, tmp_path
    )
    before = next(unit for unit in packed["children"] if "Before." in unit["canonical_content"])
    after = next(unit for unit in packed["children"] if "After." in unit["canonical_content"])
    assert before["chunk_id"] != after["chunk_id"]
    assert "After." not in before["canonical_content"]


@pytest.mark.parametrize(
    "barrier",
    [
        "```python\nx=1\n```\n",
        "| A |\n|---|\n| x |\n",
        "---\n",
        "> quoted\n",
        "    code\n",
        "<div>html</div>\n",
    ],
)
def test_protected_blocks_remain_barriers(tmp_path, setup, barrier):
    tokenizer, contract = setup
    packed, _ = cswp.pack(
        fixture_manifest(
            tmp_path, "# Title\n\nBefore.\n\n" + barrier + "\nAfter.\n", tokenizer
        ),
        tokenizer,
        contract,
        tmp_path,
    )
    before = next(unit for unit in packed["children"] if "Before." in unit["canonical_content"])
    after = next(unit for unit in packed["children"] if "After." in unit["canonical_content"])
    assert before["chunk_id"] != after["chunk_id"]


def test_whitespace_seams_are_not_omitted(tmp_path, setup):
    tokenizer, contract = setup
    text = "# Title\n\nFirst.\n\n\nSecond.\n"
    packed, _ = cswp.pack(
        fixture_manifest(tmp_path, text, tokenizer), tokenizer, contract, tmp_path
    )
    body = next(unit for unit in packed["children"] if "First." in unit["canonical_content"])
    assert "First.\n\n\nSecond." in body["canonical_content"]


def test_no_unit_exceeds_254_and_overflow_splits(tmp_path, setup):
    tokenizer, contract = setup
    text = "# Title\n\n" + " ".join(["hello"] * 400) + "\n"
    packed, report = cswp.pack(
        fixture_manifest(tmp_path, text, tokenizer), tokenizer, contract, tmp_path
    )
    assert report["children_above_254"] == 0
    assert len(packed["children"]) >= 2
    assert all(
        unit["embedding_content_token_count"] <= 254 for unit in packed["children"]
    )


def test_left_overlap_is_labelled_and_does_not_steal_core(tmp_path, setup):
    tokenizer, contract = setup
    text = (
        "# Title\n\n"
        + " ".join(["alpha"] * 180)
        + "\n\n"
        + " ".join(["beta"] * 180)
        + "\n"
    )
    packed, _ = cswp.pack(
        fixture_manifest(tmp_path, text, tokenizer), tokenizer, contract, tmp_path
    )
    overlapped = [
        unit
        for unit in packed["children"]
        if unit["retrieval_char_start"] < unit["core_char_start"]
    ]
    assert overlapped
    raw = (tmp_path / "fixture.md").read_text()
    for unit in overlapped:
        assert any(segment["role"] == "left_overlap" for segment in unit["structural_segments"])
        assert unit["canonical_content"] == raw[unit["core_char_start"] : unit["core_char_end"]]


def test_q07_q14_q23_are_coverable_and_q09_has_no_seam_gaps(setup):
    tokenizer, contract = setup
    packed, report = cswp.pack(shadow.build_corpus(tokenizer)[0], tokenizer, contract)
    assert report["source_coverage_gaps"] == 0
    assert report["children_above_254"] == 0
    oracle = load_oracle()
    spans = gold_spans()
    for query_id in ("Q07", "Q09", "Q14", "Q23"):
        for span in spans[query_id]:
            assert packed_covers(packed["children"], span, oracle)


def test_frozen_build_is_deterministic(setup):
    tokenizer, contract = setup
    b, _ = shadow.build_corpus(tokenizer)
    first = cswp.pack(b, tokenizer, contract)
    second = cswp.pack(b, tokenizer, contract)
    assert first == second
    assert first[1]["provenance_failures"] == 0


def test_offline_socket_attempt_is_fatal():
    code = (
        "from scripts.phase5a2e_cswp import ab; ab.install_offline_guard(); "
        "import os,socket; assert os.environ['LANGSMITH_TRACING']=='false'; "
        "socket.create_connection(('example.com',443))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=cswp.ROOT
    )
    assert result.returncode != 0
    assert "OFFLINE_NETWORK_FORBIDDEN" in result.stderr


def test_artifacts_bind_frozen_contract_and_unchanged_controls():
    initial = json.loads((cswp.OUTPUT / "experiment_manifest.json").read_text())
    effective = cswp.freeze(cswp.OUTPUT)
    assert initial == effective
    assert initial["required_parent"] == cswp.STARTING_HEAD
    assert initial["packing_contract"] == json.loads(cswp.CONTRACT.read_text())
    assert initial["production_retrieval_changed"] is False
    assert initial["qdrant_mutated"] is False
    assert initial["semantic_chunking"] is False
    assert initial["provider_calls"] == initial["external_network_calls"] == 0
    declared = initial["manifest_sha256"]
    observed = {key: value for key, value in initial.items() if key != "manifest_sha256"}
    assert ab.sha256_json(observed) == declared


def test_two_full_reruns_include_identical_embeddings_ranks_and_packing():
    first = json.loads((cswp.OUTPUT / "run-1/determinism_report.json").read_text())
    second = json.loads((cswp.OUTPUT / "run-2/determinism_report.json").read_text())
    assert first == second
    result = json.loads((cswp.OUTPUT / "results.json").read_text())
    assert ab.sha256_json(result) == first["result_sha256"]
    manifest = json.loads((cswp.OUTPUT / "candidate_cswp_manifest.json").read_text())
    assert ab.sha256_json(manifest) == first["cswp_fingerprint"]
    assert (
        manifest["frozen_B_fingerprint"]
        == "a9402fea0c4fbbae6613346370c0759b0239d3d870a386526980e1c40815e852"
    )


def test_complete_four_arm_metrics_and_every_query_classification():
    result = json.loads((cswp.OUTPUT / "results.json").read_text())
    assert result["absence_metrics"] is None
    assert (
        len([row for row in result["per_query_comparison"] if row["gating_eligible"]])
        == 33
    )
    for arm in ("A", "B", "B-PACKED", "CSWP"):
        for channel in ("dense", "bm25", "hybrid"):
            metrics = result["arms"][arm]["aggregate_metrics"][channel][
                "gating_33_excludes_q25_q26"
            ]
            for k in ab.K_VALUES:
                for prefix in ("recall", "precision", "ndcg", "evidence_span_recall"):
                    assert f"{prefix}@{k}" in metrics
                assert "unique_sources" in result["diversity"][arm][channel][str(k)]
            assert "mrr@10" in metrics
    for row in result["per_query_comparison"]:
        for channel in row["channels"].values():
            for control in ("A", "B", "B-PACKED"):
                assert channel["vs_" + control]["classification"] == ab._classification(
                    channel["metrics"][control], channel["metrics"]["CSWP"]
                )
    hybrid = {
        name: result["arms"][name]["aggregate_metrics"]["hybrid"][
            "gating_33_excludes_q25_q26"
        ]
        for name in ("A", "B", "B-PACKED", "CSWP")
    }
    assert hybrid["A"]["evidence_span_recall@5"] == 0.84848485
    assert hybrid["B"]["evidence_span_recall@5"] == 0.51515152
    assert hybrid["B-PACKED"]["evidence_span_recall@5"] == 0.5
    assert hybrid["CSWP"]["evidence_span_recall@5"] == 0.78787879
    assert hybrid["A"]["mrr@10"] == 0.97979798
    assert hybrid["CSWP"]["mrr@10"] == 0.95707071
    assert hybrid["A"]["ndcg@5"] == 0.97094024
    assert hybrid["CSWP"]["ndcg@5"] == 0.95478073
    gating = result["gating_queries"]["A"]
    assert gating["IMPROVED"] == ["Q11", "Q15", "Q30", "Q31"]
    assert gating["REGRESSED"] == ["Q09", "Q13", "Q16", "Q17", "Q22", "Q34"]
    assert result["CSWP_ready_to_advance"] is False
