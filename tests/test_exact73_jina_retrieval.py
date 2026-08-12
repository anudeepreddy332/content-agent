"""Deterministic tests for the exact-73 embedding-only experiment contract."""

from pathlib import Path

import pytest

from experiments.exact73_jina_retrieval import (
    BASELINE_SPEC,
    CANDIDATE_SPEC,
    EXPECTED_FIXTURE_FINGERPRINT,
    FrozenChunk,
    _rrf,
    candidate_acceptance_failures,
    hash_remote_modeling_code,
    _build_local_jina_package,
    local_feasibility_failures,
    load_exact73_fixture,
    semantic_fingerprint,
    validate_exact73_fixture,
)


def test_historical_fixture_is_exactly_the_required_73_chunk_baseline():
    root = Path(__file__).resolve().parents[1]
    chunks = load_exact73_fixture(root / "kb" / "seed_docs")
    assert len(chunks) == 73
    assert len({chunk.source for chunk in chunks}) == 20
    assert len({chunk.identity for chunk in chunks}) == 73
    assert semantic_fingerprint(chunks) == EXPECTED_FIXTURE_FINGERPRINT


def test_fixture_validation_rejects_any_payload_drift():
    chunks = [FrozenChunk("source", 0, "original")] * 73
    with pytest.raises(ValueError, match="20 sources"):
        validate_exact73_fixture(chunks)


def test_rrf_uses_historical_k_60_and_deterministically_resolves_equal_scores():
    dense = [
        {"source": "z", "chunk_index": 0, "text": "z", "distance": 0.1},
        {"source": "a", "chunk_index": 0, "text": "a", "distance": 0.2},
    ]
    bm25 = [FrozenChunk("a", 0, "a"), FrozenChunk("z", 0, "z")]
    results = _rrf(dense, bm25, depth=2)
    assert [row["source"] for row in results] == ["a", "z"]
    assert results[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 60)


def test_candidate_gate_rejects_a_source_recall_loss_without_changing_aggregate_metrics():
    baseline = {
        "per_query": [{"query": "q", "at": {"3": {"source_recall": 1.0, "concept_pass": 1.0, "ndcg": 1.0}, "5": {"source_recall": 1.0, "concept_pass": 1.0, "ndcg": 1.0}}}]
    }
    passing_metrics = {
        "@1": {"hit": 1.0},
        "@3": {"hit": 1.0, "source_recall": 1.0, "ndcg": 1.0, "concept_coverage": 1.0, "concept_pass": 1.0},
        "@5": {"hit": 1.0, "source_recall": 1.0, "ndcg": 1.0, "concept_coverage": 1.0, "concept_pass": 1.0},
        "mrr": 1.0,
    }
    candidate = {"metrics": passing_metrics, "per_query": [{"query": "q", "at": {"3": {"source_recall": 0.5, "concept_pass": 1.0, "ndcg": 1.0}, "5": {"source_recall": 1.0, "concept_pass": 1.0, "ndcg": 1.0}}}]}
    assert candidate_acceptance_failures(baseline, candidate) == ["'q': source-recall loss at @3"]


def test_model_contract_constants_hold_only_the_embedding_variable():
    assert BASELINE_SPEC.dimension == 384
    assert CANDIDATE_SPEC.dimension == 512
    assert CANDIDATE_SPEC.model_id == "jinaai/jina-embeddings-v2-small-en"
    assert CANDIDATE_SPEC.revision == "1c993a952ef47cdd9e3576c1f22f935e5252f40c"
    assert CANDIDATE_SPEC.max_sequence_length == 8192


def test_remote_modeling_code_hashes_every_shipped_python_file(tmp_path):
    (tmp_path / "modeling_jina.py").write_text("class Model: pass\n", encoding="utf-8")
    nested = tmp_path / "subpackage"
    nested.mkdir()
    (nested / "layers.py").write_text("VALUE = 1\n", encoding="utf-8")
    hashes = hash_remote_modeling_code(tmp_path)
    assert set(hashes) == {"modeling_jina.py", "subpackage/layers.py"}
    assert all(len(digest) == 64 for digest in hashes.values())


def test_local_feasibility_fails_closed_on_any_predeclared_limit():
    collection = {"peak_rss_bytes": 8 * 1024**3 + 1, "corpus_embedding_seconds": 120.1}
    failures = local_feasibility_failures(collection, 500.1)
    assert len(failures) == 3


def test_local_jina_package_rewrites_only_auto_map_to_pre_attested_local_code(tmp_path):
    model = tmp_path / "model"
    code = tmp_path / "code"
    model.mkdir()
    code.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    original_modeling = (
        "from transformers.pytorch_utils import (\n"
        "    apply_chunking_to_forward,\n"
        "    find_pruneable_heads_and_indices,\n"
        "    prune_linear_layer,\n"
        ")\n"
    )
    (code / "modeling_bert.py").write_text(original_modeling, encoding="utf-8")
    (code / "configuration_bert.py").write_text(
        "from transformers.onnx import OnnxConfig\n        super().__init__(pad_token_id=pad_token_id, **kwargs)\n",
        encoding="utf-8",
    )
    config = {"auto_map": {"AutoModel": "jinaai/jina-bert-implementation--modeling_bert.JinaBertModel"}}
    package = _build_local_jina_package(model, code, config)
    localized = __import__("json").loads((package / "config.json").read_text())
    assert localized["auto_map"]["AutoModel"] == "modeling_bert.JinaBertModel"
    modeling_text = (package / "modeling_bert.py").read_text()
    assert "find_pruneable_heads_and_indices" in modeling_text
    assert "compatibility shim" in modeling_text
    assert "Compatibility shim" in (package / "configuration_bert.py").read_text()
    config_text = (package / "configuration_bert.py").read_text()
    assert "self.is_decoder = False" in config_text
    assert "self.add_cross_attention = False" in config_text
    assert (package / "weights.bin").is_symlink()
