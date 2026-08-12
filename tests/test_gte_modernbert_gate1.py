"""Deterministic contract tests for the isolated GTE Gate 1 experiment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.nodes import _build_source_context
from experiments.gte_modernbert import (
    EXPERIMENT_MARKER,
    GTE_MODERNBERT_SPEC,
    chunk_document,
    validate_collection,
    validate_model_contract,
)
from qdrant_client.models import Distance


class CharacterTokenizer:
    """Tiny tokenizer fixture with native offsets and two special tokens."""

    name_or_path = "Alibaba-NLP/gte-modernbert-base"
    model_max_length = 8192

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        token_ids = [ord(character) for character in text]
        if add_special_tokens:
            token_ids = [-1, *token_ids, -2]
        result = {"input_ids": token_ids}
        if return_offsets_mapping:
            assert add_special_tokens is False
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_pinned_embedding_spec_is_the_single_gte_contract():
    assert GTE_MODERNBERT_SPEC.model_id == "Alibaba-NLP/gte-modernbert-base"
    assert GTE_MODERNBERT_SPEC.revision == "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    assert GTE_MODERNBERT_SPEC.max_sequence_length == 8192
    assert GTE_MODERNBERT_SPEC.output_dimension == 768
    assert (GTE_MODERNBERT_SPEC.chunk_size, GTE_MODERNBERT_SPEC.chunk_overlap) == (512, 64)


def test_tokenizer_native_chunking_preserves_contiguous_offsets_and_overlap():
    tokenizer = CharacterTokenizer()
    text = "x" * 1_000
    chunks = chunk_document(text, "source.md", tokenizer)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [(chunk.content_token_start, chunk.content_token_end) for chunk in chunks] == [
        (0, 512), (448, 960), (896, 1000)
    ]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [(0, 512), (448, 960), (896, 1000)]
    assert all(chunk.text == text[chunk.start_char:chunk.end_char] for chunk in chunks)
    assert all(chunk.model_input_tokens <= 8192 for chunk in chunks)
    assert all(any(chunk.start_char <= index < chunk.end_char for chunk in chunks) for index in range(len(text)))
    assert chunks[0].content_token_end - chunks[1].content_token_start == 64
    assert chunks[1].content_token_end - chunks[2].content_token_start == 64


def test_model_limit_accounting_includes_actual_special_tokens():
    tokenizer = CharacterTokenizer()
    chunks = chunk_document("x" * 512, "source.md", tokenizer)
    assert chunks[0].model_input_tokens == 514


def test_gte_chunks_bypass_only_the_legacy_kb_character_clip():
    intact = "A" * 3_500
    results = [
        {"source": f"gte-{index}.md", "text": intact + str(index), "experiment": EXPERIMENT_MARKER}
        for index in range(5)
    ]
    draft_context = _build_source_context([], results[:3])
    verifier_context = _build_source_context([], results[:5])

    assert draft_context.count("[KB]") == 3
    assert verifier_context.count("[KB]") == 5
    assert all(result["text"] in verifier_context for result in results)
    assert len(verifier_context) > 5 * 2_000
    legacy_context = _build_source_context([], [{"source": "legacy.md", "text": intact}])
    assert intact not in legacy_context


def test_experiment_has_no_context_reconstruction_component():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "experiments" / "context_assembly.py").exists()
    assert "context_assembly" not in (root / "experiments" / "gte_modernbert.py").read_text()


def test_model_contract_rejects_wrong_dimension_or_limit():
    tokenizer = CharacterTokenizer()
    encoder = SimpleNamespace(
        tokenizer=tokenizer,
        max_seq_length=8192,
        get_embedding_dimension=lambda: 768,
        _modules={},
    )
    assert validate_model_contract(encoder)["output_dimension"] == 768
    encoder.max_seq_length = 8_191
    with pytest.raises(ValueError, match="max sequence length"):
        validate_model_contract(encoder)


def test_collection_validation_rejects_duplicate_payload_identity():
    point_payload = {
        "source": "a.md",
        "chunk_index": 0,
        "text": "non-empty",
        "experiment": EXPERIMENT_MARKER,
        "embedding_spec": asdict(GTE_MODERNBERT_SPEC),
    }
    client = SimpleNamespace(
        get_collection=lambda _: SimpleNamespace(
            points_count=2,
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=768, distance=Distance.COSINE))),
        ),
        scroll=lambda **_: ([SimpleNamespace(payload=point_payload), SimpleNamespace(payload=point_payload)], None),
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_collection(client, [("a.md", "text")], 2, "hash")


def test_collection_validation_rejects_mismatched_embedding_payload_spec():
    payload = {
        "source": "a.md",
        "chunk_index": 0,
        "text": "non-empty",
        "experiment": EXPERIMENT_MARKER,
        "embedding_spec": {"not": "the pinned spec"},
    }
    client = SimpleNamespace(
        get_collection=lambda _: SimpleNamespace(
            points_count=1,
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=768, distance=Distance.COSINE))),
        ),
        scroll=lambda **_: ([SimpleNamespace(payload=payload)], None),
    )
    with pytest.raises(ValueError, match="embedding specification"):
        validate_collection(client, [("a.md", "text")], 1, "hash")
