"""Deterministic coverage for child-window retrieval context assembly."""
import json
from types import SimpleNamespace

import agent.nodes as nodes
from main import _write_telemetry
from tools.context_assembly import assemble_evidence_windows, context_budget_stats
from tools.save_to_kb import _chunk_text


def _children(source: str, count: int) -> list[dict]:
    return [
        {"source": source, "chunk_index": index, "text": f"{source}-child-{index}: evidence."}
        for index in range(count)
    ]


def _seed(source: str, index: int, score: float = 0.03) -> dict:
    return {
        "source": source,
        "chunk_index": index,
        "text": f"{source}-child-{index}: evidence.",
        "distance": 0.2,
        "rrf_score": score,
    }


class _CharacterTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2

    def __call__(self, text: str, **kwargs) -> dict:
        assert kwargs["add_special_tokens"] is False
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["truncation"] is False
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_minilm_safe_child_chunk_invariant():
    encoder = SimpleNamespace(tokenizer=_CharacterTokenizer(), max_seq_length=256)
    chunks = _chunk_text("x" * 600, encoder)
    assert [len(chunk) for chunk in chunks] == [224, 224, 216]
    assert all(len(chunk) + 2 <= encoder.max_seq_length for chunk in chunks)


def test_child_order_and_middle_neighbor_expansion_are_deterministic():
    source_children = {"a": list(reversed(_children("a", 5)))}
    window = assemble_evidence_windows([_seed("a", 2)], source_children)[0]
    assert window["chunk_indices"] == [1, 2, 3]
    assert window["seed_chunk_indices"] == [2]
    assert window["best_child_rank"] == 1


def test_first_and_last_chunk_expansion_stay_within_document_bounds():
    source_children = {"a": _children("a", 4)}
    first = assemble_evidence_windows([_seed("a", 0)], source_children)[0]
    last = assemble_evidence_windows([_seed("a", 3)], source_children)[0]
    assert first["chunk_indices"] == [0, 1]
    assert last["chunk_indices"] == [2, 3]


def test_overlapping_seed_windows_merge_and_consolidate_siblings():
    source_children = {"a": _children("a", 6)}
    windows = assemble_evidence_windows([_seed("a", 2), _seed("a", 3)], source_children)
    assert len(windows) == 1
    assert windows[0]["chunk_indices"] == [1, 2, 3, 4]
    assert windows[0]["seed_chunk_indices"] == [2, 3]
    assert windows[0]["seed_ranks"] == [1, 2]


def test_overlap_text_is_not_repeated_in_expanded_window():
    overlap = " shared overlap passage that is long enough "
    source_children = {
        "a": [
            {"source": "a", "chunk_index": 0, "text": "first" + overlap},
            {"source": "a", "chunk_index": 1, "text": overlap + "second"},
        ]
    }
    window = assemble_evidence_windows([_seed("a", 0)], source_children)[0]
    assert window["text"].count(overlap.strip()) == 1
    assert window["duplicate_chars_removed"] >= len(overlap)


def test_no_cross_source_expansion_and_stable_window_ranking():
    source_children = {"a": _children("a", 3), "b": _children("b", 3)}
    windows = assemble_evidence_windows([
        _seed("b", 1, score=0.02),
        _seed("a", 1, score=0.09),
    ], source_children)
    assert [window["source"] for window in windows] == ["b", "a"]
    assert "a-child" not in windows[0]["text"]
    assert "b-child" not in windows[1]["text"]


def test_context_budget_cap_is_deterministic():
    source_children = {"a": [{"source": "a", "chunk_index": 0, "text": "x" * 100}]}
    windows = assemble_evidence_windows(
        [_seed("a", 0)],
        source_children,
        max_window_chars=24,
    )
    assert windows[0]["context_chars"] == 100
    assert windows[0]["seed_budget_exceeded"] is True
    assert windows[0]["seed_budget_overflow_chars"] == 76
    stats = context_budget_stats(windows, n_windows=1)
    assert stats["evidence_windows"] == 1
    assert stats["total_context_chars"] == 100
    assert stats["estimated_context_tokens"] == 25
    assert stats["max_window_chars"] == 24


def test_middle_seed_survives_cap_before_outer_neighbors():
    source_children = {
        "a": [
            {"source": "a", "chunk_index": 0, "text": "L" * 120},
            {"source": "a", "chunk_index": 1, "text": "MIDDLE-SEED-EVIDENCE"},
            {"source": "a", "chunk_index": 2, "text": "R" * 120},
        ]
    }
    window = assemble_evidence_windows([_seed("a", 1)], source_children, max_window_chars=100)[0]
    assert "MIDDLE-SEED-EVIDENCE" in window["text"]
    assert window["chunk_indices"] == [1]
    assert window["outer_neighbor_chars_omitted"] > 0
    assert window["context_chars"] <= 100


def test_first_and_last_seed_survive_cap_pressure():
    source_children = {
        "a": [
            {"source": "a", "chunk_index": 0, "text": "FIRST-SEED"},
            {"source": "a", "chunk_index": 1, "text": "x" * 120},
            {"source": "a", "chunk_index": 2, "text": "LAST-SEED"},
        ]
    }
    first = assemble_evidence_windows([_seed("a", 0)], source_children, max_window_chars=40)[0]
    last = assemble_evidence_windows([_seed("a", 2)], source_children, max_window_chars=40)[0]
    assert "FIRST-SEED" in first["text"]
    assert "LAST-SEED" in last["text"]


def test_multiple_seed_chunks_are_never_silently_cut_when_they_exceed_budget():
    source_children = {
        "a": [
            {"source": "a", "chunk_index": 0, "text": "FIRST-SEED" * 10},
            {"source": "a", "chunk_index": 1, "text": "SECOND-SEED" * 10},
            {"source": "a", "chunk_index": 2, "text": "outer-neighbour"},
        ]
    }
    window = assemble_evidence_windows(
        [_seed("a", 0), _seed("a", 1)], source_children, max_window_chars=80,
    )[0]
    assert "FIRST-SEED" in window["text"]
    assert "SECOND-SEED" in window["text"]
    assert window["seed_budget_exceeded"] is True
    assert window["seed_budget_overflow_chars"] > 0


def test_react_style_seed_region_keeps_late_required_evidence_under_cap_pressure():
    source_children = {
        "react": [
            {"source": "react", "chunk_index": 0, "text": "intro " * 300},
            {"source": "react", "chunk_index": 1, "text": "hallucination grounding"},
            {"source": "react", "chunk_index": 2, "text": "filler " * 300},
            {"source": "react", "chunk_index": 3, "text": "fabricated observations action parsing"},
        ]
    }
    window = assemble_evidence_windows(
        [_seed("react", 1), _seed("react", 3)], source_children, max_window_chars=120,
    )[0]
    for concept in ("hallucination", "grounding", "fabricated observations", "action parsing"):
        assert concept in window["text"]
    assert window["seed_budget_exceeded"] is False


def test_retrieve_node_uses_ten_children_and_preserves_three_five_consumer_budgets(
    base_state,
    monkeypatch,
):
    queried_depths = []
    children = [_seed("a", index) for index in range(10)]
    windows = [
        {**_seed("a", index), "context_chars": 100, "duplicate_chars_removed": 0, "truncated_chars": 0}
        for index in range(5)
    ]
    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query, max_results=5, force_refresh=False: [
            {"title": query, "url": f"https://example.com/{query}", "content": "evidence", "score": 0.9}
        ],
    )
    monkeypatch.setattr(
        nodes,
        "query_kb",
        lambda query, n_results=5: queried_depths.append(n_results) or children,
    )
    monkeypatch.setattr(nodes, "assemble_child_context", lambda child_hits, n_windows: windows)

    result = nodes.retrieve_node(base_state)

    assert queried_depths == [10]
    assert result["kb_results"] == windows
    assert result["kb_context_stats"]["draft"]["evidence_windows"] == 3
    assert result["kb_context_stats"]["verifier"]["evidence_windows"] == 5


def test_telemetry_keeps_child_depth_window_identity_and_context_budgets(base_state, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    state = {
        **base_state,
        "kb_results": [{
            **_seed("a", 2),
            "chunk_indices": [1, 2, 3],
        }],
        "kb_context_stats": {
            "candidate_children": 10,
            "candidate_unique_sources": 4,
            "draft": {"evidence_windows": 3, "total_context_chars": 7000},
            "verifier": {"evidence_windows": 5, "total_context_chars": 11000},
        },
    }
    path = _write_telemetry(state)
    record = json.loads(path.read_text())
    assert record["kb_context_stats"]["candidate_children"] == 10
    assert record["kb_results"][0]["chunk_indices"] == [1, 2, 3]
