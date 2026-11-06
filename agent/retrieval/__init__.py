"""Production-owned CSWP hybrid retrieval primitives (dense + BM25 + RRF + expansion)."""

from agent.retrieval.chunks import EvalChunk, normalize_interval, source_offsets
from agent.retrieval.encoder import (
    EMBEDDING_DIMENSION,
    ENCODER_MODEL,
    MODEL_REVISION,
    get_encoder,
    resolve_local_model_snapshot,
)
from agent.retrieval.expansion import (
    PACK_BUDGET_CL100K,
    assert_seed_preservation_invariant,
    cl100k,
    cl100k_count,
    dedupe_groups,
    expand_seeds,
    flatten,
    pack_units_seed_first,
)
from agent.retrieval.fusion import (
    CANDIDATE_K,
    RRF_CONSTANT,
    rank_bm25,
    rank_dense,
    reciprocal_rank_fusion,
)

__all__ = [
    "CANDIDATE_K",
    "EMBEDDING_DIMENSION",
    "ENCODER_MODEL",
    "EvalChunk",
    "MODEL_REVISION",
    "PACK_BUDGET_CL100K",
    "RRF_CONSTANT",
    "assert_seed_preservation_invariant",
    "cl100k",
    "cl100k_count",
    "dedupe_groups",
    "expand_seeds",
    "flatten",
    "get_encoder",
    "normalize_interval",
    "pack_units_seed_first",
    "rank_bm25",
    "rank_dense",
    "reciprocal_rank_fusion",
    "resolve_local_model_snapshot",
    "source_offsets",
]
