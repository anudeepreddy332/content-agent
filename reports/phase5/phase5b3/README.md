# Phase 5B3 — Production Qualified RAG Integration

Live-wires the qualified Contiguous Structural Window Packing (CSWP) retrieval
path into production `retrieve_node`, `draft_node`, and verifier evidence
manifest assembly.

## Live path

CSWP manifest (local, fingerprint-gated) → hybrid dense (all-MiniLM-L6-v2) +
BM25 (raw lexical) → Reciprocal Rank Fusion (RRF, k=60) top-5 seeds → ±1
neighbor expansion → seed-first pack (≤2000 cl100k_base) →
`DRAFTER_PACKED_EVIDENCE_V1` drafter context + full verifier manifest.

Production Qdrant is not read or written on this path.

## Run

```bash
uv run python scripts/phase5b3_production_integration.py freeze
uv run python scripts/phase5b3_production_integration.py evaluate
```

## Parent

Required HEAD: `912342ac0e735af152833d21cf1c8e821532a600`
