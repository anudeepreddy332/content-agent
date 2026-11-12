# Phase 5B2C — Seed-First Packing + Combined Exposure Requalification

Integrates **SEED_FIRST** packing into Candidate C (bounded ±1 neighbor expansion
after frozen Contiguous Structural Window Packing retrieval) and requalifies the
**DRAFTER_PACKED_EVIDENCE_V1** shadow contract on the new packed evidence identity.

## Pack policy

1. **Pass 1:** unique frozen top-5 SEED units in retrieval-rank order.
2. **Pass 2:** remaining 2000 cl100k budget for deduplicated ±1 neighbors
   (seed rank → source order → chunk_id).

Invariant: supplementary neighbors never evict an original retrieved seed when the
seed set fits the global budget.

## Run

```bash
uv run python scripts/phase5b2c_seed_first_pack.py freeze
uv run python scripts/phase5b2c_seed_first_pack.py evaluate
```

## Parent

Required HEAD: `41261a1a12ffba8a8051debc84b27055985c1b4f`

Production `draft_node` is **not** modified.
