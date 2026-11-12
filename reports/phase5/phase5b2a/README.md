# Phase 5B2A — DRAFTER_PACKED_EVIDENCE_V1 Shadow Contract

Offline shadow evaluation of a production-compatible drafter KB exposure contract
that consumes Candidate-C PACKED evidence without the legacy top-3 gate or secondary
2000-character clipping.

## Arms

| Arm | Contract | KB groups | Clip |
|-----|----------|-----------|------|
| Legacy control | `LEGACY_DRAFTER_KB_V1` | top-3 seed ranks | 2000 chars / seed group |
| Shadow | `DRAFTER_PACKED_EVIDENCE_V1` | all packed groups | none (upstream 2000 cl100k pack only) |

Production `draft_node` is **not** modified.

## Run

```bash
uv run python scripts/phase5b2a_drafter_pack_contract.py freeze
uv run python scripts/phase5b2a_drafter_pack_contract.py evaluate
```

## Parent

Required HEAD: `fe6d35970e9c2b4ed5e494a2cea07a0287fc3f4b`

Upstream: Phase 5B1 Candidate-C contract (`evals/fixtures/phase5b1_candidate_c_contract.json`).
