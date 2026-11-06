# Phase 5B1 — Candidate C v1 bounded adjacent-neighbor expansion

**Evaluation complete; do not advance Candidate C v1.** Frozen CSWP hybrid/RRF top-5 ranks are unchanged. ±1 same-document neighbor expansion raises gating exact-span coverage from retrieved `0.78787879` to expanded `0.93939394`. Packed (2000 cl100k) and verifier-exposed both sit at `0.92424242`. Drafter-exposed is `0.75757576` because production still receives only the top-3 retrieved seed groups. This is not full parent-child retrieval and is not a production promotion.

## Frozen identity

Required parent: `f67b07054a8823ffbc6b3ebc23c3233f9e5e7496`. The checkpoint was pushed to `origin/feature/phase5-rag-hardening` before implementation. CSWP fingerprint `439ccdf81e2aff1bc0bf3448734cb71f774bc8d3e7619dd1e4b51b2685b79bb3`. Seeds are the archived 5A2E hybrid top-5. Pack budget is a standalone 2000 `cl100k_base` ceiling, not A's per-query count. Legacy A top-5 cl100k exposure is reported for comparison only (gating median 1856, max 2000).

## Five exposure layers (33 gating queries)

| Layer | Evidence recall | Source recall | Median cl100k |
|---|---:|---:|---:|
| retrieved | 0.78787879 | 0.98484848 | 986 |
| expanded | 0.93939394 | 0.98484848 | 1578 |
| packed | 0.92424242 | 0.98484848 | 1578 |
| drafter-exposed | 0.75757576 | 0.95454545 | 993 |
| verifier-exposed | 0.92424242 | 0.98484848 | 1578 |

Expanded coverage is not retrieval Recall@5. Median expansion multiplier is `1.693`. Packed max 1988 / p95 1985. Eleven queries hit `PACK_BUDGET_EXHAUSTED`. Mean duplicate chunk_ids removed: 4.67. Mean overlapping expanded source chars: 1188. Two runs share result SHA-256 `74b2dac5179630690ecc12741613318ba9c5ec1384569e01425d35f8fe3b08c9`.

## Focus queries

| Query | retrieved | expanded | packed | drafter | verifier |
|---|---:|---:|---:|---:|---:|
| Q09 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Q13 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Q19 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| Q21 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| Q22 | 0.5 | 1.0 | 0.5 | 0.5 | 0.5 |
| Q30 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |

Q13 is the demonstrated P0: the missing gold unit is an immediate neighbor of a frozen top-5 seed and survives pack, drafter K=3, and verifier K=5. Q09 recovers at expanded/packed/verifier but not drafter, because the completing neighbor is attached to retrieved rank 4. Q22 recovers only at expanded; the 2000-token skip-and-continue pack drops a complementary piece. Q19/Q21/Q30 remain unsolved, as preregistered.

## Decision

Do not advance. Expanded-only gains are not sufficient. Next: inspect Q22 pack-order/complementary-span loss and the drafter K=3 miss on Q09 before any parent-section experiment, reranker change, or production wiring.
