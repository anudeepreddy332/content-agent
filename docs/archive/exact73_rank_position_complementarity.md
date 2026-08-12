# Exact-73 Rank-Position Complementarity Diagnostic

**Scope:** `DIAGNOSTIC ONLY — NOT CANDIDATE ARCHITECTURE`

No production retrieval change. No embedding reload. No RRF/BM25/threshold tuning.

## Provenance

| Field | Value |
| --- | --- |
| Evidence artifact | `outputs/exact73_channel_ablation/result.json` |
| Evidence SHA-256 | `5c5a9cf1bf6ccafef3f028b01f216e434079a89751eb2cdffa4ef7ece78e4207` |
| Hardened ablation commit | `b9f9d1d` |
| Logical base | `794851d` |
| Fixture fingerprint | `4a1d5d1d67b56867c71497cb58ed4964d356a122a14d47ef822c227dba5924e4` |
| Chunks / sources / queries | 73 / 20 / 30 |
| Depths | dense=10, BM25=10, fused=5, RRF k=60 |
| Diagnostic result SHA-256 | `6d6a73dac1c01a0101d4f85dcce67ff2b9884d3ee4b44f0b12981d59d71f76d4` |

## Historical RRF Reproduction

Independent reconstruction from stored dense/BM25 top-10 rankings matched stored MiniLM RRF and GTE RRF top-5 identities for all 30 queries.

`reproduction.ok = true`

## Common-Identity Statistics

| Statistic | Value |
| --- | ---: |
| Mean shared dense identities @ top-10 | 6.767 |
| Min / max shared | 4 / 10 |
| Queries with zero shared | 0 |

Diagnostic arms used **exact shared membership** with **original ranks preserved** (no compression).

## Replay Results

| Arm | @3 hit | @3 recall | @3 nDCG | @3 cov | @3 pass | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM-rank replay | 0.967 | **0.933** | 0.936 | 0.806 | 0.800 | 0.975 |
| GTE-rank replay | 0.967 | **0.933** | **0.938** | 0.797 | 0.800 | 0.975 |

| Depth | Metric | MiniLM-rank | GTE-rank |
| --- | --- | ---: | ---: |
| @1 | hit / recall / nDCG | 0.967 / 0.883 / 0.967 | 0.967 / 0.883 / 0.967 |
| @5 | hit / recall / nDCG | 1.000 / 0.967 / 0.953 | 1.000 / 0.967 / 0.956 |

Δ(MiniLM − GTE) @3: recall **0.000**, nDCG **−0.0027**

## Per-Query Attribution Summary

Predeclared labels:

| Label | Count (/30) |
| --- | ---: |
| PRESERVED_BY_POSITION | 1 |
| ELIMINATED | 28 |
| REVERSED | 0 |
| MEMBERSHIP_REQUIRED | 1 |

- `PRESERVED_BY_POSITION`: ridge/SVR objective comparison query
- `MEMBERSHIP_REQUIRED`: chatbot/LangGraph+RAG query (historical MiniLM fused expected identity outside shared dense set)

## Interpretation

Historical MiniLM fused advantage @3 (recall 0.950 vs GTE 0.933) **does not survive** when dense membership is equalized to the MiniLM∩GTE intersection and only original dense ranks vary.

Rank position among shared identities is **insufficient** to explain MiniLM’s historical fused advantage.

This does **not** prove rank position has zero effect; it proves position-among-shared-identities is not a sufficient explanation. Next isolated mechanism: membership / query-composition differences (non-shared dense identities), not RRF calibration.

## Classification

`RANK_POSITION_COMPLEMENTARITY_NOT_SUFFICIENT`

## Repeatability

Two consecutive diagnostic runs: `rankings_and_metrics_identical=true`
