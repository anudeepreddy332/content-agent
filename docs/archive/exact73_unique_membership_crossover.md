# Exact-73 Unique Dense-Membership Crossover Diagnostic

**Scope:** `DIAGNOSTIC ONLY — NOT CANDIDATE ARCHITECTURE`  
**Cutover:** `DIAGNOSTIC ONLY — NO ARCHITECTURE CUTOVER AUTHORIZED`

No production retrieval change. No embedding reload. No RRF/BM25/threshold tuning.

## Provenance / Reproduction

| Field | Value |
| --- | --- |
| Evidence artifact | `outputs/exact73_channel_ablation/result.json` |
| Evidence SHA-256 | `5c5a9cf1bf6ccafef3f028b01f216e434079a89751eb2cdffa4ef7ece78e4207` |
| Rank-position parent commit | `4f6637e` |
| Fixture fingerprint | `4a1d5d1d67b56867c71497cb58ed4964d356a122a14d47ef822c227dba5924e4` |
| Chunks / sources / queries | 73 / 20 / 30 |
| Depths | dense=10, BM25=10, fused=5, RRF k=60 |
| Historical RRF reconstruction | **exact match** (30/30 MiniLM + GTE) |
| Shared-only reference | stored rank-position result SHA `6d6a73dac1c01a0101d4f85dcce67ff2b9884d3ee4b44f0b12981d59d71f76d4` |
| Diagnostic result SHA-256 | `3cabc9ba558e76161d0213cec382a37932c32c7e524d9384f46e8bd0c13c93d2` |

## Unique-Set Statistics

| Statistic | Value |
| --- | ---: |
| Mean unique identities / query | 3.233 |
| Min / max unique | 0 / 6 |
| Zero-unique queries | 1 (treated as exactly one empty permutation) |

## Four Crossover Aggregates

| Combination | @3 hit | @3 recall | @3 nDCG | @3 cov | @3 pass | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM anchor + MiniLM membership | 1.000 | **0.950** | **0.948** | 0.822 | 0.833 | 0.983 |
| MiniLM anchor + GTE membership | 0.967 | 0.933 | 0.933 | 0.806 | 0.800 | 0.972 |
| GTE anchor + MiniLM membership | 1.000 | **0.950** | **0.951** | 0.814 | 0.833 | 0.983 |
| GTE anchor + GTE membership | 0.967 | 0.933 | 0.927 | 0.797 | 0.800 | 0.961 |

## MiniLM − GTE Membership Deltas

| Anchor | Δ recall@3 | Δ nDCG@3 |
| --- | ---: | ---: |
| MiniLM shared-rank skeleton | +0.016667 | +0.014945 |
| GTE shared-rank skeleton | +0.016667 | +0.023610 |
| Cross-anchor average | **+0.016667** | **+0.019278** |

## Effect Size vs Historical Gap

| Quantity | Value |
| --- | ---: |
| Historical Δ recall@3 (MiniLM RRF − GTE RRF) | +0.016667 |
| Historical Δ nDCG@3 | +0.022521 |
| `membership_explained_fraction_recall` | **1.000** |
| `membership_explained_fraction_ndcg` | **0.856** |

Interpretation boundary: `UNIQUE_MEMBERSHIP_SUPPORTED` means unique membership has causally supported influence under the predeclared crossover. The observed cross-anchor Δrecall equals the full historical fused recall gap, and ΔnDCG explains ~86% of the historical nDCG gap — large enough to plausibly explain a meaningful portion of the historical fused advantage. This still does **not** authorize architecture cutover by itself.

## Shared-Only Comparisons (Harm vs Less Useful)

Relative to shared-only replay on the same anchor:

| Anchor | MiniLM uniques Δrecall / ΔnDCG | GTE uniques Δrecall / ΔnDCG |
| --- | --- | --- |
| MiniLM | **+0.0167 / +0.0129** | 0.000 / −0.0021 |
| GTE | **+0.0167 / +0.0129** | 0.000 / −0.0107 |

- MiniLM-unique identities **add useful signal** vs shared-only.
- GTE-unique identities do **not** improve recall vs shared-only and slightly reduce nDCG.
- GTE uniques are therefore **less beneficial** than MiniLM uniques; they are weakly nDCG-degrading, not strongly recall-harmful.

## Per-Query Attribution Summary

| Label | MiniLM anchor | GTE anchor |
| --- | ---: | ---: |
| MINILM_MEMBERSHIP_WIN | 2 | 2 |
| GTE_MEMBERSHIP_WIN | 0 | 0 |
| TIE | 28 | 28 |

- Dual-anchor MiniLM wins: **2 / 30**
- Winner stable across anchors: **30 / 30**
- Concentration: advantage is query-localized (2 queries), not broadly distributed.
- Dual-win queries include one with MiniLM-only expected source `langgraph-state-machines`, and one membership-sensitive hard query without a MiniLM-only expected-source label in the unique set.

## Source Duplication / Concentration

Across crossover arms at @3, MiniLM membership yields slightly higher `unique_sources` and fewer `duplicate_slots` than GTE membership under both anchors (e.g. MiniLM-anchor MiniLM membership unique_sources@3 ≈ 1.549 vs GTE membership ≈ 1.507). No evidence that MiniLM wins via pathological source concentration.

## Final Classification

**`UNIQUE_MEMBERSHIP_SUPPORTED`**

All six predeclared SUPPORTED gates passed:
1–4. MiniLM membership strictly exceeds GTE membership on recall@3 and nDCG@3 under both anchors  
5. ≥2 dual-anchor MiniLM wins  
6. Attribution coherent with MiniLM-only expected-source/retrieval value on dual-win queries  

**DIAGNOSTIC ONLY — NO ARCHITECTURE CUTOVER AUTHORIZED.**

## Repeatability

Two consecutive runs: `rankings_and_metrics_identical=true`
