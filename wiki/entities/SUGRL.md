---
title: SUGRL
type: entity
kind: method
venue: AAAI 2022
authors: Mo, Peng, Xu, Shi, Zhu
url: https://ojs.aaai.org/index.php/AAAI/article/view/20748
code: https://github.com/YujieMo/SUGRL
tags: [method, baseline, starting-point, ingested]
created: 2026-04-21
updated: 2026-04-21
sources: [[VALIDATION_ORIGINAL_CODE]]
ingested_from: raw/papers/SUGRL.pdf
---

# SUGRL — Simple Unsupervised Graph Representation Learning

Mo et al., AAAI 2022. Our **starting-point method**. AD-SSL begins as a modification of SUGRL's decoupled-precompute + shared-weight MLP/GCN architecture.

## Architecture (from paper §Method)

Three embedding types:

- **Anchor `H`** — MLP on raw features X (semantic info). Two-layer FC with dropout + ReLU.
- **Structural positive `H+`** — GCN on adjacency Â = D̂^{-1/2}(A+I)D̂^{-1/2}. Uses Â^l X form.
- **Neighbor positive `H̃+`** — average of m=5 sampled 1-hop neighbor embeddings.
- **Negative `H−`** — row-shuffle of anchor H (not a GCN forward on corrupted graph — the key efficiency trick).

**Shared weights between MLP and GCN encoders** (paper §"Positive Embedding Generation", footnote on eq. 4). AD-SSL inherits this — same encoder for all depths — but extends it to multi-depth precomputed features.

## Multiplet loss (eq. 12)

`L = ω1·L_S + ω2·L_N + L_U`

- **Triplet on structural**: `L_S = (1/k) Σᵢ max(0, d(h,h+)² − d(h,h⁻ᵢ)² + α)`
- **Triplet on neighbor**: `L_N = (1/k) Σⱼ max(0, d(h,h̃+)² − d(h,h⁻ⱼ)² + α)`
- **Upper-bound regularizer**: `L_U = −(1/k) Σᵢ min(0, d(h,h+)² − d(h,h⁻ᵢ)² + α + β)`

Hyperparameters: margins α (safe distance) and β (upper-bound slack); weights ω1, ω2. L_U **only** pushed on structural positives, not neighbor — explicit design choice in paper.

## Key properties

- No augmentation, no discriminator.
- Decoupled precompute inheritance from [[SGC]].
- Fast: trains small graphs in seconds; ogbn-arxiv ~6s full-batch (per §Efficiency Analysis).
- Low-dim embeddings (128-d) sufficient for best accuracy — many competitors need 512.
- Linear evaluation from DGI (2-layer LogReg on frozen h_p).

## Key weakness

Fixed k=1 propagation depth. Paper does **not** ablate depth. See [[Prepropx Depth Finding]] — moving to k=3 gives +0.80 on ogbn-arxiv for free.

## ogbn-arxiv numbers (from paper Table 2)

| Variant | Accuracy | Time |
|---|---:|---:|
| SUGRL (full-batch, k=1) | **68.8 ± 0.4** | 0.1 min |
| **SUGRL-batch (k=1)** | **69.3 ± 0.2** | 0.2 min |
| Our repro (full-batch, k=1) | 68.77 ± 0.13 | — |
| Same method with k=3 ([[Prepropx Depth Finding]]) | 69.57 ± 0.05 | — |

**Important**: SUGRL-batch (mini-batch variant in same paper) already reaches 69.3 on arxiv. Our "baseline correction" comparison should cite either 68.8 (full-batch, what most citations use) or 69.3 (batch, which the authors themselves report as higher). Flag this when writing related work — don't cherry-pick the lower number.

## Negative sampling detail

SUGRL's `np.random.permutation` produces a **derangement**, not i.i.d. sampling. On [[PubMed]] this hurts by ~1.37 — see [[Preliminary Validation - 168 Runs]] §4.4. One-line fix; appendix material.

## Role in AD-SSL

- Architectural ancestor: decoupled precompute + shared-weight MLP/GCN.
- Baseline in main results table.
- Target to beat: must clear **SUGRL-k=3 (~69.57)** or SUGRL-batch (69.3) on ogbn-arxiv, not just SUGRL-k=1 (68.8).

## What the paper does NOT claim

- No depth analysis. No multi-depth views. No per-node weighting. No bootstrap/BYOL machinery. All four of these are AD-SSL's contribution regions.
