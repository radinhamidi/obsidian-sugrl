---
title: BLNN
type: entity
kind: method
venue: arXiv 2024-08-09 (arXiv:2408.05087)
authors: [Liu, Zhang, He, Zheng, Zhao]
affiliation: Nanjing University (+ Tsinghua SIGS)
url: https://arxiv.org/abs/2408.05087
code: https://github.com/Cloudy1225/BLNN
tags: [method, baseline, bootstrap, bgrl-derivative]
created: 2026-04-21
updated: 2026-04-21
---

# BLNN — Bootstrap Latents of Nodes and Neighbors

Liu et al. (Nanjing University), arXiv Aug 2024. Direct **BGRL extension**: keeps BGRL's online/target/predictor stack and augmentation pair, and adds a **second loss term** that aligns each anchor node with its graph neighbors, weighted by an attention-computed supportiveness score. Motivated by the homophily intuition that neighbors are noisy positive pairs.

## Mechanism (§4)

Standard BGRL machinery — two augmented views `(A¹, X¹)`, `(A², X²)`, online encoder `f_θ`, target `f_ϕ` (EMA of θ), predictor `p_θ`, cosine alignment of online prediction `Z¹` with target `H²`.

**New term — Bootstrap Latents of Neighbors (Eq. 3)**:

`L_BLNN = −(1/n) Σ_i ⟨z¹_i, h²_i⟩ − (1/n) Σ_i Σ_{j∈N_i} w_j ⟨z¹_i, h²_j⟩`

(cosines normalized; `N_i` = 1-hop neighbors of i).

**Supportiveness attention (Eq. 4)**:
`w_j = softmax_j( (h¹_i · h²_j / ‖h¹_i‖‖h²_j‖) / τ )`

One-line summary: "BGRL + neighbor-positive alignment weighted by anchor-neighbor cosine similarity, softmaxed over the 1-hop neighborhood."

Complexity overhead vs BGRL: `O(|E|)` — sparse in real graphs.

## Empirical scope

**Evaluated on 5 small to medium-small graphs only**: WikiCS (11.7k), Photo (7.6k), Computer (13.7k), CS (18.3k), Physics (34.5k). **No ogbn-arxiv / ogbn-products / papers100M.**

Table 2 headline (node classification, accuracy %, 20 random 1:1:8 splits):

| | WikiCS | Photo | Computer | CS | Physics |
|---|---:|---:|---:|---:|---:|
| BGRL | 79.98 ± 0.10 | 93.17 ± 0.30 | 90.34 ± 0.19 | 93.31 ± 0.13 | 95.73 ± 0.05 |
| **BLNN** | **80.48 ± 0.52** | **93.54 ± 0.23** | **91.02 ± 0.23** | **93.61 ± 0.15** | **95.86 ± 0.10** |
| GraphMAE | 79.54 ± 0.58 | 92.98 ± 0.35 | 89.88 ± 0.10 | 93.08 ± 0.17 | 95.40 ± 0.06 |

Gains are small (+0.5, +0.37, +0.68, +0.30, +0.13) but consistent across datasets. Ablation (Table 5) shows the *supportiveness-weighted* variant beats raw-neighbor (unweighted) variant — attention is doing work.

## Relevance to AD-SSL

**Not a preempt.** BLNN extends BGRL *along a different axis* than AD-SSL:

| Axis | BLNN | AD-SSL |
|---|---|---|
| Multi-depth views? | No — 1-hop neighbors only | Yes — K precomputed `Â^k X` |
| Encoder | Full GNN (BGRL's) | MLP on precomputed features |
| Adaptivity | Attention over neighbors (spatial, 1-hop) | Per-node α_{i,k} over depths |
| Training cost | BGRL + O(|E|) attention | Strictly cheaper (no GNN pass) |
| Scale claim | 5 small graphs | ogbn-arxiv / products / (papers100M) |

The conceptual overlap — "enrich BYOL-style targets with structural information" — is real but the mechanism is 1-hop spatial, not multi-depth. BLNN is the **spatial** analog of what AD-SSL does in the **depth** dimension.

## Reviewer-defence implications

- Add to related work as "concurrent BGRL extension exploring spatial-attention positives; orthogonal to our multi-depth direction."
- If we need a headline number at their datasets (WikiCS / Photo / Computer / CS / Physics), those would be additional small-graph sanity checks, not headline benchmarks.
- BLNN inherits BGRL's augmentation requirement — AD-SSL's augmentation-free property is still a clean differentiator vs BLNN.

## Relevant links

- [[BGRL]] — direct parent method
- [[AFGRL]] (not yet ingested) — augmentation-free bootstrap; BLNN cites it as the "no augmentation" alternative
- [[GraphACL]] — uses asymmetric predictor + two-hop monophily; different spatial granularity
- [[GraphMAE]] / [[GraphMAE2]] — non-contrastive alternative in the same dataset panel
