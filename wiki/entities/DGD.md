---
title: DGD - Decoupled Graph Discrimination
type: entity
tags: [neurips-2026, baseline, graph-ssl, group-discrimination, decoupled]
created: 2026-04-24
updated: 2026-04-24
sources: []
---

# DGD — Decoupled Graph Discrimination

Neurocomputing 2024. Surfaced during 2026-04-24 literature audit. Positioning: **low pre-emption risk** but worth citing as the closest "decoupled-GCN + SSL" precedent. Different loss family (BCE group discrimination, not InfoNCE) and different use of hops (aggregate, not contrast).

**Citation placeholder (verify):** "When decoupled GCN meets group discrimination: A special graph contrastive learning framework", Neurocomputing, 2024. ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0925231224007239. OpenReview: https://openreview.net/forum?id=6Cu2sK0yfS. PDF not yet retrieved.

## One-sentence claim

Extends [[GGD]] (group discrimination, NeurIPS 2022) by replacing the GNN encoder with a one-layer MLP acting on pre-propagated (multi-hop-aggregated) features, plus a "partial corruption" negative-sampling strategy.

## Method (reconstructed from abstract)

1. **Decoupled propagation**: aggregates features across multiple hops of message passing (SGC/SIGN-style), producing a pre-propagated feature matrix.
2. **Encoder**: one-layer MLP transforms the aggregated features. No GCN.
3. **Pretext task**: group discrimination (BCE-style) — distinguish real-group embeddings from corrupted-negative-group embeddings.
4. **Negative sampling**: "partial corruption" — controls the complexity of the pretext task to extract fine structural information.

## Architectural differentiators vs. D6c

| Axis | D6c (ours) | DGD |
|---|---|---|
| Propagation timing | Precompute `Â^k X`, keep **K separate depths** | Precompute, **aggregate into single embedding** |
| Hops as views? | **Yes — hops are the contrast axis** | No — hops are aggregated, not contrasted |
| Loss | Flat cross-depth InfoNCE | **BCE group discrimination** |
| Per-depth parameters | W_k per depth | Single MLP on aggregated features |
| Negatives | Other nodes (any depth) | Corrupted-group samples |

## Pre-emption risk: LOW

DGD shares D6c's "decouple propagation from learning, use MLP not GCN" efficiency stance, but:
1. **Hops aggregated, not contrasted.** DGD collapses the multi-hop signal into one embedding before the loss. D6c's entire mechanism is keeping depths separate and using them as views.
2. **BCE, not InfoNCE.** Different contrastive objective family. DGD is a GGD descendant; D6c is closer to [[SimCLR]] / [[DINO]] in loss shape.
3. **No per-depth parameters.** DGD's single MLP cannot lift weak depths differentially the way D6c's W_k ablation shows (see [[Idea Ledger]] C3 / D6a-b fail modes).

DGD is a useful cite for "prior art on decoupled-GCN + SSL" but is not a contrastive-axis competitor.

## What we need to do

- [ ] Ingest the PDF, record benchmark numbers.
- [ ] Add row to [[Competitive Landscape 2026]] under the "decoupled / SGC-adjacent SSL" group, alongside [[GGD]] and [[SUGRL]].
- [ ] One-paragraph related-work cite: "DGD pioneered decoupled-GCN SSL with group discrimination; we take the decoupling further by using per-depth hop views contrastively rather than aggregating them."

## Related wiki pages

- [[GGD]] — parent method; group-discrimination loss origin.
- [[MHVGCL]] — the actual medium-risk pre-emption; MHVGCL has the InfoNCE-over-hops loss.
- [[Thesis]] — D6c method paper.
- [[Idea Ledger]] — D6c Live row.
- [[Competitive Landscape 2026]] — needs DGD row.
