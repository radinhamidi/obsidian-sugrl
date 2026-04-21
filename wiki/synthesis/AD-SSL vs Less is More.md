---
title: AD-SSL vs Less is More — Differentiation
type: synthesis
tags: [neurips-2026, novelty, concurrent-work, high-priority]
created: 2026-04-21
updated: 2026-04-21
sources: [[Less is More]], [[AD-SSL]]
---

# AD-SSL vs "Less is More" — Differentiation

Dedicated analysis because [[Less is More]] is the closest concurrent work and will likely surface in every NeurIPS 2026 review. This page must be kept current — update on every new arxiv version of theirs.

## Side-by-side

| Axis | Less is More (GCN-MLP) | AD-SSL |
|---|---|---|
| **Views** | exactly 2: 1 MLP + 1 k-layer GCN | **K precomputed depths** (default K=4, k ∈ {1,2,4,8}) |
| **Propagation** | k-layer GCN forward **during training** | one-time sparse matmul, then **no GCN forward** |
| **View weighting** | global scalar β (tuned or 0.5) | **per-node** α_{i,k}, learned from cross-depth consistency |
| **Loss** | direct cosine alignment | **BGRL-style bootstrap** across depth pairs (EMA target) |
| **Negatives** | none | none |
| **Augmentation** | none | none |
| **Motivating principle** | feature noise vs structural noise cancellation | underexploited depth axis (Prepropx finding) |
| **OGB-scale evidence** | Arxiv-year (169k), Roman-empire (22k). No standard ogbn-arxiv/products. | target: ogbn-arxiv, ogbn-products, stretch: ogbn-papers100M |
| **Headline claim** | SOTA on heterophilic, robust to attacks | BGRL-accuracy at GGD-cost on ogbn-arxiv |

## Calibration: what they do at scale

They DO evaluate at ~169k-node scale on Arxiv-year (a task-different variant of the Lim et al. 2021 non-homophilous benchmark): 46.15 ± 0.08, training 3.96 s. That's fast. Storage is 44 GB, which is high and hints at dense-matrix intermediates. They do **not** evaluate on standard ogbn-arxiv (category prediction) or ogbn-products.

Translation for our framing: we can't say "doesn't run at OGB scale." We can say "doesn't evaluate on standard OGB benchmarks; evaluates on Arxiv-year which shares node count but not task."

## Their strongest claim vs ours

**Theirs**: Two weakly-correlated views (MLP-on-features + GCN-on-structure) are sufficient because cancellation in β·Z_s + (1−β)·Z_f suppresses independent noise. Minimal model, minimal complexity, state of the art on heterophilic.

**Ours**: More than two correlated-at-short-range-but-diverging-with-depth views provide a *richer* signal; **per-node learned mixture** over those views captures structural-scale heterogeneity that a global β cannot.

These claims are **not in conflict** — they address different axes. Their noise-decoupling theory could even apply within our framework (each Z_k vs Z_{k'} pair has its own correlation structure). We can acknowledge their theoretical contribution while staking our own.

## Required evidence gaps

To defend against "you're just adding depth on top of Less is More":

1. **Reproduce GCN-MLP at standard OGB benchmarks** (ogbn-arxiv, ogbn-products) so the comparison is apples-to-apples in our main table. Coding Agent owns this.
2. **Ablate to 2 views (k=1 MLP-ish + k=2 GCN)** of AD-SSL to show the gap between "our framework with their view count" and "our framework with multi-depth". If 2-view AD-SSL ≈ Less is More, our advantage IS the multi-depth machinery, which is defensible. If 2-view AD-SSL ≪ Less is More, we have a problem.
3. **Per-node α ablation on heterophilic benchmarks** — show that learned α helps specifically on the heterophilic nodes where they show gains.
4. **Scale study to ogbn-papers100M**. They did not. This locks in the scalability claim.

## Framing in paper

In related work:
> "Concurrent with our work, [Zhao et al. 2026] propose GCN-MLP, a two-view augmentation-free GCL method using feature-noise-vs-structural-noise decoupling. Our approach differs in three ways: (1) we use K > 2 precomputed depth views instead of a single GCN + MLP pair, avoiding per-epoch GNN forwards entirely; (2) we learn per-node mixture weights over depth views rather than a global aggregation coefficient; (3) we adopt a bootstrap-style EMA loss rather than direct cosine alignment. Empirically, AD-SSL operates at OGB scale (Table X), which [Zhao et al.] do not evaluate."

Keep this honest. "Do not evaluate" not "do not scale to."

## Watch items

- ICLR 2026 decision (accept/reject changes citation weight).
- arxiv v4+: any addition of multi-depth, per-node weighting, or OGB datasets.
- Related group: Ji et al., ICLR 2025, "Rethinking graph neural networks from a geometric perspective of node features." Same last-author (Tay). Theoretical foundation for their noise argument. Should ingest.
