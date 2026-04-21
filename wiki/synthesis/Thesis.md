---
title: Thesis
type: synthesis
tags: [neurips-2026, thesis, ad-ssl]
created: 2026-04-21
updated: 2026-04-21
sources: [[RESEARCH_AGENT_ONBOARDING]]
---

# Thesis — AD-SSL for Scalable Unsupervised Graph Representation Learning

**Paper codename:** AD-SSL (Adaptive-Depth Decoupled Self-Supervised Learning)
**Target venue:** NeurIPS 2026
**Internal vault name:** SUGRL (legacy; refers to the starting-point method, not the final contribution)

## One-sentence claim

AD-SSL matches the accuracy of expensive unsupervised graph SSL methods (e.g. [[BGRL]], [[GraphMAE]]) on ogbn-arxiv (~71) at the wall-clock cost of cheap methods (e.g. [[GGD]], seconds). **No published method currently occupies this Pareto point.** See [[Pareto Gap]].

## Mechanism (how it works)

1. **Pre-compute** `X_k = Â^k X` for k ∈ {1, 2, 4, 8}. One-time sparse matmul. Inherits the decoupled precompute trick from [[SGC]]. See [[Decoupled Precompute]].
2. **Shared MLP encoder** maps each X_k → Z_k. Same weights across all depths.
3. **Bootstrap loss** (BYOL-style, online vs EMA target) aligns Z_k with Z_{k'} across depth pairs. See [[Bootstrap Loss]].
4. **Per-node adaptive depth weighting** via group-relative ranking: for each node, each depth is scored by how well it aligns with the consensus of the other depths; softmax gives weights. See [[Multi-Depth Views]] and [[Adaptive Depth Weighting]].
5. **Inference:** Z_final = weighted average of Z_k.

No augmentation, no negatives, no GNN forward pass during training. Per-epoch cost `O(N·d²)`.

## Why this is novel

- Multi-depth precomputed features as contrastive views — not done in unsupervised GRL.
- Per-node adaptive depth weighting via group-relative ranking — not done in graph SSL.
- Supervised adaptive-depth work exists ([[GPRGNN]], [[APPNP]], [[ATP]]) but not in the unsupervised regime.

See [[Novelty Verification Checklist]] (to be written before submission).

## Why we believe this will work

From [[Preliminary Validation - 168 Runs]]: six SUGRL-style sampling tweaks failed to move ogbn-arxiv (all within ±0.13 of baseline). **The only thing that moved ogbn-arxiv was changing propagation depth** (k=1 → k=3, +0.80 with 3/3 seeds positive). See [[Prepropx Depth Finding]]. Depth is the axis to exploit; AD-SSL makes depth learnable per-node.

## Four insights being ablated

Internal naming (does **not** go in paper):

| Insight | Internal name | What it does |
|---|---|---|
| A1 | GRPO-style | Per-node view weighting from cross-depth consistency |
| A2 | KTO-style | Binary quality signal (kNN match graph neighbors?), asymmetric loss |
| A3 | SimPO-style | Test simpler losses (MSE, InfoNCE) |
| A4 | Online-DPO-style | EMA-smoothed iterative refinement of depth preferences |

Baseline **B0** = uniform depth weights, bootstrap cosine loss, no refinement. See [[Ablation Plan - AD-SSL B0 A1-A4]].

## Outcome scenarios

- **Optimistic:** B0 ~70, A1 → ~71. Headline: "multi-depth views + adaptive weighting = BGRL accuracy at GGD cost."
- **Realistic:** B0 ~69.5, one insight adds +0.3–0.5. Headline: "depth is the underexploited axis; even uniform multi-depth is competitive."
- **Pessimistic:** B0 ≈ SUGRL-k=3 (~69.5), no insight helps. MLP encoder is the bottleneck → pivot.

Gates captured in [[Project Phases and Decision Gates]].

## Known risks

- **[[Less is More]]** (arxiv 2509.25742 v3, ICLR 2026 submission) — closest concurrent work. GCN + MLP as 2 complementary views with global β weighting and direct cosine loss. Differences with AD-SSL: (1) 2 views vs our K; (2) global β vs our per-node α_{i,k}; (3) direct alignment vs our bootstrap. Scale: they hit Arxiv-year (169k nodes, different task) but not standard ogbn-arxiv/products. See [[AD-SSL vs Less is More]].
- MLP-only encoder may be the ceiling, not depth diversity.
- Reviewers may collapse us to "GPRGNN + BGRL" — our defense in [[Reviewer Attacks and Defenses]].
