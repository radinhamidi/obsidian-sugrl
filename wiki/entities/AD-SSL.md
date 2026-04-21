---
title: AD-SSL
type: entity
kind: method-ours
tags: [neurips-2026, ad-ssl, method]
created: 2026-04-21
updated: 2026-04-21
---

# AD-SSL — Adaptive-Depth Decoupled Self-Supervised Learning

Our proposed method for NeurIPS 2026. See [[Thesis]] for the full pitch.

## One line

Multi-depth precomputed node features as contrastive views + per-node adaptive depth weighting + BYOL-style bootstrap loss, trained by a shared MLP encoder — matching [[BGRL]] accuracy at [[GGD]] cost.

## Architecture

1. Pre-compute `X_k = Â^k X` for k ∈ {1, 2, 4, 8}. See [[Decoupled Precompute]].
2. Shared MLP encoder: X_k → Z_k.
3. [[Bootstrap Loss]] aligns online Z_k with EMA-target Z_{k'} across depth pairs.
4. Per-node group-relative weighting (see [[Adaptive Depth Weighting]], [[Multi-Depth Views]]).
5. Inference: Z_final = Σ_k α_k · Z_k.

No augmentation. No negatives. No GNN forward pass during training.

## Components

- **Encoder**: MLP (default width/depth mirrors SUGRL [512, 128]).
- **Target**: EMA copy of encoder.
- **Weighting head**: lightweight MLP or parameter-free cross-depth consistency score (ablated in [[Ablation Plan - AD-SSL B0 A1-A4]]).

## Per-epoch cost

`O(N · d²)` — dominated by MLP forward/backward over all N nodes at all K depths. Propagation is a one-time preprocessing cost (~1 second on ogbn-arxiv).

## Theoretical grounding (one-liner pointers)

- Aggregation-reduces-variance and simplex-contraction geometry: [[Rethinking graph neural networks from a geometric perspective of node features]] (Ji et al., ICLR 2025) — used in [[Multi-Depth Views]] and [[Oversmoothing]].
- Spectral low-pass interpretation of `Â^k`: SGC lineage (see [[Decoupled Precompute]]).

Full motivation lives in [[Thesis]].

## Status

Under ablation by Coding Agent. See [[Project Phases and Decision Gates]].

## Competitors

- [[SUGRL]] — our starting point (single fixed depth).
- [[GGD]] — efficiency champion to match.
- [[BGRL]] / [[GraphMAE]] / [[GraphMAE2]] — accuracy ceiling to approach.
- [[Less is More]] — closest concurrent architecture.

## Open design questions

- Is uniform weighting (B0) enough, or does A1 (per-node) matter?
- Is bootstrap necessary (A3 simpler-loss ablation)?
- Is EMA refinement (A4) worth the extra complexity?
- Depth set sensitivity — does {1,2,4,8} dominate denser sets like {1,2,3,4,5}?
