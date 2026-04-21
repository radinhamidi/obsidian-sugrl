---
title: Decoupled Precompute
type: concept
tags: [concept, architecture]
created: 2026-04-21
updated: 2026-04-21
---

# Decoupled Precompute

Compute propagation once, train an MLP on the propagated features. No GNN forward pass during training.

## Provenance

Introduced by [[SGC]] (ICML 2019). Adopted by [[SUGRL]] (AAAI 2022) for unsupervised GRL at k=1. AD-SSL extends it to multiple k values simultaneously.

## Why it's fast

Per epoch: one MLP forward/backward pass per node. No sparse matmul, no neighbor sampling, no two-encoder augmentation machinery.

## Formal

`X_k = Â^k · X`, where Â = D̂^{-1/2}(A+I)D̂^{-1/2}. Computed once, stored, reused across every epoch.

## Trade-off

Loses the ability to learn propagation jointly with features (GCN-style) in exchange for wall-clock wins. AD-SSL recovers some of this by learning per-node **mixing weights** over pre-computed depths — see [[Multi-Depth Views]] and [[Adaptive Depth Weighting]].
