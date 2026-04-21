---
title: Prepropx Depth Finding
type: experiment
tags: [neurips-2026, preliminary, depth, motivation]
created: 2026-04-21
updated: 2026-04-21
sources: [[VALIDATION_ORIGINAL_CODE]]
---

# Prepropx Depth Finding

The single finding from [[Preliminary Validation - 168 Runs]] that motivates the entire AD-SSL paper.

## What it is

[[SUGRL]]'s published `train_OGB.py` applies **one** sparse matmul `(A+I)_norm @ X` before training (k=1 hops) then runs an MLP on the smoothed features. This decoupled precompute pattern is inherited from [[SGC]] (ICML 2019). SUGRL's paper does **not** ablate the depth axis — k=1 is hardcoded.

We ran the same unmodified SUGRL method with k=3 and k=4 total hops. Nothing else changed: same model, same optimizer, same loss, same hyperparameters. Training-time cost is identical (propagation happens once at preprocessing, ~1 extra second).

## Numbers (ogbn-arxiv, 3 seeds)

| Variant | Accuracy (mean ± std) | Δ vs SUGRL-k=1 | Seeds positive |
|---|---:|---:|---:|
| SUGRL baseline (k=1, published) | 68.77 ± 0.13 | — | ref |
| `prepropx2` (k=3) | **69.57 ± 0.05** | **+0.80 ± 0.08** | **3/3 ✅** |
| `prepropx3` (k=4) | 69.53 ± 0.09 | +0.77 ± 0.21 | 3/3 ✅ |

A depth sweep k ∈ {1..6} (reported in the source as `EXPLORATION_REPORT.md`) shows a clean U-curve peaking at k=3, dropping below baseline past k=5 due to [[Oversmoothing]].

## Why this is load-bearing

- All 6 brainstorm ideas (sampling tweaks) fail to move ogbn-arxiv: within ±0.13 of baseline.
- Depth alone gives +0.80 at zero training-time cost.
- Conclusion: the fixed-depth encoder is the real bottleneck in [[SUGRL]], not the sampling. The axis to exploit is depth, not negatives.

This becomes AD-SSL's motivation: if a scalar depth change from 1 to 3 gives +0.80 for free, a **per-node learned** distribution over depths {1, 2, 4, 8} should give more — and without hyperparameter sweeping. See [[Multi-Depth Views]] and [[Adaptive Depth Weighting]].

## Why this is **not** a contribution

- Multi-hop propagation is standard GNN practice (every 2–3 layer GCN already does it).
- Precomputing `A^k X` before an MLP is the [[SGC]] trick, published 2019.
- We just noticed SUGRL's published k=1 default is under-tuned for a 169k-node graph and re-ran with k=3.
- This is a hyperparameter sweep, not a method. Goes in the appendix as a *baseline correction*.

## Implications for paper framing

- Published SUGRL numbers on ogbn-arxiv (68.8) underestimate the method. Fair comparison should cite SUGRL-k=3 (69.57). Mention this explicitly so reviewers don't claim we padded our deltas.
- AD-SSL's competitor on the "precomputed-features" frontier is SUGRL-k=3 (~69.57), not SUGRL-k=1 (~68.77). Use the corrected number as the reference point.
