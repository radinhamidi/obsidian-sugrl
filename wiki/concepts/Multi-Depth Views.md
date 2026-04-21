---
title: Multi-Depth Views
type: concept
tags: [concept, method-ad-ssl]
created: 2026-04-21
updated: 2026-04-21
---

# Multi-Depth Views

Core conceptual primitive of AD-SSL. Instead of augmentation, use pre-computed features at multiple propagation depths as **natural, complementary views** for contrastive / bootstrap learning.

## Views

`{X_1, X_2, X_4, X_8}` where `X_k = Â^k X`. Each depth:

- Captures a different neighborhood radius.
- Low-pass filters the graph signal with progressively wider bandwidth (spectral interpretation).
- k=1 emphasizes local features; k=8 emphasizes broad structural patterns.

## Why this is a valid contrastive signal

A node's identity should be roughly consistent across depths (same class should emerge regardless of how wide the lens). The across-depth **disagreement** is precisely the structural-vs-semantic tension — training to bring views into agreement forces the encoder to learn structure-robust representations.

## Prior art

- [[PolyGCL]] uses spectral polynomial filters as contrastive views (supervised learnable filters).
- [[BGRL]] uses augmentation-based views.
- AD-SSL uses **parameter-free depth views** — no augmentation, no spectral parameters, no per-epoch cost.

## Depth set choice

{1, 2, 4, 8}. Exponential spacing covers the useful U-curve region from [[Prepropx Depth Finding]] without redundant adjacent values. May shrink to {1, 2, 4} if k=8 gets negligible weight in ablation.
