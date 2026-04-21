---
title: GPRGNN
type: entity
kind: method
venue: ICLR 2021
url: https://arxiv.org/abs/2006.07988
tags: [method, prior-art, adaptive-depth, supervised]
created: 2026-04-21
updated: 2026-04-21
---

# GPRGNN — Generalized PageRank Graph Neural Network

Chien et al., ICLR 2021. Learned polynomial propagation coefficients.

## Key idea

Parameterize node representation as `Σ_k γ_k · A^k X` with learnable `γ_k`. The coefficients can capture both homophily and heterophily patterns.

## Relevance to AD-SSL

**Conceptual cousin**. Supervised. Learns depth coefficients globally (one γ_k per depth, shared across all nodes).

AD-SSL differs:
- Unsupervised.
- **Per-node** weights (not global). See [[Adaptive Depth Weighting]].
- Decoupled (precompute `A^k X` once, then MLP).

Reviewer defense: "this is just GPRGNN" → see [[Reviewer Attacks and Defenses]].
