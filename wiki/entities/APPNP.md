---
title: APPNP
type: entity
kind: method
venue: ICLR 2019
url: https://arxiv.org/abs/1810.05997
tags: [method, prior-art, propagation, supervised]
created: 2026-04-21
updated: 2026-04-21
---

# APPNP — Approximate Personalized Propagation of Neural Predictions

Klicpera et al., ICLR 2019. Teleport-based propagation: `Z = (1-α) Â Z + α H` iterated, equivalent to personalized PageRank on neural predictions.

## Relevance to AD-SSL

Conceptual prior for "mixing multiple propagation scales." APPNP's teleport parameter α controls effective depth via a geometric series. AD-SSL replaces the fixed geometric mixture with a **learned, per-node, discrete** mixture over {1, 2, 4, 8}.

Related-work citation, not a direct baseline.
