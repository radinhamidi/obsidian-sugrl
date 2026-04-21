---
title: Oversmoothing
type: concept
tags: [concept, theory, propagation]
created: 2026-04-21
updated: 2026-04-21
---

# Oversmoothing

Repeated graph propagation (`Â^k X` as k → ∞) converges to a degree-weighted average, flattening node distinctions. Empirically, node classification accuracy collapses past some k.

## In our work

[[Prepropx Depth Finding]]: SUGRL's U-curve over k ∈ {1..6} peaks at k=3 and drops below baseline past k=5. This is oversmoothing at the level of linear probe accuracy.

## Implication for AD-SSL

- Depth set {1, 2, 4, 8} chosen below the oversmoothing cliff.
- k=8 may already be too deep for some graphs — if ablation shows it gets zero weight consistently, drop it.
- [[Adaptive Depth Weighting]] gives the model a way to escape from too-smooth views on a per-node basis — a node that would oversmooth at k=8 can weight k=1 heavily. This is a potential *defense* of choosing larger k (the model can ignore them) but shouldn't be oversold.
