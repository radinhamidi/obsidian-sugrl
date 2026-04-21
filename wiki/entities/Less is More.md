---
title: Less is More
type: entity
kind: method
venue: arxiv / ICLR 2026 submission
url: https://arxiv.org/abs/2509.25742
tags: [method, concurrent-work, high-priority]
created: 2026-04-21
updated: 2026-04-21
---

# "Less is More: Towards Simple Graph Contrastive Learning"

arxiv 2509.25742. ICLR 2026 submission. **Closest concurrent work to AD-SSL.**

## Architecture (as reported)

- GCN + MLP as complementary views.
- No augmentation.
- No negatives.

## Focus

- Heterophily.
- Robustness.
- **Small scale** — does NOT run at OGB scale.

## Our differentiation

| Axis | Less is More | AD-SSL |
|---|---|---|
| Scale | small-scale heterophily | OGB scale (ogbn-arxiv, products, ideally 100M) |
| Views | 1 GCN + 1 MLP | K precomputed depths {1, 2, 4, 8}, shared MLP |
| Weighting | fixed | per-node adaptive |
| Cost | per-epoch GNN forward | no per-epoch GNN forward (decoupled precompute) |

## Required action

- **Reproduce at OGB scale** as a baseline. Phase 1 Coding Agent work. See [[Project Phases and Decision Gates]].
- **Monitor ICLR 2026 acceptance** — accepted → must cite and differentiate explicitly; rejected → cite with weaker framing.

## Risk

If they add multi-depth views or push to OGB scale in a revision before NeurIPS 2026 deadline, our novelty narrows. Monitor arxiv for v2+ of the paper.
