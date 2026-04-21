---
title: SUGRL
type: entity
kind: method
venue: AAAI 2022
url: https://ojs.aaai.org/index.php/AAAI/article/view/20748
code: https://github.com/YujieMo/SUGRL
tags: [method, baseline, starting-point]
created: 2026-04-21
updated: 2026-04-21
---

# SUGRL — Simple Unsupervised Graph Representation Learning

Mo et al., AAAI 2022. The **starting point** for AD-SSL; our method begins as a modification of SUGRL's decoupled-precompute + MLP architecture.

## Architecture

- **Anchor**: MLP on raw features X → H (captures semantic info).
- **Structural positive**: shared-weight GCN on adjacency A → H+ (1-hop smoothed).
- **Neighbor positive**: average of 5 sampled neighbor embeddings → H̃+.
- **Negative**: row-shuffled anchor embeddings → H−.
- **Loss**: margin-based triplet (structural + neighbor) + upper-bound regularizer.

## Key properties

- No augmentation, no discriminator.
- Fast: trains ogbn-arxiv in ~6 seconds.
- Baseline reproduced in our env within 1–2σ on all 6 datasets (see [[VALIDATION_ORIGINAL_CODE]]).

## Key weakness

**Fixed k=1 propagation depth.** Only one structural scale. Paper does not ablate depth; the published `train_OGB.py` uses k=1. See [[Prepropx Depth Finding]] — moving to k=3 gives +0.80 on ogbn-arxiv for free.

## Accuracy on ogbn-arxiv

- Published: 68.8 ± 0.4 (k=1)
- Our reproduction (k=1): 68.77 ± 0.13
- Same method with k=3: 69.57 ± 0.05 (baseline correction, not a new method)

## Negative sampling detail

SUGRL's `np.random.permutation` produces a **derangement**, not i.i.d. sampling. On [[PubMed]] this derangement constraint seems to hurt — replacing it with per-anchor i.i.d. sampling gives +1.37 (3/3 seeds). See [[Preliminary Validation - 168 Runs]] §4.4. Appendix-worthy, not AD-SSL-critical.

## Role in AD-SSL

- Architectural ancestor: decoupled precompute + MLP pattern.
- Baseline in main results table.
- Target to beat: must clear SUGRL-k=3 (not k=1) in ogbn-arxiv.
