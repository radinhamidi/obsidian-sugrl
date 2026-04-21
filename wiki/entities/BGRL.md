---
title: BGRL
type: entity
kind: method
venue: ICLR 2022
url: https://arxiv.org/abs/2102.06514
tags: [method, baseline, accuracy-ceiling]
created: 2026-04-21
updated: 2026-04-21
---

# BGRL — Bootstrap Your Own Graph Latents

Thakoor et al., ICLR 2022. BYOL-style self-supervised learning for graphs. **Accuracy ceiling** and conceptual parent of AD-SSL's [[Bootstrap Loss]].

## Key idea

Bootstrap-style training à la BYOL:
- Online encoder processes augmented view 1.
- Target encoder (EMA of online) processes augmented view 2.
- Predict target from online in latent space.
- No negatives.

## Accuracy on ogbn-arxiv

~71.6 ± 0.3 — the number AD-SSL must approach to claim "BGRL-level accuracy."

## Cost

Seconds-to-minutes (O(N·d²) per epoch, but two GNN encoders + augmentations). Order of magnitude slower than [[GGD]] or [[SUGRL]]-k=3.

## Role in AD-SSL

- **Accuracy target**: we want ≥71 on ogbn-arxiv.
- **Loss inspiration**: AD-SSL's bootstrap across depth pairs is BGRL-flavored. Key difference: BGRL uses augmentations as views; we use multi-depth precomputed features as views.
- Baseline in main results and Pareto figure.

## Open items

- [ ] Ingest the paper PDF when available. Cover augmentation choices, exact loss, OGB-specific config.
