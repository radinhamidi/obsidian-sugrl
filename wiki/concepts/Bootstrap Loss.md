---
title: Bootstrap Loss
type: concept
tags: [concept, method-ad-ssl, loss]
created: 2026-04-21
updated: 2026-04-21
---

# Bootstrap Loss

BYOL-style training signal. Online network predicts target network's output; target is an EMA of online. No negatives.

## Provenance

- BYOL (image SSL, NeurIPS 2020).
- [[BGRL]] (graph SSL, ICLR 2022).
- AD-SSL uses it across **depth pairs** rather than augmentation pairs.

## Formulation (AD-SSL variant)

For each depth pair (k, k'):

`L_{k,k'} = − cos( predictor(online(X_k)), stop_grad(target(X_{k'})) )`

Target network = EMA of online (decay 0.99 → 0.999 warmup, mirror BGRL).

Total loss: `Σ_{k≠k'} w_{k,k'} · L_{k,k'}` where `w` may depend on adaptive weighting (A1).

## Why it works without negatives

The stop-grad + EMA target prevents representational collapse without explicit negatives. Theoretically studied for image SSL; carries over to graphs empirically (BGRL, DGI-variants).

## Ablation against simpler losses

A3 (SimPO-style) tests whether bootstrap is necessary. Alternatives: MSE between views, InfoNCE with random negatives. If MSE suffices, we remove the EMA machinery and simplify the paper. See [[Ablation Plan - AD-SSL B0 A1-A4]].
