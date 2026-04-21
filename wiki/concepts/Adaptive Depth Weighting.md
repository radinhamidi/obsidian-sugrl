---
title: Adaptive Depth Weighting
type: concept
tags: [concept, method-ad-ssl]
created: 2026-04-21
updated: 2026-04-21
---

# Adaptive Depth Weighting

Per-node, per-depth weights α_{i,k} determining how AD-SSL mixes the K depth views at inference (and possibly during training loss weighting).

## Mechanism (A1 / GRPO-style)

For each node i and each depth k:
1. Compute the **consensus** of the other depths at node i.
2. Score depth k by its alignment (cosine) with that consensus.
3. Softmax over k to get α_{i,k}.

Nodes whose local structure is best captured at shallow depths get high α_1; nodes that benefit from broader context get high α_k for larger k. The mechanism is parameter-free (no learned gate) or lightly parameterized (single softmax temperature).

## Inference

`Z_i = Σ_k α_{i,k} · Z_{i,k}` where `Z_{i,k}` is the encoder output on `X_k` at node i.

## Why "group-relative"

Scoring uses the *group* of other depths as the reference (not an absolute target). This is inspired by GRPO-style relative reward computation, but the connection to RLHF is just an intuition pump — nothing from RL ends up in the paper.

## Alternatives ablated

- **Uniform (B0)**: α_{i,k} = 1/K. Baseline.
- **A4 (EMA-refined)**: iteratively smooth α over epochs with EMA.
- **Learned gate**: small MLP predicts α from features. Currently not planned but could be A5 if A1 underperforms.

## Prior art

- [[GPRGNN]] — learned **global** γ_k (not per-node), supervised.
- [[ATP]] — per-node learned weights, supervised. Closest. Differentiation: AD-SSL's group-relative scoring vs ATP's learned gate.

See [[Ablation Plan - AD-SSL B0 A1-A4]].
