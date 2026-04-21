---
title: APPNP
type: entity
kind: method
venue: ICLR 2019
url: https://arxiv.org/abs/1810.05997
tags: [method, prior-art, propagation, supervised, decoupled]
created: 2026-04-21
updated: 2026-04-21
---

# APPNP — Approximate Personalized Propagation of Neural Predictions

Klicpera, Bojchevski, Günnemann (TU Munich). ICLR 2019. First to **decouple prediction from propagation** with a principled (personalized-PageRank) propagation scheme.

## Mechanism (exact, Eq. 4)

1. Per-node prediction `H = f_θ(X)` (MLP, no graph).
2. Iterated propagation with teleport α:
   ```
   Z⁽⁰⁾ = H
   Z⁽ᵏ⁺¹⁾ = (1−α) Â Z⁽ᵏ⁾ + α H
   Z⁽ᴷ⁾ = softmax((1−α) Â Z⁽ᴷ⁻¹⁾ + α H)
   ```
3. Fixed point (K → ∞) is exact PPNP: `Z = α(I − (1−α)Â)⁻¹ H` (Eq. 2–3).

Propagation is **parameter-free** — no training cost in the propagation steps. K=10 iterations + teleport α=0.1 (Cora-ML/Citeseer/PubMed) or α=0.2 (MS Academic).

## Key insight (for AD-SSL framing)

APPNP's teleport creates a **geometric mixture over depths**:
```
Z = α Σ_{k=0}^∞ (1−α)^k Âᵏ H
```
i.e. weight of depth-k view is `α(1−α)^k`. This is a **fixed, global, geometric-decaying** mixture over all depths.

**AD-SSL generalises this along two axes:**
1. **Per-node** rather than global weights: α_{i,k} learned per-node.
2. **Learned / non-monotone** rather than geometric: different nodes can prefer different depths (high α_1 for heterophilic, high α_4 for homophilic clusters).

This is the cleanest positioning in the related-work paragraph — APPNP is our geometric-mixture ancestor; AD-SSL learns the mixture.

## Numbers on Cora-ML / Citeseer / PubMed / MS Academic (Table 2)

| Dataset | APPNP | GCN | Notes |
|---|---:|---:|---|
| Citeseer | 75.73 ± 0.30 | 73.59 ± 0.30 | +2.14 |
| Cora-ML | 85.09 ± 0.25 | 83.41 ± 0.25 | +1.68 |
| PubMed | 79.73 ± 0.31 | 78.68 ± 0.28 | +1.05 |
| MS Academic | 93.27 ± 0.08 | 91.65 ± 0.09 | +1.62 |

Evaluation protocol is rigorous: 100 random splits × initializations, bootstrapped CIs, paired t-tests (§5). A good template for our own eval protocol.

## Role in AD-SSL

- **Closest conceptual prior** for multi-depth propagation as a mixture with learnable mixing weight.
- **Supervised only.** APPNP uses labels for the MLP; AD-SSL has no labels. The propagation-as-mixture idea ports directly.
- **Parameter-free propagation** is shared — AD-SSL's `Â^k X` precompute is even cheaper (k fixed set, not power iteration at each forward).
- **Evaluation protocol**: adopt their 100-run bootstrap protocol for our small-graph sanity checks.

## Differences from AD-SSL

| | APPNP | AD-SSL |
|---|---|---|
| Mixture | Geometric `α(1−α)^k`, global | Learned per-node α_{i,k}, discrete K views |
| Propagation | Power iteration at forward | Precomputed {Â¹X, Â²X, Â⁴X, Â⁸X} once |
| Task | Supervised classification | SSL representation |
| Depth range | K iterations at teleport rate α | Explicit depth grid |

## Reproduction note

Official: `https://github.com/benedekrozemberczki/APPNP` (PyTorch port) or `github.com/klicperajo/ppnp` (TF, original). α=0.1, K=10, 2-layer MLP with hidden 64, dropout 0.5, L2 1e-3. Not a Phase 1 baseline — cited in related work for propagation-mixture framing.
