---
title: SGC
type: entity
kind: method
venue: ICML 2019
url: https://arxiv.org/abs/1902.07153
tags: [method, prior-art, decoupled, spectral]
created: 2026-04-21
updated: 2026-04-21
---

# SGC — Simplifying Graph Convolutional Networks

Wu, Zhang, de Souza Jr., Fifty, Yu, Weinberger (Cornell). ICML 2019. The foundational **[[Decoupled Precompute]]** method AD-SSL inherits.

## Mechanism (exact)

Collapse K-layer GCN by removing all nonlinearities and reparameterising `Θ = Θ⁽¹⁾Θ⁽²⁾…Θ⁽ᴷ⁾`:

```
Ŷ_SGC = softmax(Sᴷ X Θ)    where S = D̃^(−½) Ã D̃^(−½), Ã = A + I
```

`Sᴷ X` is **parameter-free** — computed once as a preprocessing step. Training reduces to multi-class logistic regression on the propagated features.

## Spectral interpretation (§3.2)

With renormalized adjacency `S̃_adj`, filter coefficients are `ĝ(λ̃_i) = (1 − λ̃_i)^K` where `λ̃_i` are eigenvalues of the augmented normalized Laplacian. SGC = **fixed low-pass filter, amplitude `(1−λ)^K`**. Larger K → sharper low-pass → smoother features.

**This matters for AD-SSL**: different K values correspond to different low-pass cutoffs, i.e., different spectral views of the same signal. Our multi-depth `{Â^k X}_k` stack is a family of filters with different cutoffs — and per-node α is a *learned, data-dependent* mixture over these filters.

## Numbers (Table 2–3)

| Dataset | SGC | GCN | Notes |
|---|---:|---:|---|
| Cora | 81.0 ± 0.0 | 81.5 | match within 1σ |
| Citeseer | 71.9 ± 0.1 | 70.3 | **+1.6 over GCN** (less overfitting) |
| Pubmed | 78.9 ± 0.0 | 79.0 | match |
| Reddit (unsupervised→LR) | 94.9 (F1) | OOM | beats SAGE variants |

K tuned per dataset; K=2 is the default. SGC trains in **~2 L-BFGS steps** on Reddit; on Pubmed it is **~28× faster than GCN** (Fig 3).

## Role in AD-SSL

- **Direct precursor.** AD-SSL's architecture is literally "SGC-style precompute, but with K views at K depths simultaneously, plus SSL loss, plus per-node mixture α."
- **[[SUGRL]]** used the same single-depth `A^k X` precompute pattern for unsupervised GRL at k=1.
- **Cost anchor.** Any claim about AD-SSL's speed should reference SGC's precomputation regime — the wall-clock budget of AD-SSL training is "SGC precompute + K·MLP passes with bootstrap loss."
- **Spectral framing for differentiation.** Reviewers who object "multi-depth is just K SGCs" can be answered: AD-SSL learns per-node α over spectral depths; SGC uses a single fixed K. This is the same relationship as supervised MLP → adaptive mixture of experts.

## Differences from AD-SSL

| | SGC | AD-SSL |
|---|---|---|
| Propagation | Single Sᴷ, fixed K | K views {S¹X, S²X, …, Sᴷ X} |
| Training | Supervised LR | SSL bootstrap across depth pairs |
| Output | softmax(Sᴷ X Θ) | `Σ_k α_{i,k} Z_k` with per-node α |
| Task | Supervised classification | Unsupervised representation |

## Reproduction note

Official: `https://github.com/Tiiiger/SGC`. K=2, Adam lr=0.2, weight_decay tuned per dataset. On ogbn-arxiv (not in the original paper), community reproductions place SGC around **66.9 ± 0.08** (per [[GraphMAE2]] Table 3) — well below BGRL 70.5 and our 71-range targets. SGC-style precompute is a speed floor, not an accuracy ceiling.
