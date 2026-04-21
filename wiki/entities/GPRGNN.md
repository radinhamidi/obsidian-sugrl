---
title: GPR-GNN
type: entity
kind: method
venue: ICLR 2021
url: https://arxiv.org/abs/2006.07988
tags: [method, prior-art, decoupled, polynomial-filter, heterophily, critical-prior-art]
created: 2026-04-21
updated: 2026-04-21
---

# GPR-GNN — Generalized PageRank Graph Neural Network

Chien, Peng, Li, Milenkovic (UIUC). ICLR 2021. **Critical prior art for AD-SSL** — the closest published antecedent of our multi-depth learnable mixture.

## Mechanism (exact, Eq. 1)

```
H⁽⁰⁾_i = f_θ(X_i)              # MLP on features
H⁽ᵏ⁾  = Ã_sym H⁽ᵏ⁻¹⁾           # power propagation
Z     = Σ_{k=0..K} γ_k H⁽ᵏ⁾    # learned mixture over depths
P̂    = softmax(Z)
```

The **`γ_k` are learned end-to-end with θ**. K is a fixed truncation (typically 10).

## Key theoretical facts

1. **Equivalence to polynomial graph filters**: `Σ_k γ_k Ã^k_sym` is a degree-K polynomial filter. Any graph filter can be approximated by a polynomial filter, so learning γ is learning the optimal filter.
2. **APPNP and SGC are special cases**: APPNP fixes `γ_k = α(1−α)^k`; SGC fixes `γ_k = δ_{kK}` (one-hot at depth K).
3. **Homophily/heterophily universality**: Learned γ_k can be **negative**, enabling high-pass behavior. On heterophilic Texas: alternating-sign γ with dampening (Fig 1c). On homophilic Cora: monotone positive (Fig 1b).
4. **Over-smoothing escape**: unlike APPNP's label-independent escape, GPR-GNN's γ_k is trained using label signal, so the mitigation is "guided by node-label information" (Thm 4.2).

## Numbers (Table 2, small graphs)

GPR-GNN matches or beats all baselines (GCN, GAT, APPNP, JKNet, SGC, MLP) across **10 datasets** including homophilic (Cora/Citeseer/PubMed/Computers/Photo) and heterophilic (Chameleon/Squirrel/Actor/Texas/Cornell).

No OGB-scale numbers in the original paper — it's a small-graph supervised study. **Flag**: important gap when citing GPR-GNN as a ceiling at ogbn-arxiv scale.

## **Danger: this is the prior art a reviewer will cite against AD-SSL**

Architecture is nearly identical: MLP on features → power-propagate → **learned mixture over depths** → output.

**Exact differences (the ones that must do real work):**

| | GPR-GNN | AD-SSL |
|---|---|---|
| Mixture weights | **Global** scalar γ_k (one per depth) | **Per-node** α_{i,k} (N × K matrix) |
| Training signal | **Supervised** cross-entropy | **SSL bootstrap** across depth pairs + EMA target |
| View structure | Depths `{H⁽⁰⁾, H⁽¹⁾, …, H⁽ᴷ⁾}` from same MLP output | Depths from precomputed `{Â^k X}` fed to shared MLP |
| Output | `Σ_k γ_k H⁽ᵏ⁾` used as logits | Representations used with linear probe / fine-tune |
| Depth grid | Dense k=0..K (K=10) | Sparse and wide {1, 2, 4, 8} |
| Negative weights | Allowed (heterophily) | Softmax α ≥ 0 currently — **open design question** |

## Reviewer-defence map

1. **"AD-SSL = GPR-GNN + per-node weights + SSL loss."** True in spirit; own it. Defence:
   - Per-node adaptivity: polynomial filter vs locally-adapted filter bank. See [[Adaptive Depth Weighting]].
   - SSL bootstrap over depth pairs provides training signal GPR-GNN can't produce (no labels).
   - Our experimental scale (ogbn-arxiv + Pareto-cost framing) is different.

2. **"Reproduce GPR-GNN with SSL and call it done."** We **must** run an ablation with a single global γ_k across all nodes (SSL version of GPR-GNN). If per-node α gives a meaningful gain over global γ, we have the story. If not, AD-SSL collapses to "SSL-GPR-GNN."

3. **"Why no negative weights like GPR-GNN for heterophily?"** Currently softmax α ≥ 0. On homophilic ogbn-arxiv this should be fine. On heterophilic benchmarks it may cap us. Open question — see [[Ablation Plan - AD-SSL B0 A1-A4]].

## Must-do in ablations

- **Global-γ baseline inside our harness.** SSL training, MLP on precomputed `Â^k X`, but **shared γ_k across all nodes**. The tightest "is per-node α doing work?" test.
- **Heterophily check** on Chameleon/Squirrel/Actor (with and without negative-allowing α). If we can't match GPR-GNN here, we scope to homophilic.

## Reproduction note

Official: `https://github.com/jianhao2016/GPRGNN`. K=10, 2-layer MLP hidden 64, per-dataset dropout, Adam. Not a Phase 1 accuracy baseline but **must be implemented as the global-γ ablation** (A_global in our ablation table).
