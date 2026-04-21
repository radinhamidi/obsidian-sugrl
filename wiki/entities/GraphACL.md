---
title: GraphACL
type: entity
kind: method
venue: NeurIPS 2023
url: https://arxiv.org/abs/2310.18884
tags: [method, baseline, augmentation-free, heterophily, asymmetric-byol]
created: 2026-04-21
updated: 2026-04-21
---

# GraphACL — Simple and Asymmetric Graph Contrastive Learning without Augmentations

Xiao, Zhu, Chen, Wang (Penn State + Zhejiang). NeurIPS 2023, arXiv:2310.18884. **Augmentation-free, BYOL-style, targets heterophily**. Directly relevant to AD-SSL's "no augmentation" claim.

## Mechanism (exact, Eq. 3–4)

Two encoders: online `f_θ` and target `f_ξ` (EMA, stop-grad on target). Predictor `g_ϕ` used only on online side.

**Prediction loss** — for each node v, predict each one-hop neighbor u's target representation:
```
L_PRE = (1/|V|) Σ_v (1/|N(v)|) Σ_{u ∈ N(v)} ‖g_ϕ(v) − u‖²₂
  where v = f_θ(G)[v], u = f_ξ(G)[u]
  ξ ← λξ + (1−λ)θ    (EMA)
```

**Uniformity regularizer** (collapse prevention, Eq. 4):
```
L_UNI = −(1/|V|²) Σ_v Σ_{v⁻} ‖v − v⁻‖²₂
```
approximated with K random negatives per node.

Total: `L = L_PRE + λ_reg · L_UNI`.

## Key insight: two-hop monophily

GraphACL does **not** pull neighbors together directly (which would assume homophily). Instead: two nodes `v₁, v₂` that share a common neighbor `u` both reconstruct `u`'s target representation, which *implicitly* aligns 2-hop neighbors. This is the "monophily" (shared-neighborhood structure) signal and holds under both homophily and heterophily.

Theoretically proven (Thm 1, 2 in §5) to maximise MI with one-hop context and capture two-hop monophily.

## Numbers (Table 2)

Heterophilic (accuracy with linear probe):

| Dataset | GraphACL | best prior |
|---|---:|---:|
| Squirrel | **54.05** | 52.94 (SUGRL) |
| Chameleon | **69.12** | 68.74 |
| Texas | **71.08** | 62.11 |
| Actor | **30.03** | 32.55 (MVGRL beats it here) |
| **Arxiv-year** | **47.21** | 45.80 |

Homophilic:

| Dataset | GraphACL | best prior |
|---|---:|---:|
| Cora | 84.20 | 84.00 |
| Citeseer | 73.63 | 73.26 |
| Pubmed | 82.02 | 81.82 |
| Computers | 89.80 | 89.69 |
| Photo | 93.31 | 93.15 |
| **ogbn-arxiv** | **71.72 ± 0.26** | 71.64 (BGRL) |

**Important: GraphACL matches BGRL on ogbn-arxiv (71.72 vs 71.64) without augmentation.** This is the same accuracy tier as our Pareto-gap upper anchor. It *also* wins on all heterophilic benchmarks.

## Role in AD-SSL

- **Augmentation-free peer**. GraphACL is the cleanest SOTA competitor that does *not* use feature/edge masking. AD-SSL also uses no augmentations — we need a clear differentiation.
- **Heterophily benchmark setter.** If we claim anything about heterophily, we must compare against GraphACL on Squirrel/Chameleon/Actor/Texas/Arxiv-year. GraphACL is the current number to beat on those.
- **Loss-form comparison point.** GraphACL's BYOL-without-aug structure is structurally similar to what AD-SSL's bootstrap across depth pairs reduces to if K=2. We need to make sure our loss is meaningfully different.

## Differences from AD-SSL

| | GraphACL | AD-SSL |
|---|---|---|
| Views | 1 (central) vs 1-hop neighbors from target | K depth-propagated views `{Â^k X}` |
| Encoder | Full GNN (GAT/GCN) | MLP on precomputed features |
| Cost per step | GNN forward × 2 (online + target) | Zero GNN forwards (precompute once) |
| Positives | One-hop neighbors (structure-defined) | Same node at different depths (depth-defined) |
| Heterophily mechanism | Asymmetric predictor → two-hop monophily | Per-node α allows per-node depth preference |

## Reviewer-defence map

1. **"GraphACL is already augmentation-free and gets 71.72 on ogbn-arxiv — why do we need AD-SSL?"** Answer: cost. GraphACL runs a full GNN forward × 2 per step; AD-SSL has zero GNN forwards in the training loop. Same accuracy at GGD-cost is the Pareto claim. See [[Pareto Gap]].

2. **"GraphACL + depth-mixture = AD-SSL."** Risk: reviewer proposes this as obvious extension. Defence: (a) our multi-depth is a distinct view source (propagation depths, not structural neighbors); (b) we don't need the uniformity regularizer because cross-depth alignment is not collapse-prone in the same way; (c) our cost structure is fundamentally different.

3. **Heterophily framing.** If we scope to ogbn-arxiv (homophilic), we must still run against GraphACL on 1–2 heterophilic datasets or explicitly scope out heterophily. Don't over-claim.

## Reproduction note

Official: no link given in abstract; code URL in paper. λ (EMA) and K (negatives) per-dataset. Not a Phase 1 baseline but **must appear in Phase 2 accuracy comparison** at 71.72 on ogbn-arxiv, and in any heterophily table we include.
