---
title: GGD
type: entity
kind: method
venue: NeurIPS 2022
url: https://arxiv.org/abs/2206.01535
tags: [method, baseline, efficiency, pareto-anchor]
created: 2026-04-21
updated: 2026-04-21
---

# GGD — Graph Group Discrimination

Zheng, Pan, Lee, Zheng, Yu (Monash/Griffith/La Trobe/UIC), NeurIPS 2022. **Efficiency champion** and the left anchor of our [[Pareto Gap]].

## Key idea

A simple **binary cross-entropy** between two groups of node embeddings: positive group from the real graph, negative group from a node-shuffled (feature-permuted) corruption of the same graph. No similarity between node pairs, no summary vector, no InfoNCE, no negatives-per-node sampling.

Originates from the authors' re-analysis of DGI: the "summary vector" in DGI is effectively a near-constant under Xavier init + Sigmoid, so DGI's training signal is really just *group discrimination* in disguise. GGD strips that insight to its minimum.

**Loss (their Eq. 3):**
```
L_BCE = -(1/2N) Σ_{i=1..2N} [y_i log ŷ_i + (1-y_i) log(1-ŷ_i)]
ŷ_i = agg(h_i)        # sum aggregation → a scalar per node
y_i ∈ {1 for positive group, 0 for corrupted group}
```
Theorem 1: optimising L is equivalent to maximising `JS(P_pos ‖ P_neg)` — standard JS-divergence framing.

## Architecture

1. Optional augmentation on (X, A) via feature + edge dropout → (X̂, Â).
2. Corrupt X̂ by shuffling node order → (X̃, Ã) (same technique as DGI/MVGRL).
3. Siamese GCN encoder + MLP projector (shared weights) → embeddings for positive and negative groups.
4. Aggregate each embedding to a scalar with sum (mean and linear close seconds; see Tbl 11), compute BCE.

## Inference-time global embedding (important for AD-SSL)

GGD does **not** just use the encoder output. At inference they compute:
```
H_final = H_θ + A^n · H_θ       # n = 5 for all datasets
```
i.e. they re-propagate the frozen encoder output through the graph power. This is conceptually adjacent to [[Multi-Depth Views]] — they inject one extra "depth-n" view at inference (additively, fixed n, unweighted). AD-SSL generalises: K learnable depths during training with per-node weighting.

Ablation (Tbl 17): removing the `A^n` power costs 0.9–1.2 points on Cora/PubMed — non-trivial. Using only `H_θ` still beats 6 baselines on 4/5 datasets.

## Numbers on ogbn-arxiv (Table 8 in paper)

| Setting | Hidden | Epochs | Accuracy | Pre | Tr/epoch | Total(T) |
|---|---:|---:|---:|---:|---:|---:|
| GGD | 256 | 1 | **70.3 ± 0.3** | 6.26s | 0.18s | 0.18s |
| GGD | 1500 | 1 | **71.6 ± 0.5** | 6.26s | 0.95s | 0.95s |

Memory: 4,513 MB @ h=256 (69.8% less than GBT).
Hyperparameters (Tbl 15): lr=5e-5, hidden=1500, 3 GCN conv layers, 1 MLP projector layer, graph-power n=5.

## Numbers on ogbn-products (Tbl 9)

GGD 1 epoch: **75.7 ± 0.4** test, 12m46s total, 4,391 MB. Beats BGRL (64.0, 100 epoch) and GBT (70.5, 100 epoch).

## Numbers on ogbn-papers100M (Tbl 10)

| Method | Test acc | Time (1 epoch) | Memory |
|---|---:|---:|---:|
| BGRL (1 epoch) | 62.1 ± 0.3 | 26h 28m | 14,057 MB |
| GBT (1 epoch) | 61.5 ± 0.5 | 24h 38m | 13,185 MB |
| **GGD (1 epoch)** | **63.5 ± 0.5** | **9h 15m** | **4,105 MB** (68.9% less) |

GGD also converges in ~1 epoch on large OGB datasets (Fig. 4) — their framing is "GD trains the edge-distribution signal, GCL gets distracted by node-specific detail."

## Complexity

`O(N·D·(L + L·D + K·D))` — linear in N. L = #GCN layers, K = #MLP layers. Same asymptotic as SGC-style decoupled methods when L is small.

## Role in AD-SSL

- **Left anchor of the Pareto figure.** Our headline claim depends on beating or matching GGD-1500's 71.6 at comparable cost.
- **Baseline in Phase 1.** Reproduce GGD-256 (70.3) and GGD-1500 (71.6) in our env. Must match within 1σ before running AD-SSL.
- **Calibration for our "multi-depth is the axis" claim.** GGD already bakes in one A^n power at inference — they get a free ~1 point from it. AD-SSL extends this from one fixed depth (post-hoc) to K learnable depths (during training, per-node).

## Open risks this paper raises for our thesis

- **The Pareto gap is narrower than our onboarding framed it.** Onboarding positioned BGRL ~71.6 / GraphMAE ~71.7 as "expensive 71s" and GGD as "fast but lower." Actually GGD-1500 already hits 71.6 in 0.95s. Our target (71 at < 60s) is *barely above* GGD's existing number. AD-SSL must either (a) push past 71.6, (b) beat GGD's cost at matched accuracy, or (c) add clear structural advantages (per-node adaptivity, theoretical grounding, heterophily wins). See [[Pareto Gap]] for how we should re-frame.
- **The `A^n H_θ` trick is a cheap baseline for multi-depth.** A reviewer can say: "Your multi-depth is just GGD's inference-time power trick applied during training." We need the per-node α mechanism and the bootstrap loss to do real work in ablations, not just the multi-depth views.
- **GGD is one epoch on OGB.** Our training time must also be measured in the "1-epoch to convergence" regime if we want a fair cost comparison.

## References to follow up

- `https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination` — official code (for reproduction).
- ATP's HPC edge masking could stack with GGD's corruption → potentially useful preprocessing. See [[ATP]].
