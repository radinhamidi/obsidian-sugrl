---
title: GraphMAE
type: entity
kind: method
venue: KDD 2022
url: https://arxiv.org/abs/2205.10803
tags: [method, baseline, generative, accuracy-ceiling]
created: 2026-04-21
updated: 2026-04-21
---

# GraphMAE — Self-Supervised Masked Graph Autoencoders

Hou, Liu, Cen, Dong, Yang, Wang, Tang (Tsinghua + Alibaba DAMO + BirenTech). KDD 2022, arXiv v3 2022-07-13. **Generative SSL** on graphs via masked feature reconstruction — the first GAE that consistently matches contrastive methods on node/graph classification. Accuracy-ceiling competitor alongside [[BGRL]] / [[GraphMAE2]] for [[AD-SSL]]'s Pareto story.

## Mechanism (exact)

Four design choices, all ablated in Table 4:

1. **Mask input features.** Sample `Ṽ ⊂ V` uniformly at random (mask ratio 0.5 on ogbn-arxiv). Replace their features with a learnable `[MASK]` token `x_[M] ∈ R^d`. Use BERT's "random-substitution" (small probability of replacing with another random token, not leaving unchanged — "leave-unchanged" hurts).
2. **GNN encoder `f_E`.** GAT is the default for node classification; GIN for graph classification. `H = f_E(A, X̃)`.
3. **Re-mask the code.** Before decoding, replace `H[i]` for `v_i ∈ Ṽ` with a *separate* decoder mask token `h_[DM] ∈ R^d_h`. The decoder must therefore reconstruct from the neighborhood, not the node itself.
4. **GNN decoder `f_D`.** Single-layer GNN (not MLP). Output `Z = f_D(A, H̃)`.

**Loss — Scaled Cosine Error (Eq. 2):**

```
L_SCE = (1/|Ṽ|) Σ_{v_i ∈ Ṽ}  (1 − (x_i · z_i) / (‖x_i‖ · ‖z_i‖))^γ,   γ ≥ 1
```

Cosine (not MSE) handles varying feature norms; the `γ` exponent down-weights easy samples (focal-loss style). γ=3 on ogbn-arxiv.

## ogbn-arxiv hyperparameters (Table 7)

| Param | Value |
|---|---|
| Encoder | GAT (also tested GCN: 71.87 ± 0.21 — actually *better* than GAT's 71.75 ± 0.17 — see §A.2) |
| Decoder | single-layer GNN, re-masked input |
| Mask ratio | 0.5 |
| Replacing rate | 0.0 (no random-substitution on ogbn-arxiv) |
| Scaling factor γ | 3 |
| hidden_size | 1024 |
| weight_decay | 0 |
| max_epoch | 1000 |
| Optimizer | Adam (β₁=0.9, β₂=0.999, ε=1e-8), lr 0.001, cosine decay, no warmup |
| Activation | PReLU |

**Note for AD-SSL design**: mask ratio 0.5 on a relatively small-feature graph (d=128 for ogbn-arxiv). Our multi-depth precompute gives K different "views" without needing a mask — a cleaner design for graphs where feature corruption is hard to interpret (the paper itself flags this in §2.1: "without a theoretical understanding of handcrafted graph augmentation strategies, it remains unverified whether they are label-invariant").

## Numbers on ogbn-arxiv (Table 1)

| Method | Test |
|---|---:|
| GCN (supervised) | 71.74 ± 0.29 |
| GAT (supervised) | 72.10 ± 0.13 |
| DGI | 70.34 ± 0.16 |
| GRACE | 71.51 ± 0.11 |
| [[BGRL]] | 71.64 ± 0.12 |
| CCA-SSG | 71.24 ± 0.20 |
| **GraphMAE (GAT)** | **71.75 ± 0.17** |
| **GraphMAE (GCN)** | **71.87 ± 0.21** |

GraphMAE (GCN) on ogbn-arxiv: **71.87** — slightly higher than our original entry of ~71.7 and in fact beats GraphMAE-GAT here. Cite 71.87 for "best GraphMAE on ogbn-arxiv". Matches supervised GAT; essentially identical to [[BGRL]] accuracy-wise.

## Node classification (Table 1, non-OGB)

| | Cora | CiteSeer | PubMed | PPI | Reddit |
|---|---:|---:|---:|---:|---:|
| GraphMAE | 84.2 ± 0.4 | 73.4 ± 0.4 | 81.1 ± 0.4 | 74.50 ± 0.29 | 96.01 ± 0.08 |
| BGRL | 82.7 ± 0.6 | 71.1 ± 0.8 | 79.6 ± 0.5 | 73.63 ± 0.16 | 94.22 ± 0.03 |
| CCA-SSG | 84.0 ± 0.4 | 73.1 ± 0.3 | 81.0 ± 0.4 | 73.34 ± 0.17 | 95.07 ± 0.02 |

GraphMAE is SOTA or tied-SOTA on 5/6 node-classification benchmarks at the time of publication.

## Graph classification (Table 2) and transfer learning (Table 3)

Also SOTA/competitive on 7 graph-classification benchmarks (IMDB-B, IMDB-M, PROTEINS, COLLAB, MUTAG, REDDIT-B, NCI1) and best average (73.8) on 8 MoleculeNet transfer tasks. Not directly relevant to AD-SSL's OGB scale-story but matters for related-work coverage: GraphMAE is a *general* SSL framework, not a narrow node-classification trick.

## Complexity

Paper reports `O(N)` run-time memory (Figure 1a) — no N² negatives or adjacency reconstruction. Cost is dominated by the encoder+decoder GNN passes per epoch (2 GNN forwards per step: encoder over masked X̃, decoder over re-masked H̃). 1000 epochs on ogbn-arxiv at hidden=1024 on a 3090 — expensive but not negative-sample-prohibitive. Exact wall-clock not reported in the paper.

## Role in AD-SSL

- **Accuracy ceiling on ogbn-arxiv: 71.87.** Co-anchor with BGRL (71.64) and [[GraphMAE2]] for the upper end of the Pareto frontier. See [[Pareto Gap]].
- **Generative competitor.** Related-work paragraph needs a clean mechanism contrast — AD-SSL is *not* masked reconstruction; it's bootstrap on multi-depth views. The common ground is "no negatives, no feature augmentation heuristics"; the divergence is reconstructing-vs-contrasting and single-view-masked-input vs K-view-precomputed.
- **Scaled cosine error is an interesting ablation for our bootstrap loss.** [[BGRL]] uses plain cosine; GraphMAE uses `(1-cos)^γ`. Worth testing `γ > 1` in the AD-SSL depth-pair bootstrap loss to down-weight easy depth-pairs — if some pairs are trivially aligned (adjacent depths), the γ-scaling could focus gradient on harder (distant-depth) pairs. Flag as an experiment for the Coding Agent (Ablation A4 candidate).

## Differences from AD-SSL (related-work paragraph)

| | GraphMAE | AD-SSL |
|---|---|---|
| Objective | Masked feature reconstruction | Cross-depth bootstrap alignment |
| Views | 1 corrupted view (random-masked X) | K deterministic multi-depth views `Â^k X` |
| Encoder | GNN (GAT/GCN, 2 layers) | MLP only |
| Decoder | GNN | None (linear probe on mixture) |
| Loss | Scaled Cosine Error on raw features | `-cos(p(Z_k), Z_{k'}.detach())` pairwise across depths |
| Cost per step | 2 GNN forwards + 1 GNN backward | 0 GNN forwards |
| Inference | Frozen encoder on un-masked graph | Weighted sum `Σ_k α_{i,k} Z_k` |

**Reviewer attack to expect:** "AD-SSL is GraphMAE with depth-pair views instead of feature masking." Defence: (a) no reconstruction objective — we do not learn to predict raw features; (b) no GNN in the hot loop; (c) per-node depth-mixture α is a *learned inference-time output*, not just a training trick.

## Reproduction note

Official code: `https://github.com/THUDM/GraphMAE`. Phase 1 reproduction target: **71.8 ± 0.3 on ogbn-arxiv** using GAT encoder hidden=1024, mask=0.5, γ=3, 1000 epochs Adam lr=1e-3 cosine. If we can match this within our harness, the Pareto comparison is fair.

## Evidence gap flagged

- **GCN > GAT on ogbn-arxiv for GraphMAE** (71.87 vs 71.75) — the paper uses GAT in the main table but GCN is actually their best OGB result. We should cite the GCN number for a fair ceiling.
- Wall-clock not reported — for the Pareto figure we must time GraphMAE in our harness. Given 1000 epochs at hidden=1024 with a GNN decoder, expect it to be substantially slower than [[GGD]]-1500 (which hits 71.6 in under 1s). This is the core of our efficiency story.
- No ogbn-products / ogbn-papers100M numbers in the KDD paper. [[GraphMAE2]] fills this gap — see that page when ingested.
