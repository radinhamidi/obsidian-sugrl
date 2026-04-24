---
title: DGD - Decoupled Graph Discrimination
type: entity
tags: [neurips-2026, baseline, graph-ssl, group-discrimination, decoupled]
created: 2026-04-24
updated: 2026-04-24
sources: []
---

# DGD — Decoupled Graph Discrimination

Gao, Luo, Qi, et al. "When decoupled GCN meets group discrimination: A special graph contrastive learning framework." *Neurocomputing* 596 (2024) 127952.

PDF verified 2026-04-24 via `raw/papers/DGD.pdf` + `pdftotext -layout` extraction. All claims below traceable to the PDF.

## One-sentence claim

Extends [[GGD]] (group discrimination, NeurIPS 2022) by replacing the GCN encoder with a decoupled MLP pipeline: **fixed-k message passing applied to raw features before training**, a shared-weight MLP encoder, a novel **partial corruption** negative-sampling strategy, and a BCE group-discrimination loss. At inference, an additional `A^m H_θ` skip merges post-encoding global propagation into the final embedding.

## Method (verified from PDF §3)

**Training pipeline (§3.4):**
1. **Augmentation**: feature masking `X → X̂` on the clean graph.
2. **Partial corruption** (§3.3): split `X̂` into two row partitions by parameter α ∈ (0,1); shuffle rows of each part independently to yield corrupted graphs `G̃1, G̃2`; merge back to form `X̃_p` (negative group features). The positive group features are `X̂_p = P(X̂, A)`.
3. **Message passing view** (§3.2, Eq. 3): `X_k = (A + I)^k X_0` — **a single fixed k**, chosen small for efficiency. Applied to X̂ and to the corrupted matrices to produce `X̂_p` and `X̃_p`.
4. **MLP encoder** (§3.4): shared-weight `f_θ` (primary MLP) + `p_θ` (projector, single-layer MLP) + `h_θ` (aggregation head with learnable linear parameters). Both positive `X̂_p` and negative `X̃_p` are fed to the same encoder.
5. **Loss (§3.4, Eq. 2)**: Binary Cross-Entropy on 2N-dim prediction vector (N nodes × 2 groups, labels {1 for positive, 0 for negative}).
   $$\mathcal{L}_\mathrm{BCE} = -\frac{1}{2N}\sum_{i=1}^{2N} \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right]$$

**Inference pipeline (§3.5, Eqs. 3-6):**
1. `X_k = (A+I)^k X_0` — message passing on raw features with the same fixed k.
2. `H_θ = f_θ(X_k)` — pass through trained MLP encoder (local embedding).
3. `H_global = A^m H_θ` with m=10 fixed — post-encoding propagation to aggregate multi-hop global info.
4. `H = H_global + H_θ` — final embedding is local + global skip.

## Architectural differentiators vs. D6c (verified)

| Axis | D6c (ours) | DGD |
|---|---|---|
| Contrast axis | **Same node at different depths (k vs k')** | **Real node (positive group) vs. corrupted node (negative group)** — partial-corruption-based |
| Loss family | **InfoNCE** (softmax over positives/negatives) | **BCE** (2N-dim binary classification) |
| Hops at training | **K=5 depths used as distinct views** | **Single fixed k** (chosen small) |
| Hops at inference | Same K depths | Single k, plus m=10 post-hoc global-embedding skip |
| Encoder | **None** (per-depth residual W_k) | **MLP stack** (`f_θ` primary MLP + `p_θ` projector + `h_θ` aggregation head) |
| Propagation order | Precompute `Â^k X`, then residual linear projection | `(A+I)^k X_0` → MLP (propagate-then-MLP, SGC-like) |
| Augmentation | **None** | **Feature masking + partial corruption** |

**The contrast axis is the fundamental difference.** D6c contrasts same-node-at-different-depths; DGD contrasts real-node-vs-corrupted-node. These are two different SSL pretext tasks.

## Experimental scope (from PDF)

Reported datasets (Table in §4): Cora, CiteSeer, PubMed, Amazon-Computers, Amazon-Photo. **DGD does report ogbn-arxiv scale (full-neighborhood sampling per author note §4).** Direct comparison on ogbn-arxiv is feasible and should happen in efficiency benchmark (INQ-008).

## Pre-emption risk: LOW (verified)

DGD shares D6c's "decouple propagation from learning + MLP-only" efficiency stance (a common ancestor), but the contrastive objective is entirely different:
1. **BCE not InfoNCE.** Different loss family, different optimization landscape.
2. **Node-vs-corrupted-node not hop-vs-hop.** DGD treats the node as the unit of contrast; D6c treats the depth as the view axis. The two methods ask different discriminative questions.
3. **Single-k not multi-k at training.** DGD's hops are a fixed-propagation hyperparameter, not a contrastive axis. The "multi-hop" aspect only appears at inference via the `A^m H_θ` skip.

DGD is a useful cite for "prior art on decoupled-GCN + SSL" and for the "MLP-only encoder + precomputed propagation" efficiency lineage. It is not a contrastive-axis competitor.

## What we need to do

- [x] Ingest the PDF (done 2026-04-24). All claims above verified against `raw/papers/DGD.txt`.
- [ ] Add row to [[Competitive Landscape 2026]] under "decoupled / SGC-adjacent SSL" group, alongside [[GGD]] and [[SUGRL]].
- [ ] Include in efficiency benchmark (INQ-008 Config B) if public code exists.
- [ ] Related-work paragraph: "DGD extends GGD with decoupled-GCN SSL and partial corruption. Like D6c, it is MLP-only over precomputed features; unlike D6c, it retains GGD's node-vs-corrupted-node BCE contrast rather than introducing a depth-as-view axis."

## Related wiki pages

- [[GGD]] — parent method; group-discrimination loss origin.
- [[MHVGCL]] — the actual medium-risk pre-emption; MHVGCL has the InfoNCE-over-hops loss that DGD lacks.
- [[Thesis]] — D6c method paper.
- [[Idea Ledger]] — D6c Live row; reviewer-attack 3 addresses DGD.
- [[Competitive Landscape 2026]] — needs DGD row (2026-04-24 follow-up).
