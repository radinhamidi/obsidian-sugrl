---
title: ATP
type: entity
kind: method
venue: WWW 2024
authors: Xunkai Li, Jingyuan Ma, Zhengyu Wu, Daohan Su, Wentao Zhang, Rong-Hua Li, Guoren Wang
url: https://arxiv.org/abs/2402.06128
tags: [method, prior-art, adaptive-depth, supervised, ingested]
created: 2026-04-21
updated: 2026-04-21
ingested_from: raw/papers/ATP - Node-wise Adaptive Propagation.pdf
---

# ATP — Adaptive Topology-aware Propagation

Li et al., WWW 2024. **Plug-and-play node-wise propagation optimization strategy** for scalable GNNs. Closest supervised analogue to AD-SSL's per-node adaptive depth weighting; critical for novelty defense.

## Core mechanism

Two components, both applied as **pre-processing** (weight-free, offline, orthogonal to the downstream model):

### 1. High-bias Propagation Correction (HPC)

- Rank nodes by degree. Top-θ% (paper suggests 10–15%) are **High-Deg**.
- Mask a fraction of their one-hop edges with a mask token `[MASK]`.
- Justification via Thm. 1: convergence rate of k-step propagation scales as `√((2m+n)/d̃_i) · λ₂^k`. High-Deg with large d̃_i converge slowly and contribute most to oversmoothing / high-bias aggregation.
- Side effect: reduces computational cost.

### 2. Weight-free LNC (Local Node Context) encoding

Produces a **per-node scalar** r̃_i that gets plugged into the propagation kernel `D̂^(r̃−1) Â D̂^(−r̃)`:

`R̃ = C · (R_dg + R_ev + R_cu)`

- **R_dg** = Diag(d_i / (n−1)) — degree-based centrality.
- **R_ev** = top eigenvector of optimized adjacency (R0) scaled by 1/λ_max — spectral position encoding. Can be skipped for efficiency (→ "ATP/Eigen") at small accuracy cost.
- **R_cu** = local clustering coefficient — neighborhood connectivity.

Then:

`Π̃ = Σᵢ w_i · (D̂^(R̃−1) Â D̂^(−R̃))^i`

This becomes the **propagation operator** in any decoupled or message-passing GNN.

## Key properties

- **Supervised** (node classification). The r̃ optimization is weight-free but downstream training is standard supervised.
- Plug-and-play: orthogonal to the downstream GNN and orthogonal to other NP strategies (DGC, NDLS, NDM, SCARA).
- Scales to ogbn-papers100M with HPC enabling previously-OOM baselines (e.g., Cluster-GCN).
- Consistent +2–5 point gains across 12 benchmarks (Tables 1–2).

## What ATP does NOT do

- **No multi-depth views.** ATP tunes a single scalar r̃ per node in one propagation kernel; it does not maintain K separate depth representations.
- **No SSL / no contrastive loss.** Fully supervised.
- **No bootstrap / no EMA.** Pure pre-processing optimization.
- **No per-node-per-depth weighting.** The coefficient r̃ controls the shape of a single kernel, not a mixture over depths.

## Differences from AD-SSL (critical for reviewer defense)

| Axis | ATP | AD-SSL |
|---|---|---|
| Supervision | supervised | unsupervised (SSL) |
| Per-node knob | scalar r̃ in one kernel | K-dim α_{i,k} over K depth views |
| Mechanism | weight-free from degree/eigenvec/clustering | cross-depth consistency scoring (content-aware) |
| Training signal | cross-entropy on labels | bootstrap cosine across depth pairs |
| Output | one propagated `Π̃ X` | K view embeddings Z_k then weighted mix |

Reviewer attack "this is just ATP in SSL" has a clean answer: (1) weight-free closed-form encoding vs content-aware consistency scoring; (2) single-kernel vs multi-view; (3) supervised vs SSL loss. See [[Reviewer Attacks and Defenses]].

## Overlap worth acknowledging

- ATP's HPC (mask high-degree edges) is a clean, cheap **preprocessing trick**. AD-SSL could adopt it as an orthogonal improvement — they stack in their own ablations (SIGN+SCARA+ATP). Open question: do we apply HPC in our preprocessing too? Could be a small appendix ablation.
- Eigenvector centrality as a position encoding is a general graph-learning primitive; not an ATP-exclusive contribution.

## Benchmarks used

Cora, CiteSeer, PubMed, Photo, Computer, CS, Physics, ogbn-arxiv, ogbn-products, **ogbn-papers100M**, Flickr, Reddit. Strong OGB-scale evidence. Our own experiment suite should cover overlapping benchmarks.
