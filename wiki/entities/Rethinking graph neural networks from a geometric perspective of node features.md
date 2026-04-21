---
title: Rethinking graph neural networks from a geometric perspective of node features
type: entity
kind: method
venue: ICLR 2025
authors: [Feng Ji, Yanan Zhao, Kai Zhao, Hanyang Meng, Jielong Yang, Wee Peng Tay]
affiliation: NTU Singapore (Tay group) + Jiangnan University
url: https://openreview.net/forum?id=lBMRmw59Lk
tags: [method, theoretical, geometry, oversmoothing, heterophily, supervised]
created: 2026-04-21
updated: 2026-04-21
---

# Rethinking GNNs From A Geometric Perspective Of Node Features (Ji et al., ICLR 2025)

Feng Ji, Yanan Zhao, Kai Zhao, Hanyang Meng, Jielong Yang, Wee Peng Tay — NTU Singapore + Jiangnan University. ICLR 2025. **Theoretical paper** analyzing GNN behavior through a feature-centric lens (the "feature centroid simplex") rather than a topology-centric one. Same last author (Tay) as [[Less is More]]; cited there for the Prop. 2 noise argument.

## Core construction (§2)

For each label class `c`, compute the centroid `g_c = (1/n_c) Σ_{v ∈ D_c} x_v`. The convex hull of `{g_c | c ∈ C}` is the **feature centroid simplex** `Δ_g` (or its approximation `Δ_e` over expected features). Tools from **coarse geometry** (quasi-isometric embeddings) compare `Δ_g` with two canonical models:

- **Regular simplex** — all vertices equidistant → feature aggregation pulls a node's representation near its class centroid with high probability (Theorem 1a).
- **Degenerate simplex** — some vertices collapse (`u_{c_i} = u_{c_j}`) → two classes indistinguishable in feature space regardless of aggregation (Corollary 1).

Whether `Δ_e` is close to regular vs degenerate is an **intrinsic, graph-independent** property of features. This is the paper's main claim: some datasets (e.g. **Actor**) have near-degenerate simplexes, and **no aggregation mechanism can fix that** — the poor headline numbers on those datasets reflect a feature-space limit, not a GNN design flaw.

## Explanations of GNN phenomena (§4)

- **Homophily vs heterophily.** Homophilic: `e_{N_v} ≈ e_c`, aggregated feature close to class centroid → GCN works. Heterophilic: `d_{v,c}` large, aggregated feature bounded away from centroid → GCN fits to wrong vertex in probability simplex and can underperform vanilla MLP (Yan et al. 2022 replicated).
- **Oversmoothing** reinterpreted as iterated contraction of `Δ_e^{(t)} = conv({e_c^{(t)}})`: with mixing matrix `α_{c,c'}` summarizing cross-class neighbor proportions, `e_c^{(t+1)} = Σ_{c'} α_{c,c'} e_{c'}^{(t)}`. Markov chain contracts `Δ_e^{(t)}` to a single point as `t → ∞`. Geometric visualization of a result already known spectrally (Oono & Suzuki 2020).
- **Feature re-shuffling.** On datasets with regular `Δ_e` (Cora, CiteSeer), randomly re-shuffling features between same-class nodes *improves* GCN accuracy — edges within a class carry redundant feature information, so breaking that redundancy reduces variance. Supports Lee et al. 2024.

## Practical tricks (§5)

Denoted **-AE** ("almost extra") and model-agnostic:

1. **Random intra-class edge insertion.** During training, with probability η add edges between training nodes of the same class. Motivation: Theorem 1a — denser same-class neighborhoods pull aggregated features closer to `e_c`. Graph-structure modification; requires labels (supervised).
2. **Very early stopping at small E.** Train for `E` epochs where `E` is a small hyperparameter. Motivation: Theorem 1b — on heterophilic nodes, `y_v` drifts away from `e_{c_v}` during training; fitting harder means more test error. Stop before overfitting the aggregated-feature → probability-simplex map.

Additional trick **-AEN** ("AE + Normalization") — L2-normalize each feature vector to norm 1. Motivation: nodes with small-norm features distort the shape of `Δ_g` toward degeneracy; normalization regularizes the shape.

## Experimental setup

Supervised node classification on 6 heterophilic datasets: Texas, Cornell, Wisconsin, Chameleon, Squirrel, Actor. Homophilic (Cora, CiteSeer) and ogbn-arxiv results in Appendix F.1.

| Base model | -AE / -AEN behavior |
|---|---|
| GCN (-I, -II) | -AE adds 2–10 pts on heterophilic |
| GAT (-I, -II) | -AE adds 1–7 pts |
| ACM-GCN | best base model; ACM-GCN-AEN adds +38.9% on **Actor** (34.28 → 47.62) |
| GraphCON, CDE, GloGNN | small gains |

Feature normalization -AEN is **"very effective"** on ogbn-arxiv (their Appendix F.1 Table 10) — concrete number not captured here; check if needed.

## Relevance to AD-SSL

**Theoretical prior / related work citation, not a baseline.**

- **Cited by [[Less is More]] for Prop. 2** (noise argument underpinning their GCN+MLP view construction). Since Less is More is our closest concurrent work, this theoretical parent deserves a sentence in related work — same NTU group, same mathematical foundation.
- **Orthogonal lens for oversmoothing.** Their geometric framing (simplex contraction as Markov-chain mixing) is *complementary* to the spectral framing we plan to use. Could cite as "geometric interpretations (Ji et al. 2025) converge to the same conclusion from a different direction" — costs one sentence, strengthens the spectral paragraph.
- **Degenerate-simplex ceiling on Actor.** Reinforces our scope decision: heterophilic datasets have *intrinsic* feature-space limits independent of architecture, so AD-SSL's monotone-low-pass depth views cannot address them without retrofitting. Validates the out-of-scope framing in [[Thesis]] § Scope.
- **Early-stopping trick (Trick 2) is supervised.** Uses test-label-leaked validation; doesn't map directly onto our SSL early-stopping protocol ([[Splits and Protocol]]). But the underlying intuition — "stop before the model fits aggregated features to the wrong vertex" — is a useful reviewer talking point if questioned about why we stop early at all.

**Not a baseline.** Supervised, theoretical, heterophily-focused. No SSL comparison.

## Differences with AD-SSL

| Axis | This paper | AD-SSL |
|---|---|---|
| Regime | Supervised node classification | Unsupervised SSL |
| Contribution | Theoretical framework + model-agnostic tricks | New SSL method |
| Mechanism | Feature centroid simplex analysis + edge insertion + very-early-stopping + L2 normalization | Multi-depth precomputed views + per-node α mixing + bootstrap loss |
| Datasets | Heterophilic (+ homophilic/arxiv in appendix) | Homophilic only by scope |
| Per-node reasoning | No (class-level aggregates) | Yes (α_{i,k} per node) |

## Citation target paragraphs

- Related work — "Theoretical foundations": cite alongside Oono & Suzuki 2020 for oversmoothing.
- Related work — "Concurrent Tay-group work": cite together with [[Less is More]].
- Scope justification — cite Corollary 1 + Actor analysis as evidence that heterophilic benchmarks hit intrinsic feature-space limits, supporting our out-of-scope decision for v1.

## Relevant links

- [[Less is More]] — concurrent work by the same group; cites this for Prop. 2.
- [[AD-SSL vs Less is More]] — mentions this theoretical prior.
- [[Oversmoothing]] — this paper provides a geometric interpretation.
