---
title: Competitive Landscape 2026
type: synthesis
tags: [neurips-2026, related-work, baselines]
created: 2026-04-21
updated: 2026-04-21
sources: [[RESEARCH_AGENT_ONBOARDING]]
---

# Competitive Landscape (as of 2026-04-21)

Snapshot of where unsupervised graph representation learning stands on the axes AD-SSL competes on: **accuracy**, **scalability / wall-clock**, **augmentation-free-ness**, **adaptive depth**.

## Efficiency champions

| Method | Venue | ogbn-arxiv time | Key idea |
|---|---|---|---|
| [[GGD]] | NeurIPS 2022 | ~0.18 s | Binary group discrimination (real vs shuffled), no contrastive loss |
| [[SUGRL]] | AAAI 2022 | ~6 s | MLP + shared-GCN, margin triplet, fixed k=1 propagation |

GGD is ~30× faster than SUGRL and scales to ogbn-papers100M.

## Accuracy leaders (ogbn-arxiv)

All numbers verified against source PDFs (2026-04-21 audit).

| Method | Venue | Accuracy | Key idea |
|---|---|---|---|
| [[BGRL]] | ICLR 2022 | 71.64 ± 0.12 | BYOL-style bootstrap, no negatives |
| [[GraphMAE]] | KDD 2022 | 71.87 ± 0.21 (GCN) / 71.75 ± 0.17 (GAT) | Masked feature reconstruction |
| [[GraphMAE2]] | WWW 2023 | 71.95 ± 0.08 | Improved masked reconstruction (Table 9, full-graph) |
| [[GraphACL]] | NeurIPS 2023 | 71.72 ± 0.26 | Asymmetric contrastive, augmentation-free, heterophily-aware |
| [[PolyGCL]] | ICLR 2024 | not reported on ogbn-arxiv | Learnable polynomial spectral filters; reports arXiv-year 43.07 (heterophilic label variant) |
| [[BLNN]] | arXiv 2024-08 | not reported on ogbn-arxiv | BGRL + neighbor-positive alignment; 5 small graphs only |

## Concurrent work to monitor

- **[[Less is More]]** (arxiv 2509.25742, ICLR 2026 submission). GCN + MLP as complementary views, no augmentation, no negatives. Focus: heterophily + robustness at small scale. **Does not run at OGB scale.** Closest architectural cousin to AD-SSL; our differentiation is large-scale efficiency + multi-depth adaptive views (vs their single GCN + single MLP).
- **[[BLNN]]** (arXiv 2408.05087, Aug 2024, Liu et al.). BGRL + 1-hop neighbor-positive alignment with attention supportiveness. Small-graph evaluation only (WikiCS, Photo, Computer, CS, Physics). Orthogonal axis to AD-SSL — BLNN enriches positives in the *spatial* direction (1-hop neighbors), AD-SSL enriches in the *depth* direction (multi-hop precomputed views). Cite in related work; not a preempt.
- **[[GRAPHITE]]** (ICLR 2026, arXiv 2602.07256). Supervised heterophily preprocessor via feature nodes. **Out of AD-SSL scope** per [[Thesis]] § Scope; noted for citation completeness only.
- **[[GSTBench]]** (CIKM 2025, arXiv 2509.06975). Cross-dataset transferability benchmark — finds only [[GraphMAE2]]-style reconstruction transfers at papers100M scale; contrastive/bootstrap methods fail. We operate outside this regime (per-dataset training).

## Supervised adaptive-depth work (conceptual priors)

These are adaptive-depth methods but all supervised — none has been adapted to unsupervised GRL:

- [[GPRGNN]] (ICLR 2021) — learned polynomial propagation coefficients.
- [[APPNP]] (ICLR 2019) — teleport-based propagation.
- [[ATP]] (2024) — node-wise adaptive propagation.

## The empty Pareto point

No published unsupervised method simultaneously:

1. Matches BGRL/GraphMAE accuracy (~71 on ogbn-arxiv).
2. Runs at GGD-level wall-clock (seconds on ogbn-arxiv).
3. Adapts propagation depth per node rather than fixing k as a hyperparameter.

See [[Pareto Gap]].

## Monitoring cadence

- **Weekly arxiv scan** for new submissions in graph SSL, scalable GNNs, unsupervised GRL.
- Flag on: multi-depth contrastive views, same Pareto claim, direct extensions of [[SUGRL]] / [[GGD]] / [[BGRL]].
- Track acceptance of [[Less is More]] at ICLR 2026 — accepted → must cite and differentiate explicitly; rejected → cite as arxiv with weaker framing.

See [[log]] for literature-scan entries.
