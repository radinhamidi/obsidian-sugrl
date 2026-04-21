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

| Method | Venue | Accuracy | Key idea |
|---|---|---|---|
| [[BGRL]] | ICLR 2022 | ~71.6 | BYOL-style bootstrap, no negatives |
| [[GraphMAE]] | KDD 2022 | ~71.7 | Masked feature reconstruction |
| [[GraphMAE2]] | WWW 2023 | ~72.7 | Improved masked reconstruction |
| [[GraphACL]] | NeurIPS 2023 | ~70 | Asymmetric contrastive, augmentation-free, heterophily |
| [[PolyGCL]] | ICLR 2024 | ~70.5 | Learnable polynomial spectral filters as views |

## Concurrent work to monitor

- **[[Less is More]]** (arxiv 2509.25742, ICLR 2026 submission). GCN + MLP as complementary views, no augmentation, no negatives. Focus: heterophily + robustness at small scale. **Does not run at OGB scale.** Closest architectural cousin to AD-SSL; our differentiation is large-scale efficiency + multi-depth adaptive views (vs their single GCN + single MLP).

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
