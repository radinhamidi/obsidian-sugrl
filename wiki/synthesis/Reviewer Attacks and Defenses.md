---
title: Reviewer Attacks and Defenses
type: synthesis
tags: [neurips-2026, reviewers, risk]
created: 2026-04-21
updated: 2026-04-21
sources: [[RESEARCH_AGENT_ONBOARDING]]
---

# Reviewer Attacks and Defenses

Anticipated reviewer objections and our prepared answers. Each row lists the
required evidence — missing evidence is a submission blocker.

| Reviewer says | Our answer | Evidence needed |
|---|---|---|
| "This is just [[GPRGNN]] + [[BGRL]]." | GPRGNN is supervised and coupled (end-to-end). BGRL does not use multi-depth views. | Ablation: remove depth views → collapses to [[GGD]]-ish baseline. Remove bootstrap → collapses to [[SUGRL]]. |
| "Why not just sweep k per dataset?" | Learned per-node α_k must beat best fixed k without sweeping. | Per-dataset comparison: learned-α vs best-fixed-k (depth sweep). |
| "You don't beat BGRL/GraphMAE." | We don't claim SOTA accuracy — we claim a Pareto point: comparable accuracy at 10–100× lower cost. | The Pareto figure (see [[Pareto Gap]]). |
| "[[Less is More]] already did this." | Different mechanism (2 views — 1 MLP + 1 k-layer GCN — with global β weighting + direct cosine loss vs our K multi-depth views + per-node α + bootstrap loss; see [[AD-SSL vs Less is More]]). **Correction to onboarding framing**: they do evaluate at 169k-node scale (Arxiv-year), not just small graphs. Our scale advantage is evaluating on standard OGB category-prediction, which they don't. | Reproduce GCN-MLP at ogbn-arxiv + ogbn-products in our harness (Coding Agent). 2-view ablation of AD-SSL to isolate multi-depth contribution. Scale to ogbn-papers100M (they don't). |
| "This is just [[ATP]] ported to SSL." | ATP tunes a **scalar** per-node kernel coefficient r̃ from closed-form degree/eigenvector/clustering (weight-free, supervised). AD-SSL learns a **K-dim** per-node mixture over K depth views from content-based cross-depth consistency. Different mechanism, different output, different regime. See [[ATP]] §"Differences from AD-SSL". | Ablation: AD-SSL with uniform α_k (no per-node weighting) vs ATP's r̃-kernel as a pre-processor. Explicit differentiation paragraph in related work. |
| "6 ideas failed in preliminary — why will this work?" | Those were sampling tweaks on a fixed-depth encoder. The one thing that **did** work was changing depth ([[Prepropx Depth Finding]]: +0.80). AD-SSL extends exactly that insight. | The prepropx table in the appendix. |
| "Only tested on homophilic graphs." | Include at least one heterophilic benchmark (Chameleon or Squirrel). | Results on heterophilic graph in main table. |
| "Benchmarks cherry-picked / dated." | We use OGB + established homophilic/heterophilic sets + scale study. | Main table + scale study + appendix per-dataset ablations. |
| "Missing 2024–2025 baselines." | Full landscape in related work; reproduced or reported-in-table. | [[Competitive Landscape 2026]] expanded into a related-work section with all recent methods. |

## NeurIPS 2026 norms (signals from recent ICLR/NeurIPS discourse)

Reviewers increasingly value:

- Rigorous ablations (one-knob-at-a-time).
- Honest failure reporting — include what didn't work (we have this in [[Preliminary Validation - 168 Runs]]).
- Theoretical motivation, even lightweight (spectral-filter interpretation of depth-bootstrapping).
- Scaling studies at OGB / OGB-LSC scale.
- Reproducibility.

Reviewers increasingly dislike:

- Cherry-picked benchmarks.
- Missing recent baselines (2024–2025).
- Vague novelty claims.
- Methods that only work on Cora/CiteSeer.

Calibration reference: [Graph Learning Will Lose Relevance Due To Poor Benchmarks](https://arxiv.org/pdf/2502.14546) (ICLR 2025 position paper). Not a competitor — a **thermometer** for the current review climate.

## Current defensive gaps (to fix before submission)

- [ ] Heterophilic benchmark not yet in plan — add (Chameleon-filtered / Squirrel-filtered / Wisconsin).
- [ ] Reproduction of [[Less is More]] at **standard ogbn-arxiv and ogbn-products** — not started. Their paper only has Arxiv-year at that scale.
- [ ] 2-view ablation of AD-SSL (isolating multi-depth contribution from framework) — not planned yet.
- [ ] Scale study to ogbn-papers100M — depends on Coding Agent cluster availability.
- [ ] Spectral-filter interpretation of depth-bootstrapping — needs derivation.
- [ ] ATP's HPC (High-Deg edge masking) — should we adopt as preprocessing? Would strengthen scale claims and match their framework. Small appendix ablation.
