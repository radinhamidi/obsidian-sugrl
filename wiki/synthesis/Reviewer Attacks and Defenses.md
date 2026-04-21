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
| "[[Less is More]] already did this." | Different focus (heterophily + robustness at small scale vs our efficiency at OGB scale), different mechanism (single GCN+MLP vs multi-depth views with adaptive weighting). | Reproduce Less-is-More at OGB scale as a baseline; show AD-SSL outperforms on the efficiency axis. |
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

- [ ] Heterophilic benchmark not yet in plan — add.
- [ ] Reproduction of [[Less is More]] at OGB scale — not started.
- [ ] Scale study to ogbn-papers100M — depends on Coding Agent cluster availability.
- [ ] Spectral-filter interpretation of depth-bootstrapping — needs derivation.
