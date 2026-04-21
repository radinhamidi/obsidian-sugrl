---
title: Graph Learning Will Lose Relevance Due to Poor Benchmarks
type: source
tags: [position-paper, benchmarking, calibration, evaluation-integrity]
created: 2026-04-21
updated: 2026-04-21
authors: [Bechler-Speicher, Finkelshtein, Frasca, Müller, Tönshoff, Siraudin, Zaverkin, Bronstein, Niepert, Perozzi, Galkin, Morris]
venue: arXiv 2025-02-20 (2502.14546)
---

# Position paper — Poor Benchmarks

A multi-institution (Tel-Aviv, Oxford, Technion, RWTH, NEC, Stuttgart, Google, RWTH) position paper arguing the field is stagnating because benchmarks are narrow, splits/protocols are fragmented, and marginal gains are reported without statistical significance.

## Key claims (§1, Fig. 1)

1. **Lack of transformative real-world applications.** Overfocus on 2D molecular graphs and academic citation networks; combinatorial optimization, relational DBs, chip design are neglected.
2. **Datasets poorly represent underlying data** — no 3D geometric structure for molecules, manufactured synthetic graphs (ZINC) with no real-world use.
3. **Fragmented evaluation** — inconsistent splits, high-variance small-dataset results, no statistical significance testing.
4. **OGB parameter-count ceilings (500k)** are out of step with modern scaling laws.
5. **No true graph foundation model** exists; pipelines are bespoke case studies.

## Actionable recommendations (relevant to our paper)

- **When proposing new benchmarks, discuss the advantages of graph structure** (i.e., show a non-graph baseline beats/loses).
- **Include unstructured-set baselines** to demonstrate graph structure matters.
- **Shift focus beyond accuracy**: include cost, scalability, transferability, robustness.
- **Report statistical significance**, not marginal gains.

## Relevance to AD-SSL

- **Accuracy-alone claims are dead.** Our [[Pareto Gap]] framing (accuracy × wall-clock) is exactly what this paper advocates. Keep the Pareto frame front-and-center.
- **Include MLP/SGC baselines** in our result tables (they already are in Phase 1). Do not let AD-SSL results stand alone.
- **Report 95% CIs and paired t-tests** (following [[APPNP]]'s protocol). The position paper explicitly condemns "marginal gains without statistical significance" — our ±0.8 on ogbn-arxiv needs rigorous stat testing.
- **Do not over-claim ogbn-arxiv as transformative.** The paper is dismissive of citation networks. We should treat ogbn-arxiv as a controlled, standard testbed rather than a mission-critical application.
- **Scope scaling claims carefully.** Our onboarding mentioned ogbn-papers100M as an aspirational target — this paper's framing agrees that Papers100M is a more meaningful test than smaller OGB graphs.

## Reviewer-defence implications

Reviewers aligned with this paper (Bronstein, Galkin, Morris are prominent reviewers in graph learning) will penalise:
- Marginal (<1 point) accuracy gains without CIs and statistical tests.
- Results only on Cora/Citeseer/PubMed.
- Papers that don't compare to non-graph baselines.
- Claims of "foundation models" without transferability evidence.

We should proactively address all four in the AD-SSL paper. See [[Reviewer Attacks and Defenses]].
