---
title: Ablation Plan - AD-SSL B0 A1-A4
type: experiment
tags: [neurips-2026, ablation, ad-ssl, planned]
created: 2026-04-21
updated: 2026-04-21
status: planned
owner: Coding Agent
---

# AD-SSL Ablation Plan

Phase 2 experiment, currently **not started** (awaiting Phase 0 hygiene + Phase 1 baseline ladder). See [[Project Phases and Decision Gates]].

## Configurations

| Config | Weighting | Loss | Refinement |
|---|---|---|---|
| **B0** | uniform across depths | bootstrap cosine | none |
| **A1** (GRPO-style) | per-node cross-depth consistency | bootstrap cosine | none |
| **A2** (KTO-style) | uniform | asymmetric (kNN↔graph-nbrs signal) | none |
| **A3** (SimPO-style) | uniform | MSE or InfoNCE (sweep) | none |
| **A4** (Online-DPO-style) | EMA-smoothed depth preferences | bootstrap cosine | iterative |
| A1+A3, A1+A4, ... | combos | — | — |

Internal names are internal. See [[Thesis]] §"Four insights". Do not surface in paper.

**On the GRPO / KTO / SimPO / Online-DPO tags.** These are **RL-alignment analogies from the brainstorming session** — scaffolding used to import structural patterns (group-relative ranking, binary preference, simplified objective, iterative EMA preference) into the graph-SSL setting. The graph mechanism in each row stands on its own; the analogy does not go in the paper, and Coding Agent should treat the "Weighting / Loss / Refinement" columns as the authoritative spec.

## Datasets

Phase 2 core: [[ogbn-arxiv]] (the decision-gate dataset).
If gate passes, expand to [[Cora]], [[CiteSeer]], [[PubMed]], [[Photo]], [[Computers]], plus [[ogbn-products]] and (budget permitting) [[ogbn-papers100M]] for the scale study.

## Depth set

k ∈ {1, 2, 4, 8}. Rationale: exponential spacing covers the useful U-curve region identified in [[Prepropx Depth Finding]] without redundant adjacent values. May need to trim if k=8 consistently gets zero weight.

## Seeds

3 seeds for ablation screening. Scale to 5 seeds for final headline table.

## Decision criterion per config

Significance per [[Matched-Seed Delta]]: ROBUST = mean(Δ) > 0.3 AND 3/3 seeds positive (vs B0).

## Gate 1 test

After B0 + A1..A4 individually complete:
- B0 + best insight **≥ 71** on ogbn-arxiv at **< 60 s** → PASS → proceed to full experiments.
- B0 + all insights **< 70** → FAIL → MLP encoder is bottleneck → pivot discussion with Radin.

## Expected diagnostics in results JSON

Coding Agent should report per run:
- `acc_mean`, `acc_std`, per-seed accs.
- `embedding_std` (collapse detector).
- `weight_entropy` per depth (is the model actually using multiple depths, or collapsing to one?).
- Training curves (loss, linear-probe acc per 10 epochs).
- Wall-clock for training + precompute separately.

## Open questions (to raise as inquiries if still open when work starts)

- Encoder width/depth for MLP? Default: mirror SUGRL [512, 128].
- EMA decay for target network? Default: 0.99 → 0.999 linear warmup like BGRL.
- Temperature in A1's softmax over depth-weights?

## Outputs expected in the repo

- `results/ablation/SUMMARY.md` — headline table + verdicts.
- `results/ablation/<config>/<dataset>/seed_<s>/metrics.json` — per-run raw.
- Analysis memo from me once SUMMARY.md lands.
