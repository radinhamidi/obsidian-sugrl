---
title: ogbn-arxiv
type: entity
kind: dataset
tags: [dataset, ogb, homophilic, headline]
created: 2026-04-21
updated: 2026-04-21
---

# ogbn-arxiv

The **headline dataset** for AD-SSL — the decision-gate dataset in [[Project Phases and Decision Gates]].

## Stats

- Nodes: 169,343
- Edges: 1,166,243
- Task: node classification (40 classes, arxiv CS categories).
- Time-based train/val/test split (pre-2017 / 2018 / 2019+).

## Why it's the headline

- Moderate scale (large enough that sub-quadratic methods matter, small enough that full-batch is feasible).
- Time-based split stresses generalization more than random splits.
- Established unsupervised benchmark — many baselines have numbers here.
- Where the [[Pareto Gap]] is most visible.

## SUGRL reproductions

- Published SUGRL ([[SUGRL]]): 68.8 ± 0.4
- Our repro (k=1): 68.77 ± 0.13
- SUGRL-k=3 (prepropx2): 69.57 ± 0.05 — see [[Prepropx Depth Finding]]

## Competitive numbers (accuracy, 2026-04-21)

| Method | Accuracy | Cost class |
|---|---:|---|
| [[GGD]] | ? | < 1 s |
| [[SUGRL]] (k=1 / k=3) | 68.77 / 69.57 | < 1 s |
| [[GraphACL]] | ~70 | sec–min |
| [[PolyGCL]] | ~70.5 | sec–min |
| [[BGRL]] | ~71.6 | sec–min |
| [[GraphMAE]] | ~71.7 | sec–min |
| [[GraphMAE2]] | ~72.7 | sec–min |

AD-SSL target: **≥ 71 at < 60 s.**
