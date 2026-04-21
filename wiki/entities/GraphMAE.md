---
title: GraphMAE
type: entity
kind: method
venue: KDD 2022
url: https://arxiv.org/abs/2205.10803
tags: [method, baseline, generative]
created: 2026-04-21
updated: 2026-04-21
---

# GraphMAE

Hou et al., KDD 2022. Masked feature reconstruction as generative SSL on graphs. Close to [[BGRL]] on accuracy.

## Key idea

- Mask a fraction of node features.
- Encoder + decoder try to reconstruct masked features.
- Scaled cosine error loss.

## Accuracy on ogbn-arxiv

~71.7 ± 0.3.

## Role in AD-SSL

- Accuracy ceiling baseline. Pareto figure competitor.
- Ingest when PDF available.
