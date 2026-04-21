---
title: GGD
type: entity
kind: method
venue: NeurIPS 2022
url: https://arxiv.org/abs/2206.01535
tags: [method, baseline, efficiency]
created: 2026-04-21
updated: 2026-04-21
---

# GGD — Graph Group Discrimination

Zheng et al., NeurIPS 2022. **Efficiency champion** on ogbn-arxiv.

## Key idea

Binary group discrimination: real node embeddings vs shuffled-feature node embeddings. No contrastive loss, no positive-pair construction, no multi-view augmentation.

## Why it's fast

- Just a forward pass + a binary classifier on shuffled vs real.
- No augmentation.
- No negatives to sample.
- ogbn-arxiv time: ~0.18 s.
- Scales to ogbn-papers100M (reportedly the first graph SSL method to do so at reasonable cost).

## Accuracy

- ogbn-arxiv: competitive but below BGRL/GraphMAE (expected — less supervisory signal).

## Role in AD-SSL

- **Cost target**: we want to match GGD's wall-clock (seconds, not minutes) on ogbn-arxiv.
- Baseline in main results table and Pareto figure.
- Needs reproduction in our env during Phase 1. See [[Project Phases and Decision Gates]].

## Open items

- [ ] Ingest the GGD paper PDF when available in `raw/papers/`. Update this page with the exact loss formulation and their OGB numbers.
