---
title: SGC
type: entity
kind: method
venue: ICML 2019
url: https://arxiv.org/abs/1902.07153
tags: [method, prior-art, decoupled]
created: 2026-04-21
updated: 2026-04-21
---

# SGC — Simplifying Graph Convolutional Networks

Wu et al., ICML 2019. Source of the [[Decoupled Precompute]] trick AD-SSL inherits.

## Key idea

Collapse a multi-layer GCN's weight matrices into a single logistic regression on `Â^k X`. The propagation can be precomputed once; training is then just a linear (or small MLP) classifier.

## Relevance to AD-SSL

- The precompute pattern (`A^k X` as features, MLP on top) is SGC's.
- [[SUGRL]] inherited this pattern for unsupervised use at k=1.
- AD-SSL extends it to **multiple k values** as parallel views.
- Important citation — directly predates our design.
