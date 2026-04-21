---
title: PolyGCL
type: entity
kind: method
venue: ICLR 2024
url: https://proceedings.iclr.cc/paper_files/paper/2024/file/6faf3b8ed0df532c14d0fc009e451b6d-Paper-Conference.pdf
tags: [method, baseline, spectral]
created: 2026-04-21
updated: 2026-04-21
---

# PolyGCL — Polynomial Graph Contrastive Learning

ICLR 2024. Learnable polynomial spectral filters as contrastive views.

## Relevance to AD-SSL

- Uses spectral views (polynomial filters) as contrast. AD-SSL uses depth-varying smoothed features, which is a specific low-pass polynomial. The spectral-filter interpretation of AD-SSL (depth k ↔ low-pass with bandwidth ~1/k) is relevant prior art — needs careful differentiation in related work.
- Accuracy ~70.5 on ogbn-arxiv.
- One of the closest conceptual cousins — reviewers may ask for comparison.

Ingest when PDF available.
