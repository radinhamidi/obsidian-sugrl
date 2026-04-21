---
title: GSTBench — Transferability Benchmark for Graph SSL
type: source
tags: [benchmark, transferability, calibration, evaluation-integrity, foundation-models]
created: 2026-04-21
updated: 2026-04-21
authors: [Song, Hua, Xie, Liu, Long, Liu]
venue: CIKM 2025 (arXiv 2509.06975, 2025-08-28)
---

# GSTBench — Transferability Benchmark for Graph SSL

Song, Hua, Xie, Liu, Long, Liu (Michigan State + Meta Monetization AI). CIKM 2025. **First systematic benchmark for cross-dataset transferability of graph SSL**. Directly relevant to whether AD-SSL claims can extend beyond per-dataset training.

## Setup

- **Pretrain on ogbn-papers100M** (111M nodes, 1.6B edges), partitioned via METIS into ~10k subgraphs for mini-batch.
- **Downstream**: 8 datasets, 5 in-domain citation (Cora, Citeseer, WikiCS, DBLP, Pubmed) + 3 cross-domain e-commerce (Amazon Ratings, Child, Photo).
- **Feature harmonisation**: SentenceBERT embeddings for all nodes — a shared semantic space across graphs (critical for transfer).
- **Five pretraining objectives**: GraphMAE, VGAE, DGI, GRACE, Link Prediction.
- **Three adaptation strategies**: linear probe, fine-tune, in-context (few-shot: 5 train nodes/class).
- **Backbones**: standard GCN and GAT.

## Main findings (§4)

1. **Most SSL methods fail to transfer.** GRACE, DGI, VGAE, LP produce unstable or *negative* transfer (below random-init baseline) on downstream datasets.
2. **GraphMAE is the exception.** Masked feature reconstruction shows consistent downstream gains across all 8 target graphs, both in-domain and cross-domain. SSL loss monotonically decreases with pretraining → downstream accuracy goes up.
3. **Random-init is already strong** when features are LLM-derived (Cora ~74% with random-init GNN + SentenceBERT features). The bar for "pretraining helps" is much higher than it looks in weaker-feature settings.
4. **Contrastive methods over-rely on augmentations** that don't survive domain shift.
5. **Spearman correlation (GraphMAE, Fig 2)**: SSL error rank and downstream accuracy rank correlate at ρ = −0.39 (GCN) / −0.43 (GAT), p < 0.01 — pretext-task fitness does predict transfer, but loosely.

## Relevance to AD-SSL

This paper is **concrete evidence for our [[GraphMAE2]] Papers100M contradiction flag**:
- GSTBench confirms contrastive/BYOL-style methods (GRACE, DGI, and by extension BGRL) do not transfer at papers100M pretraining scale.
- GraphMAE (and by extension GraphMAE2) is the only SSL family with reliable transfer.
- **AD-SSL is bootstrap-style (contrastive cousin).** If we want to claim transferability or large-scale pretraining benefits, we are swimming against this result.

## Options for AD-SSL framing

1. **Scope claim to single-dataset training.** "AD-SSL is a fast, accurate Pareto point when trained per-dataset" — no transferability claims. Safest.
2. **Test transferability ourselves.** Pretrain AD-SSL on one OGB graph, linear-probe on another. If it works, strong result. If not, confirms contrastive-family limitation.
3. **Hybrid with a reconstruction term.** GSTBench suggests adding a feature-reconstruction component to AD-SSL's bootstrap loss may help transfer. See also [[GraphMAE2]]'s finding that feature reconstruction is the load-bearing signal.

## Reviewer-defence implications

Reviewers who know this paper will push back on any "large-scale pretraining" or "foundation model" framing of AD-SSL. We should **not** position AD-SSL as a foundation-model effort without a transfer study. The Pareto framing (per-dataset cost × accuracy) is safer and well-motivated.

## Concrete takeaways for evaluation design

- **Include linear probing with 5-shot splits** (GSTBench protocol) if we want to be read as a "foundation-model-aware" paper.
- **Use LLM-derived features** for any transfer experiments (bag-of-words is out of date).
- **Report SSL loss curves alongside validation accuracy** — the Fig 1 plot structure (SSL error vs epoch vs accuracy) is a useful diagnostic we should adopt for AD-SSL collapse monitoring.

## Code

`https://github.com/SongYYYY/GSTBench`
