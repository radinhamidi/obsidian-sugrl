---
title: GRAPHITE
type: entity
kind: method
venue: ICLR 2026 (arXiv:2602.07256, 2026-02-06)
authors: [Qiu, Li, Li, Tong]
affiliation: UIUC
url: https://arxiv.org/abs/2602.07256
code: https://github.com/q-rz/ICLR26-GRAPHITE
tags: [method, preprocessor, heterophily, graph-transformation]
created: 2026-04-21
updated: 2026-04-21
---

# GRAPHITE — Graph Homophily Booster

Qiu, Li, Li, Tong (UIUC). ICLR 2026. **Graph preprocessor**, not an SSL method. Transforms a heterophilic graph into a more-homophilic one by inserting **feature nodes** connected to graph nodes that share similar features. Then any standard (homophilic-biased) GNN is trained on the transformed graph.

## Mechanism (§3)

For a discrete feature dimension d, create one "feature node" per distinct value. Connect every original graph node to the feature nodes corresponding to its active features. This creates a bipartite feature–node augmentation of the original graph. Message passing over the union graph now routes information between originally-distant nodes that share features — increasing effective homophily.

Claim: only "a slight increase in graph size" and provably higher homophily than the original graph.

## Scope

- **Heterophilic focus**: Actor, Squirrel-F, Chameleon-F, Minesweeper (Platonov 2023 filtered variants).
- Homophilic sanity checks: Cora, CiteSeer.
- **No ogbn-arxiv / ogbn-products / papers100M.**
- Not self-supervised; assumes labels.

## Relevance to AD-SSL

**Low.** GRAPHITE is:
1. A **preprocessor** (runs once before training), not a learning method.
2. Designed for **heterophilic graphs**, which AD-SSL has scoped out (see [[Thesis]] § Scope).
3. **Supervised**, not SSL.

No overlap in mechanism, scope, or training regime. Not a baseline, not a preempt, not a reviewer attack vector.

Worth flagging only in two scenarios:
- If we later rescope AD-SSL to include heterophily, GRAPHITE is the current ICLR-2026 SOTA on Actor/Squirrel-F/Chameleon-F and would become the reference result to beat (or a preprocessor to stack).
- As a citation in the "heterophily handling" paragraph of related work to acknowledge the ICLR 2026 landscape.

## Relevant links

- [[PolyGCL]] — SSL method explicitly engineered for heterophily (spectral high-pass). The SSL-heterophily analog of what GRAPHITE does supervised.
- [[GraphACL]] — SSL heterophily via two-hop monophily.
