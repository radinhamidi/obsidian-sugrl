---
title: Pareto Gap - Fast-vs-Accurate Unsupervised GRL
type: synthesis
tags: [neurips-2026, thesis, framing]
created: 2026-04-21
updated: 2026-04-21
---

# The Pareto Gap

Framing for the paper's problem statement.

## The axes

- **x-axis:** wall-clock (or per-epoch cost) on ogbn-arxiv.
- **y-axis:** accuracy on ogbn-arxiv (linear probe on frozen embeddings).

## The frontier (sketch — corrected 2026-04-21 after GGD ingest)

```
  acc
  72 |                     ● GraphMAE2
  71 |  ● GGD-1500  ● BGRL   ● GraphMAE
  70 |  ● GGD-256         ● GraphACL  ● PolyGCL
  69 |  ● SUGRL(k=3)
  68 |  ● SUGRL(k=1)
     +————————————————————————————————— time
        0.2s   1s    10s   1min   10min+
```

**The gap is narrower than originally framed.** GGD at hidden=1500 already reaches
**71.6 ± 0.5 in 0.95s** on ogbn-arxiv (Table 8 of their paper). GGD-256 gets 70.3 in 0.18s.
So "BGRL accuracy at GGD cost" is *partially already achieved* by scaling GGD's hidden size — the
top-left region is not empty, it's sparse. AD-SSL must either:
1. **Push past 71.6** at comparable cost (then frame as "new Pareto point").
2. **Match 71.6 at materially lower cost** (e.g. smaller model, no inference-time global power).
3. **Add structural wins GGD cannot**: per-node adaptivity, heterophily robustness, theoretical grounding,
   principled multi-depth (rather than single-n post-hoc `A^n H_θ`).

See [[Competitive Landscape 2026]] for the numeric table and [[GGD]] §"Open risks this paper raises for our thesis".

## Why the gap exists

Accurate methods pay for accuracy with augmentations / generative decoders / multi-view GNN forward passes per step:

- [[BGRL]] — two GNN encoders (online + target) + augmentations.
- [[GraphMAE]] / [[GraphMAE2]] — full GNN encoder + mask-reconstruction decoder.
- [[PolyGCL]] — two spectral views (low-pass + high-pass Chebyshev), DGI-BCE on both, linear mix at downstream. Augmentation-free but uses a spectral GNN encoder per view. Wins 12/14 small real-world benchmarks; does **not** report ogbn-arxiv (only arXiv-year 43.07 under heterophilic labels).

Fast methods pay for speed with weaker inductive signal:

- [[SUGRL]] — MLP on precomputed 1-hop features, margin triplet. Fixed shallow propagation.
- [[GGD]] — binary discrimination on a single global signal. Cheapest loss, but note: GGD *does*
  inject one fixed-depth propagation at inference (`H_final = H_θ + A^5 H_θ`) — a weak, post-hoc
  version of multi-depth views. AD-SSL's pitch is that making this learnable and per-node wins.

## How AD-SSL closes the gap

- Keeps [[Decoupled Precompute]] (cost stays low).
- Replaces the fixed-k structural view with **multiple precomputed depths** — semantically richer without adding per-epoch GNN cost. See [[Multi-Depth Views]].
- Replaces augmentation / negatives with [[Bootstrap Loss]] across depth pairs.
- Adds [[Adaptive Depth Weighting]] so each node learns which propagation scale matters to it.

## Evidence from preliminary work

[[Prepropx Depth Finding]]: moving SUGRL from k=1 to k=3 gives +0.80 on ogbn-arxiv (3/3 seeds) at zero training-time cost. This alone shows depth is the underexploited axis. AD-SSL extends this from "pick the right scalar k" to "learn per-node k distribution."

## How we'll defend this framing

- **Main Pareto figure** in the paper (accuracy × wall-clock).
- **Scaling study** to ogbn-products and ideally ogbn-papers100M.
- **Fair comparison** — reproduce baselines in our environment, report their wall-clock on our hardware.

See [[Reviewer Attacks and Defenses]] for anticipated pushback.

## Scope (locked 2026-04-21)

- **Homophilic graphs only** — ogbn-arxiv, ogbn-products, Cora/Citeseer/PubMed. Same scope as every SSL baseline on this frontier ([[GGD]], [[SUGRL]], [[BGRL]], [[GraphMAE2]]). Heterophily is future work.
- **Per-dataset training** — field default. Cross-graph transfer is a separate open problem ([[GSTBench]] CIKM 2025: only [[GraphMAE2]]-style reconstruction transfers at papers100M scale; contrastive/bootstrap methods fail). We cite GSTBench and operate outside the transfer regime; **no foundation-model framing**.

See [[Thesis]] § Scope for the canonical statement.
