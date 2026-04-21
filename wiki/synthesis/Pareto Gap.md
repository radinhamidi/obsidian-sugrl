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

## The frontier (sketch)

```
  acc
  72 |                     ● GraphMAE2
  71 |              ● BGRL   ● GraphMAE
  70 |                           ● GraphACL  ● PolyGCL
  69 |  ● SUGRL(k=3 corrected)
  68 |  ● SUGRL(k=1)
     |
     |  ● GGD
     +————————————————————————————————— time
        0.2s   1s    10s   1min   10min+
```

**Top-left (accurate + fast) is empty.** This is where AD-SSL aims. See [[Competitive Landscape 2026]] for the numeric table.

## Why the gap exists

Accurate methods pay for accuracy with augmentations / generative decoders / multi-view GNN forward passes per step:

- [[BGRL]] — two GNN encoders (online + target) + augmentations.
- [[GraphMAE]] / [[GraphMAE2]] — full GNN encoder + mask-reconstruction decoder.
- [[PolyGCL]] — polynomial spectral filters, learned per view.

Fast methods pay for speed with weaker inductive signal:

- [[SUGRL]] — MLP on precomputed 1-hop features, margin triplet. Fixed shallow propagation.
- [[GGD]] — binary discrimination on a single global signal. Even less structure.

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
