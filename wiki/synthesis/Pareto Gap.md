---
title: Pareto Gap - Fast-vs-Accurate Unsupervised GRL
type: synthesis
tags: [neurips-2026, thesis, framing]
created: 2026-04-21
updated: 2026-04-24
---

# The Pareto Gap

Framing for the paper's problem statement. Numbers updated 2026-04-24 to reflect the D6c result locked in [[INQ-2026-04-23-004]].

## The axes

The Pareto figure ships in **two variants** (locked 2026-04-24 via [[INQ-2026-04-24-002]] § Config B):

- **Variant 1**: accuracy × **wall-clock** (hardware-dependent; matched on Vector A40-48GB).
- **Variant 2**: accuracy × **training FLOPs** (hardware-independent; measured via `fvcore.FlopCountAnalysis`, `flops_total ≈ 3 × flops_forward` heuristic, documented in methods footnote).

Both x-axes log-scaled. Matched-harness points (D6c + BGRL + GGD + GraphMAE2 + GraphACL) plotted as solid markers; paper-cited baselines (DGI/MVGRL/GRACE/CCA-SSG/SUGRL/PolyGCL/DGD/MHVGCL/BLNN) as faded markers with hardware-caveat footnote. y-axis: accuracy on ogbn-arxiv (linear/CE probe on frozen embeddings, official split).

## The frontier (sketch — updated 2026-04-24 with D6c arxiv number)

```
  acc
  72 |                     ● GraphMAE2
  71 |  ● GGD-1500  ● BGRL   ● GraphMAE
  70 |  ● GGD-256         ● GraphACL  ● PolyGCL
  69 |  ● SUGRL(k=3)
  68 |  ● D6c (Z_concat, 68.33 ± 0.06)  ← this paper
     +————————————————————————————————— time
        0.2s   1s    10s   1min   10min+
```

**Where D6c sits.** D6c's arxiv accuracy is 68.33 ± 0.06 (CE probe, official split, 5 seeds × 5 restarts, INQ-007 Config A) — **below** the BGRL/GraphMAE band (71.x), but delivered with **no encoder, no augmentation, and no per-epoch GNN forward pass**. The wall-clock position is MLP-on-precomputed-features: a single sparse matmul at dataset load, then an MLP-sized contrastive loop. Cost is comparable to [[GGD]] and [[SUGRL]] rather than to BGRL.

**What this reframes.** The claim shifts from "top-left Pareto point on accuracy" to **"+8.05 over the best single-depth raw feature at matching wall-clock cost"**. The Pareto win is **within the cheap-method band**: at the cost of SUGRL/GGD, D6c lifts the accuracy of precomputed multi-depth features by a margin that single-depth cheap methods structurally cannot access. This makes D6c a Pareto point for the SSL-at-MLP-cost regime, not the SOTA-accuracy regime.

AD-SSL v1 does NOT claim to push past 71.6 absolute — that was the 2026-04-21 framing before the D6c lock. If a reviewer asks why not BGRL-level accuracy, the defense is mechanism-and-cost (see [[Reviewer Attacks and Defenses]]), not a head-to-head accuracy table.

## Why the gap exists

Accurate methods pay for accuracy with augmentations / generative decoders / multi-view GNN forward passes per step:

- [[BGRL]] — two GNN encoders (online + target) + augmentations.
- [[GraphMAE]] / [[GraphMAE2]] — full GNN encoder + mask-reconstruction decoder.
- [[PolyGCL]] — two spectral views (low-pass + high-pass Chebyshev), DGI-BCE on both, linear mix at downstream. Augmentation-free but uses a spectral GNN encoder per view. Wins 12/14 small real-world benchmarks; does **not** report ogbn-arxiv (only arXiv-year 43.07 under heterophilic labels).

Fast methods pay for speed with weaker inductive signal:

- [[SUGRL]] — MLP on precomputed 1-hop features, margin triplet. Single fixed shallow depth.
- [[GGD]] — binary discrimination on a single global signal. Cheapest loss. GGD injects a single fixed propagation at inference (`H_final = H_θ + A^5 H_θ`) — a weak, post-hoc version of multi-depth views.

Neither fast-method family does **multi-scale contrast at precompute time**. That is the D6c wedge.

## How D6c sits in the gap

- Keeps [[Decoupled Precompute]] (cost stays low).
- Replaces the fixed-k structural view with **K precomputed depths** (`X_k = Â^k X`, k ∈ {0, 1, 2, 4, 8}) — semantically richer without adding per-epoch GNN cost. See [[Multi-Depth Views]].
- Contrastive signal is **cross-depth InfoNCE** on the same node across depths, without augmentation and without an encoder. `Z_k = X_k + W_k X_k` with per-depth linear residual projection.
- Readout: `Z_concat = [Z_0 ‖ ... ‖ Z_K]` (default; beats Z_mean by +3.43 on arxiv).

## Evidence (2026-04-24, 3-dataset primary result)

| Dataset | Raw best-k | D6c Z_concat | Δ |
|---|---|---|---|
| Cora | 78.87 (k=8) | 82.05 ± 0.34 | **+3.18** |
| Computers | 87.53 (k=1) | 87.96 ± 0.30 (Z_mean 88.24 ± 0.42) | +0.43 (Z_mean +0.71) |
| ogbn-arxiv | 60.28 (k=2) | **68.33 ± 0.06** | **+8.05** |

See [[Thesis]] § Three-dataset primary result and [[INQ-2026-04-23-004]] for the full numbers, matched-seed protocol, and per-seed breakdown.

The +8.05 on arxiv is the headline: moving from best-single-depth precomputed features to D6c's cross-depth contrastive projection lifts linear-probe accuracy by 8 full points at essentially identical inference cost (same feature dimensionality × K=5 concat = 5× probe input, still MLP-speed).

[[Prepropx Depth Finding]] remains the underlying motivation: depth was already the underexploited axis even for supervised fixed-k SUGRL (k=1 → k=3 gave +0.80 on arxiv). D6c extends "pick the right scalar k" to "contrast across K depths and let W_k align them."

## How we'll defend this framing

- **Main Pareto figure** in the paper (accuracy × wall-clock) that plots D6c against cheap-method baselines (SUGRL, GGD-256, GGD-1500) AND against BGRL/GraphMAE/GraphACL/PolyGCL at their true cost. D6c wins the cheap-method region; we do NOT claim global Pareto.
- **Efficiency benchmark** — reproduce baselines in our matched harness, report **wall-clock + peak GPU memory + training FLOPs + precompute cost** on Vector A40-48GB. Locked into Config B of [[INQ-2026-04-24-002]]; ports = BGRL, GGD, GraphMAE2, GraphACL.
- **Per-depth lift decomposition** — show the mechanism in numbers: raw k=0 → D6c k=0 on Cora is +29.62; raw k=8 → D6c k=8 on arxiv is +16.92. W_k is lifting weak depths, not just picking the best.

See [[Reviewer Attacks and Defenses]] for anticipated pushback (needs 2026-04-24 update).

## Scope (locked 2026-04-21, reaffirmed 2026-04-24)

- **Homophilic graphs only** — Cora, Computers, ogbn-arxiv confirmed; CiteSeer, PubMed, Photo, CS Phase-2. Same scope as every SSL baseline on this frontier ([[GGD]], [[SUGRL]], [[BGRL]], [[GraphMAE2]]). Heterophily future work — mechanism relies on homophilic simplex-collapse regime ([[Rethinking graph neural networks from a geometric perspective of node features]]).
- **Per-dataset training** — field default. Cross-graph transfer is a separate open problem ([[GSTBench]] CIKM 2025). We cite GSTBench and operate outside the transfer regime; **no foundation-model framing**.

See [[Thesis]] § Scope for the canonical statement.
