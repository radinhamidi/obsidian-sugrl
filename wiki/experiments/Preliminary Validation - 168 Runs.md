---
title: Preliminary Validation - 168 Runs
type: experiment
tags: [neurips-2026, preliminary, validation, sugrl]
created: 2026-04-21
updated: 2026-04-21
sources: [[VALIDATION_ORIGINAL_CODE]]
---

# Preliminary Validation (168 runs, 2026-04-10)

Tested 6 original-brainstorm ideas + 1 sampling control + 2 depth variants
on the **unmodified [[SUGRL]] codebase** across 6 datasets × 3 seeds.
Purpose: see which directions are worth pursuing before committing to
AD-SSL architecture.

## Setup

- Code: original SUGRL `train.py` / `train_OGB.py` with `--variant` injection.
- Hyperparameters: exactly as in `args.yaml` — no tuning.
- Datasets: [[Cora]], [[CiteSeer]], [[PubMed]], [[Photo]], [[Computers]], [[ogbn-arxiv]].
- Evaluation: linear probe on frozen L2-normalized embeddings, 2-layer LogReg, 2 restarts per seed.
- Significance: [[Matched-Seed Delta]] — `δᵢ = variant_accᵢ − baseline_accᵢ` per seed, ROBUST = `mean(δ) > 0.3` AND 3/3 seeds positive.
- Baselines reproduced within 1–2σ of paper on all 6 datasets (see §4.1 of source).

## Verdicts per idea

| # | Idea | Variant | Outcome | Notes |
|---|---|---|---|---|
| 1 | Structure-aware negatives | `struct_neg` | ❌ no robust signal | Flat to slightly negative everywhere |
| 2 | Hard negative mining | `hard_neg` | 💀 catastrophic | Cora −6.67, CiteSeer −1.97, PubMed −1.67 |
| 3 | Feature-sim positives | `feat_pos`, `feat_pos_w1` | ✅ **Computers +0.73 (3/3)** | Only works on feature-rich graphs; flat on ogbn-arxiv |
| 4 | PPR positives | `ppr_pos_sampled` | ✅ Computers +0.33 (3/3) | Small effect, one dataset |
| 5 | Degree-adaptive sampling | `deg_adapt_unpadded` | ≈ borderline | Computers +0.30 (exactly at bar) |
| 6 | Curriculum negatives | `curriculum` | 💀 catastrophic everywhere | Cora −10.77 |

## Two standout findings

### The sampling control ([[baseline_iid]])

On **[[PubMed]]**, replacing SUGRL's `np.random.permutation` negative sampler with per-anchor `rng.integers` (i.i.d.) gave **+1.37 ± 0.83 (3/3 seeds)**. Not one of the ideas — it's a one-line fix exposing that SUGRL's derangement-style sampling is suboptimal on low-degree graphs. Finding about SUGRL's baseline, not about any brainstorm idea. Logged as appendix material; not on the AD-SSL critical path.

### Pre-propagation depth ([[Prepropx Depth Finding]])

On **ogbn-arxiv**, running unmodified SUGRL with k=3 pre-propagation hops instead of k=1 gives **+0.80 ± 0.08 (3/3 seeds)** at zero training-time cost. Clean U-curve peaking at k=3, dropping below baseline past k=5 ([[Oversmoothing]]). **This is the result that motivates AD-SSL.** Full page: [[Prepropx Depth Finding]].

## Bottom line for AD-SSL

- **Sampling tweaks do not move ogbn-arxiv.** All 6 brainstorm ideas: within ±0.13 of baseline.
- **Depth moves ogbn-arxiv.** k=1 → k=3 gives +0.80, free.
- → Depth is the axis to exploit. AD-SSL makes depth **learnable per-node**.

## Source

Full source: [[VALIDATION_ORIGINAL_CODE]] (in `raw/`). All per-seed raw accuracies preserved there.
