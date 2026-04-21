---
title: BGRL
type: entity
kind: method
venue: ICLR 2022
url: https://arxiv.org/abs/2102.06514
tags: [method, baseline, accuracy-ceiling, bootstrap, pareto-anchor]
created: 2026-04-21
updated: 2026-04-21
---

# BGRL — Bootstrapped Graph Latents

Thakoor, Tallec, Azar, Azabou, Dyer, Munos, Veličković, Valko (DeepMind + Georgia Tech). ICLR 2022, arXiv v3 2023-02-20. BYOL-on-graphs. **Upper anchor of our [[Pareto Gap]]** and conceptual parent of AD-SSL's [[Bootstrap Loss]].

## Mechanism (exact)

Two GCN encoders: online `E_θ` and target `E_φ`. Two augmented views of (X, A) are produced via feature + edge masking (`T_1, T_2`).

1. `H̃_1 = E_θ(X̃_1, Ã_1)` — online encoding of view 1.
2. `H̃_2 = E_φ(X̃_2, Ã_2)` — target encoding of view 2.
3. `Z̃_1 = p_θ(H̃_1)` — predictor (MLP) produces prediction of target from online.
4. **Loss (Eq. 1):**
   ```
   ℓ(θ, φ) = -(2/N) Σ_i  Z̃_{1,i} · H̃_{2,i} / (‖Z̃_{1,i}‖ ‖H̃_{2,i}‖)
   ```
   i.e. node-wise **cosine similarity**, maximised. Symmetrized by also predicting view 1 from view 2.
5. Gradients flow **only through θ**. Target `φ ← τφ + (1−τ)θ`, τ cosine-annealed 0.99 → 1.0.

**No projector** (unlike BYOL); the predictor alone prevents augmentation-invariance collapse. No negatives.

Non-collapse is empirical (Appendix A): loss does not go to 0, embedding spread stays `O(‖h‖)`, norms bounded.

## Complexity (paper §3)

Per update step:
```
time+space = 6·C_enc·(M+N) + 4·C_pred·N + C_BGRL·N
```
— **4 encoder forward passes per step** (online ×2 for two views, target ×2) plus 2 backward passes. Linear in N (no N² negatives, unlike GRACE). On medium graphs BGRL uses **2–10× less memory** than GRACE.

Still, this is the cost we want to beat with AD-SSL: each AD-SSL step has **zero GNN forward passes** (MLP over precomputed `Â^k X`).

## Numbers on ogbn-arxiv (Table 5)

| Method | Val | Test |
|---|---:|---:|
| Random-Init | 69.90 ± 0.11 | 68.94 ± 0.15 |
| DGI | 71.26 ± 0.11 | 70.34 ± 0.16 |
| GRACE full-graph | OOM (16GB V100) | OOM |
| GRACE-subsample k=32 | 72.18 ± 0.16 | 71.18 ± 0.16 |
| GRACE-subsample k=2048 | 72.61 ± 0.15 | 71.51 ± 0.11 |
| **BGRL** | **72.53 ± 0.09** | **71.64 ± 0.12** |
| Supervised GCN | 73.00 ± 0.17 | 71.74 ± 0.29 |

BGRL essentially **matches supervised GCN** on ogbn-arxiv — a striking number.

**Training regime (corrected 2026-04-21)**: the BGRL paper itself (§4.2) runs BGRL **full-graph** on ogbn-arxiv with a 3-layer GCN on a 16GB V100; only GRACE full-graph OOMs in Table 5 — BGRL has no such footnote in its own paper. The [[GGD]] paper's own efficiency comparison (§5.2, Fig. 4 footnote) uses a *batched* BGRL implementation with neighbour sampling in *their* reproduction — that is a GGD-paper reproduction detail, not a BGRL-paper limitation. Earlier drafts of this page conflated the two — fixed.

## ogbn-arxiv hyperparameters (Table 8)

| Param | Value | Notes |
|---|---|---|
| Encoder | 3-layer GCN, hidden 256 | Same depth as supervised OGB baseline. |
| Predictor `p_θ` | MLP, hidden 256 | No projector. |
| `p_f,1`, `p_f,2` | **0.0, 0.0** | **No feature masking** on ogbn-arxiv. |
| `p_e,1`, `p_e,2` | **0.6, 0.6** | Heavy edge masking only. |
| η_base | 1e-2 | AdamW, weight decay 1e-5, warmup 1k steps, cosine anneal, 10k total steps. |
| τ_base | 0.99 → 1.0 (cosine) | EMA decay. |
| Normalisation | LayerNorm + weight standardisation | Not BatchNorm. |
| Embedding size | 256 | L2-normalised before linear probe. |

**Important (for AD-SSL design):** the augmentation strategy BGRL needs on ogbn-arxiv is *only edge dropout* — features are left untouched. This matters because AD-SSL's "augmentation" is cross-depth view selection; we don't need to bolt on feature masking to match BGRL-style views.

## Numbers on smaller datasets (Table 3)

BGRL is SOTA on 4/5: WikiCS 79.98, Amazon-Computers 90.34, Amazon-Photos 93.17, Coauthor-CS 93.31, Coauthor-Physics 95.73. Comparable to or ahead of GRACE, MVGRL, DGI — with 2-10× less memory.

## Role in AD-SSL

- **Accuracy ceiling on ogbn-arxiv: 71.64.** Our target for Phase 2 gate (see [[Project Phases and Decision Gates]]).
- **Loss template.** Our depth-pair bootstrap loss should mirror BGRL's exact form:
  `ℓ_{k,k'} = -cos(p(Z_k), Z_{k'}.detach())`, symmetrized across K depth pairs, EMA target.
- **No projector rule.** We should likely follow BGRL's design (predictor only, no projector) — their Appendix B shows projector hurts on MAG240M.
- **Collapse monitoring.** Adopt their three-plot protocol (loss, embedding spread, embedding norm) to rule out trivial solutions in AD-SSL ablations.
- **τ schedule.** Cosine 0.99 → 1.0 is a solid default. Don't train with fixed τ.
- **LayerNorm + weight standardisation on ogbn-arxiv**, not BatchNorm. Relevant because batch-norm on precomputed features behaves differently across depth.

## Differences from AD-SSL (for related-work paragraph)

| | BGRL | AD-SSL |
|---|---|---|
| Views | 2, via stochastic augmentation (edge + feature masking) | K, via deterministic multi-depth precompute `Â^k X` |
| Encoder | GCN (3-layer for OGB) | MLP only |
| Cost per step | 4 encoder forwards + 2 predictor | 0 GNN forwards, K predictor passes |
| View combination | Predict 2 from 1 (symmetrized pairwise) | Per-node α_k mixture across K depths, bootstrap across all depth pairs |
| Inference | Frozen encoder output | Weighted sum `Σ_k α_{i,k} Z_k` |

**Reviewer attack to expect:** "AD-SSL is BGRL with precomputed views." Defence: (a) per-node α across K depths vs BGRL's uniform pair alignment; (b) cost structure is fundamentally different (no GNN in the hot loop); (c) the multi-depth signal is theoretically motivated (different spectral filters), augmentations are not.

## Reproduction note

Official code: `https://github.com/nerdslab/bgrl`. MAG240M code: `github.com/deepmind/deepmind-research/tree/master/ogb_lsc/mag`. Phase 1 reproduction target: 71.6 ± 0.3 on ogbn-arxiv in our harness using the exact hyperparameters in Table 8.

## Evidence gap flagged

BGRL's frozen-linear eval on MAG240M **underperforms LabelProp** (Appendix J). Only works in semi-supervised. This is a general SSL-on-graphs failure mode at extreme scale and worth noting in our scale-study framing — we should not over-claim frozen-eval wins at papers100M scale without checking LabelProp as a baseline.
