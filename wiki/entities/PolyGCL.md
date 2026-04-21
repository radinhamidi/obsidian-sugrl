---
title: PolyGCL
type: entity
kind: method
venue: ICLR 2024
authors: [Chen, Lei, Wei]
affiliation: Renmin University of China
url: https://openreview.net/forum?id=y21ZO6M86t
code: https://github.com/ChenJY-Count/PolyGCL
tags: [method, baseline, spectral, contrastive, heterophily]
created: 2026-04-21
updated: 2026-04-21
sources: [[[PolyGCL (source)]]]
---

# PolyGCL — GCL via Learnable Spectral Polynomial Filters

Chen, Lei, Wei (RUC). ICLR 2024. Augmentation-free (DGI-style corruption only) graph contrastive learning with two spectral views — **low-pass** and **high-pass** — produced by learnable Chebyshev polynomial filters. Trained with DGI-style local-vs-global BCE loss on both views, combined at downstream time with a learnable linear mix `Z = αZ_L + βZ_H`.

## Why this matters to AD-SSL

- **Closest augmentation-free contrastive cousin** in the spectral-filter family. Like AD-SSL, it precomputes multi-order propagations in linear time `O(KE)` and feeds them through a shared MLP.
- Differs from AD-SSL in the **number and parameterization of views**: PolyGCL uses exactly two (low-pass / high-pass, both Chebyshev polynomial combinations of order K); AD-SSL uses K monotone low-pass views (depths k=1..K).
- **Learnable combination coefficients α, β are global** (one pair for the whole graph), not per-node. This is the same limitation AD-SSL's per-node α targets — prior-art differentiation is clean.
- Heterophily claim is strong (§5.3, Table 2): PolyGCL wins 12/14 small real-world benchmarks *without augmentations*. This is the bar AD-SSL has to clear if we want to claim heterophily.

## Method — verbatim from §4

**Step 1 — Chebyshev reparameterization (Eq. 1).** Filter coefficients `w_k` are produced from learnable values `γ_j` at Chebyshev nodes `x_j = cos((j+½)π/(K+1))`:
`w_k = (2/(K+1)) Σ_{j=0..K} γ_j T_k(x_j)`.

**Step 2 — Prefix constraints for monotonicity (Eq. 2).** Separate high-pass and low-pass parameter sequences:
- `γ_i^H = Σ_{j≤i} γ_j` (prefix sum → non-decreasing ⇒ high-pass)
- `γ_i^L = γ_0 − Σ_{1≤j≤i} γ_j` (prefix difference → non-increasing ⇒ low-pass)

(Algorithm 1 applies ReLU to γ_j before the prefix.)

**Step 3 — Spectral encoders (Eq. 3).**
`Z_L = f_θ(Σ_k w_k^L T_k(L̂) X)`, `Z_H = f_θ(Σ_k w_k^H T_k(L̂) X)`, with **shared MLP f_θ**.

**Step 4 — Contrastive loss (Eq. 4).** DGI-style BCE between node embedding and global summary `g = mean(Z)` where `Z = αZ_L + βZ_H`. Negatives come from `X̃ = shuffle(X)` (row-permuted features — no edge-drop, no feature-mask, no stochastic augmentation).
`L_BCE = (1/4N) Σ_i [log D(Z_L^i, g) + log(1 − D(Z̃_L^i, g)) + log D(Z_H^i, g) + log(1 − D(Z̃_H^i, g))]`
with discriminator `D(z, g) = σ(z W g^⊤)`.

**Theory (§4.3 Corollary 1, Theorem 1).** On k-regular binary-class graphs under the cSBM model, pure low-pass is optimal for homophily (`h→1`), pure high-pass for heterophily (`h→0`), and a linear combination strictly improves the Spectral Regression Loss upper bound versus low-pass-only in heterophilic regimes. Proposition 1 shows `L_BCE` lower-bounds `L_DGI`.

## Numbers (verified from PDF)

**Table 2 — real-world small benchmarks, mean accuracy (%) over 10 random 60/20/20 splits:**

| Dataset | PolyGCL | 2nd best |
|---|---:|---|
| Cora | 87.57 ± 0.62 | MVGRL 87.36 |
| Citeseer | 79.81 ± 0.85 | CCA-SSG 79.60 |
| Pubmed | 87.15 ± 0.27 | MVGRL 86.30 |
| Cornell | 82.62 ± 3.11 | GGD 80.33 |
| Texas | **88.03 ± 1.80** | CCA-SSG 87.87 |
| Wisconsin | **85.50 ± 1.88** | GREET 84.63 |
| Actor | **41.15 ± 0.88** | GREET 38.26 |
| Chameleon | **71.62 ± 0.96** | GBT 68.77 |
| Squirrel | **56.49 ± 0.72** | GBT 48.86 |

**Table 3 — larger heterophilic (Platonov et al. 2023):**

| Dataset | Metric | PolyGCL | Note |
|---|---|---:|---|
| Roman-empire | Acc | 72.97 ± 0.25 | vs GREET 72.68 |
| Amazon-ratings | Acc | 44.29 ± 0.43 | vs GBT 43.58 |
| Minesweeper | ROC AUC | 86.11 ± 0.43 | MVGRL 90.07 wins |
| Tolokers | ROC AUC | 83.73 ± 0.53 | vs MVGRL 80.86 |
| Questions | ROC AUC | 75.33 ± 0.67 | vs GBT 75.98 |

**Table 4 — arXiv-year** (Lim et al. 2021 — same node set as ogbn-arxiv but with year-based labels making it heterophilic): PolyGCL **43.07 ± 0.23**, best of all self-supervised baselines (most of which OOM).

**No ogbn-arxiv number is reported in the paper.** (A prior version of this page incorrectly estimated "~70.5 on ogbn-arxiv" from memory — that claim is withdrawn.)

## Complexity

`O(KE + N)` per epoch — linear in order K, edges E, nodes N. Polynomial propagation can be precomputed; the loss is O(N). Matches AD-SSL's decoupled-precompute cost profile.

## Differentiation from AD-SSL

| Axis | PolyGCL | AD-SSL |
|---|---|---|
| Views | 2 (low-pass, high-pass Chebyshev) | K (depths `Â^k X`, k=1..K) |
| Mixing | global learnable α, β | **per-node α_{i,k}** (novelty) |
| Loss | DGI BCE (negatives = shuffled X) | BYOL-style bootstrap across depth pairs |
| Augmentation | none (only shuffle-for-negatives) | none (depths are the views) |
| Heterophily handling | explicit high-pass filter | monotone low-pass views only — **AD-SSL has no high-pass channel** |

**Reviewer-defence risk**: PolyGCL reviewers will ask "why don't you include a high-pass view? You need it for heterophily." Candidate responses: (a) scope AD-SSL to homophilic / near-homophilic (ogbn-arxiv, citation nets) — safer; or (b) add a high-pass Chebyshev depth as one of the K views in AD-SSL (see [[Architecture Notes]]).

## Evaluation protocol used

Linear evaluation (two-stage): self-sup pretrain → freeze → linear classifier. Splits: 10 random 60/20/20 following Chien et al. 2021 (GPR-GNN's protocol). 95% CIs reported on cSBM; standard deviations on real-world. This is close to what AD-SSL should adopt for small-graph sanity checks. See [[APPNP]] for bootstrap protocol and [[Graph Learning Poor Benchmarks]] for why CIs matter.

## Relevant links

- [[SGC]] — the K=1-view low-pass endpoint of the polynomial family
- [[APPNP]] — fixed-coefficient geometric mixture `α(1−α)^k`
- [[GPRGNN]] — learnable global γ_k, supervised (PolyGCL is the self-supervised spectral-polynomial descendant)
- [[GraphMAE2]] — masked reconstruction alternative (per [[GSTBench]], more reliable at scale)
- [[GraphACL]] — augmentation-free contrastive peer (spatial not spectral)
