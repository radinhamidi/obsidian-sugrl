---
title: Thesis
type: synthesis
tags: [neurips-2026, thesis, ad-ssl]
created: 2026-04-21
updated: 2026-04-24
sources: [[RESEARCH_AGENT_ONBOARDING]]
---

# Thesis — Cross-Depth Contrastive SSL on Precomputed Multi-Depth Features

**Paper codename:** AD-SSL (legacy; "adaptive depth" framing was dropped 2026-04-24 after INQ-007 closed α-on-top-of-D6c as adding nothing). A finalized paper name is TBD; candidates: **CDC-SSL** (Cross-Depth Contrastive SSL), **DepthC-GCL** (Depth-Contrastive Graph CL). Internal vault name SUGRL remains for git history; the starting-point [[SUGRL]] method is unrelated to the final contribution.
**Target venue:** NeurIPS 2026
**Methods page:** [[Idea Ledger]] § Live D6c.

## One-sentence claim

Cross-depth instance-discrimination contrastive learning on precomputed multi-depth features `X_k = Â^k X`, with a residual `F_in`-preserving per-depth linear projection, produces a linear-probe representation that **strictly exceeds the best single-depth probe** on Cora (+3.18 on Z_concat, +3.14 on Z_mean), Computers (5/5 seeds above bar, +0.71 Z_mean, +0.43 Z_concat), and ogbn-arxiv (+8.05 on Z_concat, +4.62 on Z_mean). The method has **no encoder, no augmentation, no per-epoch GNN forward pass, and no adaptive-depth routing**; depth itself is the contrastive view axis.

## Mechanism (how the method works)

1. **Precompute** `X_k = Â^k X` for k ∈ {0, 1, 2, 4, 8}. One-time sparse matmul at dataset-load time. Inherits [[Decoupled Precompute]] from [[SGC]] / [[SIGN]].
2. **Per-depth residual linear projection**: `Z_k = X_k + W_k X_k`, where `W_k ∈ R^{F_in × F_in}` is a trainable linear map, one per depth. Residual skip guarantees `Z_k` information floor ≥ raw `X_k` — the representation cannot degrade below the raw multi-depth ensemble regardless of what InfoNCE does to `W_k`.
3. **Flat cross-depth InfoNCE** (no per-pair weighting, no entropy routing):
   - Positive pairs: `(Z_k[i], Z_{k'}[i])` for the same node `i` at any two different depths `k ≠ k'`.
   - Negative pairs: `(Z_k[i], Z_{k'}[j])` for different nodes `j ≠ i` at any depths.
   - Loss: standard InfoNCE with temperature `τ_c = 1.0`.
4. **Readout** (default): `Z_concat = [Z_0 ‖ Z_1 ‖ ... ‖ Z_K] ∈ R^{(K+1)·F_in}` fed to a linear probe. `Z_mean = (1/K) Σ_k Z_k` is a secondary readout; near-tied with concat on Cora/Computers but **concat beats mean by +3.43 on ogbn-arxiv** because arxiv's per-depth quality variation is large (raw k=2 = 60.28 vs k=8 = 50.82).

**Training hyperparameters (paper-default):** 200 epochs, Adam lr=0.01, WD=5e-4 on W_k (required — V-WD variant without WD drops Computers below the hard bar), τ_c=1.0. Matched-seed protocol: 5 seeds on Cora/Computers/arxiv per [[Splits and Protocol]], 5 linear-probe restarts per seed.

## Why this is novel

- **Depth as the contrastive view axis.** Prior graph contrastive methods use augmentations ([[GraphCL]], [[BGRL]]), spectral filter views ([[PolyGCL]]), or fixed structural views like adjacency-vs-PPR ([[MVGRL]]). None use propagation depth — `Â^k X` at different `k` — as the axis of contrast, and certainly none do it at precompute time with no encoder.
- **No encoder, no per-epoch GNN.** Every baseline that comes close on wall-clock cost ([[GGD]], [[SUGRL]]) operates at a single fixed propagation depth, losing the multi-scale signal. Every baseline that operates at multiple scales runs a GNN encoder per epoch per view. D6c is the first method to do multi-scale contrast at single-epoch-MLP cost.
- **Residual projection is load-bearing.** Ablations in [[INQ-2026-04-23-003]] show the residual is what makes the method work: D6a (linear d_proj=128, no residual) fails Computers by −5.14 because WD-shrinkage makes Z_k collapse to zero; D6b (linear d_proj=F_in, no residual) fails Computers by −7.21; only D6c's `Z_k = X_k + W_k X_k` preserves the raw-feature floor while learning cross-depth discriminative structure on top.

See [[Novelty Verification Checklist]] for claim-level 🔴/🟡/🟢 ablation status (needs update 2026-04-24).

## Why this works (mechanism story)

**The oversmoothing problem, reframed as a contrastive signal.** Deep propagation causes per-node features to collapse toward class centroids — the simplex collapse theorem of Ji et al. 2025 (see [[Rethinking graph neural networks from a geometric perspective of node features]]). On homophilic graphs, this means `Â^k X` at large `k` loses instance-level detail but preserves class structure; small `k` keeps detail but lacks neighborhood context. Neither extreme is ideal alone.

Cross-depth InfoNCE treats this as a feature: the *same node's* features at different depths are pulled together (positive pairs) while *different nodes at any depth* are pushed apart (negatives). This is the learnable equivalent of an ensemble over depths — W_k learns to project each depth into a shared space where the node-discrimination signal from every depth converges, even when some depths are "collapsed" (deep) and others are "raw" (shallow).

Empirically this LIFTS weak depths, sometimes dramatically: on Cora raw k=0 = 46.95 → post-D6c k=0 = 76.57 (+29.62); on arxiv raw k=8 = 50.82 → 67.74 (+16.92). On Computers where the strongest single depth (k=1) is already near-optimal, the lift is smaller but every depth still improves, and the concat/mean readout ensemble crosses the hard bar.

## Three-dataset primary result (INQ-007, 2026-04-24)

| Dataset | Seeds × restarts | Raw best-k | Raw mean-pool | D6c Z_mean | D6c Z_concat | Δ vs best-k (Z_concat) |
|---|---|---|---|---|---|---|
| Cora | 5 × 5 = 25 | 78.87 (k=8) | 76.25 | 82.01 ± 0.29 | **82.05 ± 0.34** | **+3.18** |
| Computers | 5 × 5 = 25 | 87.53 (k=1) | 86.10 | 88.24 ± 0.42 | 87.96 ± 0.30 | +0.43 (Z_mean +0.71) |
| ogbn-arxiv | 5 × 5 = 25, official split, CE probe | 60.28 (k=2) | 59.12 | 64.90 ± 0.10 | **68.33 ± 0.06** | **+8.05** |

**Key stats:**
- **Cora**: 14σ+ hard pass across seed-restart combinations.
- **Computers**: 5/5 per-seed Z_mean ∈ {87.75, 88.71, 88.34, 87.74, 88.64}, all strictly > 87.53; 3.8σ in stderr.
- **arxiv**: tightest stds observed in the project (0.06–0.10); best-val ≡ final-epoch to ±0.01.

## Scope (locked 2026-04-21, reaffirmed 2026-04-24)

- **Homophilic graphs only.** Evaluation on Cora, Computers, ogbn-arxiv (confirmed); CiteSeer, PubMed, Photo, CS planned as Phase-2 extensions. Heterophilic benchmarks (Chameleon, Squirrel, Texas, Actor, Wisconsin) are **out of scope for v1** — mechanism relies on the simplex-collapse regime of homophilic features (see Ji et al. 2025). On heterophily, different depths point in genuinely different class directions; cross-depth InfoNCE would pull apart legitimately different semantic views, likely hurting. Future work.
- **Per-dataset training.** AD-SSL is pretrained and evaluated on the same graph. Cross-graph pretrain-and-transfer is a separate problem benchmarked by [[GSTBench]]; we cite it and operate outside the transfer regime. No "foundation model" framing.

## Outcome scenarios (2026-04-24 read)

- **Realistic (current):** 3-dataset primary table above, +3–8 pts over best-single-depth, holds up on 4+ more homophilic datasets in Phase 2, delivers a clean Pareto point on ogbn-arxiv against [[GGD]] and [[BGRL]]. Headline: "depth-as-contrastive-view at precompute time beats best-single-depth on every homophilic benchmark tested, at wall-clock cost comparable to GGD."
- **Optimistic:** mechanism generalizes to ogbn-products and maybe [[Graph Learning Poor Benchmarks]] coverage improves the benchmarking rigor defense. Novel NeurIPS acceptance.
- **Pessimistic:** Phase-2 datasets show regression (e.g. Photo, where raw best-k is already ~92, leaves little room). We'd need a smaller-per-dataset-gain but more-mechanism-analysis angle — still a paper, but less headline-clear.

Gates captured in [[Project Phases and Decision Gates]] (needs update 2026-04-24).

## Things formally retired by D6c-lock (2026-04-24)

- **"Adaptive-Depth SSL" framing.** α-routing on top of D6c moves the probe by ≤+0.01 at any τ_p on any of three datasets tested (INQ-007 Config C). The adaptive-depth hook is surplus; the paper does not claim learned per-node depth routing. See [[Idea Ledger]] C4 and V2 closures.
- **The four-insight A1–A4 RL-analogy ablation plan** (GRPO/KTO/SimPO/Online-DPO analogies from [[RESEARCH_AGENT_ONBOARDING]]). The original insights were tied to learned per-node α, which is gone. Ablations for D6c are: (i) readout (Z_mean vs Z_concat), (ii) residual vs linear, (iii) K_SET (which depths), (iv) τ_c (InfoNCE temperature), (v) WD regime. See [[Ablation Plan - AD-SSL B0 A1-A4]] (needs rewrite 2026-04-24).
- **[[Bootstrap Loss]] as primary.** Early 2026-04-22 pivot from bootstrap → InfoNCE held; InfoNCE is the paper's loss. Bootstrap may return as a B-series ablation column.
- **The multi-depth MLP-encoder architecture** originally sketched in [[AD-SSL]] and [[Ablation Plan - AD-SSL B0 A1-A4]]. D6c has no encoder; it has per-depth linear projection with residual.

## Known risks and reviewer attacks (audit 2026-04-24)

- **Pre-emption risk (medium, verified 2026-04-24):** [[MHVGCL]] (Wang et al., Applied Soft Computing Jan 2025) is the closest known ancestor. PDF verified. **Shared**: identical InfoNCE loss (same-node-across-views positives, cross-node negatives, no augmentation), structurally identical concat readout `[H^(0), ..., H^(K)]`. **Differs**: (1) MHVGCL has a trainable linear encoder `H^(0) = XW + S` that is propagated per epoch via the iterative fusion `H^(k) = ELU(α·Â·H^(k-1) + (1−α)·H^(k-1))` — D6c has no encoder, uses raw `X_k = Â^k X` precomputed once. (2) MHVGCL has a single shared W; D6c has K per-depth W_k matrices with residual `Z_k = X_k + W_k X_k` (our D6a/D6b ablations show the residual per-depth is load-bearing). (3) MHVGCL's ELU-nonlinear iterated fusion loses direct access to raw X after H^(0); D6c's residual preserves a raw-feature floor at every depth, letting the simplex-collapse signal from Ji et al. 2025 surface contrastively. **Scope gap D6c exploits:** MHVGCL reports only limited-label small-graph results (Cora/CiteSeer/Photo/CS with 1–4 labels per class); **does not report ogbn-arxiv**. D6c's +8.05 arxiv result is not pre-empted. Also adjacent (verified 2026-04-24): [[MVGRL]] (adjacency vs. PPR, 2 views, GNN encoder) and [[DGD]] (decoupled-GCN + BCE group discrimination, single-k at training with post-hoc `A^10 H_θ` skip — **LOW risk**, different loss family and different contrast axis).
- **"This is just SGC + InfoNCE":** No — [[SGC]] is supervised and uses a single fixed depth; D6c uses K depths contrastively without labels.
- **Computers +0.71 is marginal.** On datasets where raw best-k is near saturation (Photo, CS likely candidates), the gain may be small. Defense: mechanism story + mean-of-means across 6+ datasets.
- **Residual is almost-trivial architecturally.** Strength: makes the method minimal and reproducible. Weakness: reviewers may ask "why does this paper exist." Defense: the ablation table (D6a/D6b without residual fail Computers hard) shows residual is load-bearing, not decoration.
- **Homophily-only scope.** Inherits from every comparable baseline ([[GGD]], [[SUGRL]], [[BGRL]], [[GraphMAE2]]). Not a differentiator risk.

Full defenses in [[Reviewer Attacks and Defenses]] (needs 2026-04-24 update).

## Related wiki pages

- [[Idea Ledger]] — live D6c entry with full INQ-003/INQ-007 numbers.
- [[Pareto Gap]] — fast-vs-accurate framing, updated 2026-04-24 with the new result.
- [[Competitive Landscape 2026]] — baseline numeric table.
- [[Reviewer Attacks and Defenses]] — anticipated objections (needs update).
- [[Novelty Verification Checklist]] — per-claim ablation status (needs update).
- [[Rethinking graph neural networks from a geometric perspective of node features]] — Ji et al. 2025 simplex collapse theorem, the mechanism backbone.
- [[INQ-2026-04-23-003]], [[INQ-2026-04-23-004]] — source inquiries for the primary result.
