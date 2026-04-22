---
title: AD-SSL v2 - Encoder-Free Design Sketch
type: synthesis
tags: [neurips-2026, ad-ssl, reframe, draft]
created: 2026-04-22
updated: 2026-04-22
status: draft
sources: [[[INQ-2026-04-22-001]], [[INQ-2026-04-22-002]]]
---

# AD-SSL v2 — Encoder-Free Design Sketch

**Status:** design draft. Triggered by three-dataset failure of encoder-based B0 in [[INQ-2026-04-22-002]]. Parallel to the BGRL-lite diagnostic currently running in that inquiry. If that diagnostic passes, this sketch becomes a secondary ablation. If it fails, this becomes the paper.

## Why we're reframing

CA's diagnostics across Cora / Computers / ogbn-arxiv:

| | Best trained Z_k | Best raw Â^{k*}X | Gap |
|---|---:|---:|---:|
| Cora | 78.49 (edge-drop, k=2) | 81.32 (k=4) | −2.83 |
| Computers | 84.92 (no-aug, k=1) | 87.49 (k=1) | −2.57 |
| ogbn-arxiv | 61.28 (no-aug, k=4) | 69.31 (k=4) | −8.03 |

**On no dataset does our trained encoder beat the best raw propagation.** CA's emerging hypothesis (#4 in [[INQ-2026-04-22-002]]): InfoNCE between highly-correlated Â^k views cannot in principle exceed a strong raw linear probe — instance discrimination learns the shared component and discards class structure.

If the hypothesis is right, no encoder-capacity / loss-tweak / augmentation will save the current architecture. The reframe: **skip the encoder. Work directly on raw Â^k X, learn only the per-node mixture α_{i,k}.**

## Core architecture (v2)

```
Pre-compute: X_k = Â^k X for k ∈ {1, 2, 4, 8}   # unchanged from v1
For each node i:
    α_{i,k} = softmax_k( g(i, k) )              # per-node depth weights
    Z(i)    = Σ_k α_{i,k} · X_k[i]              # linear mixture, no encoder
Linear probe on Z  →  downstream classification
```

**What's parameterized:** only `g(i, k)` — the per-node depth-scoring function. Candidates:

- **(P-A) Feature-conditioned MLP.** `g(i, k) = MLP_k(X_i)` — a small MLP per depth, or a shared MLP with depth embeddings. O(d·h + K·h) parameters. Deterministic function of input features.
- **(P-B) Free parameter table.** `α_{i,k}` is a direct N×K matrix. Most expressive, memory-heavy on ogbn-products, no generalization to unseen nodes. Consider only if other options fail.
- **(P-C) Attention over raw-features.** `g(i, k) = <φ(X_i), ψ_k>` — low-rank, scales well.

Default: **P-A with a shared small MLP + learned depth embedding**, unless parameter-count experiments argue otherwise.

**What's removed:**
- Encoder (shared MLP mapping X_k → Z_k). Gone.
- InfoNCE / bootstrap loss between depth views. Gone.
- EMA target network. Gone.
- Predictor. Gone.
- Augmentation. Gone.

**What's preserved:**
- Multi-depth precompute (the [[SGC]] lineage).
- Per-node adaptive mixture (the [[Adaptive Depth Weighting]] novelty claim vs [[GPRGNN]]).
- Decoupled-precompute scalability story (stronger now, nothing is learned at O(N·d²) per epoch — only the small α network).

## Candidate training signals for α

The hard question: without an encoder and without labels, what drives α?

### Signal candidates

**(S1) Cross-depth prediction consistency (confidence maximization).**
Shared linear classifier head `h: d → M` over M soft clusters.

```
p_{i,k} = softmax(h(X_k[i]))                          # per-depth prediction
p_i     = Σ_k α_{i,k} · p_{i,k}                       # mixed prediction
L_conf  = -Σ_i entropy(p_i)                           # confident mixture
L_div   = -entropy( (1/N) Σ_i p_i )                   # avoid cluster collapse
L       = L_conf + λ · L_div
```

Inspired by IIC / SCAN / Self-Labelling. No labels required. α and h learned jointly.

**(S2) Cross-depth feature alignment (BYOL-without-encoder).**
For each node, enforce that the α-mixed representation is close to each individual depth representation weighted appropriately.

```
z_i     = Σ_k α_{i,k} · X_k[i]
L_align = Σ_i Σ_k α_{i,k} · ‖ normalize(X_k[i]) - stop_grad(normalize(z_i)) ‖²
```

Risk: trivial solution α_{i,k} concentrates on one k (whichever depth matches z_i best). Needs an anti-collapse term.

**(S3) Graph-smoothness prior.**
α should vary smoothly over graph neighbors (neighbors likely share an optimal depth). No labels; pure regularizer.

```
L_smooth = Σ_{(i,j)∈E} ‖α_i - α_j‖²
```

Can't stand alone — needs combining with S1 or S2. But addresses a real inductive bias.

**(S4) Supervised α, frozen or joint.**
Drop the SSL framing. Learn α and h with 20-per-class Planetoid labels (or equivalent on larger graphs). **This is no longer SSL**, but it preserves the per-node α novelty vs GPR-GNN's global γ.

Risk: collapses the paper into "adaptive-depth supervised method" — competes with GPR-GNN directly. May still be novel *if* per-node α beats global γ at matched cost on the benchmarks where GPR-GNN is strongest. Not the originally-planned paper, but a honest one.

**(S5) No training — just probe per-depth, pick best-k per-node by probe confidence.**
No gradient steps. For each node i, compute linear probe predictions at each k, choose the k with highest confidence. The mixture is hard (one-hot) and confidence-based.

This is almost trivial to implement but requires a labeled probe — so it's a supervised method at eval time. Useful as a reference: if it matches our SSL S1/S2/S3 results, we don't need training at all.

### Ranking

My current lean, subject to user input:
1. **S1 (confidence max) as the headline SSL signal.** Closest to the original "unsupervised" framing, validated in image SSL literature, clean loss. Risk: known to require careful cluster-count and λ tuning.
2. **S3 added as regularizer.** Cheap, principled, uniquely graph-aware.
3. **S4 (supervised α) as a backup paper.** If S1+S3 underperform, the honest fallback is "adaptive-depth supervised" — still novel vs GPR-GNN, but a different paper.
4. **S5 as a baseline row** (like Â^{k*}X is now a baseline) — not the method, but important to report so we can claim SSL is doing work.
5. **S2 only if S1+S3 both fail.**

## How this fits the thesis

**Scalability claim:** **stronger.** Training cost is `O(N·(d+K))` for the α network vs the original `O(N·d²)` for the encoder — order-of-magnitude smaller. Still works with decoupled precompute. Still no per-epoch graph propagation.

**"BGRL accuracy at GGD cost" claim:** **uncertain.** Depends on whether α-mixture + S1/S3 can beat the raw Â^{k*}X baseline meaningfully. If not, the Pareto claim becomes "Â^{k*}X is already competitive with expensive SSL methods; our α-mixture is a small additional improvement."

**Novelty framing (revised):** The contribution shifts from "SSL method" to "the propagation baseline is stronger than the field admits; we characterize this and introduce a minimal adaptive-depth mixture that exceeds it." A1 (per-node α) is still the key mechanism; A2/A3/A4 become less central or get rewritten.

## Open questions

- Can S1 beat Â^{k*}X on Cora/Computers/arxiv by ≥1 pt? **Blocks decision.** Need a quick prototype run before committing to the reframe.
- Does P-A (MLP-parameterized α) generalize to unseen nodes, or do we need P-B (free table) for transductive benchmarks? Benchmark-specific.
- How do we handle the **reviewer response**: "this is just GPR-GNN with per-node weights and no labels"? See [[GPRGNN]] — their differences table still works but with weaker gaps (no encoder vs their MLP, α-only training vs their end-to-end). [[Reviewer Attacks and Defenses]] will need a rewrite.
- **Ablation plan needs full rewrite.** A1 vs uniform α becomes the core ablation. A2/A3/A4 need re-interpretation — A3's "swap bootstrap for InfoNCE/MSE" is now "swap S1 for S2/S3/S4", different question.

## What would kill this reframe

- If even S1+S3+S5 taken together can't beat Â^{k*}X on any dataset, per-node α isn't doing work over the best single depth — the entire depth-adaptivity claim fails, and the paper has no method.
- If Â^{k*}X under proper 5-trial protocol turns out to be noticeably lower than CA's 3-seed numbers (e.g. if our Cora Â⁴X=81.32 becomes 78 under varied splits), the baseline we're reframing around isn't as strong as we think, and the gap reopens for an encoder.

Both are real risks. The reframe is cheap to prototype (weeks not months) so we can de-risk fast.

## Next action

**Blocked on:** BGRL-lite diagnostic in [[INQ-2026-04-22-002]]. If that passes, this sketch gets shelved as a secondary ablation. If it fails, this becomes the paper — and the first implementation task is a 2-day S1 prototype (α-MLP + confidence-max loss) on Cora to check whether this direction has legs.

## Related

- [[INQ-2026-04-22-001]] — B0 collapse evidence.
- [[INQ-2026-04-22-002]] — wide-encoder failure + edge-dropout split result.
- [[Thesis]] — original thesis; will need partial rewrite if this direction locks.
- [[Adaptive Depth Weighting]] — concept unchanged; implementation shifts from over-Z_k to over-X_k.
- [[GPRGNN]] — closest supervised analog; novelty arguments need revision for v2.
