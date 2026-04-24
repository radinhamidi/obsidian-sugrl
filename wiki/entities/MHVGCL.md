---
title: MHVGCL - Multi-Hop Views Graph Contrastive Learning
type: entity
tags: [neurips-2026, baseline, pre-emption-risk, graph-ssl, contrastive]
created: 2026-04-24
updated: 2026-04-24
sources: []
---

# MHVGCL — Robust Graph Contrastive Learning with Multi-Hop Views

**⚠ UNVERIFIED — abstract-level reading only.** PDF not retrieved as of 2026-04-24. Paper is published in Applied Soft Computing (paywalled ScienceDirect). Tried: ScienceDirect (paywall), ResearchGate (403 blocked), arxiv (no preprint found in 2026-04-24 search). Everything below is inferred from the abstract + secondary web-search paraphrases. **The architectural claims (MLP-before-propagation, single-shared-head, per-epoch cost) are my interpretation of abstract wording, not read from the paper's method section.** Before citing this in the paper or acting on the pre-emption defense, the PDF must be obtained (author email, institutional access, or wait for preprint) and re-audited via `pdftotext -layout` per CLAUDE.md audit discipline.

Wu et al., Applied Soft Computing, January 2025. Surfaced during 2026-04-24 literature audit after D6c-lock as a **candidate architectural ancestor (unconfirmed)** to our AD-SSL / D6c method. Tentatively placed above [[MVGRL]] on the pre-emption-risk ladder pending PDF verification.

**Citation (unverified):** "Robust graph contrastive learning with multi-hop views for node classification", Applied Soft Computing, 2025. ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S1568494625000948.

## One-sentence claim (abstract-level, unverified)

Generates multiple contrastive views by iteratively propagating an MLP-encoded "base view" through multi-hop message passing (`MLP(X) → Â · MLP(X) → Â² · MLP(X) → ...`), then maximizes same-node agreement across hops with a contrastive loss, minimizes cross-node agreement. **Propagation order (MLP-before-vs-after) inferred from abstract phrasing "generates augmented heads by exploiting multi-hop information, obtained iteratively from a single output head" — not verified against the method section.**

## Method (reconstructed from abstract + web-search paraphrases — UNVERIFIED)

1. **Base view**: `H_base = MLP(X)` — a learned linear/MLP transformation of raw features.
2. **Multi-hop views**: `H_k = Â^k H_base` for k ∈ {1, 2, ..., K} (order: **MLP then propagate**, APPNP-style).
3. **Contrastive loss**: InfoNCE-style multi-hop contrastive loss. Positive pairs = same node across hop views; negatives = different nodes across any hops.
4. **Shared output head**: a single shared encoder/head across all hops — **not** per-hop parameters.
5. **Variants**: MHVGCL-MV (all augmented views), MHVGCL-2V (only first + last view, cost reduction).

## Architectural differentiators vs. D6c

| Axis | D6c (ours) | MHVGCL |
|---|---|---|
| Propagation timing | Precompute `Â^k X` at dataset load (one-time sparse matmul) | Per-epoch: `Â^k MLP(X)` recomputed every epoch |
| Encoder order | **No encoder.** Per-depth residual `Z_k = X_k + W_k X_k` | MLP-before-propagate (APPNP order) |
| Per-depth parameters | **W_k per depth** (one trainable linear map per k) | **Single shared head** across all hops |
| Inputs contrasted | Raw features at different depths | Encoder-output views at different hops |
| Augmentation | None | None |
| Loss | Flat cross-depth InfoNCE | InfoNCE-style multi-hop contrastive |
| Per-epoch cost | MLP-level (no GNN forward) | GNN forward per epoch (K hops × encoder) |

**The differentiator that matters for the paper:**
- D6c is **encoder-free + precompute-only**. Every per-epoch step is MLP-speed because `X_k` is fixed at load time.
- MHVGCL runs `Â^k` through a trainable MLP per epoch. This is a GNN encoder (just written as MLP + Â^k composition). Its per-epoch cost scales with K × (GCN forward).
- D6c learns per-depth `W_k` **after** propagation, letting each depth be projected independently into a shared contrastive space. MHVGCL's shared head cannot do this — every depth passes through the same parameters.

## Pre-emption risk: MEDIUM

Reviewers may collapse the distinction and ask "how is D6c different from MHVGCL?" The sharp answer is:
1. **Precompute vs. per-epoch**: D6c's cost is structurally lower (MLP-speed); this is the Pareto point. MHVGCL sits in the BGRL/GraphMAE cost band.
2. **Per-depth W_k vs. shared head**: D6c ablations show the residual `Z_k = X_k + W_k X_k` is load-bearing (D6a and D6b without residual fail Computers). MHVGCL's shared-head architecture is equivalent in spirit to our D1-family variants, which all hard-failed (W_k collapse under shared head + homophily, per [[INQ-2026-04-23-001]] and [[Idea Ledger]] C3).
3. **Contrast axis**: D6c contrasts **raw** depths (`Â^k X`), surfacing the simplex-collapse signal ([[Rethinking graph neural networks from a geometric perspective of node features]]) at the contrastive level. MHVGCL contrasts **encoder-output** depths, which mixes encoder inductive bias with hop-signal.

## What we need to do

- [ ] Ingest the PDF (`pdftotext -layout`), verify loss form, read reported numbers on Cora / Citeseer / Photo / Computers / ogbn-arxiv (if reported).
- [ ] If MHVGCL benchmarks ogbn-arxiv, add to [[Competitive Landscape 2026]] and compare directly to D6c's 68.33.
- [ ] If MHVGCL does NOT benchmark ogbn-arxiv (common for Applied Soft Computing scope), this is an exposure gap we can exploit: D6c reports arxiv, they don't.
- [ ] Write a short contrast paragraph for the paper's related-work section with the three-way differentiator above.
- [ ] Add MHVGCL to the efficiency-benchmark inquiry (INQ-008 draft) as a B-tier baseline if implementable.

## Related wiki pages

- [[Thesis]] — D6c method paper, cites this as closest ancestor in Known Risks.
- [[Idea Ledger]] — D6c Live row, MHVGCL flagged in reviewer attacks.
- [[MVGRL]] — previously thought to be closest ancestor; demoted after MHVGCL surfaced.
- [[DGD]] — related "decoupled-GCN + contrastive" precedent, low pre-emption risk (BCE not InfoNCE).
- [[Competitive Landscape 2026]] — needs MHVGCL row.
