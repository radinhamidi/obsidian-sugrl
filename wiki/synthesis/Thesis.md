---
title: Thesis
type: synthesis
tags: [neurips-2026, thesis, ad-ssl]
created: 2026-04-21
updated: 2026-04-21
sources: [[RESEARCH_AGENT_ONBOARDING]]
---

# Thesis — AD-SSL for Scalable Unsupervised Graph Representation Learning

**Paper codename:** AD-SSL (Adaptive-Depth Decoupled Self-Supervised Learning)
**Target venue:** NeurIPS 2026
**Internal vault name:** SUGRL (legacy; refers to the starting-point method, not the final contribution)

## One-sentence claim

AD-SSL matches the accuracy of expensive unsupervised graph SSL methods (e.g. [[BGRL]], [[GraphMAE]]) on ogbn-arxiv (~71) at the wall-clock cost of cheap methods (e.g. [[GGD]], seconds). **No published method currently occupies this Pareto point.** See [[Pareto Gap]].

## Mechanism (how it works)

1. **Pre-compute** `X_k = Â^k X` for k ∈ {1, 2, 4, 8}. One-time sparse matmul. Inherits the decoupled precompute trick from [[SGC]]. See [[Decoupled Precompute]].
2. **Shared MLP encoder** maps each X_k → Z_k. Same weights across all depths.
3. **Bootstrap loss** (BYOL-style, online vs EMA target) aligns Z_k with Z_{k'} across depth pairs. See [[Bootstrap Loss]].
4. **Per-node adaptive depth weighting** via group-relative ranking: for each node, each depth is scored by how well it aligns with the consensus of the other depths; softmax gives weights. See [[Multi-Depth Views]] and [[Adaptive Depth Weighting]].
5. **Inference:** Z_final = weighted average of Z_k.

No augmentation, no negatives, no GNN forward pass during training. Per-epoch cost `O(N·d²)`.

## Why this is novel

- Multi-depth precomputed features as contrastive views — not done in unsupervised GRL.
- Per-node adaptive depth weighting via group-relative ranking — not done in graph SSL.
- Supervised adaptive-depth work exists ([[GPRGNN]], [[APPNP]], [[ATP]]) but not in the unsupervised regime.

See [[Novelty Verification Checklist]].

## Why we believe this will work

From [[Preliminary Validation - 168 Runs]]: six SUGRL-style sampling tweaks failed to move ogbn-arxiv (all within ±0.13 of baseline). **The only thing that moved ogbn-arxiv was changing propagation depth** (k=1 → k=3, +0.80 with 3/3 seeds positive). See [[Prepropx Depth Finding]]. Depth is the axis to exploit; AD-SSL makes depth learnable per-node.

## Four insights being ablated

Internal naming (does **not** go in paper). The "GRPO / KTO / SimPO / Online-DPO" tags are **RL-alignment analogies used during brainstorming** to import structural patterns (group-relative ranking, binary preference, simplified objective, iterative EMA preference) into graph SSL. The graph mechanism stands on its own — the analogy is scaffolding only.

Baseline **B0** = uniform depth weights, bootstrap cosine loss, no refinement. See [[Ablation Plan - AD-SSL B0 A1-A4]].

| Insight | Brainstorm tag (RL analogy) | What it does in AD-SSL | Maps to [[Novelty Verification Checklist]] |
|---|---|---|---|
| A1 | GRPO-style | Per-node α_{i,k} weighting from cross-depth consistency (analogous to group-relative advantage across a set of candidates) | Claim 1: per-node adaptive depth. 🔴 global-γ SSL ablation, 🔴 best-fixed-k sweep. |
| A2 | KTO-style | Binary quality signal (does the node's kNN in embedding space match its graph neighbors?), asymmetric loss on good-vs-bad pairs | Claim 2 variant: alternative to bootstrap. Paired with 🔴 bootstrap-vs-DGI-BCE swap as a "what loss is the right self-sup signal" study. |
| A3 | SimPO-style | Swap bootstrap for simpler losses (MSE, InfoNCE) to isolate whether EMA target + predictor is actually load-bearing | Claim 2: bootstrap justification. 🔴 bootstrap-vs-DGI-BCE swap covers the InfoNCE side; MSE variant is an extra column. |
| A4 | Online-DPO-style | EMA-smoothed iterative refinement of α_{i,k} across epochs (preference-style update rather than direct gradient) | Claim 1 extension: stability of per-node α. Currently 🟡 — gated on A1 passing. |

## Outcome scenarios

- **Optimistic:** B0 ~70, A1 → ~71. Headline: "multi-depth views + adaptive weighting = BGRL accuracy at GGD cost."
- **Realistic:** B0 ~69.5, one insight adds +0.3–0.5. Headline: "depth is the underexploited axis; even uniform multi-depth is competitive."
- **Pessimistic:** B0 ≈ SUGRL-k=3 (~69.5), no insight helps. MLP encoder is the bottleneck → pivot.

Gates captured in [[Project Phases and Decision Gates]].

## Scope (locked 2026-04-21)

- **Graph regime: homophilic graphs only.** Evaluation on ogbn-arxiv, ogbn-products, Cora/Citeseer/PubMed. Heterophilic benchmarks (Chameleon, Squirrel, Texas, Actor, Wisconsin) are **out of scope for v1** — clearly marked as future work. This matches the scope of every baseline on our Pareto frontier: [[GGD]], [[SUGRL]], [[BGRL]], [[GraphMAE2]] all restrict to homophilic graphs. Methods that do claim heterophily ([[PolyGCL]], [[GraphACL]]) engineer for it explicitly (spectral high-pass channel / asymmetric predictor) — AD-SSL's monotone-low-pass depth views do not, and retrofitting a high-pass view is an architectural change we defer.
- **Training regime: per-dataset training.** AD-SSL is pretrained and evaluated on the same graph, following the field default (every SSL baseline above does the same). Cross-graph pretrain-and-transfer is a **separate open problem** benchmarked by [[GSTBench]] (CIKM 2025), which finds contrastive methods fail at it and only masked-reconstruction ([[GraphMAE2]]) transfers. We cite GSTBench and position AD-SSL outside that regime. No "foundation model" framing.

These two decisions determine the shape of the main table and the introduction; they also remove Claim 5 (heterophily) and the "transfer probe" defensive gap from [[Novelty Verification Checklist]].

## Known risks

- **[[Less is More]]** (arxiv 2509.25742 v3, ICLR 2026 submission) — closest concurrent work. GCN + MLP as 2 complementary views with global β weighting and direct cosine loss. Differences with AD-SSL: (1) 2 views vs our K; (2) global β vs our per-node α_{i,k}; (3) direct alignment vs our bootstrap. Scale: they hit Arxiv-year (169k nodes, different task) but not standard ogbn-arxiv/products. See [[AD-SSL vs Less is More]].
- MLP-only encoder may be the ceiling, not depth diversity.
- Reviewers may collapse us to "GPRGNN + BGRL" — our defense in [[Reviewer Attacks and Defenses]].
