---
title: Less is More
type: entity
kind: method
venue: arxiv / ICLR 2026 submission (v3, 2026-03-20)
authors: Yanan Zhao, Feng Ji, Jingyang Dai, Jiaze Ma, Wee Peng Tay (NTU Singapore)
url: https://arxiv.org/abs/2509.25742
tags: [method, concurrent-work, high-priority, ingested]
created: 2026-04-21
updated: 2026-04-21
ingested_from: raw/papers/Less is More.pdf
---

# "Less is More: Towards Simple Graph Contrastive Learning"

Zhao, Ji, Dai, Ma, Tay (NTU Singapore). arxiv 2509.25742 **v3, 20 Mar 2026**. ICLR 2026 submission. Closest architectural cousin to AD-SSL; high-priority monitor.

## Core principle (novel-to-them)

> "Cancellation is stronger in the sum of two vectors when they are less correlated."

GCL framed as **noise decoupling**: construct two views whose *noise components* (feature noise, structural noise) are weakly correlated. When β·Z_s + (1−β)·Z_f is computed for downstream use, independent noise partially cancels while correlated signal survives. Theoretical support in Prop. 1 + Thm. 1 (Appendix B) using graph signal processing on the normalized Laplacian.

## Architecture (exact)

- **Z_s** = k-layer GCN on (A, X). Captures structural noise. `H^(ℓ+1) = σ(Ã H^(ℓ) W^(ℓ))`.
- **Z_f** = 1-layer MLP on X. Isolates feature noise.
- **Loss**: standard cosine-mean contrastive (from [[BGRL]]/Thakoor):
  `L = 1 − (1/N) Σᵢ ⟨Z_s,i, Z_f,i⟩ / (||Z_s,i||·||Z_f,i||)`
- **Downstream**: `Z = β·Z_s + (1−β)·Z_f`, β ∈ {0.5, tuned}.

**Defaults**: k = 2 GCN layers, L = 1 MLP layer, hidden dim swept 128–2048.

## Crucial: exactly 2 views, always

k is the **number of GCN layers**, not the number of views. There is always one GCN view and one MLP view. No multi-depth, no multi-scale, no per-node weighting. β is a **global** scalar across all nodes.

## No augmentation, no negatives

- Loss is cosine-alignment-only — no negatives, no InfoNCE.
- Paper explicitly ablates adding augmentation / negatives (Table 5) and shows no gains → Occam's razor argument.

## Performance highlights (from paper)

**Heterophilic datasets (where they win, Table 2):**
- Wisconsin 85.29, Cornell 71.35, Texas 78.38, Actor 36.79.
- Roman-empire (22k nodes) **78.21**. Arxiv-year (169k) **46.15**.

**Homophilic (Table 3): middling**
- Cora 77.26 (lower than many baselines, e.g. SDMG 83.60).
- Citeseer 70.12, PubMed 79.00, Computer 87.65, Photo 93.41.

**Scale evidence**:
- Arxiv-year: 169k nodes. Training 3.96 s, inference 1.89 s, storage **44.4 GB** (note: very high).
- Roman-empire: 22k nodes. Competitive and fast.
- **Does NOT report standard ogbn-arxiv / ogbn-products / ogbn-papers100M.** Arxiv-year is a different task (publication-year prediction from Lim et al. 2021), not OGB's category prediction.

## Implications for AD-SSL differentiation

Our onboarding said "does not run at OGB scale" — this is technically correct but soft. A careful reviewer will note they do hit 169k nodes (Arxiv-year) at <4 s training. See [[AD-SSL vs Less is More]] for the sharpened differentiation.

Main differentiators (all intact):
1. **Multi-depth views** (K=4) vs their 2-view (MLP + one k-layer GCN).
2. **Per-node adaptive weighting** α_{i,k} vs their global scalar β.
3. **Bootstrap loss** across depth pairs vs their direct cosine alignment.
4. **No GCN forward during training** (decoupled precompute) vs their k-layer GCN forward per epoch.

Robustness angle: they claim strong robustness under adversarial attacks (Table 6, 7) via noise-decoupling. If reviewers demand robustness, we'd need similar evaluation.

## What to monitor

- ICLR 2026 acceptance decision.
- v4+ of the paper (especially: any extension to multi-depth, per-node weighting, or standard OGB datasets).
- Ji et al. 2025 ([[Rethinking graph neural networks from a geometric perspective of node features]], ICLR 2025) — cited heavily for Prop. 2, **confirmed same group (Tay, NTU)**. Fully ingested 2026-04-21; see that page for the feature-centroid-simplex framework underpinning their noise argument.

## Risks to our novelty

Low-to-moderate, as of 2026-04-21:
- Their noise-decoupling theory is orthogonal to our multi-depth mechanism — both could be cited as different mitigation strategies.
- If v4 adds multi-depth GCN views with the same cosine loss, our per-node adaptive weighting + bootstrap remains differentiated.
- If they show OGB scale in a revision, our "scale study" claim weakens — we must execute ogbn-products / ogbn-papers100M to lock this in.

## Required Coding Agent work

Reproduce GCN-MLP with their official config on:
- Arxiv-year (direct from paper)
- **ogbn-arxiv** (not in paper — we provide the number)
- **ogbn-products** (not in paper — we provide the number)

These three rows go in our main table as the "[[Less is More]] (reproduced)" baseline.
