---
title: Idea Ledger
type: synthesis
tags: [neurips-2026, ad-ssl, idea-ledger]
created: 2026-04-23
updated: 2026-04-23
---

# Idea Ledger

Append-only record of hypotheses tested, hypotheses queued, and surprising side observations. Mandated by [[Research Agent Operating Protocol]]. When a direction closes, pull the next candidate from here rather than starting from a blank page.

Status values: `live`, `testing`, `closed-falsified`, `closed-superseded`, `backlog`.

## Closed — hypotheses tested, direction falsified

### C1 — Encoder-based B0 (InfoNCE between Â^k views) beats Â^{k*} X
- **Status:** closed-falsified — 2026-04-22.
- **Evidence:** [[INQ-2026-04-22-001]], [[INQ-2026-04-22-002]]. Across 6 encoder configurations (strict, wide, predictor+EMA, edge-dropout, edge-dropout+BGRL-lite) on Cora / Computers / ogbn-arxiv, no trained encoder beat the parameter-free best-depth linear probe.
- **Mechanism CA identified:** InfoNCE between highly-correlated Â^k views learns the shared component and discards class structure; on pre-propagated features the "shared component" already is class structure, and the encoder destroys it.

### C2 — Per-node α over raw Â^k X (Track 2 v2) beats Â^{k*} X under SSL loss S1+S3
- **Status:** closed-falsified — 2026-04-22.
- **Evidence:** [[INQ-2026-04-22-003]]. Primary + V1–V6 variations all failed pass bar on Cora + Computers. α never leaves uniform under L_S1 regardless of M, β, λ, or α parameterization. Supervised V5 moves α only on Cora, and to the wrong depth.
- **Mechanism (CA's diagnosis, confirmed by V3 λ=0 + V4 free-table + V6 global γ):** shared-head L_S1 has a depth-symmetry when p_{ik} are similar across k, which they are on homophilic graphs (→ see [[Rethinking graph neural networks from a geometric perspective of node features]] simplex collapse). No gradient to α under any α parameterization. See INQ-003 RESPONSE for full analysis.

## Live — currently being tested

### L1 — D1: break the symmetry via X_0 view + per-depth projection W_k
- **Status:** live, inquiry pending (INQ-2026-04-23-001 to be filed 2026-04-23).
- **Hypothesis:** adding raw-feature view X_0 (outside the simplex-collapse regime) and/or per-depth learnable projection W_k before a shared classifier head provides depth-distinguishable p_{ik}, breaking CA's L_S1 symmetry. Expected outcome: α moves off uniform, mixture probe approaches or exceeds Â^{k*} X on Cora + Computers.
- **Contribution sentence if passes:** "Raw-feature views and per-depth projections above Â^k X recover learnable SSL signal that uniform mixing and encoder-based SSL miss; per-node α provides meaningful depth adaptivity on homophilic graphs."
- **Falsifying observation:** if W_k and X_0 both fail to break uniform α, or if they move α but the mixture does not beat Â^{k*} X, the broader "learnable combination of depths beats best-single-depth" claim is dead on homophilic graphs and we pull from the backlog.

## Backlog — queued alternatives from 2026-04-22 post-mortem slate

### B1 — D2: hard / sparse routing (Gumbel-softmax top-k) instead of soft α
- **Premise:** the pathology may be "soft averaging of near-identical things"; a hard top-1 or top-2 routing has fundamentally different gradient dynamics.
- **Next step if L1 fails:** implement Gumbel-softmax α with temperature anneal; run primary + M sweep on Cora + Computers.

### B2 — D3: reframe around "dataset-conditional propagation budget"
- **Premise:** per-dataset optimal depth varies by 7× across our three datasets (see O2 below). Build a training-free per-node depth oracle that matches SSL.
- **Next step if L1 fails and D2 fails:** characterize best-per-node-k across 6+ homophilic datasets; compare to supervised and SSL α.
- **Venue fit:** NeurIPS 2026 possible; more naturally TMLR or a benchmark track.

### B3 — D4: heterophily scope extension
- **Premise:** the feature-centroid-simplex collapse is *homophily-specific*. On heterophily, different depths point in genuinely different class directions. Track 2 architecture may work there as-is.
- **Status:** requires scope-unlock from 2026-04-21 scope lock. Flagged for later if in-scope options exhaust.

### B4 — input-feature augmentation as the view axis
- **Premise:** instead of propagation-depth views, use feature-dropout / attribute-masking views of raw X. Cross-depth SSL replaced by cross-augmentation SSL.

### B5 — predictive SSL ("predict Â^{k+1} X from Â^k X")
- **Premise:** reconstruction task instead of instance-discrimination. Closer to GraphMAE but decoupled.

### B6 — learnable propagation coefficients (unsupervised PolyGCL/GPR-GNN-flavored)
- **Premise:** instead of fixed Â^k X precompute + mixture, learn `Σ_k β_k Â^k X` with β per-depth or per-node via SSL.

### B7 — packaging-only: efficiency-delta claim paired with any L1–B6
- **Premise:** if any of the above passes, pair the accuracy claim with wall-clock comparison to per-epoch-propagation SSL methods.

## Surprising side observations — revisit when framing shifts

### O1 — low-label supervised training under-uses deep propagation
- **Observed in:** INQ-003 V5 supervised Cora. 140-label supervised α routed to k=1 (Â^1 X = 73.81), but the test-optimal single depth is k=8 (Â^8 X = 78.91). A ~5-pt generalization gap between training-CE minimum and test-minimum over depths.
- **Revisit when:** framing shifts toward "SSL overcomes label-scarcity-induced bias in supervised propagation selection."

### O2 — per-dataset optimal depth varies by 7× and is non-monotonic across datasets
- **Observed in:** [[INQ-2026-04-22-001]] + INQ-003 raw-probe tables. Cora k=1→k=8 monotonically **improves** (73.81 → 78.91). Computers k=1→k=8 monotonically **degrades** (87.51 → 76.27). ogbn-arxiv k=4 peaks. Three datasets, three qualitatively different curves.
- **Revisit when:** framing shifts toward "dataset-conditional depth selection" (→ B2/D3).

## Related

- [[Research Agent Operating Protocol]] — mandates this ledger.
- [[INQ-2026-04-22-001]], [[INQ-2026-04-22-002]], [[INQ-2026-04-22-003]] — source inquiries for closed rows.
- [[Thesis]] — current contribution sentence; update when a live row passes or closes.
