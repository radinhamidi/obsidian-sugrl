---
title: Idea Ledger
type: synthesis
tags: [neurips-2026, ad-ssl, idea-ledger]
created: 2026-04-23
updated: 2026-04-23
last_response: 2026-04-23 (CA returned INQ-004; D1 closed-falsified; L1 → C3)
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

### C3 — D1: X_0 view + per-depth linear W_k breaks the L_S1 symmetry
- **Status:** closed-falsified — 2026-04-23.
- **Evidence:** [[INQ-2026-04-23-001]]. Primary + V1–V6 variations all hard-fail on Cora + Computers. Best variant on Cora: V2 (no-W_k) at 76.25 — exactly the raw mean-pool floor, still 5 pts below the 81.3 pass bar. V3 (nonlinear W_k) catastrophically below random-7 (16.48 Cora) confirming the hard-constraint on encoder-shaped W_k.
- **Mechanism (CA's new finding):** the L_S1 symmetry is **structural, not parametric**. Under uniform α + shared head, ∂L/∂W_k is depth-symmetric (differs only by X_k − X_{k'}, small on homophilic graphs) and small in magnitude (L_conf/L_div near floor). Weight decay (5e-4) then dominates every W_k individually, shrinking ~50,000× from xavier init to ~2.7e-4 Frobenius. At W_k → 0: p_ik → softmax(h(0)) = constant → Var_k(p_ik) ≡ 0 → α-gradient strictly zero. System parks at an absorbing fixed point.
- **Cross-evidence rejecting alternative diagnoses:** V5 (supervised CE) matches D1-primary to 0.03 pts on Cora → **loss-agnostic / architectural** pathology. V6/best-k one-hot satisfies every pre-registered symmetry signal by fiat yet Z-probe stays at Z_0 level → α is a bystander, W_k collapse is the sole driver. V6/free-table α gives same Z-probe as MLP α → α parameterization is not the bottleneck.
- **Generalized lesson:** any architecture whose contribution to L_S1 is permutation-symmetric across k on homophilic graphs will have its k-distinguishing weights annihilated by weight decay. Giving α or W_k more expressive form does not help; the break has to happen in the *loss* (not the architecture) or in the *routing discretization* (hard routing).
- **Open gate before declaring mechanism confirmed at every layer:** CA flagged one ≤5-line controlled test that distinguishes "WD dominates a small non-zero data gradient" from "data gradient on W_k is literally zero." Queued as **H4/D1'** in backlog.

## Live — currently being tested

_None — awaiting Radin's pick from the 2026-04-23 post-mortem slate (D1'/D2/D3/D4)._

## Backlog — candidates for next live direction

Current slate for Radin to pick from (2026-04-23 post-mortem; see chat for full contribution-sentence / min-exp / venue / reviewer-objection write-up):

### H4 / D1' — WD-exclusion controlled test (cheap diagnostic gate)
- **Premise:** CA's W_k-collapse mechanism is gated on weight decay. Excluding W_k from WD (≤5-line param-group change) distinguishes row-2 (hyperparam) from row-5 (hypothesis genuinely wrong). If W_k stays at xavier ~15 *and* Z-probe moves (either direction), the D1-family is alive with WD-exclusion — possibly above V2 floor. If W_k still collapses without WD (i.e., ∂L/∂W_k is itself zero), D1-family is truly dead and M1 is fully confirmed.
- **Contribution alone:** diagnostic / negative-result-with-teeth. Not a paper on its own.
- **Min experiment:** D1 primary + V1, WD-excluded on W_k, 3 seeds × Cora + Computers. ~1 hour of CA time.
- **Recommendation:** run regardless of which larger direction we pick; the cost is too low not to close this gate.

### B1 / H6 / D2 — Hard routing (Gumbel-softmax top-k)
- **Premise:** the L_S1 symmetry requires `p_i` to be a *linear sum* over k. Gumbel-softmax top-1 or top-2 with temperature anneal makes L_S1 non-linear in α: for each node only one (or two) depth(s) contribute per step, so the depth-symmetric gradient that drove M1+M2 no longer holds. Gradient to W_k becomes per-node sparse and depth-specific.
- **Contribution sentence:** "Per-node discrete depth routing via Gumbel-softmax breaks the depth-symmetry of soft-mixing SSL losses and recovers learnable per-node adaptive depth on homophilic graphs."
- **Min experiment:** Gumbel-softmax α with τ: 5.0 → 0.5 anneal. Primary + τ-schedule sweep on Cora + Computers. If mixture probe > raw best-k on either, extend to ogbn-arxiv.
- **Venue fit:** NeurIPS 2026 possible. Clean architectural novelty vs APPNP/GPR-GNN/ATP (all soft-route), PolyGCL (spectral), BGRL/GraphMAE/GGD (no depth routing at all). The C2+C3 falsifications become the prior-art claim that motivates hard routing.
- **Strongest reviewer objection:** "Gumbel-softmax is well-known; what's your contribution?" Answer: unsupervised application + characterization of why soft routing fundamentally cannot work under homophily (M1/M2/M3) + the fix. Needs reviewer-ready write-up of the symmetry theorem.

### H5 / D3 — Direct loss on Z_k, bypass shared head
- **Premise:** the collapse is gated on `h(Z_k)` being the only path from architecture to loss. If we add a loss term that acts on Z_k itself (e.g., cross-depth contrastive: pull Z_k[i] toward Z_{k'}[i] for same-node, push Z_k[i] from Z_k[j] for different nodes at same depth), ∂L/∂W_k acquires a non-zero, depth-asymmetric gradient that does not depend on uniformity of α.
- **Contribution sentence:** "Depth-discriminative self-supervision (cross-depth contrastive on projected features) breaks the shared-head symmetry in adaptive-depth SSL."
- **Min experiment:** Add depth-contrastive InfoNCE term on Z_k (positive: same node different depths; negative: different nodes same depth). Train α simultaneously. Cora + Computers. λ_contrast sweep.
- **Venue fit:** adjacent to GraphACL / multi-view contrastive lineage. Novelty story weaker — needs clean differentiator from PolyGCL and any existing depth-contrastive work we haven't yet surveyed.
- **Strongest reviewer objection:** "This is multi-view contrastive with depth as the view axis; within epsilon of existing work." Needs literature audit on "depth-as-view" contrastive papers before claiming novelty.

### B2 / D4 — Reframe around training-free per-node depth oracle
- **Premise:** two sessions of negative results on adaptive-depth SSL under homophily suggest the contribution may live in *characterizing* why training fails and providing a training-free oracle that matches or exceeds SSL methods. Per-dataset optimal depth varies 7× (O2); per-node optimal depth likely varies with local homophily + neighborhood class entropy.
- **Contribution sentence:** "No SSL formulation under homophily reliably beats per-node best-k; we characterize a training-free per-node depth oracle (local homophily + feature stability) that matches or exceeds trained adaptive-depth methods at a fraction of the compute."
- **Min experiment:** engineered per-node depth heuristic f(degree, local homophily, neighborhood class entropy) on 6+ homophilic datasets. Benchmark vs best-fixed-k, supervised GPR-GNN α, any SSL α method we have.
- **Venue fit:** TMLR or NeurIPS benchmark track more than NeurIPS main. Could pair with workshop version. Contribution is the characterization, not the heuristic itself.
- **Strongest reviewer objection:** "This isn't machine learning, it's a heuristic." Answer: the paper's novelty is the *negative characterization* of when training helps and when it doesn't — itself a substantial finding given the growing pile of ICLR/NeurIPS submissions in adaptive-depth SSL.

## Deferred backlog — not on current slate, queued for later

### B3 — heterophily scope extension
- **Premise:** the feature-centroid-simplex collapse is *homophily-specific*. On heterophily, different depths point in genuinely different class directions. Track 2 architecture may work there as-is.
- **Status:** requires scope-unlock from 2026-04-21 scope lock. Flagged for later if in-scope options exhaust.

### B4 — input-feature augmentation as the view axis
- **Premise:** instead of propagation-depth views, use feature-dropout / attribute-masking views of raw X. Cross-depth SSL replaced by cross-augmentation SSL.

### B5 — predictive SSL ("predict Â^{k+1} X from Â^k X")
- **Premise:** reconstruction task instead of instance-discrimination. Closer to GraphMAE but decoupled.

### B6 — learnable propagation coefficients (unsupervised PolyGCL/GPR-GNN-flavored)
- **Premise:** instead of fixed Â^k X precompute + mixture, learn `Σ_k β_k Â^k X` with β per-depth or per-node via SSL.

### B7 — packaging-only: efficiency-delta claim paired with any passing direction
- **Premise:** if any of the above passes, pair the accuracy claim with wall-clock comparison to per-epoch-propagation SSL methods.

## Surprising side observations — revisit when framing shifts

### O1 — low-label supervised training under-uses deep propagation
- **Observed in:** INQ-003 V5 supervised Cora. 140-label supervised α routed to k=1 (Â^1 X = 73.81), but the test-optimal single depth is k=8 (Â^8 X = 78.91). A ~5-pt generalization gap between training-CE minimum and test-minimum over depths.
- **Revisit when:** framing shifts toward "SSL overcomes label-scarcity-induced bias in supervised propagation selection."

### O2 — per-dataset optimal depth varies by 7× and is non-monotonic across datasets
- **Observed in:** [[INQ-2026-04-22-001]] + INQ-003 raw-probe tables. Cora k=1→k=8 monotonically **improves** (73.81 → 78.91). Computers k=1→k=8 monotonically **degrades** (87.51 → 76.27). ogbn-arxiv k=4 peaks. Three datasets, three qualitatively different curves.
- **Revisit when:** framing shifts toward "dataset-conditional depth selection" (→ B2/D4).

### O3 — α-scorer routes nodes structurally even when Z is constant
- **Observed in:** INQ-004 V6/free-table α on Cora. Dom-k counts across 2708 nodes: {k=0: 847, k=1: 654, k=2: 496, k=4: 406, k=8: 302} with mild homophily correlation (−0.012) and class spread (0.068). All five depths non-trivially represented. Yet Var_k(p_ik) = 0 exactly, because W_k → 0 → p_ik = softmax(h(0)) = constant.
- **Interpretation:** the α-scorer (MLP on Z_k + depth embedding) *has* the capacity to learn per-node structure from Z even when downstream is broken. If α is given a training signal that doesn't route through h(0), the routing may land on meaningful per-node depths without architectural surgery.
- **Revisit when:** exploring loss-surgery directions (D3, or any direction where α is trained against a supervisor that is not p_i).

### O4 — V3 below-random + high-variance is the signature of a dead architecture
- **Observed in:** INQ-004 V3 nonlinear W_k on Cora. Z-probe 16.48 ± 4.17 (7-class random ≈ 14.3). Per-seed variance is an order of magnitude larger than any non-dead variant (±0.2–1.0 elsewhere). Mechanism: ReLU + WD produces near-zero pre-activations; seed-dependent h(b) lands at a slightly informative constant for some seeds, pure noise for others.
- **Revisit when:** diagnosing any future experiment that produces "below-random accuracy + huge seed variance" — likely an architecture that bottoms out at a seed-dependent constant, not a genuinely trainable network.

## Related

- [[Research Agent Operating Protocol]] — mandates this ledger.
- [[INQ-2026-04-22-001]], [[INQ-2026-04-22-002]], [[INQ-2026-04-22-003]] — source inquiries for closed rows.
- [[Thesis]] — current contribution sentence; update when a live row passes or closes.
