---
title: Idea Ledger
type: synthesis
tags: [neurips-2026, ad-ssl, idea-ledger]
created: 2026-04-23
updated: 2026-04-23
last_response: 2026-04-23 (CA returned INQ-006 [INQ-2026-04-23-003]; **D6c hard-passes both Cora +2.93 and Computers +0.73** — first simultaneous double-pass in the project; V2-E1 τ_p sweep confirmed entropy-from-kmeans on raw X_k is structurally dead on these datasets regardless of τ)
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
- **Open gate before declaring mechanism confirmed at every layer:** CA flagged one ≤5-line controlled test that distinguishes "WD dominates a small non-zero data gradient" from "data gradient on W_k is literally zero." Queued as **H4/D1'** in backlog — **now closed, see D1' outcome below.**

### D1' (bundled with INQ-005) — WD-exclusion test on D1 architecture (linear W_k only)
- **Status:** closed — **linear W_k** D1 dead both ways, 2026-04-23. **Does NOT close nonlinear-W_k or skip-connection D1 variations** — those were foreclosed by the (now-retired) linear-only constraint. See D1-family open variations below.
- **Evidence:** [[INQ-2026-04-23-002]] D1' section. Excluding W_k from weight decay replaces WD-shrinkage collapse with an opposite degenerate solution: ||W_k||_F grows to 100–300 from xavier ~15 (17–20× growth); cos(W_k, W_k') ≥ 0.5 with some pairs ≥ 0.95 (co-linear). **α now passes every movement diagnostic** (mean-std across k = 0.28, frac α-entropy < 0.8·lnK = 1.00) — but Z-probe crashes: Cora 55.18 ± 4.95 (vs D1-primary 59.60 and raw mean-pool floor 76.25); Computers 40.43 ± 0.37 (vs D1-primary 76.25 and raw 86.11).
- **Reading:** D1 architecture has two absorbing fixed points. WD-dominant (INQ-004): W_k → 0, p_ik = softmax(h(0)) = constant, α trivially uniform, Z stuck at best single-depth floor. WD-free (D1'): W_k explodes, shared-head softmax saturates per-depth, α freely spreads to "pick the right class" — but mixing saturated heads is not meaningful depth selection, so probe collapses below random. Both degenerate; no middle ground under L_S1 + shared head on homophilic features. Mechanism M1 (shared-head depth-symmetry) and M2 (near-zero data gradient on W_k) are jointly confirmed.
- **Row-2 vs row-5 gap (INQ-004 post-mortem):** closed. WD is not just the fast path to the collapse — the architecture is structurally degenerate regardless of WD.

### C4 — Entropy-driven per-node depth routing (E1/E2/E3/E4)
- **Status:** **partially tested — entropy-as-signal never actually tested under a valid spec.** Direction not yet falsified; spec bug masked the test. 2026-04-23.
- **Evidence:** [[INQ-2026-04-23-002]]. E1 (training-free k-means entropy α), E2 (learnable α, entropy-min loss, frozen H), E3 (cross-depth InfoNCE + λ·L_ent), E4 (supervised ridge entropy ceiling). All four reported by CA as hard-fail under pre-registered compound rule.
- **Honest RA read of the numbers (disagrees with CA's verdict labels):**
  1. **E3 Cora probe = 81.73 ± 0.27 vs hard bar 78.87 is a GENUINE +2.86 pt probe pass at ~10σ.** CA called it hard-fail because α stayed uniform, but the pre-registered compound rule was defensive against a different failure mode (V6 best-k oracle). The Cora probe pass is real; the story is just not the α-driven one we wrote.
  2. **E3-V-WD Cora = 80.22 ± 0.57 also passes the hard bar** (+1.35, ~2.4σ).
  3. **The "flat H across k" finding is a τ_p=1.0 spec artifact, not a theorem.** Cora E1 H values (1.946 × 5 depths) equal log(7) = 1.9459 to 4 decimals — the softmax is numerically uniform over classes. Same on Computers (2.302 ≈ log(10)). τ_p=1.0 on k-means distances pre-saturated entropy regardless of underlying cluster quality. **Entropy-as-signal was never given a temperature at which it could respond.** (The spec bug is mine — I wrote `τ_p = τ_α = 1.0 default`.)
  4. **E4's "failed flip test" uses ridge-probe softmax as a confidence source.** Ridge is trained for MSE on 20-per-class labels, not for calibrated probability. With high-dim features + few labels, test-node ridge logits reflect feature-space overfitting geometry, not class confidence. Q5's "E4 is the supervised ceiling" framing was wrong: the ceiling was never supervised-well-calibrated; it was supervised-poorly-operationalized.
  5. **E3 Computers failure (79.48) is a projection bottleneck, not an entropy or α failure.** Raw Â¹X = 87.49; projecting 767 → 128 strictly loses info. E3-V-WD Z_k probes (81/84/84/83/80) are 2–4 pts below their raw counterparts even with healthy W_k. The 128-dim projection is the wrong architectural choice here; swap to d_proj = F_in or identity-skip.
- **What is confirmed dead (updated 2026-04-23 by [[INQ-2026-04-23-003]] V2-E1 τ_p sweep):** entropy-from-kmeans on raw X_k as an α-routing signal is **structurally dead on Cora + Computers across all tested τ_p ∈ {0.001, 0.01, 0.05, 0.1, 1.0}**. Computers: argmin_k H locks on k=0 at 98.7–99.98% at every τ_p (never flips to k=1 where the signal lives). Cora: argmin flips between k=0 (low τ) and k=8 (high τ) but α never sharpens enough to matter (α mean-std 0.0002 at τ=1.0 → Z = raw mean-pool). None of the 5 τ settings hard-pass either dataset; best probe (Cora τ_p=0.05 = 76.33; Computers τ_p=1.0 = 86.12) stays at or below raw mean-pool floor. corr(argmin-k, degree / local-homophily / label-entropy) ≈ 0 at every τ_p. **E1 is closed.** E2 and E4 inherit E1's dead signal on raw X_k → also closed on raw X_k. See O7 update below.
- **What is NOT confirmed dead:** entropy-from-kmeans on **cross-depth-InfoNCE-transformed features Z_k** (not raw X_k). Z_k after D6c training has fundamentally different k-means structure than X_k; that signal has not been tested. Queued as Config C in [[INQ-2026-04-23-004]] (D6c+α).
- **What is confirmed alive — MAJOR (updated 2026-04-23):** cross-depth InfoNCE with residual projection is a working SSL pretext. See D6c under Live below.

## Live — currently being tested

### D6c — Cross-depth InfoNCE with residual F_in-preserving projection (no α at training)
- **Status:** **live, primary-pass 2026-04-23.** First configuration in the project to hard-pass both Cora and Computers simultaneously. Extensions running in [[INQ-2026-04-23-004]].
- **Evidence (3 seeds, from [[INQ-2026-04-23-003]]):** Cora Z_mean 81.80 ± 0.20 (bar 78.87, +2.93 at ~14σ across-seed); Computers Z_mean 88.26 ± 0.40 (bar 87.53, +0.73 at ~1.8σ across-seed, ~3.2σ in stderr). Z_concat within 0.15 pts of Z_mean on both datasets. Every per-depth Z_k probe beats its raw Â^k X counterpart on both datasets.
- **Architecture:** `Z_k = X_k + W_k X_k`, W_k ∈ R^{F_in × F_in} linear one per k ∈ {0,1,2,4,8}. Flat cross-depth InfoNCE: positive pairs (Z_k[i], Z_{k'}[i]) for k ≠ k'; negative pairs (Z_k[i], Z_{k'}[j]) for j ≠ i. No α, no L_ent, no confidence weighting. 200 epochs, Adam lr=0.01, WD=5e-4, τ_c=1.0. Readout: Z_mean = (1/K) Σ_k Z_k.
- **Mechanism (RA independent read):** residual guarantees Z_k information floor ≥ X_k (can't drop below raw signal); cross-depth InfoNCE pushes each W_k to encode cross-depth-discriminative structure on top of the floor. The outcome is not "residual rescued k=1" — it's a clean ensemble lift: D6a (no residual) HURT Computers k=1 by −4.50, D6c lifted every depth (+0.71 at k=1, +5.13 at k=0, +2.04 at k=2, +4.52 at k=4, +6.23 at k=8). Different mechanism: D6a's linear projection under WD shrinks W_k so Z_k ≈ W_k X_k ≈ 0; D6c's residual lets the InfoNCE gradient shape a useful W_k transformation regardless of ||W_k||_F. Depth-decorrelation also differs: cos(W_k, W_k') drops to 0.04 (k=4 vs k=8) on Cora, indicating genuinely depth-distinct projections on Cora; stays high 0.78–0.99 on Computers (residual+InfoNCE couldn't decouple depths there, but every depth still improved).
- **Caveat / open:** Computers +0.73 is tight (1.8σ across-seed at 3 seeds). 5-seed confirmation queued as Config B in [[INQ-2026-04-23-004]]. ogbn-arxiv extension queued as Config A.
- **Open thesis question:** does α on top of D6c help or hurt? If α-free D6c is the best-performing variant, the paper pivots from "adaptive-depth SSL" to "cross-depth InfoNCE on pre-propagated features with residual projection" — still novel, but thesis framing changes. Queued as Config C in [[INQ-2026-04-23-004]].
- **Contribution sentence (current draft):** "Cross-depth instance-discrimination contrastive learning on precomputed multi-depth features, with a residual F_in-preserving per-depth projection, produces a depth-diverse representation that hard-passes best-single-depth linear probes on Cora and Computers. The depth itself — not an augmentation view — is the contrastive axis; the method operates fully at precompute time with no encoder."
- **Reviewer attack:** "How does this differ from multi-view contrastive methods that use depth as a view?" Needs literature audit on prior "Â^k vs Â^{k'} contrastive at precompute" work. Queued.

## Backlog — candidates for next live direction

Current slate (2026-04-23 INQ-005 post-mortem). D6 and V2 are the RA's top picks based on the independent read of INQ-005 numbers.

### D6 — Cross-depth InfoNCE as multi-depth SSL pretext (no α, no mixing)
- **Premise:** cash in the latent E3-Cora pass (O5 below). Drop α entirely. Train W_k per depth via cross-depth InfoNCE (positive pairs: same node, different depths; negative: different nodes, same depth). Probe either mean-pool `(1/K) Σ_k Z_k` OR concat `[Z_0 ‖ Z_1 ‖ ... ‖ Z_K]`.
- **Constraint retirement:** the "W_k must be linear-only" rule inherited from INQ-001/002 (encoder-destroys-signal) does **NOT** apply to cross-depth contrastive. INQ-001/002's mechanism requires highly-correlated same-depth views that an encoder can collapse to a shared component; cross-depth views are genuinely different. D6 variations may and should include nonlinear W_k, skip/residual connections, d_proj = F_in, MLP projections, per-depth nonlinear heads. See `feedback_no_constraint_propagation.md`.
- **Variations opened by retirement:**
  - D6a: linear W_k, d_proj = 128 (= INQ-005 E3 baseline, Cora already passes)
  - D6b: linear W_k, d_proj = F_in (information-preserving; expected to close Computers gap)
  - D6c: skip-connection `Z_k = X_k + W_k X_k` with d_proj = F_in (residual preserves info while adding depth-specific transform)
  - D6d: nonlinear W_k = small MLP (e.g. F_in → h → F_in with ReLU + residual)
  - D6e: per-depth nonlinear head + concat readout `[Z_0 ‖ Z_1 ‖ ... ‖ Z_K]` (gives probe all depths as separate features)
- **Contribution sentence:** "Cross-depth instance-discrimination contrastive learning on precomputed multi-depth features produces a depth-diverse representation that equals or beats the best single-depth probe, without per-node adaptive routing."
- **Min experiment:** E3 with α-free readout across D6a/D6b/D6c on Cora + Computers, 3 seeds. If D6c or D6b closes Computers, extend to D6d/D6e + ogbn-arxiv. ~6 hours CA time.
- **Venue fit:** NeurIPS 2026. Differentiator from PolyGCL (spectral filter, not contrastive), BGRL/GraphMAE (single-depth views), GraphACL (same-depth InfoNCE on augmented views). Novelty: *depth* as the contrastive axis, done at precompute time with no encoder.
- **Strongest reviewer objection:** "How is this different from multi-view contrastive with depth as view?" Answer: prior work uses augmented views at a fixed depth; we use the propagation depth itself as the view, decoupled from training compute. Needs audit of any paper already doing "Â^k vs Â^{k'} contrastive" — flag for literature check.
- **Status:** strongest signal in hand (already +2.86 pts on Cora from INQ-005 E3). Top pick.

### V2 — Retest entropy family with tuned τ_p (CLOSED 2026-04-23, on raw X_k)
- **Status:** closed-falsified for entropy-from-kmeans on **raw X_k**. See C4 update above and O7 below. V2-E1 τ_p sweep ([[INQ-2026-04-23-003]]) tested τ_p ∈ {0.001, 0.01, 0.05, 0.1, 1.0} on Cora + Computers. At no τ_p does the probe hard-pass; on Computers the argmin-k distribution locks to k=0 at 98.7–99.98% at every τ_p (structural dataset property, not a temperature issue). argmin-k has zero correlation with degree/local-homophily/label-entropy at best-probe τ_p.
- **What remains open:** entropy-from-kmeans on cross-depth-InfoNCE-transformed Z_k (different context, different cluster geometry). Running as Config C in [[INQ-2026-04-23-004]]. Constraint non-propagation rule applies: V2-E1 raw-X_k failure does NOT auto-transfer to Z_k.
- **CE-probe replacement for E4:** not queued. E4's ridge-softmax was the defensibly-supervised ceiling; the "ceiling" direction only matters if the unsupervised signal responds, and it doesn't. If Config C surfaces a responsive unsupervised entropy signal on Z_k, revisit an E4-CE probe on Z_k at that point.

### H4 / D1' — linear-W_k WD-exclusion controlled test (CLOSED 2026-04-23)
- See "Closed" section above. Confirmed linear-W_k D1 dead both ways. Not a live candidate IN THE LINEAR REGIME.

### D1-family reopened — nonlinear / skip-connection / residual W_k variations
- **Status:** reopened 2026-04-23 after retiring the linear-only constraint (see `feedback_no_constraint_propagation.md`). INQ-004 V3 (nonlinear W_k that crashed to random) is ONE data point with specific arch + WD + shared-head; not a blanket ban.
- **Variations to consider if we return to D1 family:**
  - D1-NL-a: D1 architecture with `W_k = MLP(F_in → h → F_in)` + residual, linear classifier head, L_S1 loss. Tests whether nonlinear-W_k-with-residual escapes the collapse (residual guarantees ||W_k X + X|| stays ≥ ||X||, so p_ik cannot collapse to softmax(h(0))).
  - D1-NL-b: D1 with `Z_k = X_k + W_k X_k` (linear skip), which is a strict superset of V2 (no W_k) and should probe-dominate V2.
  - D1-NL-c: D1 with per-depth MLP head instead of shared head (breaks the M1 shared-head symmetry directly).
- **Note:** D1 family's fundamental problem (L_S1 depth symmetry + shared head) is separate from whether W_k is linear. O6 (two absorbing fixed points under shared head) is the real killer. So D1-NL-c (per-depth head) is the only one of these that attacks the actual mechanism; D1-NL-a/b only address the W_k collapse surface symptom. Priority: D1-NL-c > D1-NL-a > D1-NL-b if we reopen this direction at all.

### D5 / B1 / H6 / D2 — Hard routing (Gumbel-softmax top-k)
- **Premise:** the L_S1 symmetry requires `p_i` to be a *linear sum* over k. Gumbel-softmax top-1 or top-2 with temperature anneal makes L_S1 non-linear in α: for each node only one (or two) depth(s) contribute per step, so the depth-symmetric gradient that drove M1+M2 no longer holds. Gradient to W_k becomes per-node sparse and depth-specific.
- **Contribution sentence:** "Per-node discrete depth routing via Gumbel-softmax breaks the depth-symmetry of soft-mixing SSL losses and recovers learnable per-node adaptive depth on homophilic graphs."
- **Min experiment:** Gumbel-softmax α with τ: 5.0 → 0.5 anneal. Primary + τ-schedule sweep on Cora + Computers. If mixture probe > raw best-k on either, extend to ogbn-arxiv.
- **Venue fit:** NeurIPS 2026 possible. Clean architectural novelty vs APPNP/GPR-GNN/ATP (all soft-route), PolyGCL (spectral), BGRL/GraphMAE/GGD (no depth routing at all). The C2+C3 falsifications become the prior-art claim that motivates hard routing.
- **Strongest reviewer objection:** "Gumbel-softmax is well-known; what's your contribution?" Answer: unsupervised application + characterization of why soft routing fundamentally cannot work under homophily (M1/M2/M3) + the fix. Needs reviewer-ready write-up of the symmetry theorem.

### D7 — Across-k entropy (node-prediction disagreement across depths)
- **Premise:** INQ-005's entropy spec computed H over M classes for a fixed depth k, giving K entropies per node. The alternative: for each node i, look at the K predictions `p_i0, p_i1, ..., p_iK` (one distribution per depth), compute inter-depth disagreement (e.g., mean pairwise KL, or H over the *concatenated* prediction distribution). This flips the asymmetry direction: a node whose predictions AGREE across depths is "easy, use any depth"; a node whose predictions DISAGREE is "ambiguous, needs careful depth selection." Structurally sidesteps the τ_p saturation problem (entropy is over K values, not M class probabilities).
- **Contribution sentence:** "Per-node depth uncertainty measured as cross-depth prediction disagreement produces a calibrated signal for adaptive depth routing in SSL."
- **Min experiment:** compute cross-depth KL matrix per node; route α by 1/max disagreement or similar; probe on Cora + Computers.
- **Venue fit:** NeurIPS 2026 if it works. Novelty: novel entropy operationalization for depth routing.

### H5 / D3 — Direct loss on Z_k, bypass shared head
- **Premise:** the collapse is gated on `h(Z_k)` being the only path from architecture to loss. If we add a loss term that acts on Z_k itself (e.g., cross-depth contrastive: pull Z_k[i] toward Z_{k'}[i] for same-node, push Z_k[i] from Z_k[j] for different nodes at same depth), ∂L/∂W_k acquires a non-zero, depth-asymmetric gradient that does not depend on uniformity of α.
- **Contribution sentence:** "Depth-discriminative self-supervision (cross-depth contrastive on projected features) breaks the shared-head symmetry in adaptive-depth SSL."
- **Min experiment:** Add depth-contrastive InfoNCE term on Z_k (positive: same node different depths; negative: different nodes same depth). Train α simultaneously. Cora + Computers. λ_contrast sweep.
- **Venue fit:** adjacent to GraphACL / multi-view contrastive lineage. Novelty story weaker — needs clean differentiator from PolyGCL and any existing depth-contrastive work we haven't yet surveyed.
- **Strongest reviewer objection:** "This is multi-view contrastive with depth as the view axis; within epsilon of existing work." Needs literature audit on "depth-as-view" contrastive papers before claiming novelty.

### B2 / D4 / D8 — Reframe around training-free per-node depth oracle
- **Premise:** two sessions of negative results on adaptive-depth SSL under homophily suggest the contribution may live in *characterizing* why training fails and providing a training-free oracle that matches or exceeds SSL methods. Per-dataset optimal depth varies 7× (O2); per-node optimal depth likely varies with local homophily + neighborhood class entropy.
- **Contribution sentence:** "No SSL formulation under homophily reliably beats per-node best-k; we characterize a training-free per-node depth oracle (local homophily + feature stability) that matches or exceeds trained adaptive-depth methods at a fraction of the compute."
- **Min experiment:** engineered per-node depth heuristic f(degree, local homophily, neighborhood class entropy) on 6+ homophilic datasets. Benchmark vs best-fixed-k, supervised GPR-GNN α, any SSL α method we have.
- **Venue fit:** TMLR or NeurIPS benchmark track more than NeurIPS main. Could pair with workshop version. Contribution is the characterization, not the heuristic itself.
- **Strongest reviewer objection:** "This isn't machine learning, it's a heuristic." Answer: the paper's novelty is the *negative characterization* of when training helps and when it doesn't — itself a substantial finding given the growing pile of ICLR/NeurIPS submissions in adaptive-depth SSL.

## Deferred backlog — not on current slate, queued for later

### B3 / D9 — heterophily scope extension
- **Premise:** the feature-centroid-simplex collapse is *homophily-specific*. On heterophily, different depths point in genuinely different class directions. Track 2 architecture may work there as-is.
- **Status:** requires scope-unlock from 2026-04-21 scope lock. Flagged for later if in-scope options exhaust. INQ-005's finding that every pathology we've hit traces back to simplex collapse strengthens the mechanistic case.

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

### O5 — Cross-depth InfoNCE levels weak depths on Cora (E3 W_k lift, α-uniform)
- **Observed in:** [[INQ-2026-04-23-002]] E3 primary + E3-V-WD. On Cora, cross-depth InfoNCE trained W_k takes every depth's individual probe from raw → post-training: k=0 (46.79 → 75.68, +28.9 pts), k=1 (73.89 → 80.65, +6.8), k=2 (78.22 → 80.99), k=4 (77.91 → 80.67), k=8 (78.86 → 80.67). Mean-pool probe = 81.73 ± 0.27, beating hard bar 78.87 by +2.86 at ~10σ. α uniform, so the gain is entirely from W_k.
- **On Computers, opposite direction:** raw k=1 = 87.49, post-training Z_1 = 79.40 (E3 primary) or 84.53 (E3-V-WD) — projection bottleneck when raw features are already strong at 767-dim.
- **Interpretation:** cross-depth InfoNCE as a *multi-depth SSL pretext* is itself a method. Drop α, use concat or skip-connection readout. Separate paper direction, see D6 in current slate.

### O6 — D1 architecture has two absorbing fixed points under L_S1 + shared head
- **Observed in:** [[INQ-2026-04-23-002]] D1' section. Removing W_k from WD does not rescue D1 — it flips the collapse from WD-shrinkage (||W_k||_F → 2.7e-4) to WD-free-explosion (||W_k||_F → 100-300, 17-20× xavier). In the second regime, α passes every movement diagnostic (mean-std 0.28, frac concentrated 1.00) — yet Z-probe crashes to Cora 55 / Computers 40 because shared-head softmax saturates per-depth and mixing saturated heads is not meaningful depth routing.
- **Interpretation:** L_S1 + shared head on homophilic features admits no non-degenerate W_k geometry. The INQ-004 row-2/row-5 gap closes on row-5 (hypothesis genuinely wrong), not row-2 (hyperparam).

### O7 — τ_p=1.0 on k-means distances saturates p_ik to class-uniform (and fixing it doesn't help on raw X_k)
- **Observed in:** [[INQ-2026-04-23-002]] E1/E2 per-depth H tables (saturation); [[INQ-2026-04-23-003]] V2-E1 τ_p sweep (fixing saturation doesn't rescue signal on raw X_k).
- **Saturation:** Cora H ≡ 1.946 = log(7) to 4 decimals on every depth and seed at τ_p=1.0; Computers H ≡ 2.302 ≈ log(10). Fully saturated.
- **Sweep result (raw X_k, [[INQ-2026-04-23-003]]):** saturation breaks at τ_p ≤ 0.1 (spread > 0.01), but the argmin-k signal is structurally wrong regardless of τ: Computers argmin_k=0 at 98.7–99.98% at every τ_p (should prefer k=1); Cora argmin flips to k=8 at high τ_p but α is too soft to matter. Best probe at any τ_p stays at or below raw mean-pool floor on both datasets.
- **Interpretation:** τ_p=1.0 was a spec bug that masked a deeper fact — entropy-from-kmeans on L1-row-normalized raw X_k tracks feature concentration, not class structure, on these datasets. Temperature only changes the sharpness of a wrong signal. Non-propagation: this conclusion is local to raw X_k; Z_k (post-InfoNCE) has different cluster geometry and needs its own test.
- **Revisit when:** specifying any softmax-over-distance signal — always (a) log the pre-softmax distance scale and (b) sweep τ *before* drawing a verdict, so spec bugs aren't confused with structural signal failures.

### O4 — V3 below-random + high-variance is the signature of a dead architecture
- **Observed in:** INQ-004 V3 nonlinear W_k on Cora. Z-probe 16.48 ± 4.17 (7-class random ≈ 14.3). Per-seed variance is an order of magnitude larger than any non-dead variant (±0.2–1.0 elsewhere). Mechanism: ReLU + WD produces near-zero pre-activations; seed-dependent h(b) lands at a slightly informative constant for some seeds, pure noise for others.
- **Revisit when:** diagnosing any future experiment that produces "below-random accuracy + huge seed variance" — likely an architecture that bottoms out at a seed-dependent constant, not a genuinely trainable network.

## Related

- [[Research Agent Operating Protocol]] — mandates this ledger.
- [[INQ-2026-04-22-001]], [[INQ-2026-04-22-002]], [[INQ-2026-04-22-003]] — source inquiries for closed rows.
- [[Thesis]] — current contribution sentence; update when a live row passes or closes.
