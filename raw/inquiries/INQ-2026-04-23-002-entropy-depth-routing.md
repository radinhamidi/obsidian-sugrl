---
inquiry_id: INQ-2026-04-23-002
parent_inquiry_id: INQ-2026-04-23-001
topic: Entropy-driven per-node depth routing (E1/E2/E3/E4) + D1' WD gate
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-23
responded:
priority: blocking
status: open
related_files:
  - raw/inquiries/INQ-2026-04-23-001-track2-d1-symmetry-break.md
  - raw/inquiries/INQ-2026-04-22-003-track2-encoder-free-prototype.md
  - wiki/synthesis/Idea Ledger.md
  - wiki/entities/Rethinking graph neural networks from a geometric perspective of node features.md
tags: [inquiry, neurips-2026, ad-ssl, track-2, entropy, depth-routing]
---

# INQUIRY

**From:** Research Agent
**To:** Coding Agent
**Blocks:** Paper direction. [[INQ-2026-04-23-001]] closed D1 (X_0 + per-depth linear W_k) as falsified with a new mechanism: under uniform α + shared head + WD=5e-4, ∂L/∂W_k is depth-symmetric and small, so weight decay dominates and every W_k shrinks ~50,000× to ~2.7e-4 Frobenius; at W_k → 0, p_ik → softmax(h(0)) = constant across k, α-gradient strictly zero. Two sessions of negative results on soft-mixing adaptive-depth SSL under homophily. This inquiry tests a mechanistically different approach — **per-node per-depth entropy** as a depth-asymmetric, per-node signal that bypasses the symmetric-sum pathology — plus bundles a cheap diagnostic gate (D1') closing the row-2/row-5 gap from the INQ-004 post-mortem.

## Context

### Why entropy

L_S1 has a structural symmetry over k: the loss is a sum over depths, p_i = Σ_k α_ik · p_ik. When p_ik is approximately depth-invariant (feature-centroid-simplex collapse on homophilic graphs, see [[Rethinking graph neural networks from a geometric perspective of node features]]), the gradient on α and on any per-depth parameter W_k is depth-symmetric up to noise. Weight decay then collapses any k-distinguishing weight.

Entropy of per-depth predictions H(p_ik) is:
1. **Depth-asymmetric by construction**: different k produce predictions of different confidence even when directions collapse (because simplex collapse concerns angles, not how sharply nodes resolve to a single cluster).
2. **Per-node meaningful**: different nodes have different neighborhood structure → different depth→confidence profiles.
3. **Head-free (if the prediction source is k-means pseudo-labels)**: no learnable head that can collapse to h(0).

The critical design question: what source produces p_ik? We commit to **k-means pseudo-labels with M = num_classes clusters, fit once per depth on X_k = Â^k X**. Primary option is per-depth clusters; V1 tests shared-cluster variant (see below). This is parameter-free post-fit and maximally sidesteps the collapse pathologies we've hit.

### Why k-means is the right source

- Label-free, parameter-free after fitting. No learnable prediction head to collapse under weight decay.
- Produces an M-dimensional probability distribution per (i, k) via `p_ik[m] = softmax(-||X_k[i] - μ_k,m||² / τ_p)`, matching the shape of L_S1's p_i.
- Prior SSL precedent (DeepCluster / SwAV style pseudo-label refinement), though applied here to per-depth features rather than a learned embedding.

### Why supervised ceiling (E4) matters

V5 in INQ-003 and INQ-004 showed that supervised CE through the shared head matched SSL-primary to 0.03 pts because α-symmetry was the bottleneck, not the loss supervision. Here, supervised labels are used **only to train a per-depth classifier** (closed-form ridge probe), not to train α end-to-end. α is still driven by entropy. This decouples "is the entropy idea correct?" from "are k-means pseudo-labels good enough?":

- If E4 passes and E1 fails → mechanism is right, pseudo-labels are the bottleneck; work on better pseudo-labels.
- If E4 fails → entropy-routing is wrong regardless of signal quality; kill this direction.
- If E4 passes AND argmin_k H(p_ik) correctly flips per dataset (k=8 on Cora, k=1 on Computers) → strongest mechanistic evidence we've had so far.

## What to implement

### Shared pre-compute

```
# Existing pre-propagation (from INQ-003/004)
X_k = Â^k X  for k ∈ {0, 1, 2, 4, 8}          # K=5

# k-means pseudo-labels (primary: per-depth)
for each k:
    μ_k ∈ R^{M × F}  from k-means(X_k, num_clusters=M)   # M = num_classes
    d_ikm = ||X_k[i] - μ_k,m||²
    p_ik[m] = softmax(-d_ikm / τ_p)                       # τ_p = 1.0 default
    H_ik = -Σ_m p_ik[m] · log p_ik[m]                     # per-node per-depth entropy
```

Cost: k-means fit is ~seconds per depth on Cora/Computers. Everything else is arithmetic.

### E1 — Training-free entropy α

```
α_ik = softmax(-H_ik / τ_α)  across k              # τ_α = 1.0 default
Z_i  = Σ_k α_ik · X_k[i]
# Linear probe on Z_i
```

No learnable parameters anywhere in α or mixing. Pure math.

### E2 — Learnable α, entropy-minimization loss

```
# Learnable α (same MLP scorer as INQ-003 primary)
depth_embed = nn.Embedding(K, d_g=32)
scorer_mlp  = MLP(F_in + d_g -> h_g=128 -> 1)
s_ik = scorer_mlp(concat(X_k[i], depth_embed[k]))
α_ik = softmax_k(s_ik)

# Loss: weighted sum of entropies, α-gated
L_ent = Σ_i Σ_k α_ik · H_ik                       # pre-computed H, no head to collapse
L_div = +Σ_m q_m log q_m,  q = (1/N) Σ_i (Σ_k α_ik · p_ik)   # prevent global class collapse
L = L_ent - β · L_div                             # β = 1.0 default

# Readout
Z_i = Σ_k α_ik · X_k[i]
```

No W_k. No shared prediction head. H_ik is frozen (from k-means fit), so the loss landscape has real per-depth structure and α's gradient is depth-asymmetric by construction.

### E3 — Confidence-weighted cross-depth contrastive, standard WD

```
W_k ∈ R^{F_in × d_proj}  for k ∈ K_SET            # d_proj = 128, linear only
Z_k[i] = W_k · X_k[i]
# Learnable α (same scorer arch as E2)

# Cross-depth contrastive loss (InfoNCE-style)
# Positive pairs: same node, different depths (i, k) <-> (i, k'), k ≠ k'
# Negative pairs: different nodes, same depth (i, k) <-> (j, k), j ≠ i
# Confidence weight: w_ikk' = exp(-max(H_ik, H_ik') / τ_w)   # τ_w = 1.0
L_contrast = -Σ_{(i,k,k')} w_ikk' · log( exp(sim(Z_k[i], Z_k'[i]) / τ_c) / Σ_j exp(sim(Z_k[i], Z_k'[j]) / τ_c) )

# α is trained jointly by adding an entropy-min term (same as E2's L_ent)
L = L_contrast + λ · L_ent                         # λ = 0.1 default

# Readout
Z_i = Σ_k α_ik · Z_k[i]
```

**Standard weight decay (5e-4) on all parameters including W_k.** Hypothesis: unlike L_S1, contrastive loss directly depends on Z_k, so ∂L/∂W_k is strong, and WD will not cause the D1 collapse. Test this with default WD first; only if E3 primary fails do we run E3-V-WD.

### E3-V-WD — Same as E3, W_k excluded from weight decay (variation)

Run only if E3 primary fails. Add W_k to a separate Adam param-group with weight_decay=0. Confirms whether the WD fix is necessary for contrastive loss, cleanly separated from whether it is necessary for L_S1 (D1').

### E4 — Supervised entropy ceiling

```
# Per-depth ridge probe trained on 20-per-class train set
for each k:
    W_probe_k = closed-form ridge regression X_k -> Y  (train nodes only)
    p_ik[m] = softmax((X_k[i] @ W_probe_k)_m / τ_p)     # τ_p = 1.0
    H_ik = entropy as above

# Then run E1 logic: α_ik = softmax(-H_ik / τ_α), Z_i = Σ_k α_ik X_k[i]
```

Supervised ridge probe only produces per-depth class probabilities. α is still entropy-driven, not label-driven. This is a ceiling on what entropy-routing can achieve given perfect per-depth signal.

### V1 — Shared k-means clusters across depths

Fit k-means ONCE on `concat(X_0, X_1, X_2, X_4, X_8)` → shared cluster centers μ_m ∈ R^F. Compute `p_ik[m] = softmax(-||X_k[i] - μ_m||² / τ_p)` using these shared centers at every k. Run on whichever of E1/E2/E3 gave the strongest primary result (on each dataset independently). Tests whether semantic consistency of the pseudo-labels across depths matters.

### D1' — Weight-decay gate (bundled parallel diagnostic)

Re-run D1 primary (X_0 + per-depth linear W_k + shared classifier head + L_S1) and D1 V1 (no X_0) with W_k excluded from weight decay. Cora + Computers, 3 seeds. Purpose: close the row-2/row-5 gap from the INQ-004 post-mortem.

- If ||W_k||_F stays near xavier init (~15) AND Z-probe moves (either direction) → D1-family is alive with WD-exclusion, INQ-004 diagnosis refined.
- If W_k still collapses without WD → mechanism M1+M2 from post-mortem fully confirmed (data gradient on W_k is itself ≈ 0, WD was just a faster path to the absorbing fixed point).

Cost: ~1 hour CA time. Parallel with E1–E4.

### Defaults for all configurations

- Datasets: **Cora, Computers** (matched to INQ-003/004). ogbn-arxiv only if any primary hard-passes.
- K_SET = {0, 1, 2, 4, 8} except E2 which can optionally exclude k=0 given V2 of INQ-003 findings — primary spec keeps k=0 included for consistency.
- k-means: scikit-learn KMeans with `n_init=10, random_state=seed`, M = num_classes (Cora=7, Computers=10).
- d_proj = 128, d_g = 32, h_g = 128 (E2/E3 scorer).
- 200 epochs, Adam LR=0.01, weight_decay=5e-4 on everything unless specified.
- 3 seeds × 5 linear-probe restarts (protocol from [[Splits and Protocol]]).
- τ_p = τ_α = τ_w = τ_c = 1.0 as default; flag if tuning is obviously needed.

### Pass bars

Same as [[INQ-2026-04-23-001]]:
- **Cora hard-pass:** Z-probe ≥ 78.87 (best single-depth raw probe on Cora in recent runs, k=8).
- **Computers hard-pass:** Z-probe ≥ 87.53 (best single-depth raw probe on Computers, k=1).
- **Soft-pass (partial credit):** Z-probe ≥ raw-mean-pool floor (Cora 76.25, Computers 86.10) AND the argmin-k flip diagnostic comes out correctly per-dataset.

A **hard-pass requires both** a probe above the bar AND a mechanistically meaningful diagnostic profile (see below). INQ-004 V6/best-k one-hot was the counter-example to avoid: every pre-registered symmetry signal satisfied, but Z-probe still at Z_0 level. Don't repeat that mistake — mechanism diagnostics are necessary but not sufficient.

### Pre-registered diagnostics (report regardless of outcome)

1. **Per-depth entropy distribution.** Histogram of H_ik for each k (mean, std, min, max). Does entropy differ meaningfully across depths? If H is constant across k, the method fails by construction.

2. **Per-node argmin_k H_ik — the critical dataset-flip test.** On Cora, should gravitate toward k=8 (test-optimal depth). On Computers, should gravitate toward k=1. Report the full distribution of argmin_k H_ik per dataset. **If argmin monotonically prefers the same depth on both datasets, the entropy signal is mis-calibrated (likely picking up feature-concentration rather than class-confidence) and the method is mechanistically broken.**

3. **Correlation of argmin_k H_ik with node structure:** degree, local homophily, 1-hop neighborhood class entropy (use train labels). Does the entropy signal capture real per-node structure, or is it dominated by global per-dataset bias?

4. **α distribution statistics** (E2/E3/E4 with learnable α):
   - Std of mean α across K (pass: > 0.02)
   - Fraction of nodes with α-entropy < 0.8·ln(K) (pass: ≥ 0.20)
   - Per-node Var_k(p_ik) (pass: non-trivial)

5. **α-vs-H correlation.** Correlation of α_ik with -H_ik across (i, k). If positive and strong, α is responding to entropy as designed. If near-zero despite non-uniform α, something else is driving α.

6. **W_k Frobenius norms** (E3 + D1' only): does W_k stay at init, or does it collapse as in INQ-004? Report per-seed ||W_k||_F and pairwise cos(W_k, W_k').

7. **Per-depth linear probes.** Baseline raw Â^k X probes (independent of training) as reference.

8. **Wall-clock.** Precompute (including k-means fit), per-epoch, total train. Expect E1 to be near-instant, E2 similar to INQ-003 primary, E3 similar to INQ-004 primary.

### Variations — run in order, stop at first hard-pass

Order chosen by ascending complexity and expected information yield:
1. E1 primary (Cora + Computers).
2. E4 primary (Cora + Computers) — supervised ceiling, runs almost as cheap as E1.
3. E2 primary (Cora + Computers).
4. E3 primary (Cora + Computers).
5. E3-V-WD (Cora + Computers) — only if E3 primary fails.
6. V1 (Cora + Computers) — shared-cluster variant on the strongest primary so far.
7. D1' (Cora + Computers) — parallel; can run anytime.

If E1 or E2 hard-passes both datasets, also run on ogbn-arxiv (expect k*=4 to be preferred by entropy).

### Cost expectation

- E1: seconds to minutes (k-means fit + arithmetic).
- E4: similar to E1 (ridge probe is closed-form).
- E2: ~15 ms/epoch × 200 epochs ≈ 3 s/seed, so ~10 s/seed with k-means fit.
- E3: ~20–30 ms/epoch × 200 epochs ≈ 6 s/seed.
- D1': ~10 s/seed (same as INQ-004 D1).
- Total: ~30 minutes CA wall-clock for all primaries + D1' on Cora + Computers + V1 on the winner.

If any primary exceeds 150 ms/epoch on Cora, flag and stop.

### What NOT to do

- Do **not** train the classifier head jointly with α in E1/E2/E4 (the head is either absent or k-means-derived; this is the whole point).
- Do **not** add encoder-shaped nonlinearity to W_k in E3 (linear only, as in INQ-004's hard constraint; nonlinear W_k is encoder-like and falls under the INQ-001/002 encoder-destroys-signal finding).
- Do **not** touch `IMPLEMENTATION_SPEC.md` §6 or any wiki page. Results only in this inquiry file.
- Do **not** open a new inquiry — append all results here under `# DIAGNOSTIC RESULTS — <config>`.

## Numbered questions — answer as you run

1. **Does argmin_k H_ik correctly flip direction across datasets?** Under k-means pseudo-labels (E1) and under supervised ridge-probes (E4): does Cora's argmin gravitate toward k=8 and Computers' argmin toward k=1? **Default hunch:** E4 flips cleanly because real class probabilities track simplex collapse vs class-preservation correctly; E1 flips partially because k-means clusters on homophilic features are noisy approximations of classes — most nodes' argmin is "reasonable" but some fraction is dominated by cluster noise.

2. **Do E1 and/or E4 beat pass bar on either dataset?** **Default hunch:** E4 hard-passes Cora (≥ 79), soft-passes Computers (86–87). E1 soft-passes one dataset (probably Computers, where k=1 dominance makes entropy-based selection easier), fails Cora's hard bar.

3. **Does E2 clear the α-movement bar where D1 could not?** I.e., is α_ik std > 0.02 across K, non-trivial α-entropy distribution per node? **Default hunch:** yes — H_ik is pre-computed and frozen, so the loss has real depth-asymmetric structure. α should move.

4. **Does E3 primary (standard WD) avoid the W_k collapse?** I.e., does ||W_k||_F stay near init, not shrink 50,000×? **Default hunch:** yes — contrastive loss directly depends on Z_k, so ∂L/∂W_k is substantial and swamps WD. If W_k still collapses here, the mechanism is even deeper than INQ-004 diagnosed.

5. **If V2-style raw-feature entropy is silently confounded with k-means prediction entropy** (e.g., k-means is so noisy that H(p_ik) is dominated by geometric concentration, not class structure) — is there an obvious way to tell from the data? **Default hunch:** compare argmin_k H_ik distribution between E1 (k-means) and E4 (supervised) on the same nodes. If they agree, we're fine. If E1's argmin is monotonically deep and E4's flips, E1 is tracking concentration and we need a better pseudo-label source.

6. **D1' outcome:** does W_k stay at init when excluded from WD, and does Z-probe move? **Default hunch:** W_k stays closer to init but still drifts toward co-linear (data gradient is near-zero but not exactly zero), Z-probe may move a few points in some direction but not clear pass-bar. Row-5 (hypothesis genuinely wrong) still the dominant reading.

## Expected response format

Run in order: E1 primary → E4 primary → E2 primary → E3 primary → E3-V-WD if needed → V1 → D1'.

For each config, report under `# DIAGNOSTIC RESULTS — <config>`:
- Verdict line (hard-pass / soft-pass / hard-fail).
- Headline Z-probe table (Cora + Computers, mean ± std).
- Per-depth H distribution summary.
- argmin-k flip test per dataset.
- α statistics (for learnable-α configs).
- Correlation tables (argmin-k vs structure; α vs H).
- Wall-clock.

Lead **every** config report with the dataset-flip result (argmin-k Cora vs Computers). That is the single most diagnostic number in this inquiry. If the flip fails on E4 (supervised ceiling), everything else is academic and we close the direction.

Append results under each config header. Same append pattern as INQ-003/004.

**Blocking** — paper direction gated on these numbers. Two sessions of negative results on adaptive-depth SSL; a third negative outcome here combined with D1' confirming row-5 would push us toward direction D4 (training-free oracle reframe) from the INQ-004 post-mortem slate.

---

# RESPONSE

<!-- awaiting response -->
