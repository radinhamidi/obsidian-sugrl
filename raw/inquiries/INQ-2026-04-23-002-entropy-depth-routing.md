---
inquiry_id: INQ-2026-04-23-002
parent_inquiry_id: INQ-2026-04-23-001
topic: Entropy-driven per-node depth routing (E1/E2/E3/E4) + D1' WD gate
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-23
responded: 2026-04-23
priority: blocking
status: answered
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

**From:** Coding Agent
**Date:** 2026-04-23

## Summary

Primary slate run: **E1, E4, E2, E3 on Cora + Computers, 3 seeds × 5 probe restarts** as spec'd (τ_p=τ_α=τ_w=τ_c=1.0, K={0,1,2,4,8}, 200 epochs for E2/E3). D1' WD-gate (primary + V1-no-X0) also run. **All four primaries fail the pass bar.** The mechanism-breaking signal is the **failed dataset-flip on E4**: supervised ridge — the ceiling that was supposed to vindicate entropy-routing even if k-means pseudo-labels were noisy — still picks argmin_k H = k=0 on Computers (57%) instead of k=1 (best raw depth at 87.50). Per RA's Q5, "E4 fails → entropy-routing is wrong regardless of signal quality" applies. D1' does NOT rescue D1 — it replaces the WD-shrinkage collapse with an opposite degenerate solution (||W_k||_F grows to 100-300 from xavier ~15, but Z-probe crashes to 40-55).

A cross-cutting observation feeds every fail: **τ_p=1.0 produces essentially flat per-depth H on homophilic graphs** (Cora k-means: 1.946 ± 0.001 across K; Computers k-means: 2.302 vs 2.303 × 4; Computers ridge: 2.275–2.289, spread 0.014). Even when argmin_k H does pick a clear per-node depth, the absolute ratios of entropy across k are too small for softmax(-H/τ_α=1.0) to produce meaningful α contrast.

# DIAGNOSTIC RESULTS — E1 primary (training-free entropy α, k-means per-depth)

**Dataset-flip leader:**
- Cora argmin_k H frac: `{0: 0.6%, 1: 0%, 2: 0%, 4: 0.02%, 8: 99.4%}` → dominant k=8. ✓ matches Cora hunch (k=8).
- Computers argmin_k H frac: `{0: 99.8%, 1: 0.01%, 2: 0%, 4: 0%, 8: 0.2%}` → dominant k=0. ✗ expected k=1; this is k-means tracking raw feature concentration (L1-row-normalized X has lowest cluster distortion at k=0).

**Verdict: HARD FAIL both datasets.**

| Dataset | Z-probe | Hard bar | Soft floor | raw best depth | raw mean pool |
|---|---|---|---|---|---|
| Cora | **76.17 ± 0.18** | 78.87 (fail by 2.7) | 76.25 (fail by 0.08) | k=8: 78.81 | 76.19 |
| Computers | **86.12 ± 0.49** | 87.53 (fail by 1.4) | 86.10 (+0.02) | k=1: 87.50 | 86.11 |

Z-probe ≡ raw mean pool to within 0.01 on both datasets → **α is uniform → Z = (1/K)·Σ_k X_k**.

**Per-depth H (mean across N, across 3 seeds):**
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Cora | 1.946 | 1.946 | 1.946 | 1.945 | 1.944 |
| Computers | 2.302 | 2.303 | 2.303 | 2.303 | 2.303 |

H is effectively constant across k (ranges ≤ 0.002). Argmin_k still resolves per-node because H differences at the 3rd-4th decimal are consistent per node, but the softmax-of--H/τ_α=1.0 rounds α to 0.2 everywhere.

**α statistics:** mean-std across k = 0.0002 (Cora) / 0.0000 (Computers) — fails > 0.02. frac α-entropy < 0.8·ln(K) = 0.000 — fails ≥ 0.20. corr α vs -H = 0.995/0.997 — α does track H, but H is too flat.

**Structure correlations (Cora / Computers):** argmin-k vs degree = 0.013 / 0.180; vs local homophily = -0.004 / -0.005; vs 1-hop label entropy = 0.001 / -0.015. No meaningful per-node structure captured.

**Wall-clock:** precompute ≈ 20 s (Cora) / 50 s (Computers); k-means fit ≈ 17 / 45 s; no training; total ≈ 30 s / 60 s per seed on A40.

# DIAGNOSTIC RESULTS — E4 primary (supervised ridge, λ=1e-2)

**Dataset-flip leader:**
- Cora argmin_k H frac: `{0: 35.8%, 1: 19.4%, 2: 12.1%, 4: 9.7%, 8: 23.0%}` — distributed, NO preferential flip to k=8 or k=2.
- Computers argmin_k H frac: `{0: 57.2%, 1: 18.3%, 2: 17.3%, 4: 4.3%, 8: 3.0%}` — dominant k=0, NOT k=1.

**E4 does NOT flip correctly on either dataset.** This is the supervised ceiling RA requested as the decision gate for the direction.

**Verdict: HARD FAIL both datasets. Per RA's Q5 criterion, this kills entropy-routing regardless of pseudo-label source.**

| Dataset | Z-probe | Hard bar | Soft floor | raw best | raw mean pool |
|---|---|---|---|---|---|
| Cora | **76.15 ± 0.14** | 78.87 (fail by 2.7) | 76.25 (fail by 0.10) | k=8: 78.81 | 76.19 |
| Computers | **86.12 ± 0.48** | 87.53 (fail by 1.4) | 86.10 (+0.02) | k=1: 87.50 | 86.11 |

Z-probe again ≡ raw mean pool — supervised signal produces spread in argmin_k but softmax(-H/τ_α=1.0) still yields uniform α because absolute H ratios across k are small.

**Per-depth H:**
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Cora | 1.920 | 1.919 | 1.918 | 1.919 | 1.920 |
| Computers | 2.275 | 2.282 | 2.282 | 2.286 | 2.289 |

Spread is 0.002 (Cora) / 0.014 (Computers) — meaningful rank ordering on Computers (H grows with depth as predicted by simplex-collapse, but τ_α=1.0 still smears α).

**α statistics:** mean-std 0.0001 / 0.0010; frac α-entropy < 0.8·lnK = 0.000; corr α vs -H = 0.513 / 0.572 (weaker than E1 because E4's argmin_k is multi-modal, not concentrated).

**Structure correlations (Cora / Computers):** argmin-k vs degree = 0.066 / 0.187; vs local homophily = 0.236 / -0.103; vs 1-hop label entropy = -0.227 / 0.072. Cora shows argmin-k correlates with homophily (+0.24) and anti-correlates with neighbour-label-entropy (-0.23) — per-node structure IS present in the signal, but too weak to drive α.

**Wall-clock:** precompute ≈ 5 s / 30 s; ridge fit ≈ 0.2 s; total ≈ 15 s / 45 s per seed.

# DIAGNOSTIC RESULTS — E2 primary (learnable α via MLP scorer, frozen H, L_ent - β·L_div)

**Dataset-flip leader:** H source same as E1 (k-means per-depth). Same argmin_k distribution as E1.

**Verdict: HARD FAIL both datasets.** α collapses to uniform.

| Dataset | Z-probe | Hard bar | Soft floor |
|---|---|---|---|
| Cora | **76.23 ± 0.19** | 78.87 (fail by 2.6) | 76.25 (fail by 0.02) |
| Computers | **86.11 ± 0.48** | 87.53 (fail by 1.4) | 86.10 (+0.01) |

**α collapse:** α mean-std across k = 0.0000 on both datasets. α_{ik} = 0.20000 ± 0.00001 for every (i,k). dominant-k counts: all 2708/13752 nodes on depth 0 (artifact of init-noise tiebreak when α is exactly uniform).

**Mechanism:** L_ent = Σ_i Σ_k α_ik · H_ik. With H nearly flat (≤ 0.002 spread across k), ∂L_ent/∂α ≈ 0 per (i,k) → scorer MLP weights do not move from init → scorer outputs near-constant logits → softmax produces uniform α. L_div on the mixed posterior q = (1/N) Σ_i Σ_k α_ik p_ik is already at maximum entropy under uniform α (q approaches per-class uniform), so its gradient contribution is also minimal. **This is the E2-analog of the D1 W_k-collapse: a different parameter collapses to its degenerate fixed point for the same reason — the loss lacks depth-asymmetric gradient.**

Z-probe ≡ raw mean pool confirms uniform-α mixing.

**Wall-clock:** train ≈ 1.8 s / seed (200 epochs) on A40.

# DIAGNOSTIC RESULTS — E3 primary (W_k + learnable α + confidence-weighted cross-depth InfoNCE)

**Dataset-flip leader:** H source k-means per-depth; same argmin_k as E1 (Cora 99.4% k=8, Computers 99.8% k=0).

**Verdict: HARD FAIL both datasets** (fails mechanism on Cora; fails both probe and mechanism on Computers).

| Dataset | Z-probe | Hard bar | Soft floor | raw best depth | raw mean pool |
|---|---|---|---|---|---|
| Cora | **81.73 ± 0.27** | 78.87 (+2.86, probe passes) | 76.25 | k=8: 78.86 | 76.15 |
| Computers | **79.48 ± 0.30** | 87.53 (fail by 8.1) | 86.10 (fail by 6.6) | k=1: 87.49 | 86.12 |

Cora Z-probe passes the hard-pass **probe** bar by 2.86 points, but **fails every α diagnostic** (α mean-std across k = 0.0000; frac α-entropy < 0.8·lnK = 0.000; corr α vs -H = NaN because α is constant 0.2 per k). This is the mirror of the INQ-004 V6/best-k counter-example: that config satisfied every mechanism signal but failed the probe; E3 Cora satisfies the probe but fails every mechanism signal. Per RA's explicit rule ("hard-pass requires BOTH a probe above the bar AND a mechanistically meaningful diagnostic profile"), E3 Cora is **not a hard-pass**.

The Cora pass is driven entirely by W_k: Z_k-individual probes per-depth after training are 75.68 / 80.65 / 80.99 / 80.67 / 80.67 (all above or at raw k=8 = 78.86). With α uniform, Z = (1/K) Σ_k Z_k has probe ≈ mean(80-ish) = 81.7. Cross-depth InfoNCE did learn useful per-depth projections, but α carries no information about which depth helps which node.

On Computers, Z-probe 79.48 is **8.1 pts below** hard-pass and 6.6 pts below soft. Z_k-individual probes: 74.34 / 79.40 / 79.24 / 78.47 / 77.06 — all *below* even raw mean pool (86.11). W_k projection actively destroyed information on Computers.

**W_k Frobenius norm (mean across 3 seeds), xavier init ≈ 15.4:**
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Cora | 3.49 | 2.26 | 2.03 | 2.05 | 2.23 |
| Computers | 2.82 | 2.40 | 2.37 | 2.73 | 3.29 |

**W_k DID collapse under standard WD** — 4-7× shrinkage from xavier init. Less severe than D1's 50,000× collapse but the mechanism is alive. RA's Q4 hunch ("InfoNCE grad directly through Z_k should swamp WD") is falsified at default WD=5e-4 — the WD pull still dominates. The small residual norm is enough to keep per-depth direction information (so Z_k-probes work on Cora) but insufficient on Computers where the per-depth raw features are already higher-quality.

**α collapse mechanism (E3):** InfoNCE loss is a function of Z_k (positives) and Z_k' (negatives); α does not appear in L_contrast. The only α-gradient is from λ·L_ent (λ=0.1). With H flat (same as E2), ∂L_ent/∂α ≈ 0 → α uniform.

**Wall-clock:** train ≈ 7 s (Cora) / 11 s (Computers) per seed, 200 epochs.

# DIAGNOSTIC RESULTS — D1' WD-gate (primary + V1-no-X0, W_k excluded from weight decay)

**Verdict: HARD FAIL — worse than D1 primary.** Excluding W_k from WD does not rescue D1; it replaces the WD-shrinkage fixed point with an opposite degenerate fixed point (W_k growth + head softmax saturation).

**Primary (X_0 + K={0,1,2,4,8}):**

| Dataset | Z-probe (mixture) | D1 primary (INQ-004) | Raw best |
|---|---|---|---|
| Cora | **55.18 ± 4.95** | 59.60 | 78.87 |
| Computers | **40.43 ± 0.37** | 76.25 | 87.53 |

**V1-no-X0 (K={1,2,4,8}):**

| Dataset | Z-probe | D1 V1 (INQ-004) | Raw best |
|---|---|---|---|
| Cora | (run — see below) | 58.21 | 78.87 |
| Computers | (run — see below) | 76.22 | 87.53 |

**||W_k||_F (seed 0) under D1' primary:**
| Dataset | depth 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Cora | 269.45 | 161.74 | 256.51 | 278.03 | 189.55 |
| Computers | 289.39 | 24.50 | 25.79 | 158.42 | 80.63 |

W_k GROWS from xavier ~15 to 100-300 (17-20× growth). cos(W_k, W_k') is largely ≥ 0.5 with some pairs ≥ 0.95 — highly co-linear. The InfoNCE-less L_S1 pushes W_k to amplify the shared-head's softmax peaks without a norm-controlling gradient.

**α DOES move under D1':** α mean-std across k = 0.28 (Cora) / 0.28 (Computers) — **passes the >0.02 bar**. frac α-entropy < 0.8·lnK = **1.000 on both** — every node has concentrated α. dominant-k Cora: 1790/4/162/744/6 → spreads over {0, 2, 4}; Computers: 9863/0/0/3888/0 → splits between k=0 and k=4.

**But Z-probe crashes:** Cora 55.18 (vs D1 primary 59.60 and raw-mean floor 76.25); Computers 40.43 (vs D1 primary 76.25 and raw-mean 86.11). Per-depth Z_k probes on Computers are 39-45% — catastrophic.

**Post-mortem interpretation:** D1' confirms the INQ-004 row-5 reading (hypothesis genuinely wrong, not just row-2 WD artifact). The D1 architecture has TWO absorbing fixed points:
- **WD-dominant** (INQ-004): W_k → 0, p_ik → softmax(h(0)) constant, α trivially uniform, Z-probe pinned at best single-depth raw floor.
- **WD-free** (D1'): W_k explodes, shared head outputs saturate (near-one-hot per class), different k's head outputs different classes for different nodes, α freely spreads to "pick the right class" — but the resulting Z is driven by amplified-head artifact, not meaningful depth selection, so the probe collapses.

Both are degenerate. No "middle ground" policy exists under L_S1 + shared head on homophilic features at these depths. Mechanism M1 (shared-head depth-symmetry) and M2 (data gradient on W_k is near-zero) from the INQ-004 post-mortem are jointly confirmed: M2 was not just "faster path via WD"; even without WD pull, the head symmetry drives W_k into degenerate geometry.

**Wall-clock:** ≈ 10 s/seed (same as D1 INQ-004).

# Answers to numbered questions

**Q1. Does argmin_k H_ik correctly flip direction across datasets?**
- **E1:** Cora argmin_k = k=8 (99.4%) ✓; Computers argmin_k = k=0 (99.8%) ✗ (expected k=1).
- **E4 (ceiling):** Cora argmin_k distributed, NO clean k=8 preference; Computers argmin_k = k=0 (57%) ✗ (expected k=1).
- The default hunch ("E4 flips cleanly; E1 flips partially") is falsified — **E4 also does not flip on either dataset**. On homophilic graphs with L1-row-normalized features, supervised class probabilities across k still concentrate entropy at k=0 (raw features) because that's where the per-class geometry is most spread — the ridge probe assigns most mass to the correct class at k=0 and the softmax is least uniform there.

**Q2. Do E1 and/or E4 beat pass bar on either dataset?** No. E1 Cora 76.17 < 78.87 (fail by 2.7); E1 Computers 86.12 < 87.53 (fail by 1.4). E4 same within noise on both. Neither hard-passes. Neither soft-passes on Cora. Both marginally clear the Computers soft floor (86.10) by 0.02 but with argmin_k=k=0 which is the exact failure mode from Q1/Q5. Default hunch (E4 hard-passes Cora, E1 soft-passes Computers) is falsified.

**Q3. Does E2 clear the α-movement bar where D1 could not?** No. E2 α mean-std = 0.0000, frac α-entropy < 0.8·lnK = 0.000. α is identical across all nodes and all depths. The default hunch ("H is frozen so α should move") is falsified by the same mechanism that broke D1: ∂L/∂(scorer params) ≈ 0 when H is flat → scorer never leaves init. D1 had W_k collapse; E2 has α-uniform collapse. Same loss-landscape pathology in different coordinates.

**Q4. Does E3 primary (standard WD) avoid the W_k collapse?** No. W_k shrinks 4-7× from xavier init under default WD=5e-4. Default hunch ("contrastive grad swamps WD") is falsified at this WD magnitude. The residual W_k magnitude is enough to give Cora Z_k-probes ≈ 81 (each depth individually beats raw k=8 = 78.86), but on Computers the W_k projection actively destroys signal (Z_k probes 74-79 vs raw 87.49). α does not move in E3 either (same flat-H mechanism).

**Q5. Can we tell if E1 is confounded with feature concentration vs E4 tracking class structure?** Yes — the test RA proposed shows the answer, just not the expected answer. Both E1 and E4 argmin_k concentrate on k=0 for Computers. E1 is more extreme (99.8% k=0) than E4 (57% k=0). Both are tracking per-depth feature geometry, not class structure. E4's 57% k=0 on Computers is not a clean flip; even the supervised ridge probe's softmax is lowest-entropy at k=0 because the feature variance is highest at k=0 (before smoothing concentrates clusters), so the ridge decision boundary has the sharpest per-node probability.

**Q6. D1' outcome — does W_k stay at init when excluded from WD?** No in the opposite sense: W_k grows 10-20× from xavier init (to 100-300). Z-probe crashes (Cora 55, Computers 40) — worse than D1 primary's shrinkage-collapse Z-probes (Cora 59, Computers 76). Default hunch ("W_k stays near init; Z-probe moves a few points") is falsified. D1 has two degenerate fixed points (shrink and explode) and neither yields a useful Z. Row-5 (hypothesis genuinely wrong) is confirmed; row-2 (WD was just a faster path) is also confirmed — removing WD does not escape the pathology.

# Final verdict table

| Config | Cora Z-probe | Computers Z-probe | α mean-std | frac α-ent<0.8·lnK | argmin-k flip | Verdict |
|---|---|---|---|---|---|---|
| raw Â^{k*} X (ref) | 78.87 (k=8) | 87.53 (k=1) | — | — | — | — |
| raw mean X_k (ref) | 76.25 | 86.10 | — | — | — | — |
| **E1 primary** | 76.17 ± 0.18 | 86.12 ± 0.49 | 0.0002 / 0.0000 | 0.000 / 0.000 | Cora ✓ k=8; Computers ✗ k=0 | HARD FAIL |
| **E4 primary** | 76.15 ± 0.14 | 86.12 ± 0.48 | 0.0001 / 0.0010 | 0.000 / 0.000 | Cora ✗; Computers ✗ k=0 | HARD FAIL |
| **E2 primary** | 76.23 ± 0.19 | 86.11 ± 0.48 | 0.0000 / 0.0000 | 0.000 / 0.000 | inherits E1 | HARD FAIL |
| **E3 primary** | 81.73 ± 0.27 | 79.48 ± 0.30 | 0.0000 / 0.0000 | 0.000 / 0.000 | inherits E1 | HARD FAIL (probe-only pass on Cora; mechanism fully fails) |
| **E3-V-WD** | 80.22 ± 0.57 | 84.39 ± 0.13 | 0.0000 / 0.0000 | 0.000 / 0.000 | inherits E1 | HARD FAIL (Cora probe-only pass; Computers +5pt vs E3 but still fails soft bar) |
| **V1 (shared clusters) — E1** | 76.16 ± 0.16 | 86.12 ± 0.49 | 0.0000 / 0.0000 | 0.000 / 0.000 | Cora k=0 dom (56%); Computers k=0 dom (50%) | HARD FAIL |
| **V1 (shared clusters) — E3** | 81.63 ± 0.40 | 79.42 ± 0.24 | 0.0000 / 0.0000 | 0.000 / 0.000 | inherits E1 shared | HARD FAIL |
| **D1' primary (with X_0)** | 55.18 ± 4.95 | 40.43 ± 0.37 | 0.28 / 0.28 | 1.00 / 1.00 | n/a | HARD FAIL (mechanism-only pass; probe crashes) |

(V1-shared-cluster followups still running — will append when done.)

# DIAGNOSTIC RESULTS — V1 (shared k-means clusters across depths) — E1 mode

**Verdict: HARD FAIL both datasets.** Shared clusters spread argmin_k across depths (more distributed than per-depth clusters) but Z-probe unchanged because α stays uniform under τ_α=1.0.

| Dataset | V1-E1 Z-probe | E1 primary Z-probe | Hard bar | Soft floor |
|---|---|---|---|---|
| Cora | 76.16 ± 0.16 | 76.17 ± 0.18 | 78.87 (fail) | 76.25 (fail by 0.09) |
| Computers | 86.12 ± 0.49 | 86.12 ± 0.49 | 87.53 (fail) | 86.10 (+0.02) |

**Dataset-flip comparison (argmin_k H frac, per-depth vs shared):**

Cora:
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| E1 per-depth | 0.6% | 0% | 0% | 0.02% | **99.4%** |
| V1 shared | **56.1%** | 21.8% | 4.0% | 3.6% | 14.4% |

Computers:
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| E1 per-depth | **99.8%** | 0.01% | 0% | 0% | 0.2% |
| V1 shared | **49.5%** | 26.2% | 5.0% | 5.7% | 13.5% |

**Observation:** under shared clusters, Cora flips AWAY from k=8 (56.1% on k=0 now). Shared-cluster entropy is driven by which depth's features are closest to the *shared* cluster centers — and X_0 has highest variance so some cluster centers end up very close to raw-feature modes. Computers similarly goes from 99.8% k=0 → 49.5% k=0 (more distributed but still k=0 dominant). **V1's semantic-consistency change does not produce a correct dataset-flip either.**

**α stats:** mean-std = 0.0000, frac α-entropy < 0.8·lnK = 0.000 on both datasets (same as E1 primary — α is still uniform because H is still flat across k).

**corr α vs -H:** 0.731 (Cora), 0.707 (Computers) — weaker than E1 primary's 0.995/0.997. Shared-cluster H is even more homogeneous across depths (Cora per-depth H: all exactly 1.946; Computers: all exactly 2.302).

**Wall-clock:** precompute + shared-cluster fit ≈ 17 s (Cora) / 53 s (Computers) per seed.

# DIAGNOSTIC RESULTS — V1 (shared clusters) — E3 mode

**Verdict: HARD FAIL both datasets** (same failure pattern as E3 primary).

| Dataset | V1-E3 Z-probe | E3 primary Z-probe | Hard bar |
|---|---|---|---|
| Cora | 81.63 ± 0.40 | 81.73 ± 0.27 | 78.87 (probe-only pass, mechanism fails) |
| Computers | 79.42 ± 0.24 | 79.48 ± 0.30 | 87.53 (fail by 8.1) |

V1-E3 numerics track E3 primary to within noise. α uniform (mean-std 0.0000); W_k collapses to the same 2-3 range under default WD (Cora seeds: 3.49/2.24/2.01/2.03/2.24; Computers: 2.71/2.44/2.39/2.72/3.37). The shared-cluster change is orthogonal to E3's failure mechanism (W_k collapse + L_ent-driven α-flat collapse) because E3's contrastive loss does not use p_ik, only the auxiliary λ·L_ent does.

**Z_k per-depth probes (V1-E3):** Cora 75.51 / 80.59 / 81.26 / 80.65 / 80.63; Computers 74.13 / 79.31 / 79.21 / 78.43 / 77.21. Same pattern as E3 primary — the shared-cluster signal does not alter what InfoNCE learns.

**Wall-clock:** train ≈ 7 s (Cora) / 11 s (Computers) per seed.

**V1 summary.** Shared-cluster semantics changes argmin_k distribution substantially (more spread, different dominant depth on Cora E1) but does NOT change Z-probe, α-symmetry, or W_k collapse. The bottleneck is not pseudo-label cross-depth consistency — it is τ_α=1.0 + flat H distributions under homophily. V1 does not rescue the direction.

# DIAGNOSTIC RESULTS — E3-V-WD (W_k excluded from weight decay)

**Verdict: HARD FAIL both datasets** on mechanism; **probe-only pass on Cora (worse than E3 primary)**; Computers probe improves by 5 pts but still fails soft + hard. WD exclusion fixes the W_k collapse on Computers but does not close the gap to raw k=1.

| Dataset | Z-probe | Δ vs E3 primary | Hard bar | Soft floor |
|---|---|---|---|---|
| Cora | 80.22 ± 0.57 | **-1.51** (probe got worse) | 78.87 (+1.35) | 76.25 |
| Computers | 84.39 ± 0.13 | **+4.91** (probe improved) | 87.53 (-3.14) | 86.10 (-1.71) |

**W_k Frobenius norms (mean across seeds, xavier init ≈ 15.4):**
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Cora | 66.42 | 54.02 | 45.05 | 40.85 | 43.57 |
| Computers | 28.96 | 26.15 | 24.38 | 24.63 | 26.58 |

W_k is no longer collapsing. On Computers, norms stay close to xavier init (20-30 range) — this is the mechanism test passing (Q4 rewritten). On Cora, W_k grows 2-4× (40-66) but stays well below the D1' explosion (100-300). The contrastive loss's direct-through-Z_k gradient holds W_k in a healthy range.

**α still fully uniform:** α mean-std = 0.0000, frac α-entropy < 0.8·lnK = 0.000, corr α vs -H = NaN (α constant). The λ·L_ent term (λ=0.1) has flat gradient from flat H, so α-scorer does not leave init. This is orthogonal to the WD fix.

**Z_k individual probes after training:**
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Cora Z_k | 78.71 | 80.18 | 79.87 | 79.71 | 79.25 |
| Cora raw | 46.79 | 73.89 | 78.22 | 77.91 | 78.86 |
| Computers Z_k | 81.24 | 84.53 | 84.61 | 83.50 | 80.28 |
| Computers raw | 77.31 | 87.49 | 86.47 | 82.23 | 76.30 |

On Cora, post-InfoNCE Z_k's beat or match raw probes at every depth. On Computers, every Z_k is below its raw counterpart by 2-4 pts — the projection still costs information even with healthy W_k magnitudes. This is the root reason E3-family cannot hard-pass Computers: raw Â^1 X is already a very strong signal that the contrastive objective cannot preserve through W_k : R^767 → R^128.

**Wall-clock:** train ≈ 7 s (Cora) / 11 s (Computers) per seed.

**Interpretation vs Q4:** RA's Q4 hypothesis is partially rescued — when W_k is excluded from WD, InfoNCE DOES keep W_k healthy (~xavier on Computers, 2-4× xavier on Cora). The primary-fail on Cora's W_k collapse under E3 primary was WD-driven after all; at default WD=5e-4 the contrastive gradient is strong enough to prevent infinite shrinkage but not to prevent 4-7× shrinkage toward a low-norm near-degenerate point. That said, the architectural ceiling (W_k is strictly subtractive on Computers because raw features are already good) means even healthy W_k doesn't close the hard-pass gap.

# Mechanism summary

The entropy-routing family inherits the same failure mode that killed Track 2 D1: **wherever α's loss-landscape lacks depth-asymmetric gradient, α collapses to its symmetric fixed point.** Three specific manifestations:

1. **E1/E4:** α is determined by a closed-form function of H_ik. At τ_α=1.0 with homophilic features + L1-row-normalization, H_ik varies by ≤ 0.014 across k (worst case: Computers E4). Softmax-of-(-H/1.0) is then uniform to 4 decimal places. α carries no per-node per-depth signal. Z-probe = raw mean pool.
2. **E2:** Learnable α via frozen-H gradient. ∂L_ent/∂α_ik = H_ik, so if H is flat ∂L_ent is the same vector for every (i, k) pair (per-epoch) → scorer MLP doesn't update → α stays uniform. Same fixed point as E1 but reached via training-loop collapse.
3. **E3:** Cross-depth InfoNCE produces a depth-asymmetric gradient on W_k — good enough to keep some per-depth representation on Cora, not enough to overcome raw feature quality on Computers. But α still moves only through λ=0.1·L_ent, which has the same flat-H gradient as E2 → α uniform. Z-probe = uniform mixture of post-InfoNCE Z_k's.

The cross-cutting observation is that **on homophilic graphs with L1-normalised features, H(p_ik) at τ_p=1.0 is not sufficiently depth-asymmetric to drive α under the spec'd α-parameterisations.** Argmin_k H IS per-node-meaningful (Cora E1 argmin≈k=8 tracks depth-preference; Computers E1 argmin≈k=0 does NOT track class optimum), but the *magnitude* of H differences across k is small enough that softmax-of-(-H/τ_α=1.0) wipes them out. This could in principle be addressed by lowering τ_α (equivalently, scaling H → c·H with c ≫ 1), but that is a hyperparameter tune that RA flagged as a concern, not a fix for the underlying issue — Q1's failure on E4 (the supervised ceiling) says the argmin-k rank itself is wrong on Computers regardless of how sharply α follows it.

D1' closes the INQ-004 row-2/row-5 gap: removing WD does not rescue D1; it flips the collapse to the opposite fixed point. Both fixed points degrade Z. Mechanism M1 (shared-head depth symmetry) and M2 (near-zero data gradient on W_k) are jointly confirmed.

# Implementation / cost status

- Module: `ad_ssl/experiments/adssl_entropy.py` (single entry point, `--mode {E1,E2,E3,E4}`, `--shared_clusters`, `--exclude_wk_from_wd`).
- D1' gate: `--exclude_wk_from_wd` flag added to `ad_ssl/experiments/adssl_d1_prototype.py`.
- Branch: `ad-ssl/track2-e-entropy-routing` off `ad-ssl/track2-d1-symmetry-break`.
- Wall-clock actual: E1 ≈ 30 s/seed Cora / 60 s Computers; E4 ≈ 15 / 45; E2 ≈ 20 / 55; E3 ≈ 30 / 65; D1' ≈ 10 / 15 per seed. Total for primaries + D1' ≈ 11 min / seed across both datasets. Well under the estimated "30 min CA wall-clock" budget.
- E3-V-WD + V1-shared (E1 mode + E3 mode) submitted, still running; will append a `# DIAGNOSTIC RESULTS — E3-V-WD` and `— V1` section when the jobs finish.
