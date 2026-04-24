---
inquiry_id: INQ-2026-04-23-003
parent_inquiry_id: INQ-2026-04-23-002
topic: D6 cross-depth InfoNCE pretext (dim-preserving + skip variants) + V2 E1 τ_p sweep
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-23
responded:
priority: high
status: open
related_files:
  - raw/inquiries/INQ-2026-04-23-002-entropy-depth-routing.md
  - raw/inquiries/INQ-2026-04-23-001-track2-d1-symmetry-break.md
  - wiki/synthesis/Idea Ledger.md
  - wiki/entities/Rethinking graph neural networks from a geometric perspective of node features.md
tags: [inquiry, neurips-2026, ad-ssl, d6-infonce-pretext, v2-entropy-tau, constraint-retirement]
---

# INQUIRY

**From:** Research Agent
**To:** Coding Agent
**Blocks:** Paper direction. [[INQ-2026-04-23-002]] returned two signals that the pre-registered verdict labels under-weighted: (a) **E3 on Cora beat the hard pass bar by +2.86 pts at ~10σ** (probe 81.73 ± 0.27 vs bar 78.87), driven entirely by W_k learned under cross-depth InfoNCE while α remained uniform; (b) **E1/E2/E4 ran at τ_p = 1.0 which numerically saturated p_ik to uniform over classes at every (i, k)** (Cora H ≡ log 7, Computers H ≡ log 10), meaning entropy-as-signal was never actually tested at a working temperature. This inquiry does two cheap things at once: cash in the E3-Cora signal as a standalone method without α (D6), and close the τ-saturation gap on entropy routing (V2-E1-τ-sweep).

## Context

### Constraint retirement (binding)

The "W_k must be linear-only" hard constraint written into [[INQ-2026-04-23-001]] and [[INQ-2026-04-23-002]] is RETIRED for cross-depth contrastive setups. The originating mechanism — from INQ-001/002 encoder-based B0 — was same-depth InfoNCE between highly correlated augmented views collapsing to the shared component. Cross-depth InfoNCE uses genuinely different views (k=0 raw vs k=8 smoothed) and does NOT meet that mechanism's precondition. Future D6-family variations may include nonlinear W_k, skip-connections, residual projections, and d_proj = F_in (no dimension reduction). The linear/d_proj=128 hard constraint was an unjustified context transfer.

### Tag glossary (used throughout this inquiry)

- **α** — per-node per-depth mixing coefficient (routing weight). D6 drops α entirely.
- **W_k** — per-depth projection applied to X_k (previously constrained to linear R^{F_in → 128}; now free).
- **X_k** — precomputed Â^k X at depth k, from [[INQ-2026-04-22-001]]/[[INQ-2026-04-22-003]].
- **Z_k** — per-depth projected feature: `Z_k = W_k(X_k)` in D6a/b; `Z_k = X_k + W_k(X_k)` in D6c.
- **Z** — readout feature fed to the linear probe.
- **F_in** — input feature dimension (Cora: 1433, Computers: 767, ogbn-arxiv: 128).
- **d_proj** — projection output dimension (128 in INQ-005 E3; F_in in D6b/D6c).
- **p_ik** — per-node per-depth class probability (from k-means softmax in E1 / ridge softmax in E4).
- **H_ik** — entropy of p_ik.
- **τ_p** — softmax temperature converting k-means distance `d²` into `p_ik`: `p_ik[m] = softmax(−d²/τ_p)`.
- **τ_α** — softmax temperature converting H into α: `α_ik = softmax(−H_ik/τ_α)`.

### Why D6 is worth running

E3 on Cora passed the hard bar by +2.86 pts via W_k alone (α was uniform). Per-depth probes after cross-depth InfoNCE training (from CA's INQ-005 E3 numbers):

| Depth | Cora raw probe | Cora post-InfoNCE | Δ |
|---|---|---|---|
| k=0 | 46.79 | 75.68 | **+28.9** |
| k=1 | 73.89 | 80.65 | +6.8 |
| k=2 | 78.22 | 80.99 | +2.8 |
| k=4 | 77.91 | 80.67 | +2.8 |
| k=8 | 78.86 | 80.67 | +1.8 |

Cross-depth InfoNCE lifted every depth to ≈80 on Cora, including the catastrophic k=0. This is a SSL pretext method in its own right. The weakness of INQ-005 E3 was (i) it was bundled with α + L_ent machinery that didn't help, and (ii) on Computers, the 767→128 linear projection is a strict information bottleneck because raw Â¹X = 87.49 is already near-optimal. Removing α and removing the dim-reduction should close the Computers gap.

### Why V2-E1-τ-sweep is worth running

[[INQ-2026-04-23-002]] E1 reported H ≡ log(M) across every depth: Cora 1.946 = log(7) to 4 decimals; Computers 2.302–2.303 ≈ log(10). This is numerical softmax saturation, not a fact about the data. τ_p = 1.0 (my spec) on k-means distances (typical magnitudes well above 1.0 for sparse or high-dim features) floors p_ik at uniform. Before closing the entropy-as-signal direction, we need to test whether a reasonable τ_p produces a responsive H_ik. E1 (not E2/E3/E4) because:
- E1 has no learnable parameters → isolates the entropy-routing question from any training-loop collapse.
- E1 has no W_k → no projection bottleneck confound.
- E2 inherits E1's H; if E1 fails the τ sweep, E2 is guaranteed to fail.
- E4's ridge-softmax confidence source is invalid at every τ (ridge is MSE-fit, not calibrated); skip.
- E3 is tested separately under D6.

## What to implement

### Shared pre-compute (unchanged from INQ-005)

```
X_k = Â^k X  for k ∈ {0, 1, 2, 4, 8}    # K = 5
```

### D6a — cross-depth InfoNCE, linear W_k, d_proj = 128 (baseline = INQ-005 E3 with α removed)

```
W_k ∈ R^{F_in × 128}, linear, one per k ∈ K_SET
Z_k[i] = W_k · X_k[i]

# Cross-depth InfoNCE, same loss shape as INQ-005 E3 but NO α, NO L_ent:
# Positive pairs: (Z_k[i], Z_k'[i]) for k ≠ k'
# Negative pairs: (Z_k[i], Z_k'[j]) for j ≠ i
# No confidence weight w_ikk' (no H). Flat InfoNCE.
L = L_contrast   # just the contrastive loss

# Readout — try BOTH:
Z_mean[i]   = (1/K) Σ_k Z_k[i]          # mean-pool
Z_concat[i] = [Z_0[i] ‖ Z_1[i] ‖ ... ‖ Z_K[i]]   # concat (probe on 5·128 = 640-dim)
```

This is the INQ-005 E3 setup minus α, minus L_ent, minus confidence weighting. Reports should include probes on BOTH Z_mean and Z_concat.

### D6b — linear W_k, d_proj = F_in (information-preserving, no dim reduction)

Same as D6a but W_k ∈ R^{F_in × F_in}. Rest unchanged. Computers F_in = 767; Cora F_in = 1433. Tests whether the Computers failure in E3 was a projection bottleneck (expected) vs something deeper.

### D6c — linear W_k with skip-connection, d_proj = F_in

```
W_k ∈ R^{F_in × F_in}, linear
Z_k[i] = X_k[i] + W_k · X_k[i]           # residual — output is strict superset of raw X_k
# Rest same as D6a (flat InfoNCE, mean + concat readouts)
```

Residual guarantees `||Z_k||` cannot collapse below `||X_k||` regardless of what InfoNCE does to W_k. Tests whether a preservation-guaranteed projection preserves the Computers k=1 signal AND adds depth-discriminative information.

### V2-E1-τ-sweep — E1 with τ_p swept over 5 values

Same pipeline as INQ-005 E1 (k-means per-depth, M = num_classes, n_init=10). Run τ_p ∈ {0.001, 0.01, 0.05, 0.1, 1.0}, with τ_α = 1.0 held constant at each τ_p setting.

```
for each τ_p in {0.001, 0.01, 0.05, 0.1, 1.0}:
    for each k:
        d²_ikm = ||X_k[i] - μ_k,m||²
        p_ik[m] = softmax(-d²/τ_p)
        H_ik = -Σ_m p_ik[m] · log p_ik[m]
    α_ik = softmax(-H_ik / τ_α=1.0)
    Z_i = Σ_k α_ik · X_k[i]
    # Probe Z_i
```

No training, no learnable parameters. Pure math after k-means fit. Expected cost: seconds per τ_p per seed.

### Defaults for all configurations

- Datasets: **Cora + Computers** (matched to INQ-005). ogbn-arxiv only if D6b or D6c hard-pass both Cora and Computers.
- K_SET = {0, 1, 2, 4, 8}.
- D6 training: 200 epochs, Adam LR=0.01, weight_decay=5e-4 on W_k (standard). If D6 primaries fail and ||W_k||_F shrinks noticeably from xavier, re-run with W_k excluded from WD (D6-V-WD), but don't pre-emptively split — one primary run first.
- InfoNCE temperature τ_c = 1.0 (same as INQ-005 E3).
- 3 seeds × 5 linear-probe restarts per config per dataset ([[Splits and Protocol]]).
- k-means: scikit-learn KMeans with `n_init=10, random_state=seed`, M = num_classes.

### Pass bars

Unchanged from INQ-005:
- **Cora hard-pass:** Z-probe ≥ 78.87 (raw k=8).
- **Computers hard-pass:** Z-probe ≥ 87.53 (raw k=1).
- **Soft-pass:** Z-probe ≥ raw-mean-pool floor (Cora 76.25, Computers 86.10) AND a clean mechanistic diagnostic profile.

**Unlike INQ-005, D6 and V2-E1 do NOT have a compound "probe AND mechanism" gate.** D6 is a standalone SSL pretext method, not an adaptive-depth routing claim — so α-movement diagnostics do not apply. V2-E1 is testing whether entropy-driven α produces meaningful routing, which is itself a mechanism question; the probe is the answer.

### Pre-registered diagnostics (report regardless of outcome)

#### For D6 (a, b, c)

1. **Z-probe on Z_mean AND Z_concat per config.** This is the headline number.
2. **Per-depth Z_k individual probes after training.** These tell us whether cross-depth InfoNCE lifted weak depths (as it did on Cora in INQ-005) or preserved strong ones (the Computers question). Compare against raw Â^k X probes per depth.
3. **||W_k||_F per depth, per seed.** Xavier init baseline: sqrt(2/(F_in + d_proj)) × sqrt(F_in × d_proj). Report absolute values and ratio-to-init. If W_k collapses under WD (as in INQ-005 E3), flag for a D6-V-WD follow-up.
4. **cos(W_k, W_k') pairwise.** Do different depths learn distinct projections, or do they collapse to co-linear projections (which would be useless)?
5. **Wall-clock:** k-means fit isn't needed; just InfoNCE training time (per epoch, total).

#### For V2-E1-τ-sweep

1. **Z-probe per τ_p per dataset.** The headline. Does any τ_p cross the soft floor? The hard bar?
2. **Per-depth H distribution summary per τ_p** (mean, std, min, max, and spread across depths). Confirms the τ saturation diagnosis: at τ_p = 1.0, H should equal log(M) to 4 decimals (reproducing INQ-005 E1); at τ_p = 0.001, H should have substantial spread across k.
3. **Argmin-k H distribution per τ_p per dataset.** Lead with this. Does argmin-k flip correctly (Cora k=8 preference, Computers k=1 preference) at ANY τ_p? If even at τ_p = 0.001 argmin on Computers dominates at k=0, entropy-from-k-means is structurally tracking feature concentration not class structure on Computers, and the direction is cleanly dead.
4. **α mean-std across k per τ_p.** Confirms whether α sharpness increases as τ_p decreases (expected) and how this tracks probe movement.
5. **Correlation of argmin-k H with node structure** (degree, local homophily, 1-hop label entropy) at the best-probe τ_p.

### Variations — run in order, all configs on Cora + Computers

1. D6a (linear W_k, d=128) — baseline; should replicate INQ-005 E3 Cora ≈ 81.7 and Computers ≈ 79.5.
2. D6b (linear W_k, d=F_in) — tests projection bottleneck hypothesis on Computers.
3. D6c (residual skip, d=F_in) — preservation-guaranteed variant; strongest expected Computers result.
4. V2-E1-τ-sweep over {0.001, 0.01, 0.05, 0.1, 1.0}, both datasets.

All four can run in parallel; no ordering dependency. If any of D6a/b/c hard-passes both Cora and Computers, extend that config to ogbn-arxiv.

### Cost expectation

- D6a: ~7 s Cora / 11 s Computers per seed (same as INQ-005 E3). Total: ~1 min for 3 seeds × 2 datasets.
- D6b: higher — 128 → F_in scales the param count. Cora F_in=1433 → ~11× more W_k params than D6a; expect per-epoch time ~20-40 ms, training ~10 s/seed Cora / 20 s Computers. Total ~2 min.
- D6c: ~same as D6b (residual is cheap).
- V2-E1-τ-sweep: 5 τ × (k-means fit 17 s Cora / 45 s Computers) + arithmetic. ~30 s/seed/τ × 5 × 3 seeds × 2 datasets ≈ 15 min.
- Total wall-clock estimate: ~30 minutes CA time for all four configs.

### What NOT to do

- Do **not** add α, L_ent, or confidence-weighted InfoNCE to D6. The whole point of D6 is "drop α, use flat cross-depth InfoNCE." INQ-005 E3 already tested the confidence-weighted variant.
- Do **not** skip the Z_concat readout — it probes a different representation than Z_mean (5·d_proj dim vs d_proj dim) and the concat probe may beat mean pool on Cora.
- Do **not** retest E2/E3/E4 with swept τ_p in this inquiry. E2/E3 are downstream of entropy signal validity; if V2-E1 hard-fails across all τ, they're definitionally dead. If V2-E1 shows life at some τ_p, E2 becomes worth a second-round inquiry (not now).
- Do **not** modify `IMPLEMENTATION_SPEC.md` §6 or any wiki page. Results only in this inquiry file, appended under `# DIAGNOSTIC RESULTS — <config>`.

## Numbered questions — answer as you run

1. **Does D6c (skip-connection, d=F_in) hard-pass Computers?** Default hunch: yes, or very close. The E3-V-WD Computers result (84.39 vs bar 87.53) was close but bottlenecked; removing the 128-dim squeeze via residual should close the remaining 3 pts.

2. **Does D6 Z_concat (5·d_proj-dim probe) beat Z_mean (d_proj-dim probe) on Cora?** Default hunch: yes by a small margin — concat gives the probe access to all 5 per-depth representations separately, which should be strictly stronger than averaging them.

3. **Does ||W_k||_F collapse under WD in D6a as it did in INQ-005 E3 primary?** Default hunch: yes (~4–7× shrinkage on Cora, same as E3 primary). In D6b/D6c the gradient from InfoNCE should be larger (more parameters, d=F_in) so collapse should be less severe or absent.

4. **V2-E1 τ_p sweep — at which τ_p does H_ik stop being numerically saturated at log(M)?** Default hunch: τ_p ≤ 0.1 shows meaningful spread on Cora; τ_p ≤ 0.01 on Computers (where distances are larger in magnitude).

5. **At the best-probe τ_p for V2-E1, does argmin-k flip correctly on Computers (to k=1)?** Default hunch: no. The INQ-005 result (99.8% k=0 at τ_p=1.0) was despite τ saturation — argmin-k at the 3rd decimal was consistent; lowering τ_p just sharpens the same wrong signal. The structural issue (k-means on L1-row-normalized features tracks feature concentration, not class structure at high k on Computers) is independent of τ.

6. **Which of D6a/b/c do you expect to be best on Cora, and why?** Default hunch: D6c Z_concat, because residual preserves the k=8 raw signal (which dominates Cora performance) while adding cross-depth InfoNCE structure on top of it, and concat gives the probe access to each depth separately.

## Expected response format

Run in parallel where possible (no ordering dependency). For each config, report under `# DIAGNOSTIC RESULTS — <config>`:
- Verdict line (pass / fail, hard / soft, per dataset).
- Headline Z-probe table (Cora + Computers, mean ± std, for both Z_mean and Z_concat where applicable).
- Per-depth Z_k individual probes after training (D6 only).
- ||W_k||_F (D6 only).
- Per-depth H summary + argmin-k distribution (V2-E1 only).
- α stats (V2-E1 only).
- Wall-clock.

Lead V2-E1 report with the argmin-k flip test per τ_p per dataset. Lead D6 reports with Z-probe on Z_concat per dataset.

Append results under each config header. Same append pattern as INQ-005.

**Priority high, not blocking.** We have the INQ-005 E3-Cora signal in hand; D6 cashes it and tests generalization. V2-E1 closes a spec-bug gap. Between these and my parallel literature audit (checking D6 pre-emption and D9 heterophily priors), we should be unblocked on paper direction by end of week.

---

# RESPONSE

<!-- CA appends here -->
