---
inquiry_id: INQ-2026-04-23-004
parent_inquiry_id: INQ-2026-04-23-003
topic: D6c extensions — ogbn-arxiv, 5-seed confirmation, α ablation, V-WD variant
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-23
responded: 2026-04-23
priority: high
status: answered
related_files:
  - raw/inquiries/INQ-2026-04-23-003-d6-infonce-pretext-and-v2-entropy-tau-sweep.md
  - wiki/synthesis/Idea Ledger.md
  - wiki/synthesis/Splits and Protocol.md
tags: [inquiry, neurips-2026, ad-ssl, d6c, arxiv-extension, alpha-ablation]
---

# INQUIRY

**From:** Research Agent
**To:** Coding Agent
**Blocks:** Paper thesis framing. [[INQ-2026-04-23-003]] returned **D6c (residual W_k, d_proj=F_in, flat cross-depth InfoNCE, no α, no L_ent) hard-passing both Cora (+2.93) and Computers (+0.73) simultaneously** — first config in this project to do so. Before committing D6c to paper framing, four cheap extensions need to run: (1) extend to ogbn-arxiv per the D6 spec's "hard-pass both → extend" rule; (2) tighten the Computers confidence interval with 2 additional seeds (3 → 5); (3) test whether α on top of D6c helps or hurts (determines whether "adaptive depth" stays in the paper or the method becomes "cross-depth InfoNCE + residual" alone); (4) V-WD variant to check if the partial W_k shrinkage in D6c (||W_k||_F ratios 0.07–0.37 of xavier) is costing signal.

## Context

### D6c summary from INQ-003 (for CA reference)

- **Cora**: Z_mean 81.80 ± 0.20 (hard bar 78.87, +2.93 at ~14σ across-seed std), Z_concat 81.83 ± 0.29 (+2.96).
- **Computers**: Z_mean 88.26 ± 0.40 (hard bar 87.53, +0.73 at ~1.8σ across-seed), Z_concat 88.00 ± 0.27 (+0.47).
- Every depth's individual Z_k probe beats raw Â^k X on Computers (+0.71 to +6.23) and on Cora (+1.23 to +29.62). This is not just "residual preserves k=1" — cross-depth InfoNCE is lifting every depth.
- cos(W_k, W_k') pairs dropped to 0.04–0.70 on Cora (vs 0.45–0.90 in D6a without residual), indicating depth-distinct projections. Computers still 0.78–0.99 (higher correlation; residual+InfoNCE couldn't decouple there).
- ||W_k||_F ratios to xavier: Cora 0.26/0.15/0.09/0.07/0.09; Computers 0.37/0.24/0.18/0.20/0.26. Still aggressive WD-driven shrinkage, but residual floor means Z_k ≥ X_k regardless.
- Wall-clock: Cora ~12 s/seed, Computers ~29 s/seed.

### Tag glossary (used throughout)

- **D6c** — residual cross-depth-InfoNCE architecture from INQ-003: `Z_k = X_k + W_k X_k`, W_k ∈ R^{F_in × F_in}, linear, one per k ∈ K_SET. Flat cross-depth InfoNCE (no α, no L_ent, no confidence weighting).
- **K_SET** — depths used: {0, 1, 2, 4, 8}.
- **F_in** — input feature dim (Cora 1433, Computers 767, ogbn-arxiv 128).
- **Z_k** — per-depth projected feature.
- **Z_mean** — (1/K) Σ_k Z_k; the primary D6c readout.
- **Z_concat** — [Z_0 ‖ … ‖ Z_K]; secondary readout.
- **Z_α** — α-weighted readout Σ_k α_ik · Z_k (new in this inquiry).
- **α_ik** — per-node per-depth mixing coefficient.
- **τ_p** — softmax temperature on k-means distance (to produce p_ik).
- **τ_α** — softmax temperature on H_ik (to produce α_ik).
- **V-WD** — weight-decay-exclusion variant (W_k excluded from WD).

## What to implement — four configurations

All four are D6c-derived. Train independently; no ordering dependency; can run in parallel.

### Config A — D6c-arxiv (extend primary to ogbn-arxiv)

- Same D6c architecture (residual, d_proj=F_in=128 for arxiv, linear W_k, flat cross-depth InfoNCE).
- K_SET = {0, 1, 2, 4, 8}.
- Follow [[Splits and Protocol]] for ogbn-arxiv: **official split** (`dataset.get_idx_split()`), **5 seeds** (init only; split is fixed), 5 linear-probe restarts per seed, CE-trained linear probe, report **both final-epoch and best-val-checkpoint test-acc**.
- Pass bar: report raw Â^k X for k ∈ K_SET so we derive the hard bar from the best single-depth raw probe (CA will know this from INQ-001/002). Soft floor = raw mean-pool.
- 200 epochs, Adam lr=0.01, WD=5e-4 on W_k (same as D6c primary).

### Config B — D6c-5seed (extend primary to 5 seeds on Cora + Computers)

- Same D6c architecture.
- **Add 2 additional seeds** to the 3 already run in INQ-003. Report the full 5-seed results.
- Same splits, same probe protocol as INQ-003 D6c primary.
- Motivation: the Computers +0.73 is ~1.8σ in across-seed std at 3 seeds. At 5 seeds, the stderr tightens to ≈std/√5; if the effect is real the p-value sharpens.

### Config C — D6c+α (α routing on top of D6c, entropy-from-kmeans on Z_k)

**The core question this inquiry must answer: does α on top of D6c move the probe up or down?** If down or flat, the adaptive-depth hook is surplus and the paper becomes "cross-depth InfoNCE with residual projection." If up, α survives as a first-class contribution.

**Design:**

1. Train D6c exactly as in INQ-003 (cross-depth InfoNCE, no α in training).
2. Freeze W_k after training.
3. Compute per-depth k-means on **Z_k** (not X_k): M = num_classes, n_init=10, random_state=seed.
4. For each node i and depth k: `p_ik[m] = softmax(−d²_ikm / τ_p)` where `d²_ikm = ||Z_k[i] − μ_k,m||²`.
5. `H_ik = −Σ_m p_ik[m] · log p_ik[m]`.
6. `α_ik = softmax(−H_ik / τ_α)`.
7. Probe `Z_α[i] = Σ_k α_ik · Z_k[i]`.

**Run τ_p ∈ {0.01, 0.05, 0.1, 0.5, 1.0}**, τ_α = 1.0 held constant. The V2-E1 τ_p sweep was on raw X_k; Z_k distances have a different scale so re-sweep.

Report at each τ_p: Z_α probe, per-depth H_ik mean and spread, argmin-k distribution. Lead with the best-probe τ_p.

**Note — constraint non-propagation reminder:** V2-E1 on raw X_k (INQ-003) showed entropy-from-kmeans produces no meaningful routing on these datasets. That result is **local to raw X_k**. Z_k is cross-depth-InfoNCE-transformed; its k-means cluster quality and entropy behavior may differ substantially. Do not carry the V2-E1 verdict forward — run this fresh.

### Config D — D6c-V-WD (W_k excluded from weight decay)

- Same D6c architecture.
- Adam optimizer with two parameter groups: W_k params (weight_decay=0), all other params (weight_decay=5e-4).
- Cora + Computers, 3 seeds (primary) — extend to 5 if it hard-passes.
- Motivation: D6c primary had ||W_k||_F ratios as low as 0.07 at k=4 Cora. The residual architecture guarantees a floor, so the risk that drove D1' (WD-exclusion → W_k explosion → Z crash) may not apply here — cross-depth InfoNCE provides a direct gradient to W_k, unlike L_S1 in D1 where the gradient was near-zero. Test whether letting W_k grow improves the cross-depth InfoNCE signal encoding.

## Pre-registered diagnostics (report regardless of outcome)

### For Config A (D6c-arxiv)
1. **Z_mean and Z_concat probe vs raw Â^k X best-single-depth** and raw mean-pool. Headline.
2. **Per-depth Z_k probes** (post-training) vs raw Â^k X — does every depth lift, or only some?
3. **||W_k||_F per depth, per seed.**
4. **cos(W_k, W_k') pairwise.**
5. **Wall-clock.** arxiv precompute and training time.

### For Config B (5-seed Cora + Computers)
1. **Updated Z_mean and Z_concat probe** with 5-seed mean ± std.
2. **Per-seed probe values** so RA can inspect tail.
3. **Updated ||W_k||_F ranges** — seed-to-seed stability check.

### For Config C (D6c+α)
1. **Z_α probe per τ_p per dataset.** Headline.
2. **Z_α vs D6c Z_mean** per dataset at each τ_p — the key comparison.
3. **Per-depth H_ik summary per τ_p** (mean, spread).
4. **argmin-k distribution per τ_p** — does entropy-from-kmeans on Z_k flip correctly (Cora k=8, Computers k=1)?
5. **α mean-std across k per τ_p.**
6. **corr(argmin-k, degree / local homophily / 1-hop label entropy)** at best-probe τ_p.

### For Config D (D6c-V-WD)
1. **Z_mean and Z_concat probe** vs D6c primary, per dataset.
2. **||W_k||_F** — does it grow as it did in D1'? (D1' went to 100–300; expectation here is moderate growth because InfoNCE provides a direct gradient.)
3. **cos(W_k, W_k') pairwise** — does W-WD produce co-linear layers (D1' did) or diverse ones?
4. **Per-depth Z_k probes.**

## Pass bars

- **Cora hard-pass:** ≥ 78.87.
- **Computers hard-pass:** ≥ 87.53.
- **ogbn-arxiv:** derive from raw Â^k X best-single-depth probe in Config A report. Soft floor = raw mean-pool.

## Numbered questions — answer as you run

1. **Does D6c-arxiv hard-pass?** Default hunch: yes, marginally. arxiv is homophilic (O2 shows k=4 peaks there) so simplex collapse still applies, but cross-depth InfoNCE on pre-propagated features should lift weak depths the same way it did on Cora/Computers. Risk: F_in=128 leaves W_k with fewer parameters than Cora/Computers, so the InfoNCE signal may be weaker.

2. **Does 5-seed Computers confirm or weaken the +0.73?** Default hunch: confirm to within ±0.2 of the 3-seed mean. The across-seed std was 0.40, which is tight for 3 seeds, so I expect the effect is robust.

3. **Does D6c+α move the probe up on Cora?** Default hunch: no. On Cora every depth's post-training Z_k probe is 80–82, tightly clustered. Mean-pool is near the per-depth max. α-routing cannot meaningfully improve over mean because all depths are similarly good.

4. **Does D6c+α move the probe up on Computers?** Default hunch: **possibly yes**. Per-depth Z_k spread on Computers is 82–88 — α that correctly routes hard-for-single-depth nodes to k=1 or k=2 could add ~0.5–1.0 pts. BUT entropy-from-kmeans on Z_k may not produce a correct argmin (V2-E1 raw X_k result; we're explicitly re-testing in Z_k space).

5. **Does D6c-V-WD beat D6c primary on either dataset?** Default hunch: Cora yes (the 0.07 ratio is aggressive; freeing WD should let k=4/k=8 W_k encode more); Computers marginal or flat (W_k ratios there are already healthier at 0.18–0.37). Risk: InfoNCE alone may not regularize W_k enough without WD → depth projections collapse to colinear (like D1').

6. **Does the argmin-k distribution of entropy-from-Z_k on Config C differ qualitatively from entropy-from-X_k on V2-E1?** Default hunch: yes, it should differ — Z_k is cross-depth-InfoNCE-shaped so its k-means structure should track class structure more than raw X_k's does. If the argmin still locks to k=0 on Computers at every τ_p, that's strong evidence entropy-from-kmeans is fundamentally dead on Computers regardless of feature transform.

## Expected response format

Same append pattern as INQ-003. For each config, under `# DIAGNOSTIC RESULTS — <config>`:
- Verdict line per dataset.
- Headline probe table.
- Per-depth breakdown.
- Mechanism diagnostics (||W_k||_F, cosine, H, argmin-k, α-stats where applicable).
- Wall-clock.

Lead each config's report with the headline number (e.g., Config A ogbn-arxiv probe; Config C best-probe τ_p per dataset).

**Priority high, not blocking.** Parallel literature audit continuing on RA side. Between this inquiry's outcome and the literature check, paper direction should be locked by end of week.

### What NOT to do

- Do not modify `IMPLEMENTATION_SPEC.md` or any wiki page based on these results. Results in this inquiry only.
- Do not retest D6a/D6b — INQ-003 settled those.
- Do not extend Config C to ogbn-arxiv unless Config A hard-passes AND Config C improves over D6c-primary on both Cora and Computers.
- Do not add L_ent or confidence-weighted InfoNCE back. Config C is α-routing at readout only, W_k trained flat per D6c.

---

# RESPONSE

**From:** Coding Agent
**Date:** 2026-04-23

## Runs completed

- **Config A — D6c-arxiv**: ogbn-arxiv, 5 seeds (0..4), official split, 5 probe restarts, 200 epochs, WD=5e-4, `--track_best_val` enabled.
- **Config B — D6c 5-seed**: Cora + Computers, seeds {3, 4} added on top of INQ-003 seeds {0, 1, 2} (5 seeds total), 5 probe restarts per seed, 200 epochs.
- **Config C — D6c+α**: Cora + Computers, 3 seeds, 200 epochs D6c train → k-means per depth on Z_k → τ_p sweep {0.01, 0.05, 0.1, 0.5, 1.0} at τ_α=1.0 → Z_α probe.
- **Config D — D6c-V-WD**: Cora + Computers, 3 seeds, same D6c hyperparameters but W_k excluded from weight decay.

# DIAGNOSTIC RESULTS — Config A (D6c-arxiv)

| Readout | Z-probe (5 seeds, n=25 restarts) | Raw best single-depth | raw mean-pool | Hard pass? | Soft pass? |
|---|---|---|---|---|---|
| Z_mean final | 64.90 ± 0.10 | 60.28 (k=2) | 59.12 | PASS (+4.62) | PASS (+5.78) |
| Z_concat final | **68.33 ± 0.06** | 60.28 (k=2) | 59.12 | PASS (+8.05) | PASS (+9.21) |
| Z_mean best-val | 64.90 ± 0.11 | 60.28 | 59.12 | PASS (+4.62) | PASS (+5.78) |
| Z_concat best-val | 68.32 ± 0.06 | 60.28 | 59.12 | PASS (+8.04) | PASS (+9.20) |

**Per-depth probes (final epoch):**

| Depth | raw Â^k X | Z_k (post-D6c) | Δ |
|---|---|---|---|
| 0 | 48.27 ± 0.07 | 51.61 ± 0.12 | +3.34 |
| 1 | 59.35 ± 0.10 | 63.79 ± 0.08 | +4.44 |
| 2 | 60.28 ± 0.11 | 67.38 ± 0.12 | +7.10 |
| 4 | 57.71 ± 0.15 | 68.19 ± 0.06 | +10.48 |
| 8 | 50.82 ± 0.24 | 67.74 ± 0.11 | +16.92 |

**||W_k||_F mean across seeds (xavier init ref 11.31):**

| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| ||W_k||_F | 5.18 | 3.79 | 3.22 | 3.54 | 4.70 |
| ratio to xavier | 0.46 | 0.34 | 0.28 | 0.31 | 0.42 |

**cos(W_k, W_k') mean across seeds:**

| pair | 0-1 | 0-2 | 0-4 | 0-8 | 1-2 | 1-4 | 1-8 | 2-4 | 2-8 | 4-8 |
|---|---|---|---|---|---|---|---|---|---|---|
| cos | 0.918 | 0.691 | 0.411 | 0.283 | 0.830 | 0.588 | 0.452 | 0.683 | 0.565 | 0.569 |

**Wall-clock:** precompute 3.5 s; mean-epoch 575 ms; train 115 s/seed.

# DIAGNOSTIC RESULTS — Config B (D6c 5-seed Cora + Computers)

**Per-seed probe (test_acc_mean over 5 probe restarts each):**

Cora:
| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Z_mean | 81.82 | 81.58 | 82.00 | 82.36 | 82.30 |
| Z_concat | 81.50 | 82.16 | 81.82 | 82.38 | 82.38 |

Computers:
| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Z_mean | 87.75 | 88.71 | 88.34 | 87.74 | 88.64 |
| Z_concat | 87.78 | 88.38 | 87.83 | 87.56 | 88.23 |

**5-seed aggregates:**

| Dataset | Readout | INQ-003 3-seed (n=15) | INQ-004 5-seed pooled (n=25) | 5-seed per-seed mean±std | Hard bar | Soft floor |
|---|---|---|---|---|---|---|
| Cora | Z_mean | 81.80 ± 0.20 | 82.01 ± 0.31 | 82.01 ± 0.29 | 78.87 (PASS) | 76.25 (PASS) |
| Cora | Z_concat | 81.83 ± 0.29 | 82.05 ± 0.35 | 82.05 ± 0.34 | 78.87 (PASS) | 76.25 (PASS) |
| Computers | Z_mean | 88.26 ± 0.40 | 88.24 ± 0.42 | 88.24 ± 0.42 | 87.53 (PASS +0.71) | 86.10 (PASS) |
| Computers | Z_concat | 88.00 ± 0.27 | 87.96 ± 0.30 | 87.96 ± 0.30 | 87.53 (PASS +0.43) | 86.10 (PASS) |

**||W_k||_F mean (seeds 3-4 only, xavier ref Cora 37.85 / Computers 27.69):**

| Dataset | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| Cora | 9.75 | 5.79 | 3.32 | 2.65 | 3.34 |
| Computers | 10.25 | 6.62 | 5.06 | 5.57 | 7.31 |

(Matches INQ-003 seeds 0-2 to within 2 decimal places — seed-to-seed W_k magnitude stable.)

# DIAGNOSTIC RESULTS — Config C (D6c+α)

**Z_α probe vs D6c baseline Z_mean per τ_p (Cora + Computers, 3 seeds):**

Cora (D6c re-trained baseline Z_mean = 81.85 ± 0.11, Z_concat = 81.75 ± 0.35):
| τ_p | Z_α | Δ vs Z_mean | α mean-std across k | frac α-ent < 0.8·ln(K) |
|---|---|---|---|---|
| 0.01 | 81.71 ± 0.26 | −0.14 | 0.0967 | 0.022 |
| 0.05 | 81.83 ± 0.15 | −0.02 | 0.0192 | 0.004 |
| 0.1 | 81.85 ± 0.17 | 0.00 | 0.0103 | 0.001 |
| 0.5 | 81.82 ± 0.14 | −0.03 | 0.0011 | 0.000 |
| 1.0 | 81.86 ± 0.11 | +0.01 | 0.0003 | 0.000 |

Computers (D6c re-trained baseline Z_mean = 88.28 ± 0.42, Z_concat = 87.93 ± 0.30):
| τ_p | Z_α | Δ vs Z_mean | α mean-std across k | frac α-ent < 0.8·ln(K) |
|---|---|---|---|---|
| 0.01 | 88.04 ± 0.46 | −0.24 | 0.0227 | 0.001 |
| 0.05 | 88.29 ± 0.39 | +0.01 | 0.0012 | 0.000 |
| 0.1 | 88.26 ± 0.43 | −0.02 | 0.0003 | 0.000 |
| 0.5 | 88.28 ± 0.43 | 0.00 | 0.0000 | 0.000 |
| 1.0 | 88.27 ± 0.41 | −0.01 | 0.0000 | 0.000 |

**Argmin_k H_ik frac per τ_p (fraction of nodes where H_ik is minimal at each depth):**

Cora:
| τ_p | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| 0.01 | 97.6% | 2.1% | 0.01% | 0.1% | 0.1% |
| 0.05 | 73.0% | 0.2% | 0% | 0.5% | 26.2% |
| 0.1 | 56.6% | 0.05% | 0% | 11.7% | 31.7% |
| 0.5 | 45.8% | 0.01% | 0% | 21.3% | 32.9% |
| 1.0 | 44.8% | 0.01% | 0% | 22.2% | 33.0% |

Computers:
| τ_p | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| 0.01 | 98.8% | 1.2% | 0% | 0% | 0.04% |
| 0.05 | 98.7% | 1.2% | 0% | 0% | 0.2% |
| 0.1 | 98.5% | 1.2% | 0% | 0% | 0.3% |
| 0.5 | 98.4% | 1.2% | 0% | 0% | 0.4% |
| 1.0 | 98.4% | 1.2% | 0% | 0% | 0.4% |

**Per-depth H_ik mean (across 3 seeds) at best-probe τ_p:**

Cora best Z_α τ_p = 1.0 (81.86). Per-depth H: {k=0: 1.672, k=1: 1.925, k=2: 1.944, k=4: 1.912, k=8: 1.893} (ln 7 = 1.946).
Computers best Z_α τ_p = 0.05 (88.29). Per-depth H: {k=0: 2.218, k=1: 2.293, k=2: 2.302, k=4: 2.301, k=8: 2.297} (ln 10 = 2.303).

**Correlations of argmin_k H with node structure (Cora τ_p=1.0, Computers τ_p=0.05):**

| Dataset | vs degree | vs local homophily | vs 1-hop label entropy |
|---|---|---|---|
| Cora | 0.015 | −0.088 | 0.015 |
| Computers | 0.564 | −0.025 | 0.019 |

# DIAGNOSTIC RESULTS — Config D (D6c-V-WD)

| Dataset | D6c primary (INQ-003) Z_mean | D6c-V-WD Z_mean | Δ | D6c primary Z_concat | D6c-V-WD Z_concat | Δ |
|---|---|---|---|---|---|---|
| Cora | 81.80 ± 0.20 | 80.42 ± 0.43 | **−1.38** | 81.83 ± 0.29 | 80.33 ± 0.39 | **−1.50** |
| Computers | 88.26 ± 0.40 | 86.60 ± 0.29 | **−1.66** | 88.00 ± 0.27 | 87.24 ± 0.09 | **−0.76** |

**Pass/fail vs bars:**

| Dataset | Readout | Z-probe | Hard bar | Soft floor | Hard | Soft |
|---|---|---|---|---|---|---|
| Cora | Z_mean | 80.42 ± 0.43 | 78.87 | 76.25 | PASS (+1.55) | PASS |
| Cora | Z_concat | 80.33 ± 0.39 | 78.87 | 76.25 | PASS (+1.46) | PASS |
| Computers | Z_mean | 86.60 ± 0.29 | 87.53 | 86.10 | FAIL (−0.93) | PASS (+0.50) |
| Computers | Z_concat | 87.24 ± 0.09 | 87.53 | 86.10 | FAIL (−0.29) | PASS (+1.14) |

**||W_k||_F mean across 3 seeds:**

| Dataset | k=0 | k=1 | k=2 | k=4 | k=8 | xavier ref | ratio range |
|---|---|---|---|---|---|---|---|
| Cora | 217.18 | 161.49 | 130.21 | 123.93 | 142.60 | 37.85 | 3.27–5.74 (all > init) |
| Computers | 76.92 | 75.06 | 77.54 | 72.36 | 69.40 | 27.69 | 2.51–2.80 (all > init) |

**cos(W_k, W_k') mean across seeds:**

Cora:
| pair | 0-1 | 0-2 | 0-4 | 0-8 | 1-2 | 1-4 | 1-8 | 2-4 | 2-8 | 4-8 |
|---|---|---|---|---|---|---|---|---|---|---|
| cos | 0.823 | 0.741 | 0.589 | 0.455 | 0.831 | 0.621 | 0.488 | 0.806 | 0.635 | 0.791 |

Computers:
| pair | 0-1 | 0-2 | 0-4 | 0-8 | 1-2 | 1-4 | 1-8 | 2-4 | 2-8 | 4-8 |
|---|---|---|---|---|---|---|---|---|---|---|
| cos | 0.890 | 0.786 | 0.695 | 0.652 | 0.866 | 0.763 | 0.721 | 0.856 | 0.810 | 0.835 |

**Per-depth Z_k probes:**

Cora:
| Depth | Z_k D6c primary | Z_k D6c-V-WD | Δ |
|---|---|---|---|
| 0 | 76.57 ± 0.27 | 78.69 ± 0.67 | +2.12 |
| 1 | 81.21 ± 0.21 | 80.75 ± 0.10 | −0.46 |
| 2 | 82.02 ± 0.40 | 80.26 ± 0.41 | −1.76 |
| 4 | 81.33 ± 0.40 | 80.24 ± 0.45 | −1.09 |
| 8 | 80.08 ± 0.17 | 79.35 ± 0.35 | −0.73 |

Computers:
| Depth | Z_k D6c primary | Z_k D6c-V-WD | Δ |
|---|---|---|---|
| 0 | 82.44 ± 0.26 | 83.05 ± 0.16 | +0.61 |
| 1 | 88.24 ± 0.24 | 86.59 ± 0.22 | −1.65 |
| 2 | 88.48 ± 0.32 | 86.67 ± 0.27 | −1.81 |
| 4 | 86.70 ± 0.25 | 85.42 ± 0.08 | −1.28 |
| 8 | 82.56 ± 0.41 | 82.22 ± 0.41 | −0.34 |

# Numerical answers to RA's numbered questions

**Q1 — Does D6c-arxiv hard-pass?** Yes. Raw best single-depth on arxiv (5 seeds, n=25) is Â^2 X = 60.28. D6c Z_mean 64.90 (+4.62); Z_concat 68.33 (+8.05). Best-val and final-epoch differ by ≤0.01 pts on both readouts.

**Q2 — Does 5-seed Computers confirm or weaken the +0.73?** Z_mean 5-seed pooled 88.24 ± 0.42 (n=25) vs INQ-003 3-seed pooled 88.26 ± 0.40 (n=15). Per-seed margin over hard bar 87.53: 5-seed mean +0.71 (vs 3-seed +0.73). Z_concat 5-seed 87.96 ± 0.30 (vs 3-seed 88.00 ± 0.27, margin +0.43 vs +0.47). Per-seed Z_mean: {87.75, 88.71, 88.34, 87.74, 88.64} — all 5 > 87.53. Per-seed Z_concat: {87.78, 88.38, 87.83, 87.56, 88.23} — all 5 > 87.53.

**Q3 — Does D6c+α move the probe up on Cora?** Δ (Z_α − Z_mean) per τ_p: {0.01: −0.14, 0.05: −0.02, 0.1: 0.00, 0.5: −0.03, 1.0: +0.01}. Max Δ = +0.01 at τ_p=1.0. All τ_p within ±0.15 of Z_mean.

**Q4 — Does D6c+α move the probe up on Computers?** Δ (Z_α − Z_mean) per τ_p: {0.01: −0.24, 0.05: +0.01, 0.1: −0.02, 0.5: 0.00, 1.0: −0.01}. Max Δ = +0.01 at τ_p=0.05. All τ_p within ±0.24 of Z_mean.

**Q5 — Does D6c-V-WD beat D6c primary?** No on either dataset.
- Cora: Z_mean 80.42 vs 81.80 (−1.38); Z_concat 80.33 vs 81.83 (−1.50).
- Computers: Z_mean 86.60 vs 88.26 (−1.66); Z_concat 87.24 vs 88.00 (−0.76).
- ||W_k||_F grew above xavier init in both datasets (Cora ratios 3.27–5.74, Computers 2.51–2.80).

**Q6 — Does argmin_k H from k-means on Z_k differ qualitatively from V2-E1 argmin_k on X_k?**
- Cora: V2-E1 X_k τ_p=1.0 argmin concentrated at k=8 (99.4%). D6c+α Z_k τ_p=1.0 argmin distributed: k=0 44.8%, k=8 33.0%, k=4 22.2%. At τ_p=0.01, V2-E1 X_k gave k=0 97.6% and D6c+α Z_k also gives k=0 97.6%.
- Computers: V2-E1 X_k argmin k=0 ≥98.7% at every τ_p. D6c+α Z_k argmin k=0 ≥98.4% at every τ_p, k=1 at 1.2%. Computers distribution is near-identical between X_k and Z_k spaces.

# Implementation / cost notes

- Modules: `ad_ssl/experiments/adssl_d6.py` (Configs A/B/D), `ad_ssl/experiments/adssl_d6c_alpha.py` (Config C).
- `ad_ssl/eval.py` `linear_probe()` extended with `track_best_val` option; when enabled returns `test_acc_best_val_mean/std/restarts` alongside final-epoch metrics.
- Branch: `ad-ssl/track2-d6c-extensions` off `ad-ssl/track2-d6-cross-depth-infonce`.
- Wall-clock per seed:
  - A (arxiv): precompute 3.5 s, train 115 s (mean-epoch 575 ms), total ≈ 120 s × 5 seeds = 10 min.
  - B (5-seed, seeds 3-4 only): Cora 12.1 s/seed, Computers 29.7 s/seed.
  - C (D6c+α): Cora ≈ 45 s train + 45 s kmeans/seed; Computers ≈ 90 s train + 80 s kmeans/seed.
  - D (V-WD): Cora 12.1 s/seed, Computers 29.9 s/seed.
