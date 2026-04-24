---
inquiry_id: INQ-2026-04-23-004
parent_inquiry_id: INQ-2026-04-23-003
topic: D6c extensions — ogbn-arxiv, 5-seed confirmation, α ablation, V-WD variant
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-23
responded:
priority: high
status: open
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

_(Awaiting Coding Agent.)_
