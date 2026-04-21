---
title: Novelty Verification Checklist
type: synthesis
tags: [neurips-2026, novelty, ablations, experiment-design]
created: 2026-04-21
updated: 2026-04-21
sources: [[[GPRGNN]], [[PolyGCL]], [[GraphMAE2]], [[GraphACL]], [[APPNP]], [[SGC]], [[GGD]], [[BGRL]], [[SUGRL]], [[ATP]], [[GSTBench]]]
---

# Novelty Verification Checklist

Every claim in the AD-SSL paper that a reviewer can attack with "prior X already did this" needs a specific ablation or comparison that isolates the contribution. This page enumerates the claims, the prior art that threatens each, and the experiment that proves the differentiation. It feeds directly into Coding Agent inquiries.

## Legend

- 🔴 **Load-bearing** — if this ablation doesn't go our way, the paper's thesis collapses or must be rescoped.
- 🟡 **Defensive** — needed to close a reviewer attack but doesn't threaten the core claim.
- 🟢 **Nice-to-have** — strengthens the paper; absence is survivable.

## Claim 1 — Per-node adaptive depth weighting is novel and necessary

**Threatened by**: [[GPRGNN]] (global γ_k, supervised), [[APPNP]] (fixed `α(1−α)^k`, supervised), [[Less is More]] (global β over 2 views).

| 🔴 | Ablation | Experimental form | Passes if |
|---|---|---|---|
| 🔴 | **Global-γ SSL** (replace per-node α_{i,k} with single learned vector γ_k) | Same training, same depths, same bootstrap loss, only the α parameterization differs | Per-node ≥ global by ≥1 pt on ogbn-arxiv **with overlapping CIs excluded** |
| 🔴 | **Best fixed k** (sweep k=1..10, pick winner per-dataset) | Report best fixed-k result alongside AD-SSL | AD-SSL ≥ best-fixed-k without sweeping |
| 🟡 | **Uniform α** (α_{i,k} = 1/K) | Isolates "any weighting" from "learned weighting" | Learned > uniform |
| 🟢 | **Per-node α heatmap** on a heterogeneous graph | Qualitative figure: nodes near boundaries prefer small k, deep-community nodes prefer large k | Visually interpretable pattern |

## Claim 2 — Bootstrap across depth pairs is a valid SSL signal without augmentation

**Threatened by**: [[BGRL]] (BYOL on augmented views), [[GraphACL]] (asymmetric + uniformity, two-hop monophily), [[PolyGCL]] (DGI-BCE on two spectral views), [[GGD]] (binary discriminator).

| 🔴 | Ablation | Experimental form | Passes if |
|---|---|---|---|
| 🔴 | **Remove bootstrap** (substitute BGRL-style augmentation pair on single depth) | Fix depths, swap loss | AD-SSL ≥ BGRL-on-precomputed; more importantly, no collapse |
| 🔴 | **Swap bootstrap for DGI-BCE** (PolyGCL's loss) on same K depth views | Isolates bootstrap-vs-BCE | Bootstrap ≥ BCE OR provides a cost argument (BCE needs negatives) |
| 🟡 | **Single depth + bootstrap** (K=1) | Isolates "multi-depth" from "bootstrap" | Multi-depth > K=1 |
| 🟡 | **Collapse check** — monitor `‖z_L − z_H‖` or representation rank over training | Standard BYOL diagnostic | No rank collapse, no constant embeddings |

## Claim 3 — Decoupled precompute preserves the Pareto advantage at scale

**Threatened by**: [[SGC]] (K=1 precompute, supervised), [[SUGRL]] (K=1 precompute, triplet), [[GGD]] (ms-scale on ogbn-arxiv @ hidden=1500).

| 🔴 | Ablation | Experimental form | Passes if |
|---|---|---|---|
| 🔴 | **Wall-clock on ogbn-arxiv** (head-to-head GGD-1500, BGRL, GraphMAE2) | Same hardware, same eval protocol | AD-SSL sits on a non-dominated point of the Pareto frontier |
| 🔴 | **ogbn-products scale** | Verify precompute fits memory and scales | Completes in same order of magnitude as on arxiv |
| 🟡 | **Precompute amortization** | Report precompute time as separate column; amortizes over seeds/hparams | Precompute ≤ 1× single BGRL training run |
| 🟢 | **ogbn-papers100M** pretrain + per-dataset eval | Matches GSTBench's setup | Any result — even "we run in X hours" — is publishable |

## Claim 4 — AD-SSL is not subsumed by a reconstruction method

**Threatened by**: [[GraphMAE2]] (Papers100M 64.89, transfers per [[GSTBench]]; contrastive methods do not transfer).

| 🟡 | Ablation | Experimental form | Passes if |
|---|---|---|---|
| 🟡 | **GraphMAE2 head-to-head on ogbn-arxiv at matched cost** | Include GraphMAE2 in Pareto plot | AD-SSL offers a distinct cost point; accuracy gap ≤ 1 pt at comparable cost |
| 🟡 | **Honest transfer scoping** (text) | Introduction scopes AD-SSL to per-dataset regime (locked 2026-04-21); does not claim foundation-model framing | ✅ Done in [[Thesis]] § Scope |
| 🟢 | **Hybrid: AD-SSL + masked feature recon** | Add λ L_recon to bootstrap loss, single knob | Either improves AD-SSL, or cleanly positions as complementary in discussion |

## Claim 5 — Heterophily (OUT OF SCOPE, locked 2026-04-21)

Rescoped to homophily-only. See [[Thesis]] § Scope. Rationale: every baseline on our Pareto frontier ([[GGD]], [[SUGRL]], [[BGRL]], [[GraphMAE2]]) is homophily-only; methods that claim heterophily engineer for it with spectral high-pass channels ([[PolyGCL]]) or asymmetric predictors ([[GraphACL]]). No ablations required.

## Claim 6 — Statistical significance of all reported gains

**Threatened by**: [[Graph Learning Poor Benchmarks]] (position paper; reviewers aligned with it will reject marginal-gain claims).

| 🔴 | Infrastructure | Experimental form | Passes if |
|---|---|---|---|
| 🔴 | **95% CIs** on every headline number | 10 seeds minimum; bootstrap CIs ideally | All main-table numbers carry CIs |
| 🔴 | **Paired t-tests** vs each baseline in main table | Same splits, same seeds where possible | Report p-values; main claims have p < 0.01 |
| 🟡 | **Non-graph baseline** (MLP on raw features) | Shows graph structure is helping | MLP < AD-SSL on every dataset in the main table |

## Submission-blocker summary

If any 🔴 row above has not run and produced a positive result by submission time, the paper must be rescoped. Current 🔴-open items that don't have a coding-agent inquiry yet:

- Global-γ SSL ablation (Claim 1)
- Best-fixed-k sweep (Claim 1)
- Bootstrap-vs-DGI-BCE swap (Claim 2)
- Wall-clock head-to-head on ogbn-arxiv (Claim 3)
- ogbn-products scaling (Claim 3)
- 95% CI + paired t-test infrastructure (Claim 6)

**Next**: file an inquiry for each 🔴 item that isn't already queued. See `raw/inquiries/`.

## Queued CA inquiries (not yet filed, 2026-04-21)

The researcher already has Coding Agent studies in flight; these six are **held** pending those results. Do not file until the researcher signals. Full inquiry bodies drafted in the conversation of 2026-04-21:

1. **INQ-1 Global-γ SSL ablation** — per-node α vs single learned γ_k vector (Claim 1).
2. **INQ-2 Best-fixed-k sweep** — α as one-hot at k ∈ {1,2,3,5,8,10} (Claim 1).
3. **INQ-3 Bootstrap-vs-DGI-BCE swap** — same K views, different loss (Claim 2).
4. **INQ-4 Wall-clock Pareto head-to-head** — AD-SSL + GGD(256,1500) + BGRL + GraphMAE2 + SUGRL on ogbn-arxiv, same hardware (Claim 3, main figure).
5. **INQ-5 ogbn-products scaling** — AD-SSL end-to-end at products scale (Claim 3).
6. **INQ-6 Stats infrastructure** — 95% bootstrap CI + paired t-tests + MLP baseline row in the harness (Claim 6, submission blocker).

Dependency: INQ-6 should land first; INQ-1/2/3 can run in parallel after; INQ-4/5 depend on baseline reproductions.
