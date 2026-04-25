---
title: Reviewer Attacks and Defenses
type: synthesis
tags: [neurips-2026, reviewers, risk]
created: 2026-04-21
updated: 2026-04-24
sources: [[RESEARCH_AGENT_ONBOARDING], [[INQ-2026-04-24-002]]]
---

# Reviewer Attacks and Defenses

Anticipated reviewer objections and our prepared answers. Each row lists the
required evidence — missing evidence is a submission blocker.

| Reviewer says | Our answer | Evidence needed |
|---|---|---|
| "This is just [[GPRGNN]] + [[BGRL]]." | GPRGNN is supervised and coupled (end-to-end). BGRL does not use multi-depth views. | Ablation: remove depth views → collapses to [[GGD]]-ish baseline. Remove bootstrap → collapses to [[SUGRL]]. |
| "Why not just sweep k per dataset?" | Learned per-node α_k must beat best fixed k without sweeping. | Per-dataset comparison: learned-α vs best-fixed-k (depth sweep). |
| "You don't beat BGRL/GraphMAE." | We don't claim SOTA accuracy — we claim a Pareto point: comparable accuracy at 10–100× lower cost. | The Pareto figure (see [[Pareto Gap]]). |
| "[[GGD]] at hidden=1500 already hits 71.6 in 0.95s — your Pareto point is not empty." | True — onboarding's framing was too loose. The real claim is per-node adaptive-depth wins where a single `A^5 H_θ` cannot: heterophily, datasets where optimal depth varies per node, and smaller-model regimes. Plus our bootstrap + adaptive α has theoretical motivation (spectral mixture), GGD's inference-time power does not. | Reproduce GGD-1500 in our harness. Show AD-SSL > GGD on (a) heterophilic set, (b) matched-capacity model, (c) at least one OGB dataset. If we can't beat GGD-1500 on all three, we have a much narrower paper. |
| "[[Less is More]] already did this." | Different mechanism (2 views — 1 MLP + 1 k-layer GCN — with global β weighting + direct cosine loss vs our K multi-depth views + per-node α + bootstrap loss; see [[AD-SSL vs Less is More]]). **Correction to onboarding framing**: they do evaluate at 169k-node scale (Arxiv-year), not just small graphs. Our scale advantage is evaluating on standard OGB category-prediction, which they don't. | Reproduce GCN-MLP at ogbn-arxiv + ogbn-products in our harness (Coding Agent). 2-view ablation of AD-SSL to isolate multi-depth contribution. Scale to ogbn-papers100M (they don't). |
| "This is just [[ATP]] ported to SSL." | ATP tunes a **scalar** per-node kernel coefficient r̃ from closed-form degree/eigenvector/clustering (weight-free, supervised). AD-SSL learns a **K-dim** per-node mixture over K depth views from content-based cross-depth consistency. Different mechanism, different output, different regime. See [[ATP]] §"Differences from AD-SSL". | Ablation: AD-SSL with uniform α_k (no per-node weighting) vs ATP's r̃-kernel as a pre-processor. Explicit differentiation paragraph in related work. |
| "6 ideas failed in preliminary — why will this work?" | Those were sampling tweaks on a fixed-depth encoder. The one thing that **did** work was changing depth ([[Prepropx Depth Finding]]: +0.80). AD-SSL extends exactly that insight. | The prepropx table in the appendix. |
| "Only tested on homophilic graphs." | AD-SSL is scoped to homophilic graphs — same scope as [[GGD]], [[SUGRL]], [[BGRL]], [[GraphMAE2]], the exact Pareto frontier we compete on. Methods that claim heterophily ([[PolyGCL]], [[GraphACL]]) engineer for it explicitly with spectral high-pass channels or asymmetric predictors; our monotone-low-pass depth views do not, and we clearly mark heterophily as future work rather than retrofitting. Additional theoretical support: [[Rethinking graph neural networks from a geometric perspective of node features]] (Ji et al., ICLR 2025) Corollary 1 shows datasets like Actor have near-degenerate feature centroid simplexes — an *intrinsic* feature-space limit that no aggregation mechanism can resolve. Heterophily is as much a feature-space problem as an architecture problem. | Introduction explicitly scopes to homophily; related work cites PolyGCL/GraphACL as the heterophily-specialized line, and Ji et al. as theoretical backing for the intrinsic-limit framing. |
| "Benchmarks cherry-picked / dated." | We use OGB + established homophilic/heterophilic sets + scale study. | Main table + scale study + appendix per-dataset ablations. |
| "Missing 2024–2025 baselines." | Full landscape in related work; reproduced or reported-in-table. | [[Competitive Landscape 2026]] expanded into a related-work section with all recent methods. |
| "[[PolyGCL]] already does augmentation-free SSL with learnable spectral filters — what's new?" | PolyGCL uses **2 global spectral views** (low-pass + high-pass Chebyshev) mixed with **global** α, β. AD-SSL uses **K monotone-low-pass depth views** mixed with **per-node** α_{i,k}. Different view family, different mixing granularity. PolyGCL also needs a spectral GNN encoder per view; AD-SSL is MLP on precomputed `Â^k X`. | Explicit differentiation paragraph in related work. Ablation: AD-SSL with global α (single vector) vs per-node α (ours). If global-α closes the gap, the per-node claim dies. |
| "[[GPRGNN]] already learns γ_k across depth — per-node just makes it bigger." | GPRGNN learns **one global γ_k vector, end-to-end with labels**. AD-SSL learns **per-node α_{i,k} under SSL**. Two separate claims: (a) SSL can supervise a γ_k-like mixture without labels, (b) per-node beats global. | Global-γ SSL ablation (replace per-node α with single learned γ vector) — directly isolates the per-node claim. |
| "[[GraphMAE2]] is the only thing that transfers at papers100M scale ([[GSTBench]])." | True in GSTBench's setup. That's why we scope AD-SSL to per-dataset Pareto, not foundation-model pretraining. We also view feature reconstruction as complementary — a hybrid (bootstrap + feature recon) is a clear follow-up. | Explicit scoping in introduction; no "foundation model" framing. Optional: small transfer probe as appendix. |
| "Contrastive methods don't transfer (GSTBench)." | We're not making a transfer claim. Per-dataset Pareto in the Pareto Gap framing ([[Pareto Gap]]). | Introduction must not over-claim; related work must cite GSTBench honestly. |
| "[[BLNN]] already enriches BGRL with structural positives — what's new?" | BLNN enriches positives along the **spatial** axis (1-hop neighbors with attention). AD-SSL enriches along the **depth** axis (multi-hop precomputed views with per-node weighting). Different mechanisms, stackable in principle. BLNN also keeps BGRL's augmentation pair and full GNN encoder — AD-SSL is augmentation-free and MLP-only. | Related-work paragraph positioning BLNN as a concurrent orthogonal extension. Optional: cite BLNN numbers on their 5 small graphs if we run there. |
| "[[MHVGCL]] is the same method — same InfoNCE-over-hops loss, same concat readout." | Loss family and readout overlap is real; mechanism differs. (1) Encoder-free precompute (D6c) vs. trainable linear encoder + per-epoch propagation (MHVGCL). (2) Per-depth W_k with residual `Z_k = X_k + W_k X_k` (load-bearing — D6a/D6b without residual fail Computers by −5.14/−7.21) vs. single shared W. (3) Raw `Â^k X` contrast vs. ELU-nonlinear iterative-fusion contrast that loses raw-feature reference. (4) The cross-depth contrast axis is load-bearing for D6c — Config C.2 B2 (per-depth-independent InfoNCE) isolates it. (5) MHVGCL does not report ogbn-arxiv (few-shot 1–4 label scope); D6c +8.05 arxiv unchallenged. | **Config C.2 B2** ablation (per-depth-independent InfoNCE): if B2 degrades materially, the cross-depth axis is confirmed load-bearing. **Config B** efficiency benchmark: MHVGCL cited paper-as-reported (no public code located 2026-04-24); per-epoch-propagation cost story made via D6c precompute-once vs. encoder-methods cost band. See [[MHVGCL]] and [[INQ-2026-04-24-002]] § Config C.2. |
| "[[DGD]] is the same — decoupled GCN + SSL on precomputed features." | Different loss family (BCE group discrimination, GGD descendant) and different contrast axis (real-node-vs-corrupted-node, not depth-vs-depth). DGD uses a single fixed k at training; the m=10 `A^m H_θ` skip is a post-hoc inference merge, not a contrastive axis. **Low risk.** | Related-work paragraph: D6c shares decoupled-precompute lineage but introduces depth-as-view contrastive axis. See [[DGD]]. |
| "Marginal gains without statistical significance ([[Graph Learning Poor Benchmarks]])." | 95% CIs + paired t-tests on every reported number. 100-run bootstrap following [[APPNP]]'s protocol on small graphs. | Stats infrastructure in eval harness (Coding Agent inquiry). |

## NeurIPS 2026 norms (signals from recent ICLR/NeurIPS discourse)

Reviewers increasingly value:

- Rigorous ablations (one-knob-at-a-time).
- Honest failure reporting — include what didn't work (we have this in [[Preliminary Validation - 168 Runs]]).
- Theoretical motivation, even lightweight (spectral-filter interpretation of depth-bootstrapping).
- Scaling studies at OGB / OGB-LSC scale.
- Reproducibility.

Reviewers increasingly dislike:

- Cherry-picked benchmarks.
- Missing recent baselines (2024–2025).
- Vague novelty claims.
- Methods that only work on Cora/CiteSeer.

Calibration reference: [Graph Learning Will Lose Relevance Due To Poor Benchmarks](https://arxiv.org/pdf/2502.14546) (ICLR 2025 position paper). Not a competitor — a **thermometer** for the current review climate.

## Current defensive gaps (to fix before submission)

- [x] ~~Heterophilic benchmark — add Chameleon/Squirrel/Wisconsin.~~ **Out of scope (2026-04-21).** Matches scope of [[GGD]], [[SUGRL]], [[BGRL]], [[GraphMAE2]]; heterophily-claiming methods ([[PolyGCL]], [[GraphACL]]) engineer for it explicitly. See [[Thesis]] § Scope.
- [ ] Reproduction of [[Less is More]] at **standard ogbn-arxiv and ogbn-products** — not started. Their paper only has Arxiv-year at that scale.
- [ ] 2-view ablation of AD-SSL (isolating multi-depth contribution from framework) — not planned yet.
- [ ] Scale study to ogbn-papers100M — depends on Coding Agent cluster availability.
- [ ] Spectral-filter interpretation of depth-bootstrapping — needs derivation.
- [ ] ATP's HPC (High-Deg edge masking) — should we adopt as preprocessing? Would strengthen scale claims and match their framework. Small appendix ablation.
- [ ] Global-γ SSL ablation (no per-node α) — **load-bearing** for the novelty claim vs [[GPRGNN]]. Must be in the main ablation table.
- [ ] 95% CI + paired t-test infrastructure in eval harness — required by [[Graph Learning Poor Benchmarks]] calibration.
- [ ] Efficiency benchmark with FLOPs (not just wall-clock + memory) — locked into Config B per [[INQ-2026-04-24-002]] using `fvcore.FlopCountAnalysis` with `flops_total ≈ 3 × flops_forward` heuristic. Hardware-independent cost story complements wall-clock for the Pareto figure.
- [ ] Config C.2 B2 (per-depth-independent InfoNCE) — load-bearing for the [[MHVGCL]] defense. Must isolate that the cross-depth axis (not the loss family alone) drives D6c's gains.
- [x] ~~Small appendix transfer probe (papers100M-pretrain → one downstream).~~ **Out of scope (2026-04-21).** AD-SSL operates in the per-dataset regime, field default. See [[Thesis]] § Scope.
