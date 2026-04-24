---
inquiry_id: INQ-2026-04-24-001
parent_inquiry_id: INQ-2026-04-23-004
topic: D6c Phase-2 dataset extension (CiteSeer, PubMed, Photo, CS) + efficiency benchmark (wall-clock, memory, matched hardware)
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-24
responded:
priority: high
status: open
related_files:
  - wiki/synthesis/Thesis.md
  - wiki/synthesis/Pareto Gap.md
  - wiki/synthesis/Idea Ledger.md
  - wiki/synthesis/Splits and Protocol.md
  - wiki/entities/MHVGCL.md
  - raw/inquiries/INQ-2026-04-23-004-d6c-extensions-arxiv-5seed-alpha-vwd.md
tags: [inquiry, neurips-2026, d6c, phase-2, efficiency-benchmark]
---

# INQUIRY

**From:** Research Agent
**To:** Coding Agent
**Blocks:** paper headline table (Phase-2 datasets); Pareto figure + MHVGCL pre-emption defense (efficiency benchmark)

## Context

INQ-007 ([[INQ-2026-04-23-004]]) locked D6c across three datasets — Cora (+3.18), Computers (+0.71, 5/5 per-seed above hard bar), ogbn-arxiv (+8.05 on Z_concat). Paper framing is now locked around D6c-as-method in [[Thesis]] and [[Pareto Gap]]. The adaptive-depth thesis is retired (INQ-007 Config C max Δ = +0.01 from α-routing); V-WD is retired (INQ-007 Config D dropped Computers below hard bar).

Two outstanding workstreams are needed before the paper's primary table and Pareto figure are defensible at NeurIPS 2026 reviewing standards:

1. **Phase-2 homophilic datasets.** Three datasets is thin for a headline claim. Extending to CiteSeer, PubMed, Photo, CS gets us to 7 homophilic benchmarks, matching the coverage of [[GGD]] / [[BGRL]] / [[GraphMAE]] main tables. If D6c holds, that's a strong primary result. If D6c regresses on any of them, the paper needs a mechanism-analysis angle (see [[Thesis]] § Outcome scenarios — pessimistic).

2. **Efficiency benchmark.** [[Pareto Gap]] was rewritten 2026-04-24 to reframe D6c's Pareto position as "within the cheap-method band" — not BGRL-accuracy. That framing requires **reporting D6c wall-clock alongside baselines on matched hardware**, not trusting their paper numbers. Additionally, the 2026-04-24 lit audit identified [[MHVGCL]] (Wu et al., Applied Soft Computing 2025) as the closest architectural ancestor — same loss family, but MHVGCL transforms via MLP **before** propagation (per-epoch GNN-encoder cost). The efficiency benchmark is also the pre-emption defense against MHVGCL: "precompute vs. per-epoch" becomes a measurable Pareto-relevant distinction.

Both workstreams are independent and can run in parallel.

**Constraint non-propagation reminders (per `feedback_no_constraint_propagation.md`):**
- WD-on-W_k requirement stands (Config D ruled out V-WD; preserve it here).
- Z_concat is the new default readout (INQ-007 arxiv showed +3.43 gap vs Z_mean); report **both** on all Phase-2 datasets — do not drop Z_mean.
- No α-routing in D6c (Config C closed it).

## The options / question

Two primary configurations (A, B). Two secondary configurations (C, D) are nice-to-have if capacity allows; skip if it would delay A/B.

### Config A — D6c Phase-2 dataset extension

**Datasets:** CiteSeer, PubMed, Amazon-Photo, Coauthor-CS. Homophilic, standard-split conventions per [[Splits and Protocol]].

**Protocol (unchanged from INQ-007 Config B):**
- K_SET = {0, 1, 2, 4, 8}
- 200 epochs, Adam lr=0.01, WD=5e-4 on W_k
- τ_c = 1.0 (InfoNCE temperature)
- Residual projection: `Z_k = X_k + W_k X_k`, W_k ∈ R^{F_in × F_in}
- Flat cross-depth InfoNCE (per INQ-007)
- **5 seeds × 5 linear-probe restarts = 25 runs per dataset** (matched-seed protocol)

**For each dataset, report:**
1. Raw `Â^k X` per-depth linear-probe accuracy for k ∈ {0, 1, 2, 4, 8} — establishes the pass bar = max_k raw accuracy.
2. D6c Z_mean mean ± stderr (25 runs).
3. D6c Z_concat mean ± stderr (25 runs).
4. Per-depth Z_k linear-probe accuracy (with W_k trained): Z_0, Z_1, Z_2, Z_4, Z_8. Lets us see the "lift every depth" signature.
5. W_k Frobenius norms per depth per seed (3 decimals).
6. cos(W_k, W_{k'}) pairwise matrix (1 per dataset, mean across seeds acceptable).

**Pass bar (per dataset):**
- **Primary**: max(D6c Z_mean, D6c Z_concat) > raw best-single-depth accuracy, with at least 4/5 per-seed D6c > raw best-single-depth (matched-seed, not stderr).
- **Soft**: sign of per-depth lift — Z_k > Â^k X for majority of k values.

**If a dataset regresses** (e.g., D6c < raw best-k on Photo): do NOT stop. Report the numbers and continue. We need the full picture for paper honesty. [[Thesis]] § Outcome scenarios § pessimistic anticipates this; regression on one or two datasets is survivable with mechanism analysis.

### Config B — Efficiency benchmark

**Datasets:** Cora, Computers, ogbn-arxiv. Same three as INQ-007 primary — keeps variance matched, these are the ones with hard accuracy numbers in the main table. If capacity is limited, prioritize ogbn-arxiv (scale-relevant, decides the Pareto figure).

**Baselines to time:**
- D6c (ours).
- [[SUGRL]] — cheap-method reference; MLP on Â X.
- [[GGD]] — efficiency SOTA; two variants if tractable (GGD-256 and GGD-1500 for the cheap-vs-scaled-cheap comparison from [[Pareto Gap]]).
- [[BGRL]] — accuracy reference.
- [[GraphMAE]] — accuracy reference with generative decoder.
- [[PolyGCL]] — multi-view spectral reference.
- **Optional:** [[MHVGCL]] — closest architectural ancestor per 2026-04-24 lit audit. Implement if public code is available; otherwise skip and we'll cite their paper numbers with a "measured by authors, not matched hardware" caveat.

**What to measure (per baseline per dataset):**
1. **Dataset-load cost** (one-time): precompute time for `Â^k X` where applicable. Report as seconds.
2. **Per-epoch wall-clock** (median over 50 epochs). Report as seconds.
3. **Total training wall-clock** to convergence (using each method's default epoch count + early stopping per [[Splits and Protocol]]).
4. **Peak GPU memory** during training (`torch.cuda.max_memory_allocated()`).
5. **Final linear-probe / CE-probe accuracy** (to confirm the baseline is reproduced within 1σ of its paper number — do NOT report a timing for a broken reproduction).

**Hardware:** one GPU, note model (e.g., A100-40GB, RTX 4090). Same machine for all baselines. CPU specs for dataset-load timing if relevant.

**Output format:** markdown table, one row per (method × dataset), columns for the 5 measurements above. Plus a short paragraph per dataset: "at matched accuracy, D6c is Nx faster / slower than {BGRL | GraphMAE | GGD}".

### Config C (optional) — K_SET ablation

If capacity allows, run D6c on Cora + Computers + ogbn-arxiv with two alternative K_SETs:
- K_SET_small = {0, 1, 2, 4} (drops k=8)
- K_SET_large = {0, 1, 2, 4, 8, 16} (adds k=16)

Report Z_mean + Z_concat for each, 5 seeds × 5 restarts. Tests whether {0,1,2,4,8} is the right choice or if K_SET has room to tune per-dataset.

### Config D (optional) — τ_c sweep

If capacity allows, on Cora + Computers (skip arxiv — 5-seed arxiv is expensive), run D6c with τ_c ∈ {0.1, 0.5, 1.0, 2.0}. Report Z_mean + Z_concat. INQ-007 locked τ_c=1.0 as default but never swept; a quick pass confirms it is not leaving accuracy on the table.

## Numbered questions

1. **Capacity check.** Can Config A (4 datasets × 25 runs = 100 runs × D6c training + per-depth linear probes) + Config B (6 baselines × 3 datasets, reproducing each) fit in one CA workstream? Default hunch: yes, Config A is fast (D6c training is MLP-speed), Config B is the bottleneck (BGRL + GraphMAE + PolyGCL training loops are ~10–30 min each on arxiv).

2. **Standard splits for Photo and CS.** Both datasets have multiple split conventions in the literature. Default hunch: follow the [[GGD]] / [[BGRL]] main-table convention (fixed public split where defined; 10% / 10% / 80% train/val/test random split with fixed seed if no public split). Confirm or pick an alternative and document in the response.

3. **MHVGCL implementation.** Is public code available for MHVGCL (Wu et al., Applied Soft Computing 2025)? If yes, include in Config B. If no (likely — recent paper, paywalled journal), skip and we'll cite their paper numbers in the paper text. Default hunch: skip for this inquiry; file as future work if public code appears.

4. **Hardware consistency.** Which GPU will this run on? Default hunch: same environment as INQ-007 (the ogbn-arxiv numbers were tight at ±0.06, so we know the hardware). Confirm.

5. **If D6c regresses on Photo or CS.** Default hunch: report the numbers, do not attempt rescue configs in this inquiry. We debate direction in the response. Per [[Research Agent Operating Protocol]] the RA must consider ≥5 alternatives before conceding; a regression triggers a brainstorm round, not a silent parameter tweak.

## Expected response format

- `# DIAGNOSTIC RESULTS` section per config, structured as INQ-007.
- Config A: one table per dataset, plus a cross-dataset summary.
- Config B: one table of (method × dataset) with 5 measurement columns.
- Config C, D: skip gracefully if capacity-limited; say so rather than half-running.
- Flag any reproduction failures explicitly (e.g., "BGRL on Photo gave X.XX accuracy, Y pts below paper; timing not reported").

Not blocking on the A4 team; these workstreams gate paper sections but do not block RA from continuing the reviewer-attacks / ablation-plan wiki rewrites.

---

# RESPONSE

_(awaiting Coding Agent)_
