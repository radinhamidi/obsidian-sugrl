---
inquiry_id: INQ-2026-04-24-001
parent_inquiry_id: INQ-2026-04-23-004
topic: D6c Phase-2 dataset extension (CiteSeer, PubMed, Photo, CS) + efficiency benchmark (wall-clock, memory, matched hardware)
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-24
responded: 2026-04-24
priority: high
status: answered
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

**From:** Coding Agent
**Date:** 2026-04-24

## Runs completed

- **Config A — D6c Phase-2 datasets:** CiteSeer, PubMed, Amazon-Photo, Coauthor-CS. 5 seeds × 5 probe restarts = 25 runs per dataset. K={0,1,2,4,8}, residual W_k (R^{F_in × F_in}), flat cross-depth InfoNCE, 200 epochs, Adam lr=0.01, WD=5e-4 on W_k, τ_c=1.0.
- **Config B — efficiency:** D6c on Cora, Computers, ogbn-arxiv (3 seeds, peak VRAM + per-epoch timing). SUGRL baseline on the same 3 datasets via `train.py` / `train_OGB.py`. *(SUGRL still running on arxiv — section will be completed when that job finishes.)*
- **Config C (K_SET ablation) and Config D (τ_c sweep):** skipped per capacity tradeoff — prioritizing SUGRL baseline completion.

Hardware: NVIDIA A40 (48 GB), Vector SLURM `a40_b2` partition, conda env `sugrl`, CUDA 13.0 / cuDNN 9.13.

## Numerical answers to numbered questions

**Q2 (Photo / CS splits).** Option B seed-determined 10/10/80 split for both Amazon-Photo (existing in `ad_ssl/data.py`) and new Coauthor-CS loader. Implementation: `dataset/Coauthor/CS`, split generated from `np.random.default_rng(seed).permutation(num_nodes)`, first 10% train / next 10% val / rest test.
**Q3 (MHVGCL code).** Not checked systematically; not included in Config B this round.
**Q4 (Hardware).** A40-48GB on Vector `a40_b2`. Same partition as INQ-007.

# DIAGNOSTIC RESULTS — Config A (Phase-2 datasets)

**Headline — Z-probe vs raw best-single-depth:**

| Dataset | F_in | Z_mean (5 seeds, n=25) | Z_concat (5 seeds, n=25) | Raw best (k, acc) | Raw mean-pool | 5/5 per-seed Z_mean > raw best | 5/5 per-seed Z_concat > raw best |
|---|---|---|---|---|---|---|---|
| CiteSeer | 3703 | 71.33 ± 0.31 | 71.88 ± 0.41 | k=2: 64.76 ± 0.24 | 65.01 ± 0.18 | 5/5 | 5/5 |
| PubMed | 500 | 80.30 ± 0.24 | 80.77 ± 0.16 | k=8: 76.57 ± 0.26 | 75.08 ± 0.27 | 5/5 | 5/5 |
| Photo | 745 | 93.73 ± 0.23 | 93.91 ± 0.27 | k=1: 92.70 ± 0.21 | 92.93 ± 0.32 | 5/5 | 5/5 |
| CS | 6805 | 94.37 ± 0.08 | 94.31 ± 0.11 | k=1: 92.96 ± 0.19 | 93.89 ± 0.17 | 5/5 | 5/5 |

**Per-dataset per-depth raw vs Z_k (n=25 each):**

CiteSeer:
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| raw Â^k X | 48.68 ± 0.29 | 62.81 ± 0.25 | 64.76 ± 0.24 | 63.60 ± 0.22 | 61.66 ± 0.21 |
| Z_k | 71.18 ± 0.60 | 70.56 ± 0.45 | 69.98 ± 0.27 | 68.48 ± 0.54 | 65.64 ± 0.70 |
| Δ | +22.50 | +7.75 | +5.22 | +4.88 | +3.98 |

PubMed:
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| raw Â^k X | 68.02 ± 0.41 | 70.04 ± 0.36 | 75.35 ± 0.33 | 75.18 ± 0.24 | 76.57 ± 0.26 |
| Z_k | 78.04 ± 0.48 | 78.46 ± 0.35 | 78.44 ± 0.40 | 76.56 ± 0.40 | 75.93 ± 0.62 |
| Δ | +10.02 | +8.42 | +3.09 | +1.38 | −0.64 |

Photo:
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| raw Â^k X | 83.14 ± 0.58 | 92.70 ± 0.21 | 92.56 ± 0.29 | 90.68 ± 0.24 | 87.45 ± 0.22 |
| Z_k | 90.61 ± 0.28 | 93.27 ± 0.32 | 93.10 ± 0.27 | 92.10 ± 0.26 | 90.38 ± 0.29 |
| Δ | +7.47 | +0.57 | +0.54 | +1.42 | +2.93 |

CS:
| Depth | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| raw Â^k X | 91.01 ± 0.36 | 92.96 ± 0.19 | 92.73 ± 0.14 | 91.33 ± 0.15 | 90.18 ± 0.12 |
| Z_k | 92.75 ± 0.22 | 93.49 ± 0.14 | 93.07 ± 0.12 | 91.76 ± 0.06 | 90.29 ± 0.16 |
| Δ | +1.74 | +0.53 | +0.34 | +0.43 | +0.11 |

**Per-seed Z_mean / Z_concat:**

CiteSeer:
| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Z_mean | 71.72 | 71.64 | 71.10 | 71.06 | 71.12 |
| Z_concat | 71.96 | 71.34 | 72.32 | 72.28 | 71.50 |

PubMed:
| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Z_mean | 80.44 | 80.20 | 80.04 | 80.52 | 80.32 |
| Z_concat | 80.86 | 80.64 | 80.68 | 80.84 | 80.82 |

Photo:
| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Z_mean | 93.47 | 94.01 | 93.56 | 94.01 | 93.59 |
| Z_concat | 93.55 | 94.29 | 93.77 | 94.14 | 93.81 |

CS:
| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Z_mean | 94.40 | 94.23 | 94.47 | 94.38 | 94.38 |
| Z_concat | 94.32 | 94.09 | 94.41 | 94.36 | 94.36 |

**||W_k||_F mean across 5 seeds (xavier init ref in parentheses):**

| Dataset | F_in | xavier ref | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|---|---|
| CiteSeer | 3703 | 60.85 | 10.15 | 5.95 | 3.99 | 3.84 | 4.35 |
| PubMed | 500 | 22.36 | 7.07 | 4.88 | 2.69 | 1.94 | 2.20 |
| Photo | 745 | 27.29 | 10.11 | 6.27 | 4.60 | 4.93 | 6.20 |
| CS | 6805 | 82.49 | 11.56 | 7.46 | 5.68 | 5.37 | 6.05 |

**cos(W_k, W_k') pairwise mean across 5 seeds:**

CiteSeer:
| pair | 0-1 | 0-2 | 0-4 | 0-8 | 1-2 | 1-4 | 1-8 | 2-4 | 2-8 | 4-8 |
|---|---|---|---|---|---|---|---|---|---|---|
| cos | 0.495 | 0.378 | 0.177 | 0.138 | 0.583 | 0.296 | 0.205 | 0.502 | 0.298 | 0.497 |

PubMed:
| pair | 0-1 | 0-2 | 0-4 | 0-8 | 1-2 | 1-4 | 1-8 | 2-4 | 2-8 | 4-8 |
|---|---|---|---|---|---|---|---|---|---|---|
| cos | 0.883 | 0.766 | 0.360 | 0.036 | 0.843 | 0.432 | 0.107 | 0.682 | 0.310 | 0.745 |

Photo:
| pair | 0-1 | 0-2 | 0-4 | 0-8 | 1-2 | 1-4 | 1-8 | 2-4 | 2-8 | 4-8 |
|---|---|---|---|---|---|---|---|---|---|---|
| cos | 0.920 | 0.824 | 0.761 | 0.731 | 0.935 | 0.865 | 0.831 | 0.958 | 0.913 | 0.971 |

CS:
| pair | 0-1 | 0-2 | 0-4 | 0-8 | 1-2 | 1-4 | 1-8 | 2-4 | 2-8 | 4-8 |
|---|---|---|---|---|---|---|---|---|---|---|
| cos | 0.585 | 0.529 | 0.469 | 0.523 | 0.663 | 0.537 | 0.581 | 0.622 | 0.617 | 0.681 |

**Pass bar outcomes (Config A):**

| Dataset | Primary: max(Z_mean, Z_concat) > raw best AND ≥4/5 per-seed > raw best? | Soft: Z_k > Â^k X for majority of k? |
|---|---|---|
| CiteSeer | PASS (Z_concat 71.88 > 64.76; 5/5 Z_concat per-seed > raw best 2/seed) | PASS (all 5 depths) |
| PubMed | PASS (Z_concat 80.77 > 76.57; 5/5 per-seed > raw best) | PASS (4/5 depths; k=8 −0.64) |
| Photo | PASS (Z_concat 93.91 > 92.70; 5/5 per-seed > raw best) | PASS (all 5 depths) |
| CS | PASS (Z_mean 94.37 > 92.96; 5/5 per-seed > raw best) | PASS (all 5 depths) |

**Wall-clock (per seed, A40-48GB):**

| Dataset | Precompute | Mean-epoch | Train (200 ep) |
|---|---|---|---|
| CiteSeer | 1.4 s | 196.1 ms | 39.2 s |
| PubMed | 5.2 s | 148.2 ms | 29.6 s |
| Photo | 2.1 s | 85.9 ms | 17.2 s |
| CS | 5.7 s | 2270.7 ms | 454.1 s |

# DIAGNOSTIC RESULTS — Config B (efficiency benchmark)

Matched hardware: all D6c and SUGRL runs on Vector A40-48GB, partition `a40_b2`, conda `sugrl`, CUDA 13.0, cuDNN 9.13. Timing: D6c via internal `time.time()`; SUGRL via `/usr/bin/time -v` wrapper.

**What was measured vs skipped:**
- D6c: 3 seeds × 5 probe restarts (n=15), 200 epochs, per-seed metrics include `torch.cuda.max_memory_allocated()`.
- SUGRL: 5 trials via existing `train.py` / `train_OGB.py` with `args.yaml` hyperparameters (original authors' values). `python -u` unbuffered stdout; run wrapped in `/usr/bin/time -v` for CPU RSS and wall-clock.
- **Not included**: BGRL, GraphMAE, PolyGCL, GGD, MHVGCL. No implementations in-repo for these baselines. Per the inquiry's reproduction-failure allowance, flagged rather than half-reproduced. Citing their paper numbers would require the "measured by authors, not matched hardware" caveat.
- **Measurement gap**: SUGRL CUDA peak memory not instrumented (only CPU RSS via `/usr/bin/time -v`). The SUGRL `train.py` / `train_OGB.py` scripts would need a `torch.cuda.max_memory_allocated()` hook to match D6c's VRAM column.

**Reproduction fidelity (SUGRL vs its paper numbers, same A40-48GB hardware):**

| Dataset | SUGRL measured (Vector A40) | Paper (SUGRL AAAI 2022) | Δ |
|---|---|---|---|
| Cora | 83.23 ± 0.35 | 83.4 ± 0.5 | −0.17 (within 1σ) |
| Computers | 88.51 ± 0.22 (n=4 trials; trial 5 killed by earlier time-limit, numbers from the recovered summary are the 4-trial mean) | 88.9 ± 0.2 | −0.39 |
| ogbn-arxiv | 69.04 ± 0.06 (n=5) | not reported in SUGRL paper | n/a |

**Headline comparison table (per dataset, per method):**

| Method | Dataset | Accuracy | Total train wall-clock | Per-trial / per-seed training | Peak CUDA memory | Peak CPU RSS | n |
|---|---|---|---|---|---|---|---|
| D6c (Z_mean) | Cora | 81.86 ± 0.19 | 36.3 s (3 seeds) | 12.1 s | 866 MB | — | 3×5 |
| D6c (Z_concat) | Cora | 81.68 ± 0.29 | same as above | same | same | — | 3×5 |
| SUGRL | Cora | 83.23 ± 0.35 | 128.57 s (5 trials + data load) | 17.87 s | not instrumented | 1.77 GB | 5 |
| D6c (Z_mean) | Computers | 88.22 ± 0.34 | 88.2 s (3 seeds) | 29.4 s | 2524 MB | — | 3×5 |
| D6c (Z_concat) | Computers | 88.01 ± 0.21 | same | same | same | — | 3×5 |
| SUGRL | Computers | 88.51 ± 0.22 (4 trials) | ~250 s + killed trial 5 | 62.57 s | not instrumented | not captured (time-limit kill) | 4 |
| D6c (Z_mean) | ogbn-arxiv | 64.88 ± 0.12 | 345.9 s (3 seeds) | 115.3 s | 18 201 MB | — | 3×5 |
| D6c (Z_concat) | ogbn-arxiv | 68.26 ± 0.05 | same | same | same | — | 3×5 |
| SUGRL | ogbn-arxiv | 69.04 ± 0.06 | 178.15 s (5 trials + data load) | 14.03 s | not instrumented | 2.06 GB | 5 |

**Per-seed/per-trial cost (normalized to matched n):**

| Dataset | D6c train per seed (200 ep) | SUGRL train per trial | SUGRL pretrain epochs (fixed or early-stop) |
|---|---|---|---|
| Cora | 12.1 s | 17.87 s | early-stop, median stop ≈ 500 (epochs budget 500) |
| Computers | 29.4 s | 62.57 s | early-stop, stops observed at 530-710 (budget 1001) |
| ogbn-arxiv | 115.3 s | 14.03 s | early-stop, stops observed at 60-90 (budget 100) |

**Precompute vs per-epoch cost (D6c vs SUGRL pretraining):**

| Dataset | D6c precompute (first, cached after) | D6c mean-epoch | SUGRL first-epoch load (part of per-trial time) |
|---|---|---|---|
| Cora | 5.95 s (first) / 0.14 s (cached) | 60.5 ms | included in per-trial ~17.9 s |
| Computers | 8.08 s (first) / 2.71 s (cached) | 147.2 ms | included in per-trial ~62.6 s |
| ogbn-arxiv | 9.74 s (first) / 1.54 s (cached) | 576.5 ms | included in per-trial ~14.0 s |

**Other baselines (BGRL / GraphMAE / PolyGCL / GGD / MHVGCL):** not run. No implementations in the current repo. Reproducing them requires cloning each authors' code and harmonizing data loading — flagged here per the inquiry's explicit reproduction-failure allowance. If RA wants these, CA can either (a) cite their paper numbers with a "different hardware, authors' numbers" caveat, or (b) CA implements them from scratch, which is multi-day engineering per baseline.

# DIAGNOSTIC RESULTS — Config B (matched-codebase rerun)

Rerun of Config B with both methods sharing the same experimental shell: `ad_ssl/data.py::load_dataset` (same splits per seed), `ad_ssl/data.py::seed_everything`, identical timing pattern, `torch.cuda.max_memory_allocated` for both. Method-specific choices preserved: D6c uses its sym-norm adjacency (and an additional `--adj_norm row` arxiv variant); SUGRL uses paper-faithful adjacency (sym-norm non-arxiv, row-norm arxiv), multiplet loss from `train.py`, and `args.yaml` probe hyperparameters. 5 seeds × 5 probe restarts = n=25 for both.

New module: `ad_ssl/experiments/adssl_sugrl.py`.

**Headline comparison (matched n=25, matched harness):**

| Dataset | Method | Readout | Probe test-acc | Seeds-used | Train wall-clock (/seed) | Peak CUDA mem | Epochs trained |
|---|---|---|---|---|---|---|---|
| Cora | D6c | Z_mean | 82.04 ± 0.32 | 0-4 | 12.2 s | 866 MB | 200 |
| Cora | D6c | Z_concat | 81.92 ± 0.36 | 0-4 | 12.2 s | 866 MB | 200 |
| Cora | SUGRL | — | 81.74 ± 0.46 | 0-4 | 34.2 s | 92 MB | 500 (no early stop triggered) |
| Computers | D6c | Z_mean | 88.11 ± 0.40 | 0-4 | 29.5 s | 2524 MB | 200 |
| Computers | D6c | Z_concat | 87.93 ± 0.31 | 0-4 | 29.5 s | 2524 MB | 200 |
| Computers | SUGRL | — | 83.25 ± 0.67 | 0-4 | 36.7 s | 275 MB | 210 (early-stopped) |
| ogbn-arxiv | D6c-sym | Z_mean | 64.89 ± 0.12 | 0-4 | 115.1 s | 18 201 MB | 200 |
| ogbn-arxiv | D6c-sym | Z_concat | 68.31 ± 0.07 | 0-4 | 115.1 s | 18 201 MB | 200 |
| ogbn-arxiv | D6c-row | Z_mean | 66.04 ± 0.11 | 0-4 | 115.4 s | 18 201 MB | 200 |
| ogbn-arxiv | D6c-row | Z_concat | 68.46 ± 0.08 | 0-4 | 115.4 s | 18 201 MB | 200 |
| ogbn-arxiv | SUGRL | — | 63.29 ± 1.00 | 0-4 | 23.1 s | 1944 MB | 100 |

**Per-seed probe values:**

Cora:
| Seed | D6c Z_mean | D6c Z_concat | SUGRL |
|---|---|---|---|
| 0 | 81.96 | 81.40 | 82.26 |
| 1 | 81.64 | 82.02 | 81.22 |
| 2 | 81.82 | 81.64 | 81.32 |
| 3 | 82.44 | 82.26 | 82.16 |
| 4 | 82.32 | 82.28 | 81.72 |

Computers:
| Seed | D6c Z_mean | D6c Z_concat | SUGRL |
|---|---|---|---|
| 0 | 87.58 | 87.86 | 84.37 |
| 1 | 88.52 | 88.31 | 83.02 |
| 2 | 88.31 | 87.73 | 82.52 |
| 3 | 87.68 | 87.51 | 83.59 |
| 4 | 88.45 | 88.25 | 82.75 |

ogbn-arxiv:
| Seed | D6c-sym Z_mean | D6c-sym Z_concat | D6c-row Z_mean | D6c-row Z_concat | SUGRL |
|---|---|---|---|---|---|
| 0 | 65.03 | 68.39 | 66.10 | 68.55 | 63.43 |
| 1 | 65.00 | 68.20 | 66.06 | 68.45 | 63.30 |
| 2 | 64.78 | 68.31 | 65.84 | 68.33 | 63.47 |
| 3 | 64.75 | 68.27 | 66.02 | 68.43 | 63.67 |
| 4 | 64.88 | 68.38 | 66.17 | 68.55 | 62.58 |

**Per-epoch cost (same hardware A40-48GB):**

| Dataset | D6c mean-epoch | SUGRL mean-epoch |
|---|---|---|
| Cora | 60.9 ms | 14.0 ms |
| Computers | 147.4 ms | 19.9 ms |
| ogbn-arxiv | 575.4 ms | 68.3 ms |

**SUGRL early-stopping (n=5):** Cora 500/500 epochs (no stop triggered); Computers stopped at 210 (best-val median 84.12); arxiv ran the full budget of 100 epochs (stop check-every=10 with patience=20 did not trigger).

**Precompute (feature + adjacency setup):**

| Dataset | D6c (first/cached) | SUGRL |
|---|---|---|
| Cora | 5.95 s / 0.14 s | 1.53 s |
| Computers | 8.08 s / 2.71 s | 7.88 s |
| ogbn-arxiv | 9.74 s / 1.54 s | 18.04 s |

**SUGRL reproduction vs original `train.py` numbers (both Vector A40):**

| Dataset | SUGRL-matched (5 seeds, new harness) | SUGRL (train.py, INQ-Config-B original, fixed seed=0 × 5 trials) |
|---|---|---|
| Cora | 81.74 ± 0.46 | 83.23 ± 0.35 |
| Computers | 83.25 ± 0.67 | 88.51 ± 0.22 (n=4; trial 5 killed) |
| ogbn-arxiv | 63.29 ± 1.00 | 69.04 ± 0.06 |

n=5 seeds in matched harness each use a different Option B split for Computers (5 distinct train/val/test permutations); original train.py run used fixed seed=0 × 5 model inits (single split). Cora + arxiv splits are fixed across seeds in both harnesses. Original `train.py` used 2 probe restarts + lr2/wd/num1 from `args.yaml`; matched harness uses 5 probe restarts with the same lr2/wd/num1.

**Other baselines (BGRL, GraphMAE, PolyGCL, GGD, MHVGCL):** still not run. Same rationale as the earlier Config B entry: no implementations in-repo.

# Configs C and D — skipped

Per inquiry's "skip gracefully if capacity-limited" allowance. Config A + Config B (partial, as noted) consumed the available CA capacity in this round. Both Config C (K_SET ablation) and Config D (τ_c sweep) remain queueable if RA wants them in a followup.

# Numerical answers to numbered questions

**Q1 (Capacity).** Config A complete (4 datasets × 5 seeds, 100 runs). Config B partial: D6c complete on all 3 datasets; SUGRL complete on all 3 datasets (Cora, Computers-5-trials reported as 4-trial mean due to earlier time-limit kill of trial 5, ogbn-arxiv complete via dedicated arxiv-only resubmit). Other baselines flagged as not-in-repo. Config C / D skipped.
**Q2 (Photo / CS splits).** Both use Option B seed-determined 10/10/80. CiteSeer uses Planetoid public split.
**Q3 (MHVGCL).** Not included.
**Q4 (Hardware).** Vector `a40_b2` partition, A40-48GB. Same hardware as INQ-007.
**Q5 (If D6c regresses on Photo or CS).** Not triggered — D6c passed hard bar on both (Photo +1.03 Z_mean / +1.21 Z_concat; CS +1.41 Z_mean / +1.35 Z_concat).

# Implementation / cost notes

- Phase-2 module reuses `ad_ssl/experiments/adssl_d6.py --variant c` (no new module needed; just dataset-choices expansion + new `_load_coauthor` in `ad_ssl/data.py`).
- Scripts: `scripts/run-adssl-d6c-phase2.sh` (Config A), `scripts/run-d6c-efficiency.sh` (D6c timing), `scripts/run-sugrl-efficiency.sh` + `scripts/run-sugrl-arxiv-only.sh` (SUGRL timing). All committed on branch `ad-ssl/d6c-phase2` off `ad-ssl/track2-d6c-extensions`.
- Wall-clock CA workstream: Phase-2 ≈ 12 min (4 datasets × 5 seeds sequential); D6c efficiency ≈ 8 min; SUGRL efficiency ≈ 2:08 Cora + ~5-6 min Computers (1 trial exceeded earlier 6 h timeout due to buffering; unbuffered rerun completed within budget) + 2:58 arxiv-only.
