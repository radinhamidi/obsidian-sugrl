---
inquiry_id: INQ-2026-04-24-002
parent_inquiry_id: INQ-2026-04-24-001
topic: D6c final experiments plan — HPO, efficiency, ablation, sensitivity, mechanism
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-24
responded:
priority: high
status: open
related_files:
  - raw/inquiries/INQ-2026-04-24-001-d6c-phase2-datasets-and-efficiency-benchmark.md
  - wiki/synthesis/Thesis.md
  - wiki/synthesis/Splits and Protocol.md
  - wiki/synthesis/Pareto Gap.md
  - wiki/synthesis/Reviewer Attacks and Defenses.md
  - wiki/synthesis/Idea Ledger.md
tags: [inquiry, neurips-2026, hpo, ablation, sensitivity, efficiency, mechanism]
---

# INQUIRY

**From:** Research Agent
**To:** Coding Agent
**Blocks:** final numbers for every main-body table and figure of the NeurIPS 2026 submission. This inquiry is the master experimental plan — completion closes Phase 3 and starts paper writing.

## Context

INQ-008 established D6c passes 7/7 homophilic datasets under current (un-HPO'd) defaults and beats SUGRL by +4.86 / +5.17 on Computers / arxiv in the matched harness. Remaining work before the paper can ship:

1. **HPO on D6c** — per-dataset tuned numbers via Optuna. Baselines use their own paper-reported HPs (**Posture A**: our HPO for D6c; baselines cited as-reported since their authors already tuned. Alternative was Posture B = port-and-re-tune baselines, rejected as unnecessary cost.).
2. **Efficiency story** — wall-clock, peak memory, FLOPs, precompute cost. Needed for the Pareto figure. Matched harness for D6c + the baselines that pass the Config B port-selection criteria.
3. **Ablation** — which D6c components are load-bearing? Run at tuned HPs.
4. **Sensitivity** — HP robustness around the tuned optimum.
5. **Mechanism** — evidence for the simplex-collapse-as-contrastive-signal story.

Scope locked to homophilic graphs per [[Thesis]] § Scope (2026-04-21). 7 datasets: Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers, Coauthor-CS, ogbn-arxiv.

## Split protocol

**D6c splits (locked)**:

| Dataset | Split type | Trials | Notes |
|---|---|---|---|
| Cora | Planetoid public (20/class / 500 / 1000) | 5 seeds × 5 probe restarts | DGI convention |
| CiteSeer | Planetoid public | 5 × 5 | same |
| PubMed | Planetoid public | 5 × 5 | same |
| Amazon-Photo | 10/10/80 random (seed-permuted) | 5 × 5 | BGRL/GGD convention |
| Amazon-Computers | 10/10/80 random (seed-permuted) | 5 × 5 | same |
| Coauthor-CS | 10/10/80 random (seed-permuted) | 5 × 5 | same |
| ogbn-arxiv | OGB default masks via `dataset.get_idx_split()['train'/'valid'/'test']` | 5 × 5 | universal |

**Trial count policy: N=5 default. Bump to N=10 only if final-eval stderr > 0.5 pts on any dataset.**

## Baseline inventory

Built from a sweep of `wiki/entities/` + `wiki/synthesis/Competitive Landscape 2026.md`, 2026-04-24.

### Main-table baselines (numerically compared)

13 methods, partitioned into **classical** (pre-2022 references that every graph-SSL paper reports) and **modern** (2022+ SOTA tier).

**Classical tier**:
- DGI (ICLR 2019) — foundational local-vs-global SSL; reported in BGRL / GraphMAE as reference
- MVGRL (ICML 2020) — multi-view contrastive (adjacency ↔ PPR); reported in PolyGCL / GraphACL
- GRACE (ICML 2020) — classical contrastive; reported in BGRL / GraphMAE
- CCA-SSG (NeurIPS 2021) — canonical canonical-correlation SSL; reports all 7 of our datasets

**Modern tier**:
- SUGRL (AAAI 2022) — margin-triplet + MLP + fixed k=1 propagation (architectural predecessor of D6c at the code level)
- BGRL (ICLR 2022) — BYOL-style bootstrap; canonical reviewer-expected baseline
- GGD (NeurIPS 2022) — binary group discrimination; efficiency champion (1-epoch training)
- GraphMAE2 (WWW 2023) — improved masked-feature reconstruction; generative family; **supersedes GraphMAE (KDD 2022)** which is cited in §2 as predecessor
- GraphACL (NeurIPS 2023) — asymmetric contrastive, augmentation-free
- PolyGCL (ICLR 2024) — spectral low-pass / high-pass view SSL; closest spectral cousin to depth-view idea; Planetoid cells flagged `*(60/20/20 split)*`
- DGD (Neurocomputing 2024) — decoupled-GCN + BCE group discrimination; recent decoupled-SSL precursor
- MHVGCL (ASOC 2025) — multi-hop contrastive views with APPNP-style nonlinear fusion + InfoNCE; **medium pre-emption risk** per our analysis (same InfoNCE-over-hops loss family as D6c); Planetoid cells flagged `*(few-shot scope)*`
- BLNN (arXiv 2024) — BGRL + 1-hop neighbor-positive alignment; reports Amazon-Photo/Computers/Coauthor-CS under the dominant 10/10/80 (actually 1:1:8) protocol; no Planetoid or arxiv

### Cited but NOT baselines (related-work only)

Papers referenced in the paper's §2 Related Work but **not in the main accuracy table** because they are not graph-SSL methods on our benchmark or the scope is disjoint enough that no blank-cell representation is meaningful.

| Method | Venue | Why not a baseline |
|---|---|---|
| GraphMAE (KDD 2022) | KDD 2022 | Superseded by GraphMAE2 on overlapping datasets; cite as predecessor |
| Less is More | ICLR 2026 sub | Concurrent; heterophily + small-scale focus |
| GRAPHITE | ICLR 2026 | Supervised heterophily preprocessor; out of scope per [[Thesis]] § Scope |
| APPNP / GPRGNN / ATP | ICLR 2019 / 2021 / 2024 | Supervised adaptive-depth priors (conceptual, not SSL) |
| SGC | ICML 2019 | Supervised simple propagation |
| Rethinking… (Ji et al.) | 2025 | Theory paper (simplex-collapse framing for §3) |
| AFGRL | AAAI 2022 | Not ingested; flagged by BLNN |

## Baseline accuracy audit (reported numbers, paper protocols)

**Dominant protocols**: Planetoid public for Cora/CiteSeer/PubMed, 10/10/80 random for Amazon/CS, OGB canonical for arxiv.

Cells are **blank** where the paper used a non-dominant split or did not report. Blanks flagged `*(non-dominant split)*` or `*(not reported)*`. Follow-up inquiry will re-run any baseline we want to fill in.

| Method | Cora | CiteSeer | PubMed | Computers | Photo | CS | arxiv |
|---|---|---|---|---|---|---|---|
| **Classical tier** | | | | | | | |
| DGI (ICLR 2019) | 82.3 ± 0.6 | 71.8 ± 0.7 | 76.8 ± 0.6 | *(confirm)* | *(confirm)* | *(confirm)* | 70.34 ± 0.16 |
| MVGRL (ICML 2020) | *(confirm public)* | *(confirm public)* | *(confirm public)* | — | — | — | *(not reported)* |
| GRACE (ICML 2020) | 83.3 ± 0.4 *(split unverified)* | 72.1 ± 0.5 *(split unverified)* | 86.7 ± 0.1 *(split unverified)* | 86.25 ± 0.25 | 92.15 ± 0.24 | 92.93 ± 0.01 | 71.51 ± 0.11 *(subsample k=2048)* |
| CCA-SSG (NeurIPS 2021) | 84.00 ± 0.40 | 73.10 ± 0.30 | 81.00 ± 0.40 | 88.74 ± 0.28 | 93.14 ± 0.14 | 93.31 ± 0.22 | 71.24 ± 0.20 |
| **Modern tier** | | | | | | | |
| SUGRL (AAAI 2022) | *(confirm from INQ-008 matched harness)* | — | — | *(matched harness)* | — | — | 68.83 ± 0.40 |
| BGRL (ICLR 2022) | *(20-rnd, blank)* | *(20-rnd, blank)* | *(20-rnd, blank)* | 90.34 ± 0.19 | 93.17 ± 0.30 | 93.31 ± 0.13 | 71.64 ± 0.12 |
| GGD (NeurIPS 2022) | *(confirm)* | *(confirm)* | *(confirm)* | *(confirm)* | *(confirm)* | *(confirm)* | 71.64 ± 0.50 |
| GraphMAE2 (WWW 2023) | 84.50 | 73.40 | 81.40 | *(not reported)* | *(not reported)* | *(not reported)* | 71.95 ± 0.08 |
| GraphACL (NeurIPS 2023) | 84.20 | 73.63 | 82.02 | 89.80 | 93.31 | *(not reported)* | 71.72 ± 0.26 |
| PolyGCL (ICLR 2024) | *(60/20/20, blank)* | *(60/20/20, blank)* | *(60/20/20, blank)* | — | — | — | *(OOM, not reported)* |
| DGD (Neurocomputing 2024) | *(confirm public)* | *(confirm public)* | *(confirm public)* | *(confirm)* | *(confirm)* | — | *(not reported)* |
| MHVGCL (ASOC 2025) | *(few-shot, blank)* | *(few-shot, blank)* | *(few-shot, blank)* | *(few-shot, blank)* | — | *(few-shot, blank)* | *(not reported)* |
| BLNN (arXiv 2024) | *(not reported)* | *(not reported)* | *(not reported)* | 91.02 ± 0.23 | 93.54 ± 0.23 | 93.61 ± 0.15 | *(not reported)* |
| **Ours** | | | | | | | |
| **D6c (un-HPO'd)** | **82.89** | **71.88** | **80.77** | **89.04** | **93.91** | **94.37** | **68.46 ± 0.08** |
| **D6c (tuned)** | Config A | Config A | Config A | Config A | Config A | Config A | Config A |

**Note on cell verification**: classical-tier numbers (DGI, MVGRL, GRACE, CCA-SSG) cited from BGRL / GraphMAE / GraphACL paper tables (secondary source). Radin should verify against primary sources before paper submission. Modern-tier numbers (BGRL, GraphMAE, etc.) cited directly from primary sources (verified in `wiki/entities/*.md` ingest audits).

**GRACE split caveat**: GRACE Cora/CiteSeer/PubMed values are flagged `*(split unverified)*` because GRACE's original table used random splits, not Planetoid public splits. Radin must verify against primary source during port work or substitute a BGRL-re-reported public-split value.

**Policy on blank cells**: 
- `*(non-dominant split)*` — paper used a different protocol; re-run on dominant protocol via a follow-up inquiry if accuracy column is expected.
- `*(not reported)*` — paper did not report this dataset; cell stays blank; do not extrapolate.
- `*(confirm)*` — author's entity page does not pin the exact number; CA fills from primary source during Config B port work.

---

## Config A — Optuna HPO for D6c (per-dataset)

### Search space (per-dataset ranges)

**Continuous HPs — range scales with dataset size**:

| HP | Type | Small (Cora/CiteSeer/PubMed) | Medium (Photo/Computers/CS) | Large (arxiv) |
|---|---|---|---|---|
| τ (InfoNCE temp) | log-uniform | [0.05, 3.0] | [0.05, 3.0] | [0.01, 2.0] |
| lr | log-uniform | [1e-4, 5e-2] | [1e-5, 1e-2] | [1e-5, 5e-3] |
| weight_decay | log-uniform | [1e-6, 1e-2] | [1e-6, 1e-2] | [1e-7, 1e-3] |

**Categorical HPs — same across datasets**:

| HP | Choices | Count |
|---|---|---|
| K_set | `{0,1,2}`, `{0,1,2,4}`, `{0,1,2,4,8}`, `{0,2,4,8,16}`, `{0,1,2,3,4,5,6,7,8}`, `{0,1,2,4,8,16}`, `{0,1,2,4,8,16,32}` | 7 |
| readout | `Z_mean`, `Z_concat`, `Z_0 ⊕ Z_K` | 3 |
| adj_norm | `sym` = `D^{-1/2}(A+I)D^{-1/2}`, `row` = `D^{-1}A`, `col` = `AD^{-1}` | 3 |

**Fixed (not searched)**: hidden dim = F_in (residual invariant per Thesis). max_epochs = 10000 (ceiling only), early-stop controls.

**Justification for ranges**: each range covers the published HPs of BGRL / GraphMAE / GGD / SUGRL + safety tails so Optuna cannot underexplore by construction. Per-dataset scale-group adjustment reflects that arxiv's gradient magnitude differs from Cora's by ~2 orders (SUGRL-paper + INQ-006 confirm).

### Optuna configuration

```python
study = optuna.create_study(
    direction="maximize",                                      # val-acc
    sampler=optuna.samplers.TPESampler(seed=42),               # 1 study per dataset, 1 sampler seed
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=50,                                       # min epochs before prune eligible
        max_resource="auto",
        reduction_factor=3,
    ),
    storage=None,                                              # in-memory OK; CA may use SQLite if preferred for resume
    study_name=f"d6c_{dataset}",
    load_if_exists=True,
)
```

**1 Optuna study per dataset × 7 datasets = 7 studies total.** No study merging / no sampler-seed sensitivity check (drops 2× cost).

### Warm-start

Use `study.enqueue_trial()` to inject known-good configs before TPE takes over:
1. Current D6c default: τ=0.5, lr=1e-3, WD=5e-4, K={0,1,2,4,8}, readout=Z_concat (Cora/Photo/arxiv) or Z_mean (CS), adj_norm=sym (arxiv: row)
2. INQ-008 per-dataset best readout (use the table from INQ-008 response)

Guarantees HPO never underperforms current numbers.

### Early stopping + selection metric (UNIFIED — val-acc everywhere)

| Setting | Value | Rationale |
|---|---|---|
| max_epochs | 10000 | Ceiling only |
| early_stop_metric | **val-acc** (linear probe on val split) | Faithful to downstream task; avoids InfoNCE-loss vs downstream-acc divergence |
| eval_frequency | every 10 epochs | LogReg refit overhead negligible; gives patience a reasonable granularity |
| min_delta | 0.0 | Strict monotonicity |
| patience | 20 eval cycles (= 200 epochs w/o val-acc improvement) | BGRL convention; enough for cosine-style wiggles |
| HPO ranking metric | **val-acc** (same as early-stop) | Never test-acc during HPO |

**Never peek at test labels during HPO.**

### Fidelity protocol

| Phase | Model seeds | Probe restarts | n per trial | Configs per dataset | Total runs per dataset |
|---|---|---|---|---|---|
| Search | 3 | 3 | 9 | 400 trials | 3,600 |
| Top-5 confirmation | 5 | 5 | 25 | 5 configs | 125 |
| Final headline | 5 | 5 | 25 | 1 config | 25 |

**Probe restarts**: LogReg head refit with K different inits per model checkpoint → gives us additional variance-reduction for free (LogReg cost is negligible vs GNN forward pass). `probe_restarts=5` at final means the reported accuracy is median of 5 LogReg fits per model seed, averaged over 5 model seeds (n=25).

**Selection protocol**: ranking via mean val-acc across the 3 search seeds. Top-5 from Optuna Pareto front go to confirmation at n=25; top-1 from confirmation is the final headline.

### Deliverables for Config A

1. Per-dataset tuned HP table (7 rows × 6 HP columns: τ, lr, WD, K_set, readout, adj_norm). → Paper Appendix A.
2. Final accuracy per dataset at n=25: mean ± 95% CI via bootstrap. → Paper Table 1.
3. Optuna study DBs retained for reproducibility.
4. Top-5 confirmation table (ranking stability evidence).
5. **Comparison row**: tuned-D6c vs current-D6c (INQ-008 numbers). Quantifies HPO lift.

---

## Config B — Efficiency benchmark

### Port-selection criteria

For a baseline to be included in the matched-harness efficiency benchmark it must satisfy **all** of:

- (a) **Recency** — 2022 or later.
- (b) **SOTA-tier accuracy** on one of our benchmark datasets, as reported by the paper itself (accuracy-protocol mismatch with our dominant protocol is not a disqualifier — we re-measure in the harness anyway).
- (c) **Explicit scalability claim** — either reports ogbn-arxiv or a comparable large graph, or asserts linear-in-N complexity.
- (d) **Maintained public code** — reference implementation must exist.
- (e) **Runs at arxiv scale on our hardware** — no OOM on a single A40-48GB.

### Candidate audit (under the criteria above)

| Method | (a) ≥2022 | (b) SOTA-tier | (c) Scales | (d) Public code | (e) No OOM | Verdict |
|---|:-:|:-:|:-:|:-:|:-:|---|
| DGI (2019) | ✗ | — | — | ✓ | ✓ | cite paper |
| MVGRL (2020) | ✗ | — | — | ✓ | — | cite paper |
| GRACE (2020) | ✗ | — | subsample only on arxiv | ✓ | subsample | cite paper |
| CCA-SSG (2021) | ✗ | ✓ (71.24) | ✓ | ✓ | ✓ | cite paper |
| SUGRL (2022) | ✓ | ✗ (68.83) | ✓ | ✓ | ✓ | **cite paper** (fails b) |
| BGRL (2022) | ✓ | ✓ (71.64) | ✓ | ✓ | ✓ | **PORT** |
| GGD (2022) | ✓ | ✓ (71.64) | ✓ (papers100M) | ✓ | ✓ | **PORT** |
| GraphMAE2 (2023) | ✓ | ✓ (71.95) | ✓ (papers100M) | ✓ | ✓ | **PORT** |
| GraphACL (2023) | ✓ | ✓ (71.72) | ✓ | ✓ | ✓ | **PORT** |
| PolyGCL (2024) | ✓ | claim | ✗ (OOM arxiv) | ✓ | ✗ | cite paper |
| DGD (2024) | ✓ | ✓ on small graphs | ✓ (claims) | needs CA confirm | unknown | cite paper (pending CA confirm on (d)) |
| MHVGCL (2025) | ✓ | ✓ few-shot | ? | ✗ (none located) | — | cite paper |
| BLNN (2024) | ✓ | ✓ on small graphs | ✗ (5 small graphs only) | needs CA confirm | unknown | cite paper |

### Matched-harness set

**Ports**: D6c (native), BGRL, GGD, GraphMAE2, GraphACL — **5 methods total**.


### Metrics

Per (method, dataset) cell:

| Metric | Definition |
|---|---|
| Precompute time | One-shot cost (D6c's `Â^k X`; applies to any method with a precompute stage). Median of 3 runs. 0 if not applicable. |
| Training wall-clock / epoch | Seconds per epoch in steady state (mean of last 100 epochs). |
| Epochs to convergence | First epoch where early-stop triggered. Per seed; report mean. |
| Total training wall-clock | precompute + (epochs × wall-clock/epoch). End-to-end cost. |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` in MB. Peak across training. |
| **Training FLOPs per epoch** | Hardware-independent cost. Measured via `fvcore.FlopCountAnalysis` or equivalent. |
| **Total training FLOPs** | FLOPs/epoch × epochs to convergence + precompute FLOPs. |
| Final test accuracy | From Config A (D6c) or matched-harness re-run (ported baselines) or paper (cited-only methods, with hardware-caveat footnote). |

### Pareto figure

Two figures per dataset family (or composite): accuracy × wall-clock AND accuracy × FLOPs (both on log-x).

- **Matched-harness (solid markers, fully comparable)**: D6c, BGRL, GGD, GraphMAE2, GraphACL (+ SUGRL if retained).
- **Paper-cited (faded markers, hardware-caveat footnote)**: all other baselines from the main table.

Each matched-harness point annotated with hardware (Vector A40-48GB).

### Deliverables for Config B

1. Efficiency table (Method × Dataset × 7 metrics). → Paper Table 2 or Appendix B.
2. Pareto figure — wall-clock variant and FLOPs variant. → Paper Figure 2.
3. Precompute-cost breakdown.
4. Port validation report: each ported baseline must match its paper accuracy to within 1 pt on at least 2 datasets where paper reports — sanity check.

---

## Config C — Ablation studies

All ablations run at **Config-A-tuned D6c HPs (per dataset)** unless otherwise noted. Report mean ± 95% CI at n=25.

### C.1 Architecture ablation

| Variant | Description | Hypothesis |
|---|---|---|
| A0 | D6c full (baseline) | — |
| A1 | **No residual**: `Z_k = W_k X_k` | Large drop — D6b lost −5.14 / −7.21 on Cora/Computers (INQ-006) |
| A2 | **Shared W**: single W across all k (degenerates to D1) | Large drop — D1 hard-failed (INQ-006) |
| A3 | **No W_k**: `Z_k = X_k` (raw Â^k X baseline) | Quantifies W_k contribution vs. pure propagation |
| A4 | **Nonlinear W_k**: 2-layer MLP per depth | Tests whether nonlinearity helps (encoder-free claim) |
| A5 | **Single-k, best depth**: K=1, k = argmax raw-acc | Quantifies "multi-depth is necessary" |

Run on **all 7 datasets**.

### C.2 Loss ablation

| Variant | Description |
|---|---|
| B0 | InfoNCE same-node-across-depths (D6c default) |
| B1 | **BCE group-discrimination** (GGD-style) over Z_k | Does InfoNCE outperform BCE? |
| B2 | **Per-depth independent InfoNCE** (no cross-depth positives) | Is "depth as view" the load-bearing axis? |
| B3 | **Single-view contrastive** (Z=concat, standard feature-masking augs) | MHVGCL-style test — do deterministic depth views beat random augs? |

**B2 is the most critical**: defends against reviewer attack 1 (MHVGCL) by isolating the cross-depth contrast axis. If B2 degrades materially, the cross-depth axis is confirmed load-bearing.

Run on **Cora + Computers + arxiv** (3 representative datasets). Full 7 for the winning variant only (if non-B0 wins anywhere).

### C.3 Readout ablation

| Variant | Description |
|---|---|
| C0 | `Z_mean` |
| C1 | `Z_concat` |
| C2 | `Z_0 ⊕ Z_K` (first+last, MHVGCL-2V style) |
| C3 | `Z_0` only (no depth aggregation) |
| C4 | `Z_K` only (deepest only) |
| C5 | Learnable weighted sum `Σ α_k Z_k` with simplex-constrained α |

Note: Config A searches only over {Z_mean, Z_concat, Z_0⊕Z_K}. C.3 broadens the space to include C3/C4/C5 as post-HPO diagnostic.

Run on **all 7 datasets**.

### C.4 Depth-set (K) ablation

Fixed sweep (independent of Config A's categorical K_set choice):

| Variant | K_set |
|---|---|
| D0 | {0} |
| D1 | {0, 2} |
| D2 | {0, 1, 2, 4} |
| D3 | {0, 1, 2, 4, 8} |
| D4 | {0, 1, 2, 4, 8, 16} |
| D5 | {0, 1, 2, 4, 8, 16, 32} (arxiv only) |

Run on **all 7 datasets** (D5 on arxiv only).

### (Adjacency ablation — folded into Config A)

Previously drafted as C.5. Since adj_norm is now a first-class HP in Config A's search space, the per-dataset winner emerges from HPO directly. **No separate C.5 run.** 

### Deliverables for Config C

1. C.1 architecture table (6 variants × 7 datasets). → Paper Table 3 (headline).
2. C.2 loss table (4 variants × 3 datasets). → Appendix C.
3. C.3 readout table (6 variants × 7 datasets). → Appendix D.
4. C.4 K-sweep figure + table. → Paper Figure 3 or Appendix E.
5. Adjacency note: single paragraph + table derived from Config A.

---

## Config D — Sensitivity analysis

One-at-a-time sweeps around Config A's tuned config per dataset. Holds all other HPs at tuned values; sweeps one HP.

| HP | Sweep points |
|---|---|
| τ | tuned × {0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0} — 7 points, log-spaced |
| lr | tuned × {0.1, 0.3, 1.0, 3.0, 10.0} — 5 points |
| weight_decay | tuned × {0.01, 0.1, 1.0, 10.0, 100.0} — 5 points |
| K_set | all 7 Config-A categorical choices |
| readout | all 3 Config-A categorical choices |

Run on **Cora + Computers + arxiv** (3 representative datasets). n=5 × 5 per point.

### Deliverables for Config D

1. 5 sensitivity curves per dataset × 3 datasets = 15 plots. → Paper Figure 4 (composite).
2. "Plateau region" summary: for each HP, the range where accuracy stays within 1 pt of peak. Supports paper claim: *"D6c is robust across a wide HP range, not dependent on fine tuning."*

---

## Config E — Mechanism diagnostics

Evidence for the simplex-collapse-as-contrastive-signal framing ([[Thesis]] § Mechanism + Ji et al. 2025).

All diagnostics re-use Config A final checkpoints (no extra training).

### E.1 Per-depth lift

Per dataset, side-by-side bar charts:
- Raw `Â^k X` test-acc at each k ∈ K_set (the floor)
- Post-D6c `Z_k` test-acc at each k ∈ K_set (the lift)

Shows how much each W_k contributes.

### E.2 cos(W_k, W_k') heatmap

Per dataset: pairwise cosine similarity of `vec(W_k)` across k ∈ K_set. Values in [0, 1].

Flags where per-depth parameterization is redundant (Photo INQ-008 showed 0.73–0.97).

### E.3 ||W_k||_F / Xavier-init ratio

Per dataset: `||W_k||_F / sqrt(F_in)` at end of training. INQ-008 observed 0.07–0.37. Relevant for "W_k stays small under WD" framing.

### E.4 Alignment + uniformity (Wang & Isola 2020)

Standard SSL diagnostics:
- `L_align(τ=2) = E[||f(x) - f(y)||^2_2]` over positive pairs
- `L_unif(τ=2) = log E[exp(-2 ||f(x) - f(y)||^2_2)]` over random pairs

Pre- and post-training.

### E.5 Simplex-collapse metric (Ji et al. 2025)

Per dataset × per depth k:
- Per-class centroid on raw Â^k X and on post-D6c Z_k
- Mean intra-class distance and inter-centroid separation

**Prediction**: raw features show simplex collapse as k grows (intra shrinks, inter shrinks). Post-D6c Z_k should separate classes more than raw at every k.

### E.6 Per-node depth preference (diagnostic)

For each node, `argmax_k Z_k · w_c` (w_c = classifier weight for node's class). Distribution of argmax_k across nodes.

Run on **Cora + arxiv only**.

### Deliverables for Config E

1. E.1 × 7 datasets — 7 bar charts or 1 combined figure.
2. E.2 × 7 datasets — 7 heatmaps (appendix).
3. E.3 × 7 datasets — 1 combined bar chart.
4. E.4 alignment/uniformity table.
5. E.5 simplex-collapse table + scatter plot (raw vs D6c intra-class distance).
6. E.6 depth-preference histogram (Cora + arxiv).

---

## (Config F — Statistical significance — DEFERRED)

Bootstrap CIs already reported with every mean (Configs A–E). Paired t-tests and matched-seed-delta analysis deferred to a post-submission appendix if reviewers request. Not blocking this inquiry.

---

## Dependencies + execution order

RA specifies experiment dependencies and ordering only. CA owns all implementation and scheduling choices: parallelization, worker counts, storage backend, wall-clock estimates.

**Phase 1 — Config A (Optuna HPO for D6c)**
- One study per dataset (7 studies). Studies are independent of each other and of Phase 2.
- Per-dataset output: tuned HP config + final n=25 accuracy.

**Phase 2 — Baseline ports** (no upstream dependencies)
- Port BGRL, GGD, GraphMAE2, GraphACL into `ad_ssl` with the shared harness.
- Each port must pass Config B § port-validation (paper accuracy within 5 pt on ≥ 2 datasets it reports) before its row is used in Config B or the main table.

**Phase 3 — Configs B, C, D, E** (upstream deps listed per config)
- Config B (efficiency) — needs Phase 2 ports + Config A tuned HPs for D6c.
- Config C.1 (architecture) — needs Config A tuned HPs.
- Config C.2 (loss) — needs Config A tuned HPs.
- Config C.3 (readout) — needs Config A tuned HPs.
- Config C.4 (K-sweep) — needs Config A tuned HPs.
- Config D (sensitivity) — needs Config A tuned HPs.
- Config E (mechanism) — needs Config A final checkpoints (no extra training).


**Side notes for CA**:
- If any Phase 2 port fails validation, flag back via inquiry before starting Config B on that baseline.

## Summary of deliverables (paper-facing)

| Paper section | Source config | Deliverable |
|---|---|---|
| Main table (Table 1) | Config A | 7-dataset tuned accuracy, D6c vs baselines |
| HP table (Appendix A) | Config A | 7-row × 6-HP tuned configs |
| Pareto figure (Fig. 2) | Config B | accuracy vs wall-clock AND accuracy vs FLOPs |
| Efficiency table (Table 2) | Config B | wall-clock + memory + FLOPs + precompute |
| Architecture ablation (Table 3) | Config C.1 | 6 variants × 7 datasets |
| Loss ablation (Appendix C) | Config C.2 | 4 variants × 3 datasets |
| Readout ablation (Appendix D) | Config C.3 | 6 variants × 7 datasets |
| K-sweep (Fig. 3 / App. E) | Config C.4 | D0–D5 per dataset |
| Adjacency (App. F) | Config A | adj_norm winner per dataset |
| Sensitivity (Fig. 4) | Config D | 5-HP curves on 3 datasets |
| Mechanism (§3, Fig. 5) | Config E | per-depth lift, cos, alignment, simplex |

## Expected response format

Per config, CA should:

1. Acknowledge receipt; flag any infeasibility.
2. Propose simplifications or re-orderings.
3. Report under `# RESULTS — Phase X section.
4. Flag mid-execution findings that change downstream config specs.

---

# RESPONSE

<!-- CA will fill this in -->
