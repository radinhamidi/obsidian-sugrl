---
title: Competitive Landscape 2026
type: synthesis
tags: [neurips-2026, related-work, baselines]
created: 2026-04-21
updated: 2026-04-24
sources: [[RESEARCH_AGENT_ONBOARDING]]
---

# Competitive Landscape (as of 2026-04-24)

Snapshot of where unsupervised graph representation learning stands on the axes D6c competes on: **accuracy**, **scalability / wall-clock**, **augmentation-free-ness**, **multi-depth contrastive views**.

## Comparison posture (locked 2026-04-24 via [[INQ-2026-04-24-002]])

**Posture A**: D6c is HPO-tuned via Optuna per dataset (Config A); baselines are cited as-reported (their authors already tuned). Posture B (port-and-re-tune all baselines) was rejected as unnecessary cost. Implication for the main table: blank cells where a paper used a non-dominant split or did not report; no extrapolation. See [[INQ-2026-04-24-002]] § Baseline accuracy audit.

## Efficiency champions

| Method | Venue | ogbn-arxiv time | Key idea |
|---|---|---|---|
| [[GGD]] | NeurIPS 2022 | ~0.18 s | Binary group discrimination (real vs shuffled), no contrastive loss |
| [[SUGRL]] | AAAI 2022 | ~6 s | MLP + shared-GCN, margin triplet, fixed k=1 propagation |

GGD is ~30× faster than SUGRL and scales to ogbn-papers100M.

## Accuracy leaders (ogbn-arxiv)

All numbers verified against source PDFs (2026-04-21 audit).

| Method | Venue | Accuracy | Key idea |
|---|---|---|---|
| [[BGRL]] | ICLR 2022 | 71.64 ± 0.12 | BYOL-style bootstrap, no negatives |
| [[GraphMAE]] | KDD 2022 | 71.87 ± 0.21 (GCN) / 71.75 ± 0.17 (GAT) | Masked feature reconstruction |
| [[GraphMAE2]] | WWW 2023 | 71.95 ± 0.08 | Improved masked reconstruction (Table 9, full-graph) |
| [[GraphACL]] | NeurIPS 2023 | 71.72 ± 0.26 | Asymmetric contrastive, augmentation-free, heterophily-aware |
| [[PolyGCL]] | ICLR 2024 | not reported on ogbn-arxiv | Learnable polynomial spectral filters; reports arXiv-year 43.07 (heterophilic label variant) |
| [[BLNN]] | arXiv 2024-08 | not reported on ogbn-arxiv | BGRL + neighbor-positive alignment; 5 small graphs only |
| [[DGD]] | Neurocomputing 2024 | not reported on ogbn-arxiv | Decoupled-GCN + BCE group discrimination |
| [[MHVGCL]] | ASOC 2025 | not reported on ogbn-arxiv | Multi-hop APPNP-style nonlinear views + InfoNCE; few-shot regime |

## Main-table baseline set (locked 2026-04-24 via [[INQ-2026-04-24-002]])

13 methods, partitioned into **classical** (pre-2022 references every graph-SSL paper reports) and **modern** (2022+ SOTA tier).

**Classical tier** — DGI (ICLR 2019), MVGRL (ICML 2020), GRACE (ICML 2020), CCA-SSG (NeurIPS 2021).

**Modern tier** — [[SUGRL]] (AAAI 2022), [[BGRL]] (ICLR 2022), [[GGD]] (NeurIPS 2022), [[GraphMAE2]] (WWW 2023, supersedes [[GraphMAE]] which moves to §2 predecessor citation), [[GraphACL]] (NeurIPS 2023), [[PolyGCL]] (ICLR 2024), [[DGD]] (Neurocomputing 2024), [[MHVGCL]] (ASOC 2025), [[BLNN]] (arXiv 2024).

Cells are blank where the paper used a non-dominant split or did not report. See [[INQ-2026-04-24-002]] § Baseline accuracy audit for the full per-cell table.

## Port-selection criteria for Config B efficiency benchmark (locked 2026-04-24)

For a baseline to be **ported into the matched harness** (vs. cited paper-as-reported on accuracy axis only) it must satisfy **all** of:

- (a) **Recency** — 2022 or later.
- (b) **SOTA-tier accuracy** on at least one of our benchmark datasets, as reported by the paper itself. Accuracy-protocol mismatch with our dominant protocol is **not** a disqualifier — the harness re-measures.
- (c) **Explicit scalability claim** — reports ogbn-arxiv or comparable large graph, or asserts linear-in-N complexity.
- (d) **Maintained public code** — reference implementation must exist.
- (e) **Runs at arxiv scale on our hardware** — no OOM on a single A40-48GB.

Strict intersection yields **4 ports**: BGRL, GGD, GraphMAE2, GraphACL. Plus D6c (native). [[SUGRL]] fails (b) at 68.83 < 71 SOTA tier; retained from existing INQ-008 harness work as no-cost addition. PolyGCL fails (e) (OOM on arxiv). DGD/MHVGCL/BLNN cited paper-as-reported (no public code located, or scope mismatch with arxiv coverage). See [[INQ-2026-04-24-002]] § Config B for full audit table.

**Important**: protocol mismatch with our dominant split is NOT a port-exclusion criterion — the matched harness re-measures wall-clock and memory regardless. Earlier framing that treated "uses a different split" as exclusion was incorrect.

## Concurrent work to monitor

- **[[Less is More]]** (arxiv 2509.25742, ICLR 2026 submission). GCN + MLP as complementary views, no augmentation, no negatives. Focus: heterophily + robustness at small scale. **Does not run at OGB scale.** Closest architectural cousin to AD-SSL; our differentiation is large-scale efficiency + multi-depth adaptive views (vs their single GCN + single MLP).
- **[[BLNN]]** (arXiv 2408.05087, Aug 2024, Liu et al.). BGRL + 1-hop neighbor-positive alignment with attention supportiveness. Small-graph evaluation only (WikiCS, Photo, Computer, CS, Physics). Orthogonal axis to AD-SSL — BLNN enriches positives in the *spatial* direction (1-hop neighbors), AD-SSL enriches in the *depth* direction (multi-hop precomputed views). Cite in related work; not a preempt.
- **[[GRAPHITE]]** (ICLR 2026, arXiv 2602.07256). Supervised heterophily preprocessor via feature nodes. **Out of AD-SSL scope** per [[Thesis]] § Scope; noted for citation completeness only.
- **[[GSTBench]]** (CIKM 2025, arXiv 2509.06975). Cross-dataset transferability benchmark — finds only [[GraphMAE2]]-style reconstruction transfers at papers100M scale; contrastive/bootstrap methods fail. We operate outside this regime (per-dataset training).

## Supervised adaptive-depth work (conceptual priors)

These are adaptive-depth methods but all supervised — none has been adapted to unsupervised GRL:

- [[GPRGNN]] (ICLR 2021) — learned polynomial propagation coefficients.
- [[APPNP]] (ICLR 2019) — teleport-based propagation.
- [[ATP]] (2024) — node-wise adaptive propagation.

## The empty Pareto point

No published unsupervised method simultaneously:

1. Matches BGRL/GraphMAE accuracy (~71 on ogbn-arxiv).
2. Runs at GGD-level wall-clock (seconds on ogbn-arxiv).
3. Adapts propagation depth per node rather than fixing k as a hyperparameter.

See [[Pareto Gap]].

## Monitoring cadence

- **Weekly arxiv scan** for new submissions in graph SSL, scalable GNNs, unsupervised GRL.
- Flag on: multi-depth contrastive views, same Pareto claim, direct extensions of [[SUGRL]] / [[GGD]] / [[BGRL]].
- Track acceptance of [[Less is More]] at ICLR 2026 — accepted → must cite and differentiate explicitly; rejected → cite as arxiv with weaker framing.

See [[log]] for literature-scan entries.
