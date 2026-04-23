---
title: Index
type: index
updated: 2026-04-21
---

# SUGRL Wiki Index

Catalog of every page in this wiki. Read first when answering a query; drill into specific pages from here.

## Synthesis

- [[Thesis]] — AD-SSL one-sentence claim, mechanism, novelty, outcomes, risks.
- [[Competitive Landscape 2026]] — efficiency champions, accuracy leaders, concurrent work, supervised adaptive-depth priors.
- [[Pareto Gap]] — framing of the empty accuracy-×-cost region AD-SSL targets.
- [[Reviewer Attacks and Defenses]] — anticipated reviewer objections and required evidence.
- [[AD-SSL vs Less is More]] — dedicated differentiation page against closest concurrent work.
- [[Novelty Verification Checklist]] — per-claim ablations required to defend AD-SSL vs prior art; tagged 🔴/🟡/🟢.
- [[Project Phases and Decision Gates]] — phase map, Gate 1/2/3, agent communication rules.
- [[Splits and Protocol]] — locked train/val/test splits + early-stopping policy (Option B; 5 trials on every dataset).
- [[AD-SSL v2 - Encoder-Free Design Sketch]] — draft reframe: per-node α over raw Â^k X, no encoder; triggered by three-dataset encoder failure.
- [[Idea Ledger]] — hypotheses tested / queued / surprising observations; mandated by [[Research Agent Operating Protocol]].

## Experiments

- [[Preliminary Validation - 168 Runs]] — 2026-04-10 results: 6 brainstorm ideas + sampling control + depth variants on unmodified SUGRL.
- [[Prepropx Depth Finding]] — the +0.80 on ogbn-arxiv from k=1→k=3 that motivates the paper.
- [[Ablation Plan - AD-SSL B0 A1-A4]] — planned Phase 2 ablation (owned by Coding Agent).

## Entities — our method

- [[AD-SSL]] — our proposed method.

## Entities — baselines and prior methods

- [[SUGRL]] — starting-point method (AAAI 2022).
- [[GGD]] — efficiency champion (NeurIPS 2022).
- [[BGRL]] — accuracy ceiling, bootstrap-style (ICLR 2022).
- [[GraphMAE]] — accuracy ceiling, generative (KDD 2022).
- [[GraphMAE2]] — current accuracy leader (WWW 2023).
- [[GraphACL]] — augmentation-free, heterophily (NeurIPS 2023).
- [[PolyGCL]] — spectral polynomial views (ICLR 2024).
- [[Less is More]] — **closest concurrent work** (arxiv / ICLR 2026 submission).
- [[GPRGNN]] — learned global polynomial coefficients, supervised (ICLR 2021).
- [[APPNP]] — teleport-based propagation, supervised (ICLR 2019).
- [[ATP]] — per-node adaptive propagation, supervised (2024).
- [[SGC]] — origin of the decoupled precompute trick (ICML 2019).
- [[BLNN]] — BGRL + neighbor-positive alignment with attention supportiveness (arXiv 2024); small-graph evaluation only.
- [[GRAPHITE]] — graph preprocessor for heterophily via feature nodes (ICLR 2026); out of AD-SSL scope but flagged for completeness.
- [[Rethinking graph neural networks from a geometric perspective of node features]] — Ji et al. ICLR 2025; feature-centroid-simplex theory cited by [[Less is More]] (same Tay/NTU group).

## Entities — datasets

- [[ogbn-arxiv]] — headline decision-gate dataset.

## Concepts

- [[Decoupled Precompute]] — SGC-style preprocessing.
- [[Multi-Depth Views]] — core primitive of AD-SSL.
- [[Adaptive Depth Weighting]] — per-node α_{i,k} mechanism (A1/GRPO-style).
- [[Bootstrap Loss]] — BYOL-style training across depth pairs.
- [[Matched-Seed Delta]] — our significance criterion.
- [[Oversmoothing]] — theoretical ceiling on depth.

## Sources

- [[RESEARCH_AGENT_ONBOARDING]] — project brief (2026-04-21).
- [[Research Agent Operating Protocol]] — methodology spec (2026-04-22): how to brainstorm, diagnose failures, handle negative results. Binding.
- [[VALIDATION_ORIGINAL_CODE]] — 168-run preliminary validation report (2026-04-10).
- [[GSTBench]] — CIKM 2025 cross-dataset transferability benchmark; only GraphMAE transfers at papers100M scale.
- [[Graph Learning Poor Benchmarks]] — ICLR 2025 position paper advocating Pareto framings, CIs, non-graph baselines.
