---
title: VALIDATION_ORIGINAL_CODE
type: source
source_kind: internal-results
source_path: raw/VALIDATION_ORIGINAL_CODE.md
ingested: 2026-04-21
tags: [results, preliminary, sugrl]
---

# Source summary — Validation on Original SUGRL Codebase

Radin's 2026-04-10 report on 168 runs testing 6 brainstorm ideas + sampling control + depth variants on unmodified [[SUGRL]].

## Headline findings

1. **All 6 brainstorm ideas fail on ogbn-arxiv** (within ±0.13 of baseline).
2. **Feature-similarity positives** robust on [[Computers]] only (+0.73).
3. **Hard negatives + curriculum negatives** catastrophic everywhere.
4. **Sampling control** (`baseline_iid`) shows SUGRL's `np.random.permutation` is itself suboptimal on [[PubMed]] (+1.37).
5. **Pre-propagation depth k=1 → k=3** gives +0.80 on ogbn-arxiv for free. → [[Prepropx Depth Finding]]

## Baseline reproduction sanity

All 6 datasets reproduce SUGRL's paper numbers within 1–2σ. The validation environment is trustworthy.

## Evaluation protocol

Linear probe on L2-normalized `h_p` embeddings, 2-layer LogReg, 2 restarts × 3 seeds. Significance: [[Matched-Seed Delta]] — Δ > 0.3 AND 3/3 seeds positive = ROBUST.

## Why this document motivates AD-SSL

See [[Preliminary Validation - 168 Runs]] and [[Prepropx Depth Finding]]. Short form: sampling tweaks don't move ogbn-arxiv; depth does. AD-SSL exploits depth.

## Raw data preserved

Per-seed tables preserved verbatim in the source file under `raw/`. Do not re-derive or paraphrase the raw numbers — read the source when a specific cell is needed.
