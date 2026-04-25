---
title: Splits and Protocol
type: synthesis
tags: [neurips-2026, protocol, splits, early-stopping]
created: 2026-04-21
updated: 2026-04-24
sources: [[INQ-2026-04-21-001], [[INQ-2026-04-24-002]]]
---

# Splits and Evaluation Protocol (AD-SSL)

Single source of truth for train/val/test splits and downstream-evaluation protocol for the NeurIPS 2026 paper. Locked 2026-04-21 via [[INQ-2026-04-21-001]].

Rationale in one line: **AD-SSL competes on the modern SSL Pareto frontier ([[BGRL]], [[GraphMAE2]], [[GraphACL]], [[GGD]]); all use Option B-style conventions, so we use Option B.** [[SUGRL]]-paper-comparable numbers come from re-running SUGRL under Option B as a baseline, not from running AD-SSL under SUGRL's legacy per-class-30 protocol.

## Split policy

| Dataset family | Train | Val | Test | Seed policy |
|---|---|---|---|---|
| Planetoid (Cora, CiteSeer, PubMed) | Public `train_mask` (20/class) | Public `val_mask` (500) | Public `test_mask` (1000) | Split fixed; 5 trials vary init only. |
| Amazon (Photo, Computers) | 10 % random | 10 % random | 80 % random | **Split is seed-determined** — `args.seed` chooses both split and init. Run **5 trials**, report mean ± 95 % CI. |
| OGB (ogbn-arxiv, ogbn-products) | `dataset.get_idx_split()['train']` | `dataset.get_idx_split()['valid']` | `dataset.get_idx_split()['test']` | Default OGB masks; split fixed; 5 trials vary init only (leaderboard-comparable). Confirmed canonical 2026-04-24 via [[INQ-2026-04-24-002]]. |
| ogbn-mag | **Out of scope for main table.** | — | — | If kept for smoke tests, project to paper-paper subgraph and use `split_idx['paper']['{train,valid,test}']`. Not reported. |

Heterophilic graphs (Chameleon, Squirrel, Actor, Wisconsin, Texas): out of scope v1 per [[Thesis]] § Scope.

## Downstream-evaluation protocol

Unsupervised contrastive pretraining (AD-SSL) → frozen embeddings → **fresh LogReg head** retrained every `--eval_every` epochs on the train split, scored on the val split. Best-val-acc weights kept as the final checkpoint; test-acc reported on that checkpoint.

### Early stopping

| Knob | Default | Reason |
|---|---|---|
| `--eval_every` | 10 | Cheap; lets us track early-training dynamics. |
| `--patience` | 20 (= 200 epochs without val improvement) | Bootstrap/cosine SSL losses wiggle; [[BGRL]]'s official repo uses ≈ 20. Patience = 5 is too tight — stopping early costs 0.3–0.5 pts, which is the exact margin of our Pareto claim. |
| `--min_delta` | 0.0 | Strict monotonicity. |
| `epochs` | per-dataset value in `args.yaml` treated as **upper bound** | Matches BGRL/GraphMAE practice. Don't re-tune yet; one smoke run tells us whether current budgets are comfortable. |

### Label-leakage caveat (appendix sanity check)

Checkpoint selection via val-labels is a weak form of label leakage into SSL pretraining. [[BGRL]] and most modern SSL papers do it anyway, so we're in line with convention. Mitigation:

- Report **both** `final-epoch test-acc` and `best-val-checkpoint test-acc` on ogbn-arxiv in the appendix.
- If the gap > 0.3 pts, drop early stopping and move to fixed-budget pretraining (BGRL/GraphMAE style). Open a follow-up inquiry in that case.
- Never peek at test labels during pretraining or hyperparameter selection.

## Baseline gate — "training adds value over not training"

**Updated 2026-04-22 via [[INQ-2026-04-22-001]].** Replaces the previous absolute "≥82 on Cora" gate, which compared B0 to SUGRL's fully-trained accuracy — the wrong bar for a minimal baseline.

For every benchmark dataset, report the **parameter-free Â¹X linear-probe accuracy** (propagate raw features once with normalized adjacency, train LogReg on the train split, score on test). Call this the *Â¹X baseline*.

**B0 must clear its Â¹X baseline per dataset.** This is the minimum bar for "our trained encoder is not destructive." Absolute accuracy gates (SUGRL-matching) apply to the **final method** (B0 + best A-combination), not B0 alone.

Known values to date (from CA runs under [[INQ-2026-04-22-001]]):

| Dataset | Â¹X baseline | B0 (InfoNCE) | Status |
|---|---:|---:|---|
| Cora | 77.07 | 72.01 ± 1.66 | **below baseline** — diagnostics pending (per-depth inference, collapse stats, τ sweep) |
| Computers | (run and record) | 83.69 ± 0.16 | likely passes once Â¹X is recorded; final gate depends on baseline value |

If B0 under InfoNCE cannot clear the Â¹X baseline on a dataset after the diagnostic pass, that dataset becomes a scope-question (report-for-convention vs move to appendix vs exclude). See [[Thesis]] § Scope.

## Seed counts and reporting

All datasets: **5 trials**, report mean ± 95 % CI (bootstrap).

| Dataset | Trials | Seed varies |
|---|---|---|
| Planetoid (Cora, CiteSeer, PubMed) | 5 | init only (public split is fixed) |
| Photo, Computers | 5 | split + init (seed-determined 10/10/80) |
| ogbn-arxiv, ogbn-products | 5 | init only (official split is fixed) |
| ogbn-papers100M (if run) | 5 | init only |

95 % CI via bootstrap over seeds, per [[Graph Learning Poor Benchmarks]] calibration. Paired t-test when comparing AD-SSL configs with matched seeds — see [[Matched-Seed Delta]].

## HPO selection metric (locked 2026-04-24 via [[INQ-2026-04-24-002]])

For Config A (Optuna HPO on D6c), both **early-stop metric** and **HPO ranking metric** are unified to **val-acc** (linear probe on the val split). Never InfoNCE loss, never test-acc. Eval frequency every 10 epochs; min_delta = 0; patience 20 eval cycles (= 200 epochs without val-acc improvement); max_epochs = 10000 ceiling. **Never peek at test labels during HPO.**

Fidelity protocol: 3 seeds × 3 probe restarts (n=9) during search; 5 × 5 (n=25) for top-5 confirmation and final headline. N=5 default; bump to N=10 only if final-eval stderr > 0.5 pts on any dataset.

## Port-validation rule for matched-harness baselines (locked 2026-04-24 via [[INQ-2026-04-24-002]])

Each baseline ported into the matched harness (BGRL, GGD, GraphMAE2, GraphACL) must reproduce its paper-reported accuracy to **within 5 pts on at least 2 datasets** before its row counts in Config B or the main accuracy table. Recommended pairings (one large + one small per baseline):

- BGRL — arxiv (71.64 ± 0.12) + Computers (90.34 ± 0.19)
- GGD — arxiv (71.64 ± 0.50) + Cora
- GraphMAE2 — arxiv (71.95 ± 0.08) + Cora (84.50)
- GraphACL — arxiv (71.72 ± 0.26) + Photo (93.31)

Each port must hit at least one large-graph (arxiv) target so validation happens at scale, not only on small graphs.

## Downstream tasks not covered here

`link` (link prediction) and `clu` (clustering) keep their current single-eval-at-end behavior — out of scope for [[INQ-2026-04-21-001]]. Revisit when those tasks enter the main table.

## Related

- [[INQ-2026-04-21-001]] — the inquiry that locked splits + early stopping.
- [[INQ-2026-04-22-001]] — B0 loss swap + Â¹X baseline-gate policy.
- [[INQ-2026-04-24-002]] — master experimental plan: HPO selection metric, port-validation rule, OGB split confirmed.
- [[Ablation Plan - AD-SSL B0 A1-A4]] — consumes this protocol.
- [[Matched-Seed Delta]] — significance criterion.
- [[Graph Learning Poor Benchmarks]] — CI + paired-test calibration.
