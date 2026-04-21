# SUGRL Wiki Log

Append-only chronological log of wiki activity. Each entry starts with
`## [YYYY-MM-DD] <event> | <title>` for easy parsing
(`grep "^## \[" log.md | tail -20`).

Event types: `ingest`, `query`, `lint`, `inquiry-open`, `inquiry-answer`,
`inquiry-close`, `experiment`, `note`.

---

## [2026-04-21] note | Vault initialized

Vault scaffolded. Directory structure, `CLAUDE.md` schema, inquiry template,
and this log created. Remote: `github.com/radinhamidi/obsidian-sugrl`. To be
mounted as a submodule inside the implementation repo where the Coding Agent
operates.

## [2026-04-21] ingest | RESEARCH_AGENT_ONBOARDING

Project brief ingested. Defines method **AD-SSL** (Adaptive-Depth Decoupled
SSL), NeurIPS 2026 target, researcher/agents roles, Pareto thesis (match
BGRL accuracy at GGD cost), and four-insight ablation plan (B0 + A1–A4).

Created: [[Thesis]], [[Competitive Landscape 2026]], [[Pareto Gap]],
[[Reviewer Attacks and Defenses]], [[Project Phases and Decision Gates]],
[[Ablation Plan - AD-SSL B0 A1-A4]], [[AD-SSL]], entity stubs for all 12
landscape methods + [[ogbn-arxiv]], concept pages for
[[Decoupled Precompute]], [[Multi-Depth Views]], [[Adaptive Depth Weighting]],
[[Bootstrap Loss]], [[Matched-Seed Delta]], [[Oversmoothing]].

## [2026-04-21] ingest | VALIDATION_ORIGINAL_CODE

168-run preliminary report ingested. All 6 brainstorm sampling-ideas fail on
ogbn-arxiv; only **prepropx (k=1→k=3) +0.80** is ROBUST and motivates the
entire paper. Feature-sim positives +0.73 on Computers only; curriculum/hard
negatives catastrophic. Baselines reproduce paper within 1–2σ.

Created: [[Preliminary Validation - 168 Runs]], [[Prepropx Depth Finding]].

## [2026-04-21] ingest | SUGRL (paper PDF)

Deepened [[SUGRL]] entity page with exact multiplet loss (ω1·L_S + ω2·L_N + L_U), shared MLP/GCN weights confirmation, and the **SUGRL-batch 69.3** number from Table 2 (overlooked in onboarding — flagged in page). No thesis-level changes.

## [2026-04-21] ingest | Less is More (paper PDF)

Major update. Paper is v3 (2026-03-20) with full author list (Zhao, Ji, Dai, Ma, Tay @ NTU). Mechanism is 2-view: 1 MLP + 1 k-layer GCN, global β, direct cosine loss. **They evaluate at 169k-node scale (Arxiv-year)** — softens our onboarding's "not at OGB scale" framing. Differentiation still solid: K vs 2 views, per-node α vs global β, bootstrap vs direct cosine. Created [[AD-SSL vs Less is More]] synthesis page; updated [[Reviewer Attacks and Defenses]] with honest framing and new evidence gaps (reproduce at standard ogbn-arxiv, 2-view ablation of AD-SSL).

## [2026-04-21] ingest | ATP (paper PDF)

Deepened [[ATP]] entity page. Mechanism is 2-part: High-Deg edge masking (HPC) + weight-free scalar r̃ per node via degree+eigenvector+clustering in the kernel `D̂^(r̃−1) Â D̂^(−r̃)`. **Supervised, single kernel, scalar-per-node.** Our K-view per-node-per-depth α mixture in SSL regime is distinct. Added dedicated reviewer-defense row for "this is just ATP in SSL." ATP's HPC flagged as possible preprocessing adoption (open question).

## [2026-04-21] ingest | GGD (paper PDF)

Deepened [[GGD]] entity page with exact numbers. **Critical finding**: GGD at hidden=1500 reaches **71.6 ± 0.5 in 0.95s** on ogbn-arxiv (Table 8) — our onboarding's framing of "BGRL accuracy at GGD cost" as an empty Pareto point is too loose. Updated [[Pareto Gap]] frontier sketch, [[ogbn-arxiv]] competitive table, and added a reviewer-defense row for "GGD-1500 already closed this gap." Also noted GGD's inference-time `H_final = H_θ + A^5 H_θ` trick — a weak, post-hoc single-depth version of our multi-depth views; strengthens our motivation but also raises "is multi-depth just GGD's power trick scaled up?" as a defense-needed question.

ogbn-papers100M numbers captured: GGD 63.5 test in 9h15m vs BGRL 62.1 in 26h28m — GGD is the right cost anchor at OGB-LSC scale.

## [2026-04-21] note | Tier 1 ingest pass (4/7 complete)

Wiki now has ~30 pages covering the thesis, competitive landscape,
preliminary results, and reviewer-defense map. Ready for Coding Agent to
start Phase 1 (baseline ladder) and for literature monitoring to begin.

Open follow-ups:
- Ingest the actual paper PDFs when they land in `raw/papers/` (currently
  only project docs are in `raw/`). Entity pages are stubs pending that.
- First literature scan (April 2026 arxiv).
- Draft a [[Novelty Verification Checklist]] page (referenced but not yet
  written).
