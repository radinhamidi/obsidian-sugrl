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

## [2026-04-21] note | First ingest pass complete

Wiki now has ~30 pages covering the thesis, competitive landscape,
preliminary results, and reviewer-defense map. Ready for Coding Agent to
start Phase 1 (baseline ladder) and for literature monitoring to begin.

Open follow-ups:
- Ingest the actual paper PDFs when they land in `raw/papers/` (currently
  only project docs are in `raw/`). Entity pages are stubs pending that.
- First literature scan (April 2026 arxiv).
- Draft a [[Novelty Verification Checklist]] page (referenced but not yet
  written).
