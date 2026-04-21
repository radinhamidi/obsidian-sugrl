---
title: Project Phases and Decision Gates
type: synthesis
tags: [neurips-2026, project-mgmt]
created: 2026-04-21
updated: 2026-04-21
---

# Phases and Gates

## Phases

| Phase | Description | Owner | Status (2026-04-21) |
|---|---|---|---|
| 0 | Hygiene: early stopping, checkpoint cleanup, training infra | Coding Agent | In progress |
| 1 | Baseline ladder: reproduce [[GGD]], [[BGRL]], [[GraphMAE]], [[GraphACL]], [[PolyGCL]], [[Less is More]] | Coding Agent | Not started |
| 2 | AD-SSL ablation: B0 + A1–A4 individually and combined | Coding Agent | Not started |
| 3 | Paper writing: full draft to NeurIPS 2026 | Research Agent | Not started |

## Decision gates

**Gate 1 (end of Phase 2):**
- If B0 + best insight **≥ 71** on ogbn-arxiv at **< 60 s** wall-clock → proceed to full experiments + paper writing.
- If B0 + all insights **< 70** → MLP encoder is the bottleneck. Pivot discussion with Radin (options: swap encoder, drop paper, refocus contribution).

**Gate 2 (continuous — literature monitoring):**
- If a new paper appears claiming the same Pareto point → run overlap assessment; differentiate or pivot.
- Specific watch item: [[Less is More]] acceptance at ICLR 2026.

**Gate 3 (pre-submission):**
- [[Novelty Verification Checklist]] all green.
- All reviewer-defense evidence in place (see [[Reviewer Attacks and Defenses]]).
- Heterophilic benchmark, scale study, Pareto figure complete.

## Communication

- Coding Agent → Research Agent: commits results JSONs + markdown summaries under `results/` in the implementation repo; surfaces them to Radin and triggers my analysis.
- Research Agent → Coding Agent: proposes experiments, requests reproductions, or disambiguates design choices via `raw/inquiries/` (see `raw/inquiries/TEMPLATE.md`).
- Neither agent sends instructions directly to the other — Radin is the decision-maker.
