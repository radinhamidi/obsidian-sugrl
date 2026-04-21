---
title: RESEARCH_AGENT_ONBOARDING
type: source
source_kind: project-brief
source_path: raw/RESEARCH_AGENT_ONBOARDING.md
ingested: 2026-04-21
tags: [project-brief, onboarding]
---

# Source summary — Research Agent Onboarding

Project brief from Radin defining the Research Agent role, the AD-SSL method, and the NeurIPS 2026 plan. Self-contained.

## Key sections

1. **Project in 60 seconds** — AD-SSL thesis: match [[BGRL]] accuracy at [[GGD]] cost using multi-depth precomputed views. → [[Thesis]]
2. **People and agents** — Radin (PI) + Research Agent (me, literature/analysis/writing) + Coding Agent (implementation). Comms flow through Radin. → [[Project Phases and Decision Gates]]
3. **Starting point (SUGRL)** — [[SUGRL]] architecture; key weakness: fixed k=1 propagation depth.
4. **Preliminary experiments** — 168-run validation. → [[Preliminary Validation - 168 Runs]], [[Prepropx Depth Finding]]
5. **Literature survey** — 2026 competitive landscape. → [[Competitive Landscape 2026]]
6. **Method (AD-SSL)** — architecture, novelty, four ablation insights. → [[Thesis]], [[Ablation Plan - AD-SSL B0 A1-A4]]
7. **Reviewer expectations** — attacks + defenses. → [[Reviewer Attacks and Defenses]]
8. **Current status** — Phase 0 in progress. → [[Project Phases and Decision Gates]]
9. **Files** — repo structure for results.
10. **Style** — terse, no hype, significance criterion, direct.

## Notable constraints

- Research Agent does **not** write code or run experiments.
- Comms between agents go through Radin, not directly. **But** bidirectional inquiries via `raw/inquiries/` are sanctioned.
- Significance bar: Δ > 0.3 AND 3/3 seeds positive. Everything else is noise until proven otherwise.

## Derived wiki pages

- [[Thesis]]
- [[Competitive Landscape 2026]]
- [[Pareto Gap]]
- [[Reviewer Attacks and Defenses]]
- [[Project Phases and Decision Gates]]
- [[Preliminary Validation - 168 Runs]]
- [[Prepropx Depth Finding]]
- [[Ablation Plan - AD-SSL B0 A1-A4]]
- Entity stubs for all methods in the landscape table.
