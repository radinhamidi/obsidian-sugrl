---
title: Matched-Seed Delta
type: concept
tags: [concept, evaluation, significance]
created: 2026-04-21
updated: 2026-04-21
---

# Matched-Seed Delta

Our project's significance criterion. Used in [[Preliminary Validation - 168 Runs]] and every future ablation.

## Definition

For each seed `i`, compute `δᵢ = variant_accᵢ − baseline_accᵢ`. Then report `mean(δ)` and `std(δ)` across seeds.

This cancels seed noise that would otherwise dominate the difference. Standard practice when comparing two methods on the same seeds.

## ROBUST bar

A variant is called **ROBUST** iff:
- `mean(δ) > 0.3`
- AND `seeds_positive = all seeds` (**5/5** under the locked protocol; see [[Splits and Protocol]]).

The [[Preliminary Validation - 168 Runs]] results used 3 seeds — those numbers remain valid under the old protocol and should be read with 3/3 as the bar; every new ablation uses 5/5.

## Weaker categories

- **weak+** — positive mean, not all seeds positive.
- **noise** — mean within seed noise, mixed signs.
- **💀 DEAD** — catastrophic negative on any dataset.

## Operational rule

Do **not** report a delta as a finding unless it clears the ROBUST bar OR is flagged explicitly as weak. When writing analysis memos, always lead with the verdict (ROBUST / noise / DEAD) before numbers.
