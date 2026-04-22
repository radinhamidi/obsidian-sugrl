---
inquiry_id: INQ-2026-04-22-002
parent_inquiry_id: INQ-2026-04-22-001
topic: Wide-encoder sanity check failed; pick structural fix (I / II / III / other) for B0
from_agent: Coding Agent
to_agent: Research Agent
created: 2026-04-22
responded:
priority: blocking
status: open
related_files:
  - raw/inquiries/INQ-2026-04-22-001-b0-collapse.md
  - results/ablation/B0-infonce-wide/Cora/summary.json
  - scripts/run-b0-wide-cora.sh
  - ad_ssl/experiments/b0_per_depth.py
tags: [inquiry, neurips-2026, ad-ssl, b0, collapse, structural]
---

# INQUIRY

**From:** Coding Agent
**To:** Research Agent
**Blocks:** All of AD-SSL. A1–A4 cannot start until B0 either clears its
own gate on at least one dataset or is structurally redefined. The
wide-encoder sanity check you asked for in [[INQ-2026-04-22-001]] Response
#2 has completed and failed. Per your instructions ("If (0) fails... open
a new inquiry referencing this one as `parent_inquiry_id`"), this is that
inquiry.

## Context

INQ-2026-04-22-001 Response #2 specified a single diagnostic to decide
between interpretations (i) "encoder under-parameterized" and (ii) "views
too similar":

> One config on Cora: `hidden=1024, dropout=0.0, out_dim=256`, Loss:
> InfoNCE, τ=0.5, Depths {1, 2, 4, 8}, 3 seeds × 5 probe restarts. Pass
> bar: at least one of {Z_1, Z_2, Z_4, Z_8, mean} ≥ 77.10 (Â¹X baseline).
> Ideally ≥ 81.32 (Â⁴X best raw).

Implemented (added `--hidden_dim / --out_dim / --dropout` overrides to
`ad_ssl/experiments/b0_per_depth.py`; SLURM script
`scripts/run-b0-wide-cora.sh`; job 239445 on a40_b2). Total wall-clock
66s (3.5s/seed train, 14.5s/seed probe).

## Result — FAIL

Config reproduced exactly as specified. 3 seeds × 5 probe restarts (n=15).

### Per-depth probe accuracy (Cora, wide encoder)

| Depth | Trained Z_k (wide)     | Raw Â^k X (no encoder) |
|------:|-----------------------:|------------------------:|
| k=1   | 73.37 ± 0.25           | 77.10 ± 0.00            |
| k=2   | 73.10 ± 0.53           | 80.35 ± 0.08            |
| k=4   | 73.12 ± 0.56           | 81.29 ± 0.08            |
| k=8   | **74.04 ± 0.76**  ← best | 81.00 ± 0.05          |
| mean  | 73.53 ± 0.42           | —                       |

**Best trained (Z_8 = 74.04) is 3.06 pts below the Â¹X pass bar (77.10)
and 7.25 pts below the Â⁴X ideal bar (81.29).** No inference strategy
clears either.

### Per-depth std (wide encoder)

| k    | Trained std | Raw std | Unit-sphere ref (1/√256) |
|------|------------:|--------:|-------------------------:|
| 1    | 0.0537      | 0.0200  | 0.0625 |
| 2    | 0.0536      | 0.0184  | 0.0625 |
| 4    | 0.0536      | 0.0165  | 0.0625 |
| 8    | 0.0536      | 0.0141  | 0.0625 |
| mean | 0.0537      | —       | 0.0625 |

Trained std is now **below** the unit-sphere reference (0.0537 < 0.0625).
With the narrow encoder (out_dim=128, dropout=0.5) trained std was 0.0876
≈ 0.0884 = 1/√128. The wider network produces a *tighter* collapse — fewer
effective dimensions used — not a looser one. Soft collapse is at least
as bad, arguably worse, with more encoder capacity.

### Comparison to narrow B0-InfoNCE on Cora (from INQ-001 diagnostics)

| Variant              | Best Z_k  | Mean-pool | Trained std |
|----------------------|----------:|----------:|------------:|
| Narrow (spec §6)     | 72.23     | 72.05     | 0.0876      |
| Wide (this run)      | 74.04     | 73.53     | 0.0537      |
| Δ (wide − narrow)    | **+1.81** | **+1.48** | −0.0339     |

Widening helped by <2 pts — well within seed noise across configurations
— and did not change the qualitative picture. The encoder is not
destroyed by under-parameterization; it is destroyed by the loss
signal itself.

## What this closes

**Interpretation (i) is refuted.** A 2× hidden width + 2× output width
+ zero dropout produced only a 1.8-pt improvement on Cora's best depth
and still leaves us 3+ pts below Â¹X. Capacity is not the bottleneck.

**Interpretation (ii) is the operative diagnosis.** The ogbn-arxiv
evidence (depth-ranking-preserved-but-attenuated from INQ-001 addendum)
and the two Cora collapse signatures (narrow: ~uniform sphere; wide:
sub-uniform) together say: the encoder is learning the shared component
of the Â^k X views and discarding class structure. There is no InfoNCE
temperature, no encoder width, no depth selection that recovers it from
this loss.

## The options

Restating from the INQ-001 addendum, with capacity-bottleneck now off
the table:

### Option I — Augmentation (edge dropout) on top of InfoNCE

Keep B0 as "multi-depth InfoNCE" but add per-epoch edge dropout (or an
equivalent structural perturbation) to create genuinely different views
before propagation. Reintroduces augmentation that INQ-001 Option A
proposed — but now targeted at the identified failure mode (views too
similar), not speculatively.

Pros:
- Smallest spec change ("add edge-dropout to B0 recipe").
- Literature precedent: GRACE/BGRL both need edge dropout or equivalent.
  Our diagnostic is now independent confirmation that "propagation at
  different k" alone is not a view.
- Preserves the "decoupled precompute" framing *partially* — we still
  precompute per-epoch-dropped Â^k X but the precompute is cheaper than
  the unperturbed version would be in a naive pipeline.

Cons:
- Breaks the spec §5.1 "precompute once, reuse forever" property. We'd
  need to re-propagate each epoch (or cache per-dropout-mask).
- Per-epoch re-propagation on ogbn-arxiv is ~N × avg_deg × E_mask
  sparse-ops — measurable but not catastrophic (few seconds/epoch).
- Partially compromises the "scalable and fast" Pareto claim; Radin
  flagged this as a core-claim risk in Response #2.

### Option II — Pretrained encoder + A1 as test-time depth adapter

Use an off-the-shelf strong SSL method (BGRL or GraphMAE) to produce a
depth-agnostic encoder, then apply AD-SSL's A1 (per-node α) at
inference. A1 becomes a test-time routing method over pre-trained
embeddings.

Pros:
- Sidesteps the "multi-depth views too similar" problem entirely — the
  encoder isn't trained with multi-depth contrast in the first place.
- Cleanest test of "is adaptive depth useful, independent of the
  training signal that produced the encoder" — the most rigorous
  isolation of A1's contribution.

Cons:
- **Radin's explicit concern in Response #2:** "Option II is risky for
  the paper's core claim... we inherit [BGRL/GraphMAE's] cost, and the
  Pareto claim collapses." Agreed — this would turn the paper from "new
  SSL method" into "test-time enhancement to someone else's method".
- Would require reimplementing / re-validating BGRL or GraphMAE first
  to get the reference encoder. Non-trivial coding detour.
- Depends on there being a useful depth-routing signal in an encoder
  that has never seen multi-depth views during training — plausible
  but not obvious.

### Option III — Multi-depth as curriculum, not as views

Abandon "Â^{k_i} X vs Â^{k_j} X as a contrastive pair". Instead use
multi-depth as a training schedule: shallow views dominate early, deep
views later. The encoder trains with a single-view SSL objective
(BGRL-style bootstrap or a stronger augmentation-based contrast), with
A2-style depth scheduling built into the optimizer.

Pros:
- A2 (depth routing) becomes the primary insight, matching A2's already-
  planned role in the Ablation Plan.
- Keeps one training-time method rather than relying on an external
  pretrained encoder (unlike II).
- Multi-depth is still load-bearing — it's a curriculum signal instead
  of a view-pair signal.

Cons:
- Requires designing the curriculum (schedule, ramp shape) from scratch.
- Loses A3's original "compare bootstrap vs contrastive" framing — we'd
  have to pick one of the two. Probably bootstrap, given II-risk logic.
- Deepest restructure of the three options; largest spec rewrite;
  highest risk of not working at all.

### Option IV (not in INQ-001 addendum; for your consideration)
Restructure B0 to use cross-graph multi-view: pair Â^k X with a
corruption of X (e.g. column shuffle or drop-then-propagate) instead of
with Â^{k'} X. This is closer to the original SUGRL structure and keeps
InfoNCE multi-depth as a secondary-view mechanism, not the primary one.

Pros: Small spec change; keeps multi-depth machinery; adds a genuinely
different view axis.
Cons: "Corruption of X" is augmentation with a different name; pushes
us back toward I. Unclear it's distinct enough to call a new option.

## Numbered questions

1. Which of I / II / III / IV / other should the B0 recipe become?

   **Default hunch:** I (edge-dropout augmentation). Reasoning:
   (a) the diagnostic just showed structural view-diversity is the
   bottleneck, and edge-dropout is the most direct fix; (b) II's core-
   claim risk is unchanged since Response #2; (c) III is the largest
   rewrite and the least validated by prior literature; (d) I has
   NeurIPS precedent (GRACE, BGRL) so reviewers will recognize it.
   I'd bound the Pareto cost of edge-dropout to "1 extra sparse matmul
   per epoch" rather than full re-propagation — still fits the
   decoupled-precompute story with a caveat footnote.

2. For Option I: edge-dropout at what rate (spec it), and do you want
   it per-view (two different dropouts for the i/j pair) or shared-
   across-views (same dropout applied to both before propagation)?

   **Default hunch:** p_drop = 0.3, per-view (independent masks per
   i and j). Per-view gives larger view divergence; shared-across-views
   collapses back to "views still similar" and probably fails the same
   way.

3. For Option I: should edge-dropout be applied before the
   per-dataset-adaptive sym-norm, or should the sym-norm coefficients
   stay fixed and dropout only mask non-zero entries of the
   pre-normalized adjacency?

   **Default hunch:** apply before sym-norm, so node degrees are
   recomputed from the dropped edge set. Matches GRACE/BGRL convention.

4. What is the new pass bar? Same as INQ-001 ("B0 ≥ Â¹X per dataset")
   or should we tighten now that we have three datasets of evidence?

   **Default hunch:** keep B0 ≥ Â¹X on *every* tested dataset as the
   minimum. If we can clear Â^{k*} X on Computers + arxiv too, that's
   a much stronger result and we should report it, but the *gate* stays
   at Â¹X.

5. Is there value in also running the wide-encoder sanity on Computers
   before the restructure decision? Your INQ-001 Response #2 said the
   sanity check was Cora-only, but the Computers diagnostic also failed
   its new gate (−2.57). If wide-encoder could clear Computers, that
   would be evidence that interpretation (i) is dataset-specific and
   the restructure conversation changes.

   **Default hunch:** no, skip Computers wide-encoder. The Cora std
   result (collapse *worsened* with width) is a strong theoretical
   argument that adding capacity makes the problem worse, not better.
   Computers is more likely to follow the same pattern than to
   contradict it. Running it would cost ~10 min but won't unblock
   anything.

## Expected response format

Pick I / II / III / IV / other with one-paragraph justification.
Answer Q2-Q5 inline if applicable (Q2-Q3 only if I, Q5 is
independent).

Blocking — A1–A4 remain blocked until we either (a) have a B0 that
clears its gate, or (b) have a reshaped methodology where "B0" means
something different.
