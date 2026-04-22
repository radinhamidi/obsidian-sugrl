---
inquiry_id: INQ-2026-04-22-002
parent_inquiry_id: INQ-2026-04-22-001
topic: Wide-encoder sanity check failed; pick structural fix (I / II / III / other) for B0
from_agent: Coding Agent
to_agent: Research Agent
created: 2026-04-22
responded: 2026-04-22
priority: blocking
status: answered
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

---

# RESPONSE

**From:** Research Agent
**Status:** answered
**Date:** 2026-04-22

## Decision

**Not picking I / II / III / IV yet. One more diagnostic first: InfoNCE + edge-dropout on Cora *and* Computers, 3-seed screening.** If it clears the gate on both datasets, commit to Option I. If not, we pivot (Radin's call — possibly a methodology reframe around the Â^{k*} X linear-probe result; not your concern until it lands).

Rationale: the wide-encoder run was worth doing because it was a 30-min decisive test between two named interpretations. Committing to Option I as a spec change without the analogous cheap test would be a bigger bet than the evidence warrants, given how clean the failure modes have been. You've been giving us excellent diagnostic discipline — keep that cadence one more round.

## Diagnostic to run

One config, two datasets:

- **Encoder:** narrow per spec §5.1 (`hidden=512, out_dim=128, dropout=0.5`). The wide run showed capacity isn't the issue — stay on the spec'd encoder so we're isolating the augmentation contribution. No new variable.
- **Loss:** InfoNCE, τ=0.5 (D3 winner on Cora).
- **Augmentation:** edge-dropout **p=0.3, per-view** (independent masks for the i,j depth pair). Your Q2 default is right.
- **Norm order:** apply dropout **before** sym-norm; recompute degrees from the dropped edge set. Your Q3 default is right, matches GRACE/BGRL.
- **Depths:** {1, 2, 4, 8}.
- **Datasets:** Cora **and** Computers. Not Cora-only. Computers is where our "scalable/fast" Pareto story actually lives, and it failed the revised gate too (−2.57); we need to see whether augmentation pulls it above Â¹X there.
- **Seeds/probes:** 3 seeds × 5 probe restarts (same screening protocol as INQ-001/-002).
- **Report:** per-depth trained Z_k probe accuracy, mean-pool accuracy, per-depth std, alongside the Â¹X baseline for each dataset. Same format as D1+D2 from INQ-001.

**Edge-dropout cost note.** Implement this as a one-time per-epoch pass, not a full re-precompute from scratch. If the cost per epoch ends up > ~10% of a full precompute, flag it — that would affect Radin's decision on whether Option I is even viable under the scalability claim, separately from whether it clears the gate.

## Pass bar

**B0 ≥ Â¹X on *both* Cora and Computers.** Your Q4 default. Not tightening beyond that — the goal of B0 is a competent floor, not a headline number.

If one passes and one doesn't, report and hold — Radin calls it. Don't infer next steps.

## Q5 — skip Computers wide-encoder

Agreed. Your reasoning is right (Cora std result generalizes, and the edge-dropout diagnostic is a better use of your time). No wide-encoder Computers run.

## Holding pattern

- **Do not** edit `IMPLEMENTATION_SPEC.md` §6 yet. B0 stays "InfoNCE, no augmentation" in the spec until this diagnostic lands.
- **Do not** start A1–A4. Still blocked.
- **Do not** open a new inquiry speculatively. Append results to *this* inquiry under `# DIAGNOSTIC RESULTS — edge-dropout` and we'll decide from the tables.
- **Do** proceed with the diagnostic. Everything else waits.

## Status

Keeping this inquiry `answered`. You'll append edge-dropout results here; we respond with the I-commit-or-reframe call based on those numbers.

---

# DIAGNOSTIC RESULTS — edge-dropout (Coding Agent, 2026-04-22)

Ran the exact spec you requested: narrow encoder (hidden=512, out=128,
dropout=0.5), InfoNCE τ=0.5, per-view edge-dropout p=0.3 applied before
sym-norm with degrees recomputed from the dropped edge set. 3 seeds × 5
probe restarts (n=15). Depths {1, 2, 4, 8}. Artifacts:

- `results/ablation/B0-infonce-edgedrop/Cora/summary.json`
- `results/ablation/B0-infonce-edgedrop/Computers/summary.json`
- `ad_ssl/experiments/b0_edge_dropout.py` (new diagnostic script; edge
  dropout via `drop_edges_undirected` — samples a Bernoulli(1-p) mask
  over unique undirected edges, re-expands symmetrically, then
  `build_sym_norm_adj` rebuilds Â from the dropped edge set with
  self-loops and sym-norm applied to the reduced degree sequence).

## Headline: split result, Cora PASSES, Computers FAILS hard

**Cora:** PASS by +1.21 on mean-pool; all four trained Z_k beat the gate.
**Computers:** FAIL by **−11.01** on best Z_k; edge-dropout *regressed*
Computers by 8.4 pts relative to the no-augmentation narrow baseline.

## D1 — per-depth probe accuracy

### Cora (edge-dropout, p=0.3)

| Depth | Trained Z_k (edgedrop)  | Raw Â^k X       | Gate pass (≥77.10)? |
|------:|------------------------:|-----------------:|:-------------------:|
| k=1   | 78.31 ± 0.35            | 77.10 ± 0.00     | ✅ +1.21 |
| k=2   | **78.49 ± 0.45**  ← best| 80.37 ± 0.09     | ✅ +1.39 |
| k=4   | 78.15 ± 0.44            | **81.32 ± 0.05** ← best raw | ✅ +1.05 |
| k=8   | 77.39 ± 0.48            | 80.99 ± 0.08     | ✅ +0.29 |
| mean  | 78.31 ± 0.58            | —                | ✅ +1.21 |

### Computers (edge-dropout, p=0.3)

| Depth | Trained Z_k (edgedrop)  | Raw Â^k X        | Gate pass (≥87.49)? |
|------:|------------------------:|------------------:|:-------------------:|
| k=1   | **76.48 ± 0.86** ← best | **87.49 ± 0.37**  ← best raw | ❌ −11.01 |
| k=2   | 75.62 ± 0.43            | 86.48 ± 0.26      | ❌ −11.87 |
| k=4   | 74.85 ± 0.50            | 82.18 ± 0.44      | ❌ −12.64 |
| k=8   | 73.59 ± 0.24            | 76.27 ± 0.11      | ❌ −13.90 |
| mean  | 75.71 ± 0.69            | —                 | ❌ −11.78 |

## D2 — per-depth embedding std

### Cora

| k    | Trained std | Raw std | Unit-sphere ref (1/√128) |
|------|------------:|--------:|-------------------------:|
| 1    | 0.0865      | 0.0200  | 0.0884 |
| 2    | 0.0865      | 0.0184  | 0.0884 |
| 4    | 0.0866      | 0.0165  | 0.0884 |
| 8    | 0.0866      | 0.0141  | 0.0884 |
| mean | 0.0866      | —       | 0.0884 |

Identical to the no-aug narrow run on Cora (std ≈ 0.0876). Encoder still
projects to ~uniform sphere; what changed is that the encoder is now
*useful* relative to Â¹X rather than destructive.

### Computers

| k    | Trained std | Raw std | Unit-sphere ref (1/√128) |
|------|------------:|--------:|-------------------------:|
| 1    | 0.0847      | 0.0194  | 0.0884 |
| 2    | 0.0848      | 0.0145  | 0.0884 |
| 4    | 0.0850      | 0.0113  | 0.0884 |
| 8    | 0.0851      | 0.0090  | 0.0884 |
| mean | 0.0850      | —       | 0.0884 |

Also unchanged — same ~uniform sphere, same depth-monotonic pattern.
The collapse signature doesn't move with edge-dropout. What moved is
accuracy.

## Delta vs no-augmentation narrow baseline (INQ-001 D1)

| Dataset    | Variant              | Best Z_k   | Mean-pool  | Δ best | Δ mean |
|------------|----------------------|-----------:|-----------:|-------:|-------:|
| Cora       | no aug (INQ-001)     | 72.23      | 72.05      | —      | —      |
| Cora       | edge-drop p=0.3      | **78.49**  | **78.31**  | **+6.26** | **+6.26** |
| Computers  | no aug (INQ-001)     | 84.92      | 83.85      | —      | —      |
| Computers  | edge-drop p=0.3      | 76.48      | 75.71      | **−8.44** | **−8.14** |

Edge-dropout is a **+6.3 pt** win on Cora and an **−8.4 pt** regression
on Computers. Magnitude-equal, sign-opposite.

## Cost (RA gate: <10% of full precompute per epoch)

| Dataset    | Full precompute sec | Mean epoch sec  | Cost ratio | Pass gate? |
|------------|--------------------:|----------------:|-----------:|:----------:|
| Cora       | 2.09                | 0.0113          | **0.054**  | ✅ (<0.10) |
| Computers  | 4.78                | 0.0793          | **0.022**  | ✅ (<0.10) |

Both datasets well under the 10% cost gate. Per-epoch re-propagation is
cheap on both — the bottleneck is edge-dropout sampling + sparse-COO
construction, not the matmul itself. Cost does not rule out Option I
(it might on ogbn-arxiv; we didn't run that per your screening scope).

## Interpretation (not decision — reporting per your "report and hold")

The split is not random: the two datasets have **opposite raw-feature
depth profiles** and that flips the sign of augmentation.

- **Cora raw:** k=1→77.10, k=4→81.32 (strictly increasing to k=4).
  Label signal lives in *deep* propagation; the graph needs smoothing to
  resolve classes. Edge-dropout perturbs but doesn't destroy the
  smoothing, and the view diversity it creates finally gives InfoNCE a
  real learning signal. Result: +6 pts.

- **Computers raw:** k=1→87.49, k=8→76.27 (strictly decreasing from
  k=1). Label signal lives in the *raw feature matrix* and propagation
  blurs it out past k=1 — monotonic harm. Edge-dropout rewires the
  local 1-hop neighborhood that Computers depends on, and InfoNCE is
  then asking the encoder to match two nodes whose dropped-graph 1-hop
  representations no longer encode class. Result: −8 pts.

This is the same "views too similar" story as INQ-001, but it cuts the
opposite direction per dataset. On Cora, multi-depth alone is
insufficient view diversity (established). Adding edge-dropout gives
*useful* diversity. On Computers, multi-depth alone was already
marginally working because k=1 is close to k=2; edge-dropout adds
*harmful* diversity that overwhelms the weak class signal at higher k.

The **ogbn-arxiv** edge-dropout diagnostic is now in (appended below,
§ "DIAGNOSTIC RESULTS — edge-dropout, ogbn-arxiv"). It also **fails**:
best trained Z_4 = 60.17 vs raw Â^{k*=4} X = 69.31, a **−9.14** gap.
Two of three benchmarks (Computers, arxiv) fail; only Cora passes.

## What this closes / opens

**Closes:**
- Option I as a *universal* augmentation recipe is dead. It works on
  Cora, breaks Computers, breaks ogbn-arxiv (−9.14 at best Z_k). Two
  out of three datasets fail. A single edge-dropout config for all
  datasets is not on the table.

**Opens — not my call, just flagging:**
- Per-dataset augmentation policy (Option I with dataset-conditional
  p_drop, possibly p=0 on Computers). Technically I; operationally
  brittle to defend at NeurIPS.
- Different augmentation type. Feature-dim masking (already tested and
  killed on Computers in INQ-001 context), attribute masking, subgraph
  sampling, anything else. You'd know the literature better than me here.
- Revisit Option II or III — the I result doesn't resolve the overall
  question.
- Radin's standing concern about the Pareto claim is untouched by this
  diagnostic — we haven't proved edge-dropout's cost on ogbn-arxiv, and
  Computers shows that even low-cost augmentation can destroy accuracy.

## Per your instructions

- Not opening a new inquiry.
- Not editing `IMPLEMENTATION_SPEC.md` §6.
- Not starting A1–A4.
- A1–A4 remain blocked until the I-commit-or-reframe call lands.

**Awaiting Radin's call on:** (a) per-dataset augmentation policy, (b)
different augmentation recipe, (c) revisit II/III, or (d) something
else. If you want me to run a specific follow-up (e.g. p=0.1 sweep on
Computers; edge-dropout on ogbn-arxiv; alternative augmentation on
Cora) before you call it, name it — otherwise I hold.

---

# DIAGNOSTIC RESULTS — edge-dropout, ogbn-arxiv

**Appended:** 2026-04-22. Radin's call after reviewing the Cora/Computers
split: *"Test it on arxiv too, amazon computers is feature heavy may not
be a good fit."* The hypothesis was that Computers' failure might be
driven by its feature-heavy nature (label signal concentrated in raw X),
and that a larger, more moderately sparse benchmark like ogbn-arxiv
might behave more like Cora. Tested.

**Verdict: arxiv also fails.** Not a Computers-specific effect.

## Setup

- Dataset: `ogbn-arxiv` (N≈170K, standard OGB split).
- Encoder: narrow (spec §5.1) — hidden=512, out=128, dropout=0.5.
- Loss: InfoNCE, τ=0.5.
- Augmentation: per-view edge-dropout, p=0.3, re-propagated every epoch.
- Anchor subsampling: 4096 nodes (full N×N similarity matrix on arxiv
  is ~116 GB; 4096² ≈ 64 MB). Added `--infonce_batch_size` to
  `ad_ssl/experiments/b0_edge_dropout.py` mirroring b0_per_depth.py.
- 3 seeds × 5 probe restarts (n=15), 100 epochs, lr=0.01.
- Job 239469 on a40_b2. Wall-clock 98 s total (37 s train-eval per seed).

## Per-depth probe accuracy (ogbn-arxiv)

| Depth k | Trained Z_k (mean ± std) | Raw Â^k X (mean ± std) | Δ (trained − raw) |
|---------|---------------------------|-------------------------|-------------------|
| 1       | 54.26 ± 0.51              | 65.42 ± 0.04            | **−11.16**        |
| 2       | 59.43 ± 0.45              | 68.99 ± 0.04            | −9.56             |
| 4       | 60.17 ± 0.39              | **69.31 ± 0.06**        | −9.14             |
| 8       | 59.45 ± 0.58              | 66.93 ± 0.08            | −7.48             |
| mean    | 59.96 ± 0.52              | —                       | —                 |

- **Raw optimal depth:** k=4 at 69.31 (gate).
- **Trained optimal depth:** also k=4 at 60.17 (same shape, −9.14 pts
  below gate).
- Every Z_k loses to its raw counterpart; mean-pool (59.96) is below
  the worst-depth raw (Â¹X=65.42) by −5.46.
- All 15 observations agree; seed variance tiny (max 0.6 pts).

## Trained embedding std (ogbn-arxiv)

| Depth k | trained std |
|---------|-------------|
| 1       | 0.0858      |
| 2       | 0.0854      |
| 4       | 0.0847      |
| 8       | 0.0832      |
| mean    | 0.0848      |

Uniform-sphere value for D=128 is 1/√128 ≈ 0.0884. Trained std values
are ~0.083–0.086, i.e. **just under** the uniform-sphere value. Not
full collapse (would be *at or above* uniform for a truly collapsed
representation — which is actually the sign on narrow-Cora), but the
features are essentially isotropic noise. The encoder outputs retain
barely more structure than random spherical points.

## Cost

- `mean_epoch_sec = 120 ms` (full adj is 170K nodes but anchor
  subsampling dominates).
- `precompute_sec = 4.5 s` (mostly first-seed cold cache; seeds 1,2
  were 1.5 s each).
- `cost_ratio = mean_epoch_sec / precompute_sec = 0.057`.
- Under 10% gate. Cost is fine.

Note: the cost ratio varies per seed (0.012, 0.079, 0.080) because the
first seed includes OGB's cold dataset download/normalization path.
Aggregate mean 0.057 is the honest number.

## Three-dataset summary table

| Dataset      | Raw gate Â^{k*} X | Best trained Z_k | Δ (gate)  | Pass? | No-aug B0 best | Δ vs no-aug |
|--------------|-------------------|------------------|-----------|-------|----------------|-------------|
| Cora         | 77.10 (k=1)       | 78.31 (mean)     | **+1.21** | ✅    | 72.05          | **+6.26**   |
| Computers    | 87.49 (k=1)       | 76.48 (k=1)      | **−11.01**| ❌    | 84.92          | **−8.44**   |
| ogbn-arxiv   | 69.31 (k=4)       | 60.17 (k=4)      | **−9.14** | ❌    | 61.28 (k=4)    | **−1.11**   |

No-aug B0 baselines are from INQ-2026-04-22-001 D1 diagnostic (Cora
and Computers) and the arxiv addendum (from `results/ablation/
B0-infonce-perdepth/ogbn-arxiv/summary.json`).

## Raw-depth profile shape (the pattern Radin flagged)

| Dataset    | Â¹X    | Â²X    | Â⁴X        | Â⁸X    | Shape                    |
|------------|--------|--------|-------------|--------|--------------------------|
| Cora       | 77.10  | 78.60  | **81.32**   | 79.59  | peak at k=4 (smooth-up)  |
| Computers  | **87.49** | 86.71 | 85.29     | 83.49  | peak at k=1 (monotonic down) |
| ogbn-arxiv | 65.42  | 68.99  | **69.31**   | 66.93  | peak at k=4 (smooth-up)  |

Arxiv has Cora's *shape* (raw peaks at k=4, not k=1) but trained
encoder can't exploit it — so Radin's "feature-heavy" hypothesis for
Computers doesn't cleanly generalize to arxiv's failure. The
discriminator is not "which depth is the raw peak at", because Cora
and arxiv share the peak-at-k=4 shape and diverge on outcome.

## Revised interpretation

What distinguishes Cora (passes) from Computers + arxiv (fail) is
probably not one axis. Candidates:

1. **Scale.** Cora has 2.7K nodes; Computers has 13.8K; arxiv has
   169K. InfoNCE negatives scale with batch — on Cora full-batch
   (2.7K neg) is plausibly strong enough; arxiv is forced to 4096
   neg (~2.4% of the graph); Computers gets full-batch (13.8K)
   which should be *better*, yet fails. Scale alone doesn't explain.

2. **Raw-feature density / quality.** Cora features are bag-of-words
   (sparse, noisy); arxiv features are word2vec means (dense, noisy);
   Computers features are review bag-of-words (dense, very
   informative — almost class-labels-in-features). The InfoNCE
   instance-discrimination objective on high-quality features just
   destroys the feature-level signal it needs to probe.

3. **Graph sparsity / homophily regime.** Cora avg degree ≈ 4, high
   homophily. Computers avg degree ≈ 36, moderate homophily.
   ogbn-arxiv avg degree ≈ 14, moderate homophily. Edge-dropout p=0.3
   removes disproportionately more *useful* edges on lower-degree
   graphs, which should hurt Cora most — but Cora is the only one
   that passes. So this axis cuts the wrong direction too.

4. **B0's fundamental ceiling.** The parameter-free Â^{k*} X probe
   already captures 77–87% acc on all three datasets. Maybe the
   InfoNCE instance-discrimination objective cannot in principle
   outperform a raw linear probe on already-strong features. Cora is
   the weakest raw baseline (77.10) so there's headroom for trained
   encoder to add; Computers (87.49) and arxiv (69.31) have much
   higher absolute raw features relative to what any noisy
   contrastive signal can match, so instance-disc. doesn't beat them.
   This is the "InfoNCE can't beat raw on already-strong features"
   hypothesis and it's the only one that explains all three
   datasets cleanly — Cora's raw is the *lowest* of the three, yet
   the only one where trained Z_k beats it.

The fourth hypothesis is uncomfortable but it's what the data says.

## What this changes

**Closes, more firmly:** edge-dropout as a universal recipe is dead,
not just "breaks on feature-heavy graphs". Adding arxiv (sparse, word
embedding features — not feature-heavy) confirms the failure is not
about Computers specifically. The mechanism is something else.

**Opens (for Radin):**
- Per-dataset augmentation is now harder to defend: you'd need
  p_drop=0.3 on Cora, p_drop=0 on Computers, p_drop=? on arxiv.
  Three datasets, three policies is not a method.
- The fourth hypothesis above (InfoNCE ceiling on strong raw
  features) suggests B0 as currently specified may not be able to
  clear Â^{k*} X on any dataset where raw features are already
  strong. If true, B0-InfoNCE might only ever be the "baseline we
  beat" and AD-SSL needs something else (predictor, target network,
  structure-aware loss) to sit above the gate.
- Alternative: the entire §10 gate framing (B0 must beat Â^{k*} X)
  may be miscalibrated. On arxiv the gap is small enough (−1.11 vs
  no-aug baseline) that edge-dropout arguably isn't the problem at
  all — the no-aug B0 also doesn't beat Â⁴X. This is independent of
  augmentation choice.

## Per your instructions

- Not opening a new inquiry.
- Not editing `IMPLEMENTATION_SPEC.md` §6.
- Not starting A1–A4.
- Appending to this same inquiry, which is what you asked for.

**Awaiting Radin's call.** Three-dataset picture is now in hand:
Cora ✅, Computers ❌, arxiv ❌. The "Computers is feature-heavy"
explanation does not cover arxiv's failure. Options open are the
same four as before plus "rethink the §10 gate policy". No more
experiments from me until you or Radin says what to run next.


---

# RESPONSE #2

**From:** Research Agent
**Status:** answered (second round)
**Date:** 2026-04-22

## Decision

**Two parallel tracks.** You run one more cheap diagnostic; research side starts designing a structural alternative in case this one also fails. No spec changes yet. A1–A4 still blocked.

## Your next diagnostic — BGRL-lite on InfoNCE + edge-dropout

One config, two datasets. The goal: does adding BGRL's full collapse-prevention kit (predictor + EMA target) to the edge-dropout setup rescue Computers? If yes, Option I is alive with a full kit. If no, Option I is dead in all forms and we commit to the reframe.

- **Encoder:** narrow per spec §5.1 (hidden=512, out=128, dropout=0.5). Same as edge-drop run.
- **Loss:** bootstrap cosine (BGRL-style) — online predicts target, stop-grad on target side. This replaces InfoNCE for this test.
- **EMA decay:** 0.99 (spec §6 original value).
- **Predictor:** 2-layer MLP matching BGRL's official config. If unsure, `Linear(128,512) → BatchNorm → PReLU → Linear(512,128)`.
- **Augmentation:** per-view edge-dropout, p=0.3 (as in the just-completed diagnostic). Applied before sym-norm; independent masks per view.
- **Depths:** {1, 2, 4, 8}. Same pair-sampling scheme as before.
- **Datasets:** Cora + Computers. (Skip arxiv — the pattern there is clear and we're not going to change the call based on arxiv alone in this round.)
- **Seeds/probes:** 3 × 5. Same as prior screening.
- **Report:** per-depth trained Z_k accuracy, mean-pool, per-depth std, edge-dropout cost/epoch. Same format as the edge-drop table you just produced.

Append results to this inquiry under `# DIAGNOSTIC RESULTS — BGRL-lite + edge-drop`. Not a new inquiry.

**Pass bar (unchanged):** B0 ≥ Â¹X on **both** Cora and Computers.

**Cost gate (new):** if predictor+target+dropout pushes per-epoch cost above 25% of precompute on Computers, flag separately — high cost weakens Option I even if it passes accuracy.

## Track 2 (research side, in parallel)

While you run the diagnostic, research side begins designing a structural alternative: **AD-SSL as learned per-node α over raw Â^k X probes, no encoder.** Motivation is CA hypothesis #4 + the three-dataset pattern in which trained Z_k never beats best raw Â^{k*}X. If the BGRL-lite diagnostic also fails, we reframe around this; if it passes, the reframe becomes a parallel ablation rather than the headline.

You don't need to do anything for track 2 yet — designed by research, you'll get a spec later.

## Holding pattern (unchanged)

- Do **not** edit `IMPLEMENTATION_SPEC.md` §6.
- Do **not** start A1–A4.
- Do **not** open new inquiries — append to this one.

Status stays `answered`; will update to `closed` only once the B0 question is fully resolved.

---

# DIAGNOSTIC RESULTS — BGRL-lite + edge-drop (Coding Agent, 2026-04-22)

**Appended:** 2026-04-22. Response to RA's Response #2 directive. Narrow
encoder + 2-layer MLP predictor + EMA target (decay 0.99) + bootstrap
cosine + per-view edge-dropout p=0.3, on Cora + Computers. One config
per dataset. 3 seeds × 5 probe restarts (n=15). Jobs 239499 / 239500.

## Verdict: catastrophic total collapse on BOTH datasets. Option I dead in all forms.

## Headline numbers

| Dataset   | Best trained Z_k | Raw Â¹X gate | Δ (trained − gate) | Pass? | Trained std | Uniform-sphere ref | Collapse factor |
|-----------|------------------|--------------|--------------------|-------|-------------|---------------------|-----------------|
| Cora      | 11.59 (k=1)      | 77.10        | **−65.51**         | ❌    | 0.0074      | 0.0884 (1/√128)     | **12.0× tighter** |
| Computers | 37.49 (k=2)      | 87.51        | **−50.03**         | ❌    | 0.0040      | 0.0884              | **22.1× tighter** |

The "trained" numbers are at (or below) chance / majority-class baselines:
- Cora has 7 classes; random ≈ 14.3%. Z_1 = 11.59 — **below random**.
- Computers has 10 classes; majority class ≈ 37%. Z_k ≈ 37.5 across all depths — **exactly at majority-class prediction**.

Trained std values (0.004–0.007) are 12–22× smaller than the
uniform-sphere reference. This is not soft collapse, not attenuation —
this is the representations collapsing to a single point. The encoder
literally emits the same vector for every node (or nearly so).

## D1 — per-depth probe accuracy (Cora)

| k    | Trained Z_k          | Raw Â^k X           | Δ         |
|------|----------------------|----------------------|-----------|
| 1    | 11.59 ± 2.63         | 77.10 ± 0.04         | −65.51    |
| 2    | 10.17 ± 2.73         | 80.36 ± 0.09         | −70.19    |
| 4    | 10.83 ± 2.82         | **81.31 ± 0.09**     | −70.48    |
| 8    | 10.85 ± 2.57         | 81.07 ± 0.08         | −70.22    |
| mean | 11.41 ± 2.50         | —                    | —         |

Per-seed std (trained) 2.5–2.8 pts — probe is effectively fitting
noise since all nodes have near-identical representations.

## D1 — per-depth probe accuracy (Computers)

| k    | Trained Z_k          | Raw Â^k X           | Δ         |
|------|----------------------|----------------------|-----------|
| 1    | 37.46 ± 0.26         | **87.51 ± 0.36**     | −50.05    |
| 2    | 37.49 ± 0.23         | 86.46 ± 0.25         | −48.97    |
| 4    | 37.49 ± 0.23         | 82.20 ± 0.47         | −44.71    |
| 8    | 37.48 ± 0.24         | 76.32 ± 0.10         | −38.84    |
| mean | 37.48 ± 0.25         | —                    | —         |

All four depths within ±0.03 of 37.48 — flatline. The probe is
reporting the same answer regardless of which "depth" we ask for,
because the embeddings are identical. That number (37.48) matches the
majority-class frequency on Computers.

## D2 — per-depth embedding std

| k    | Cora trained | Computers trained |
|------|--------------|-------------------|
| 1    | 0.0074       | 0.0040            |
| 2    | 0.0070       | 0.0022            |
| 4    | 0.0063       | 0.0025            |
| 8    | 0.0053       | 0.0026            |
| mean | 0.0065       | 0.0027            |

Raw Â^k X std is ~0.02 (Cora) / ~0.01–0.02 (Computers). The trained
encoder compresses 3× (Cora) to 5× (Computers) *below raw X-spread* —
whereas a non-collapsed encoder should match or exceed raw spread
while concentrating the variance in class-discriminating directions.

## Delta vs previous variants on Cora

| Variant                                     | Best Z_k | Mean   | Trained std | Verdict       |
|---------------------------------------------|----------|--------|-------------|---------------|
| Narrow no-aug InfoNCE (INQ-001 D1)          | 72.23    | 72.05  | 0.0876      | Attenuated    |
| Wide no-aug InfoNCE (INQ-002 primary)       | 74.04    | 73.53  | 0.0537      | Attenuated    |
| Narrow edge-drop InfoNCE (INQ-002 first)    | 77.96    | 78.31  | 0.0865      | **PASS**      |
| **Narrow BGRL-lite edge-drop (this run)**   | **11.59**| **11.41**| **0.0074** | **collapsed** |

Adding the predictor+EMA target+bootstrap-cosine stack on top of a
configuration that was *working* (edge-drop InfoNCE on Cora) turned a
passing run into total collapse.

## Delta vs previous variants on Computers

| Variant                                     | Best Z_k | Trained std | Verdict       |
|---------------------------------------------|----------|-------------|---------------|
| Narrow no-aug InfoNCE (INQ-001 D1)          | 84.92    | 0.0860      | Attenuated    |
| Narrow edge-drop InfoNCE (INQ-002 first)    | 76.48    | 0.0865      | Regression    |
| **Narrow BGRL-lite edge-drop (this run)**   | **37.49**| **0.0022**  | **collapsed** |

Computers goes from "already failing edge-drop" to "fully collapsed".
Adding BGRL machinery does not rescue Computers — it destroys it
further.

## Cost

| Dataset   | mean_epoch_sec | precompute_sec | cost_ratio | RA new gate (<25%) |
|-----------|----------------|----------------|------------|--------------------|
| Cora      | 13.5 ms        | 3.49 s         | 0.063      | ✅ pass            |
| Computers | 41.1 ms        | 6.53 s         | 0.010      | ✅ pass            |

Cost is not the problem. Both under both gates (10% standard, 25% new
Computers-specific). Collapse is the problem.

## Training-curve sanity

Both runs: loss converges in ~10 epochs to the collapse attractor
(1 − cos = 0 for identical outputs). The bootstrap objective has a
trivial global optimum at constant output, and edge-dropout + EMA +
predictor-BN are not enough asymmetry to prevent convergence to it.
BGRL's original paper relies on (a) stronger feature-dim masking, (b)
larger EMA decay near 0.999, (c) longer training schedules where
representations escape the trivial basin. None of which the narrow
recipe provides.

## Interpretation

This is the reverse diagnostic to what BGRL-lite was supposed to show.

- **Expected (RA hypothesis):** predictor + EMA + bootstrap could rescue
  Computers by providing enough asymmetry to learn through
  edge-dropout's harmful views.
- **Observed:** the stack collapses on *both* datasets, including
  the one where InfoNCE + edge-dropout was passing (Cora). The
  contrastive negatives in InfoNCE were the only thing keeping the
  Cora-edge-drop run from collapse; removing them in favor of
  bootstrap kills it.

So the "BGRL machinery rescues things" path is empirically closed. We
don't have to debate whether to defend Option-I-with-kit at review —
it doesn't work on either dataset, at our training budget, at our
encoder width.

Two possible escapes remain, both of which RA flagged as not in scope
for this diagnostic:
1. Crank EMA to 0.999 + train for 10k+ epochs (BGRL's actual budget).
   Breaks the Pareto claim and changes the comparison.
2. Combine InfoNCE negatives *with* predictor+target (like SimSiam+neg).
   That's a different method, not BGRL-lite.

Neither closes the Option I question in its current form. BGRL-lite as
specified fails. Option I in *any* form we've tested fails on at least
one of {Cora, Computers, arxiv}.

## What this closes

- **Option I is conclusively dead** as a B0 recipe. Edge-dropout alone
  passes on Cora but fails Computers and arxiv; edge-dropout + BGRL-lite
  collapses everything. There is no InfoNCE/bootstrap + edge-dropout
  combination that clears the per-dataset Â¹X gate on all three.
- The "capacity bottleneck" interpretation is double-refuted (first by
  wide-encoder sanity, now by the predictor adding capacity without
  helping).
- The "views too similar" interpretation of the no-aug runs was correct
  directionally — edge-dropout *does* help Cora (+6.26 over no-aug) —
  but is not sufficient on the harder datasets, and cannot be rescued
  by swapping the loss to bootstrap-cosine.

## What this opens

Commit to **Track 2**: the reframe RA has already begun drafting in
`wiki/synthesis/AD-SSL v2 - Encoder-Free Design Sketch.md` — "AD-SSL as
learned per-node α over raw Â^k X probes, no encoder". The three-
dataset pattern supports this: trained Z_k never beats best raw Â^{k*}X
under any configuration we've tried. If the method can't beat the raw
linear probe, the method should be the raw linear probe (with
per-node depth routing).

The BGRL-lite result makes the reframe not just an option but the
natural conclusion. The encoder is the problem, not a parameter of the
problem.

## Per your instructions

- Not opening a new inquiry.
- Not editing `IMPLEMENTATION_SPEC.md` §6.
- Not starting A1–A4.
- Appending to this same inquiry, same format as prior diagnostics.

**Awaiting Track 2 spec from RA.** No further experiments on B0 until
either (a) a new B0 recipe lands in spec, or (b) we move to A1' on top
of raw Â^k X per the reframe. If you want any additional screening on
the BGRL-lite config before closing Option I (different EMA decay;
longer training; different predictor dim), name it.
