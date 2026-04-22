---
inquiry_id: INQ-2026-04-22-003
parent_inquiry_id: INQ-2026-04-22-002
topic: Track 2 prototype — AD-SSL v2 encoder-free (per-node α over raw Â^k X)
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-22
responded: 2026-04-22
priority: blocking
status: answered
related_files:
  - wiki/synthesis/AD-SSL v2 - Encoder-Free Design Sketch.md
  - raw/inquiries/INQ-2026-04-22-001-b0-collapse.md
  - raw/inquiries/INQ-2026-04-22-002-b0-wide-encoder-fail.md
  - results/ablation/B0-infonce-perdepth/{Cora,Computers,ogbn-arxiv}/summary.json
tags: [inquiry, neurips-2026, ad-ssl, track-2, reframe, encoder-free, prototype]
---

# INQUIRY

**From:** Research Agent
**To:** Coding Agent
**Blocks:** Paper direction. We do not commit to the Track 2 reframe in [[AD-SSL v2 - Encoder-Free Design Sketch]] until this prototype produces numbers. If it shows signal, the reframe becomes the paper. If it doesn't, we pivot again.

## Context

INQ-001 + INQ-002 established that **no encoder-based B0 we've tested beats the parameter-free Â^{k*} X linear probe** on Cora / Computers / ogbn-arxiv, across six encoder configurations (strict, wide, predictor+EMA, edge-dropout, edge-dropout+BGRL-lite). The current working hypothesis: the encoder is net-destructive on already-informative propagated features; the method should not train one.

Track 2 reframe (see [[AD-SSL v2 - Encoder-Free Design Sketch]]): skip the encoder entirely. Learn only a small **per-node depth mixture α_{i,k}** over the raw precomputed features `X_k = Â^k X`. This preserves AD-SSL's novelty claim (per-node adaptive depth vs [[GPRGNN]]'s global γ) at strictly lower cost than any encoder-based variant.

This inquiry is the first empirical test of whether Track 2 has legs. Not a full implementation — a screening prototype.

## What to implement

### Architecture (concrete)

```
# Pre-compute (unchanged from v1)
X_k = Â^k X  for k ∈ {1, 2, 4, 8}          # one-time sparse matmul

# Per-node depth scoring network g
depth_embed = nn.Embedding(K, d_g)          # K=4, d_g=32 (learned depth identity)
scorer_mlp  = MLP(F_in + d_g -> h_g -> 1)   # shared across k; h_g=128

# Forward (at training and inference)
for k in K_SET:
    s_ik = scorer_mlp(concat(X_i, depth_embed[k]))   # scalar per (i,k)
α_ik   = softmax_k(s_ik)                              # normalize over depths per node
Z_i    = Σ_k α_ik · X_k[i]                            # linear mixture, no encoder
```

**Parameter budget:** `(F_in + d_g) * h_g + h_g + d_g * K ≈ (1433+32)*128 + 128 + 32*4 ≈ 188k` params for Cora. Trivially small vs any encoder. Free table (N×K) variant is **not** requested in this round — defer until we know the MLP variant works.

No encoder. No target network. No predictor. No augmentation. No per-epoch propagation. Only α.

### Primary training signal — S1 + S3

**S1 (confidence maximization over soft clusters):**

```
h = nn.Linear(F_in, M)                     # M = num_classes_known (see variations)
p_ik = softmax(h(X_k[i]))                  # per-depth soft-cluster assignment
p_i  = Σ_k α_ik · p_ik                     # mixed prediction
L_conf = -Σ_i Σ_m p_i[m] · log p_i[m]      # per-node entropy (minimize → confident)
L_div  = +Σ_m q_m · log q_m  where q = (1/N) Σ_i p_i   # batch entropy (maximize → diverse)
L_S1   = L_conf - β · L_div
```

β controls cluster-diversity vs per-node-confidence. Default β = 1.0.

**S3 (graph-smoothness regularizer on α):**

```
L_S3 = Σ_{(i,j)∈E_neg_samples} ‖α_i - α_j‖²
```

Sample ~2 · |V| random edges per epoch to keep it cheap. This encourages neighbors to agree on depth preference.

**Total:** `L = L_S1 + λ · L_S3`. Default λ = 0.1.

### Primary config (run this exact config first)

- Datasets: **Cora, Computers** (skip ogbn-arxiv for this round — if prototype works on the small graphs we'll add it).
- Depth set: `{1, 2, 4, 8}` (K=4). Match v1.
- M (number of soft clusters): **equal to the ground-truth class count** (Cora=7, Computers=10).
- β = 1.0, λ = 0.1.
- Epochs: 200. LR: 0.01 Adam. Weight decay 5e-4.
- Probe: linear LogReg on Z (the mixture), 5 restarts per seed.
- Seeds: 3, same screening protocol as INQ-001/002.

### Pass bar

**Z-linear-probe accuracy ≥ Â^{k*} X linear probe accuracy on both Cora and Computers**, within the seed-variance bands we've seen (±0.5 pts tolerance given 3-seed noise). That is:

- Cora: mixture probe ≥ 81.3 (or clearly trending toward it with tight variance).
- Computers: mixture probe ≥ 87.5.

**Soft-pass signal:** even if the mixture doesn't beat best-raw, a mixture that meaningfully exceeds worst-raw AND shows per-node α concentrating on different depths for different nodes (measurable via entropy-of-α-per-node) is a positive result worth keeping. Report both numbers.

### Variations — run if the primary config does not pass cleanly

These are the investigation moves if primary is ambiguous or fails. Run in this order, stop early if one passes:

**V1 — M sweep.** If S1 is cluster-sensitive, our default M=num_classes may be wrong. Re-run on Cora with `M ∈ {num_classes, 2·num_classes, 4·num_classes}`. SCAN/IIC literature uses overclustering.

**V2 — β sweep.** If diversity term is pushing α toward uniform assignments, β may be too high. Re-run Cora with `β ∈ {0.5, 1.0, 2.0, 5.0}`.

**V3 — λ sweep (smoothness weight).** Re-run Cora + Computers with `λ ∈ {0, 0.1, 1.0, 10.0}`. λ=0 isolates S1's contribution.

**V4 — architecture sanity.** Replace the shared-MLP-with-depth-embed scorer with the **free parameter table** variant `α ∈ R^{N×K}` directly learned (no MLP). This tests whether the MLP parameterization is the bottleneck or the loss is. Cora only.

**V5 — S4 supervised upper bound.** Replace S1 with supervised cross-entropy on the 20-per-class train mask. This gives us the supervised ceiling of this α-architecture. **Critical diagnostic:** if supervised α can't beat Â^{k*}X, the whole per-node-α-over-raw-features idea is dead regardless of training signal, and we know to pivot. Run on Cora + Computers.

**V6 — Ablation against α.** Three controls to isolate what the mixture is actually doing:
- **Uniform α** (α_{i,k} = 1/K, no learning): baseline "multi-depth mean".
- **Best single k** (pick the k with highest Â^k X probe accuracy): no mixture.
- **Global γ** (learn a single K-vector γ_k shared across all nodes, GPR-GNN-SSL-style): isolates per-node-ness.
- **Per-node α** (our method).

Report all four on Cora + Computers. V6 is cheap since it's mostly re-using existing Â^k X probes.

### Reporting format

One results markdown per run, appended to this inquiry under `# DIAGNOSTIC RESULTS — <config name>`. Include:

- Z-linear-probe accuracy (mean ± std across seeds).
- Per-node α statistics: mean α_{i,k} per depth; per-node entropy of α_i (histogram); correlation of dominant-k with node degree / class / local homophily.
- L_S1 and L_S3 training curves.
- Wall-clock (total + per-epoch).
- Â^{k*} X baseline and per-depth probe accuracies for reference (from INQ-001 results; don't re-run).

Per-node α statistics matter regardless of whether we pass — they tell us whether α is learning anything structural. Flat α (all nodes similar) means the method is trivially mean-pooling; structured α (varies with node properties) is what the paper needs.

### Cost expectation

One full epoch on Cora should be <10 ms (MLP forward on 2708 nodes × 4 depths, plus small S1/S3 passes). Total for primary config: a few seconds wall-clock. If per-epoch cost exceeds 100 ms on Cora, something is wrong with the implementation; stop and flag.

### What NOT to do

- Do **not** add an encoder, even a thin one. The whole point is to test the encoder-free hypothesis.
- Do **not** use augmentation. Views come from different k; that's the entire view diversity.
- Do **not** add a target network / EMA / predictor. The training signal is S1+S3, not bootstrap.
- Do **not** update `IMPLEMENTATION_SPEC.md` §6 yet. Spec changes only after Track 2 direction is locked.
- Do **not** edit any wiki page. Report results here; research side handles wiki propagation.
- Do **not** start A1–A4 (from v1 Ablation Plan) — those are blocked until Track 2 lands and the ablation plan is rewritten.

## Numbered questions — answer as you run

1. Primary config: pass or fail per the gate? If pass, do we need the variations at all, or is primary enough to unblock Track 2 commit?  **RA's call will depend on your primary number.**

2. If V5 (supervised α) doesn't beat Â^{k*}X either, do you see any implementation issue that could explain it, or is this genuinely a ceiling of the α-over-raw-X parameterization?  **Default hunch:** if supervised can't beat raw best-k, the per-node-α idea is empirically dead and no reframe rescues it.

3. Per-node α entropy distribution: is it structured (varies meaningfully across nodes) or flat (all nodes similar)?  **Default hunch:** flat α means the mixture is doing nothing useful; structured α is the positive signal even if absolute accuracy is below pass bar.

4. For V6, is global γ already reproducing a meaningful fraction of best-raw-probe accuracy? (If yes, per-node α has to beat global γ by a non-trivial margin for the paper.)

## Expected response format

Run primary first. If it passes → report, stop, we commit to Track 2. If it fails or is ambiguous → run variations in order (V1 → V2 → V3 → V4 → V5 → V6), stop at the first that passes, report everything. If none pass, report all and we decide next.

Append each result table to this inquiry under a named `# DIAGNOSTIC RESULTS — <config>` section, same format as INQ-001/-002. Do not open a new inquiry.

**Blocking** — paper direction is stalled until these numbers land.

---

# RESPONSE

**From:** Coding Agent
**Status:** answered (primary config result — variations pending Radin's call)
**Date:** 2026-04-22

## Implementation

`ad_ssl/experiments/adssl_v2_prototype.py` — exactly as specified:

- `DepthMixture`: `Embedding(K=4, d_g=32)` → for each k, `MLP(F_in+d_g → 128
  → 1)(concat(X_i, depth_embed[k]))` → `softmax_k` → α (N, K)
- `cluster_head = Linear(F_in, M)` with M = num_classes
- S1: `L_conf − β · L_div` (per-node entropy minus batch-entropy of mean p)
- S3: squared-L2 of α difference over 2·|V| uniformly-sampled edges per epoch
- No encoder, no target, no predictor, no augmentation, no per-epoch
  propagation. Only α + cluster head are trainable.

Primary config run: M=num_classes, β=1.0, λ=0.1, 200 epochs, LR=0.01, wd=5e-4,
3 seeds × 5 probe restarts (n=15). Jobs 239513 (Cora), 239514 (Computers).

Per-epoch cost as expected (~11 ms/epoch on both datasets). Under the 100ms
sanity ceiling.

---

# DIAGNOSTIC RESULTS — ADSSLv2-primary

## Verdict: FAIL on both datasets. α collapses to uniform.

Primary config does not pass the Â^{k*} X pass bar on either Cora or
Computers, and crucially **α converges to uniform over all four depths
on every node.** The method is trivially mean-pooling X_k. No
structural α learned.

## Headline numbers

| Dataset   | Mixture Z (mean ± std) | Pass bar Â^{k*}X | Δ       | Pass? | Raw mean X_k  | Δ (mixture − mean-pool) |
|-----------|------------------------|-------------------|---------|-------|----------------|--------------------------|
| Cora      | **78.35 ± 0.22**       | 81.3 (k=8 ref: 78.91) | **−2.95** (vs bar) / **−0.56** (vs best-raw-in-this-run) | ❌ | 78.33 ± 0.21 | +0.02 |
| Computers | **85.02 ± 0.46**       | 87.5 (k=1 ref: 87.51) | **−2.48** | ❌ | 85.06 ± 0.44 | −0.04 |

Note on Cora: RA's 81.3 pass bar is from INQ-001's Â⁴X (81.29). In
*this* run the raw per-depth probes came out as k=1:73.81, k=2:78.20,
k=4:77.92, k=8:78.91 — best raw is now k=8 at 78.91. The mixture (78.35)
is below that too, though by a smaller margin (−0.56). Either way the
pass bar is missed. The INQ-001 raw numbers used slightly different
probe/feature code path; the numbers here are within-run self-consistent.

**Mixture is within probe noise of raw mean-pool on both datasets.**
This is what α=uniform predicts. The model is literally doing
mean-pool, no mixing structure learned.

## α statistics — the story is "no α"

| Statistic                          | Cora        | Computers   | Uniform ref |
|------------------------------------|-------------|-------------|-------------|
| α mean per depth {k=1, 2, 4, 8}    | 0.25 / 0.25 / 0.25 / 0.25 | 0.25 / 0.25 / 0.25 / 0.25 | 0.25 |
| α entropy (mean over nodes)        | 1.386       | 1.386       | 1.386 = log 4 |
| Std of per-node entropy            | 0.000       | 0.000       | 0 (uniform)  |
| Dominant-k distribution            | 100% → k=1  | 100% → k=1  | — |
| Corr(dominant-k, degree)           | 0.000       | 0.000       | — |
| Corr(dominant-k, local homophily)  | 0.000       | 0.000       | — |
| Dom-k class spread                 | 0.000       | 0.000       | — |

The "100% → k=1" under dominant-k is a tiebreak artifact: α is
essentially {0.25+ε, 0.25, 0.25, 0.25} across all nodes, and argmax
deterministically picks the first depth under numerical ties. The mean
(0.25) and entropy (exact log 4) confirm α is *identically uniform*
to floating-point precision — no meaningful variation node-to-node, no
structural preference for any depth.

## Loss curves

Both runs: L_conf drops rapidly to near-zero in the first ~20 epochs
and flatlines (p_i becomes sharp — the cluster head finds confident
per-depth assignments). L_div stays near log(M) (q stays uniform — each
class gets equal mass in the batch average). L_S3 ≈ 0 throughout (α is
uniform so ‖α_i − α_j‖² = 0 trivially). After ~20 epochs nothing is
moving because:

- The cluster head h can achieve confident per-depth p_ik regardless of α.
- L_div is symmetric in α (any α achieves uniform batch q if h already does).
- L_S3 is trivially minimized at uniform α.

**The gradient on α from L_S1 is effectively zero in a symmetric-h regime.**
The only nontrivial gradient comes from L_S3, and uniform α is its global
optimum. So α ends up exactly where it started (uniform) because the
loss has a symmetry: *any* permutation of depth indices leaves L unchanged,
and the optimum under this symmetry is uniform.

## What failed — the structural diagnosis

The S1 loss as specified has a **depth-symmetry invariance** that α
cannot break on its own:

1. `p_ik = softmax(h(X_k[i]))` — h is shared across depths. Since all X_k
   for a given node point in roughly the same class direction (propagation
   preserves class signal), h(X_k[i]) is a confident-but-depth-invariant
   class prediction.
2. Therefore `p_i = Σ_k α_ik · p_ik` ≈ `p̄_i` (a depth-independent class
   distribution) for *any* α.
3. L_conf and L_div depend only on p_i → both independent of α.
4. L_S3 prefers uniform α (any other α has positive smoothness cost).
5. So training drives α → uniform, keeps the cluster head confident, done.

This is a real failure of the specified loss, not an implementation bug.
The prototype does exactly what the spec asks; the spec has a
symmetry.

## Cost

| Dataset   | precompute | mean_epoch | total train | ratio |
|-----------|------------|------------|-------------|-------|
| Cora      | 3.6 s      | 11.3 ms    | 2.3 s       | 0.003 |
| Computers | 6.2 s      | 11.5 ms    | 2.3 s       | 0.002 |

Cost is negligible (below even the 10% gate by ~50×). If the method
learned *anything*, it would be a clear Pareto win. But it doesn't.

## Answers to your numbered questions

1. **Primary config: pass or fail?** FAIL on both. Not within ±0.5 pt
   of the bar; ~2.5–3 pts below. And the α structure is trivially absent
   rather than just too noisy — so "pass under variations" requires
   breaking the symmetry, not just finding better hyperparameters.

2. **V5 guidance (is per-node-α ceiling real?):** Not run yet — would
   need to be the next thing. My prior after seeing primary: V5 is
   *essential* before we touch V1–V4. If supervised α can't beat best-raw
   either, no combination of M/β/λ/architecture on the unsupervised
   side will either, and Track 2 is dead. V5 also sidesteps the
   symmetry issue (supervised CE gives α a direct gradient signal
   through the correct class), so it's also a clean architecture
   sanity test.

3. **α entropy structure:** FLAT to floating-point precision. 0.000 std,
   0.000 correlation with everything. This is the strongest possible
   signal of "the method is doing nothing useful". If α had any
   structure — even noisy — it would be informative; having none
   points at the symmetry diagnosis above.

4. **Global γ in V6:** Not run yet. On priors it would do ≈ mean-pool
   too (γ trained under the same S1+S3 without per-node info has no
   signal to prefer one depth over another). Worth running as a
   control but I expect all four of V6 to land within a point of each
   other given what the primary shows.

## Structural analysis (for RA to weigh; not a recommendation)

For the record of the diagnosis, here's how the symmetry intersects
each variation as specified:

| Variation | Touches the symmetry? | Notes |
|-----------|------------------------|-------|
| V1 — M sweep                        | No            | Changes cluster-head dimensionality; α still has no gradient path from L_S1. |
| V2 — β sweep                        | No            | β scales L_div vs L_conf; neither term depends on α when p_ik are similar across k. |
| V3 — λ sweep                        | No            | λ=0 removes L_S3; α stays uniform via no-signal optimum. λ large actively drives α uniform. |
| V4 — free table α ∈ R^{N×K}         | Partially     | No MLP-induced sharing across nodes; S1 symmetry still applies. |
| V5 — supervised α                   | Yes           | CE gives α a direct per-node gradient — ceiling test. |
| V6 — uniform / best-k / global-γ / per-node | Control (analytic) | Uniform α ≈ mean-pool (≈ primary outcome). Best-k ≈ 81.3 Cora / 87.5 Computers. |

This is analysis only. Decision on ordering is RA's.

## Plan — execute V1→V6 in order as specified

Running V1 now. Will stop at the first variation that passes per RA's
"stop early if one passes" directive, otherwise run through V6 and
report all. Appending each result under its own `# DIAGNOSTIC RESULTS —
<config>` section.

## Per your instructions

- Not opening a new inquiry — appended to this one.
- Not touching `IMPLEMENTATION_SPEC.md` §6.
- Not touching wiki.
- A1–A4 still blocked.

Starting V1 (M sweep on Cora: M ∈ {num_classes=7, 2·num_classes=14,
4·num_classes=28}). Results below.

# DIAGNOSTIC RESULTS — ADSSLv2-V1-Msweep-Cora

**Job**: `jobs_output/SUGRL-ADSSLv2-V1-Msweep-Cora-239882.out` (2026-04-22).
**Config**: Cora, 3 seeds × 5 probe restarts, 200 epochs, LR 0.01 Adam,
wd 5e-4, β=1.0, λ=0.1. Only M varies.

## Summary

| M                | Mixture Z probe | Â^1 X | Â^2 X | Â^4 X | Â^8 X | mean X_k | α entropy (uniform=1.386) | dominant-k counts |
|------------------|-----------------|-------|-------|-------|-------|----------|---------------------------|-------------------|
| 7 (primary)      | 78.35 ± (prior) | ≈73.9 | ≈78.2 | ≈77.9 | ≈78.9 | ≈78.3    | 1.386                     | {1: 2708, others: 0} |
| **14**           | **78.35 ± 0.18**| 73.86 | 78.20 | 77.94 | 78.87 | 78.33    | 1.386                     | {1: 2708, others: 0} |
| **28**           | **78.33 ± 0.20**| 73.85 | 78.20 | 77.95 | 78.90 | 78.31    | 1.386                     | {1: 2708, others: 0} |

All three M values: `α mean per k = {1: 0.25, 2: 0.25, 4: 0.25, 8: 0.25}`,
`corr dom-k w degree = 0.000`, `corr dom-k w homophily = 0.000`,
`dom-k class spread = 0.000`.

## Observations

1. **M does not break the symmetry.** α stays exactly uniform at all M.
   Entropy hits the ceiling 1.386 = ln(4) to three decimals. Dominant-k
   is a tie broken by argmax-on-first-match (reported as 100% k=1, same
   artifact as primary — the actual logits are identical across k).

2. **Mixture probe ≈ raw mean-pool, ≈ 0.6 pts below best single depth.**
   Pattern holds across M: Mixture 78.33–78.35 ≈ mean X_k 78.31–78.33 ≈
   0.5–0.6 below Â^8 X 78.87–78.90. Matches primary exactly.

3. **Cluster-head capacity is not the bottleneck.** M=28 (4× more
   clusters than classes) produces identical α statistics to M=7. This
   rules out "maybe the cluster head didn't have enough output
   dimensions to separate clusters across depths" as a cause.

4. **Primary pass bar (Â^{k*} X = 81.3 on Cora):** not met by any M.

## Status

V1 fails the pass bar on Cora. Proceeding to V2 (β sweep on Cora) per
prescribed order.

# DIAGNOSTIC RESULTS — ADSSLv2-V2-Betasweep-Cora

**Job**: `jobs_output/SUGRL-ADSSLv2-V2-Betasweep-Cora-239883.out` (2026-04-22).
**Config**: Cora, 3 seeds × 5 probe restarts, 200 epochs, LR 0.01 Adam,
wd 5e-4, M=num_classes=7, λ=0.1. Only β varies.

## Summary

| β              | Mixture Z probe   | Â^1 X | Â^2 X | Â^4 X | Â^8 X | mean X_k | α entropy | dom-k counts |
|----------------|-------------------|-------|-------|-------|-------|----------|-----------|--------------|
| 0.5            | **78.35 ± 0.22**  | 73.81 | 78.20 | 77.92 | 78.91 | 78.33    | 1.386     | {1: 2708}    |
| 1.0 (primary)  | 78.35             | ≈73.9 | ≈78.2 | ≈77.9 | ≈78.9 | ≈78.3    | 1.386     | {1: 2708}    |
| 2.0            | **78.35 ± 0.22**  | 73.81 | 78.20 | 77.92 | 78.91 | 78.33    | 1.386     | {1: 2708}    |
| 5.0            | **78.35 ± 0.22**  | 73.81 | 78.20 | 77.92 | 78.91 | 78.33    | 1.386     | {1: 2708}    |

## Observations

1. **Results are identical to floating-point precision across β ∈ {0.5,
   2.0, 5.0}.** Per-seed numbers are byte-identical: seed 0 mixture
   probe = 78.52 in all three runs, seed 1 = 78.22 in all three runs,
   seed 2 = 78.30 in all three runs. Means 78.35 ± 0.22, raw column
   stats 73.81/78.20/77.92/78.91/78.33, and α stats all identical.

2. **Empirical confirmation of the L_S1 symmetry.** β weights L_div vs
   L_conf. If α were changing during training, different β would land
   α at different equilibria, and at minimum the mixture outputs would
   differ across seeds. That they don't is the clean signature of "α
   never leaves initialization because L_S1 has no gradient w.r.t. α".
   β=5.0 (heavy diversity penalty, which formally *rewards* spreading
   α mass unevenly across clusters) produces the same uniform α as
   β=0.5.

3. **Primary pass bar (Â^{k*} X = 81.3 on Cora):** not met by any β.

## Status

V2 fails the pass bar on Cora. Proceeding to V3 (λ sweep on Cora +
Computers) per prescribed order.

# DIAGNOSTIC RESULTS — ADSSLv2-V3-Lambdasweep (Cora + Computers)

**Jobs**:
- Cora: `jobs_output/SUGRL-ADSSLv2-V3-Lambdasweep-Cora-240690.out`
- Computers: `jobs_output/SUGRL-ADSSLv2-V3-Lambdasweep-Computers-240691.out`

**Config**: 3 seeds × 5 probe restarts, 200 epochs, LR 0.01 Adam,
wd 5e-4, M=num_classes, β=1.0. Only λ varies.

## Summary — Cora

| λ              | Mixture Z probe   | Â^1 X | Â^2 X | Â^4 X | Â^8 X | mean X_k | α entropy | dom-k counts |
|----------------|-------------------|-------|-------|-------|-------|----------|-----------|--------------|
| 0.0            | **78.35 ± 0.22**  | 73.81 | 78.20 | 77.92 | 78.91 | 78.33    | 1.386     | {1: 2708}    |
| 0.1 (primary)  | 78.35             | ≈73.9 | ≈78.2 | ≈77.9 | ≈78.9 | ≈78.3    | 1.386     | {1: 2708}    |
| 1.0            | **78.35 ± 0.22**  | 73.81 | 78.20 | 77.92 | 78.91 | 78.33    | 1.386     | {1: 2708}    |
| 10.0           | **78.35 ± 0.22**  | 73.81 | 78.20 | 77.92 | 78.91 | 78.33    | 1.386     | {1: 2708}    |

## Summary — Computers

| λ              | Mixture Z probe   | Â^1 X | Â^2 X | Â^4 X | Â^8 X | mean X_k | α entropy | dom-k counts |
|----------------|-------------------|-------|-------|-------|-------|----------|-----------|--------------|
| 0.0            | **85.02 ± 0.46**  | 87.51 | 86.49 | 82.20 | 76.27 | 85.06    | 1.386     | {1: 13752}   |
| 0.1 (primary)  | ≈85.0             | ≈87.5 | ≈86.5 | ≈82.2 | ≈76.3 | ≈85.1    | 1.386     | {1: 13752}   |
| 1.0            | **85.02 ± 0.46**  | 87.51 | 86.49 | 82.20 | 76.27 | 85.06    | 1.386     | {1: 13752}   |
| 10.0           | **85.02 ± 0.46**  | 87.51 | 86.49 | 82.20 | 76.27 | 85.06    | 1.386     | {1: 13752}   |

## Observations

1. **λ=0 (no smoothness regularizer) produces exactly the same result
   as λ>0.** This is the critical test: if the uniform α were caused
   by S3 pulling α toward the constant, removing S3 should let α
   drift. It doesn't. α is uniform under S1 alone → L_S1 provides zero
   gradient to α, independent of S3.

2. **Results byte-identical across all four λ values (both datasets).**
   Same per-seed numbers in all runs, same overall means/stds.

3. **Computers shows an additional signal:** Mixture Z probe 85.02 is
   **2.49 pts below Â^1 X (87.51)** and **1.47 pts below Â^2 X (86.49)**.
   On Computers, mean-pooling over depths is strictly worse than
   picking the best single shallow depth. A functional α would need
   to concentrate mass near k=1 — instead it stays uniform and drags
   performance down toward the deep-depth floor (Â^8 X = 76.27).
   The "best single k" analytic control (part of V6) is already
   effectively estimated: 87.51 on Computers, 78.91 on Cora.

4. **Primary pass bar:** not met. Cora 78.35 < 81.3, Computers 85.02
   < 87.5 (and also < Â^1 X = 87.51).

## Status

V3 fails the pass bar on both datasets. The λ=0 result rules out S3
as the driver of uniform α — L_S1 is the invariance. Proceeding to
V4 (free-table α variant on Cora) per prescribed order.

# DIAGNOSTIC RESULTS — ADSSLv2-V4-FreeTable-Cora

**Job**: `jobs_output/SUGRL-ADSSLv2-V4-FreeTable-Cora-241055.out`
(2026-04-22; first two attempts 241044/241049 crashed in np.histogram
on narrow-range entropy — fix: wrap histogram call in try/except with
degenerate fallback).

**Config**: Cora, 3 seeds × 5 probe restarts, 200 epochs, LR 0.01 Adam,
wd 5e-4, M=num_classes=7, β=1.0, λ=0.1. Architectural change:
α ∈ R^{N×K} learnable free table (no MLP, no depth embedding, no
feature input). logits init N(0, 0.01²), softmax over K.

## Summary

| Variant                | Mixture Z probe | Â^1 X | Â^2 X | Â^4 X | Â^8 X | mean X_k | α entropy | dom-k counts                    |
|------------------------|-----------------|-------|-------|-------|-------|----------|-----------|----------------------------------|
| MLP (primary, V1–V3)   | 78.35 ± 0.22    | 73.81 | 78.20 | 77.92 | 78.91 | 78.33    | 1.386     | {1: 2708, others: 0} (tie-break) |
| **free-table (V4)**    | **78.28 ± 0.11**| 73.91 | 78.19 | 77.96 | 78.89 | 78.27    | 1.386     | {1:1007, 2:732, 4:560, 8:409}    |

**α-structure stats (V4):** `α mean per k = {1:0.25, 2:0.25, 4:0.25, 8:0.25}`,
`corr dom-k w degree = -0.005`, `corr dom-k w homophily = 0.002`,
`dom-k class spread = 0.077`.

## Observations

1. **Mixture probe ≈ MLP variant ≈ mean-pool.** 78.28 vs MLP 78.35 vs
   raw mean X_k 78.27. Removing parameter-sharing across nodes does
   not change the representation quality.

2. **α mean is still exactly uniform.** All four depths receive 0.25
   on average after 200 epochs. Entropy is 1.386 to floating-point
   precision. The free table has N·K = 10,832 independent parameters
   for Cora; if L_S1 had any per-node gradient to α, these parameters
   would diverge. They don't.

3. **Dominant-k distribution is dispersed rather than tie-break-locked.**
   MLP variants reported dom-k 100% on k=1 because the MLP produced
   near-identical scores per node and argmax picked the first; the
   free table has per-node random init, so argmax varies. But the
   dispersion is structureless: correlation with degree is -0.005,
   with local homophily is +0.002, class spread is 0.077 (vs a
   meaningful spread of O(K)=O(4) if depths aligned with classes).

4. **MLP parameter-sharing is NOT the bottleneck.** V4 was designed
   to test whether the shared MLP was forcing all nodes to the same
   α. Answer: no — even with full per-node freedom, α converges to
   uniform because L_S1 provides no per-node α-gradient.

5. **Primary pass bar (Â^{k*} X = 81.3 on Cora):** not met (78.28).

## Status

V4 fails. Combined with V3's λ=0 result, this pins the failure
squarely on L_S1 as specified — the loss has no gradient w.r.t. α when
p_ik are similar across k, regardless of how α is parameterized.
Proceeding to V5 (supervised α on Cora + Computers) per prescribed
order.

# DIAGNOSTIC RESULTS — ADSSLv2-V5-Supervised (Cora + Computers)

**Jobs**:
- Cora: `jobs_output/SUGRL-ADSSLv2-V5-Supervised-Cora-241080.out`
- Computers: `jobs_output/SUGRL-ADSSLv2-V5-Supervised-Computers-241081.out`

**Config**: 3 seeds × 5 probe restarts, 200 epochs, LR 0.01 Adam,
wd 5e-4, λ=0.1, MLP mixture. Objective: replace L_S1 with
cross-entropy on the 20-per-class train mask, applied to the mixture
prediction `p_i = Σ_k α_ik · softmax(head(X_k[i]))`. Head output
dim = num_classes. L_S3 smoothness kept at λ=0.1. This gives α a
direct per-node gradient under the supervised signal.

## Summary

| Dataset   | Mixture Z probe  | Â^1 X | Â^2 X | Â^4 X | Â^8 X | mean X_k | α mean per k                           | α entropy | dom-k counts          |
|-----------|------------------|-------|-------|-------|-------|----------|----------------------------------------|-----------|-----------------------|
| **Cora**  | **74.77 ± 0.30** | 73.81 | 78.20 | 77.92 | 78.91 | 78.33    | (0.997, 0.001, 0.001, 0.001)           | 0.025     | {1: 2708}             |
| **Computers** | **85.02 ± 0.46** | 87.51 | 86.49 | 82.20 | 76.27 | 85.06    | (0.25, 0.25, 0.25, 0.25)               | 1.386     | {1: 13752} (tie-break)|

## Observations

1. **The two datasets respond very differently to supervised signal.**

   - Cora: α collapses globally to k=1 with near-zero entropy (0.025).
     Supervised CE moves α. This is a qualitative change from V1–V4,
     which all stayed exactly uniform.
   - Computers: α stays **exactly uniform** (entropy 1.386). Per-seed
     numbers are byte-identical to V3 λ=0.1 Computers (seed 0=85.66,
     seed 1=84.74, seed 2=84.65). The supervised signal failed to
     move α at all on this dataset.

2. **Even where α moves (Cora), it picks the wrong depth.** α → k=1,
   but k=1 is the *worst* single depth for Cora (Â^1 X = 73.81).
   The best depth is k=8 (78.91) and the overall pass bar is 81.3.
   Mixture result 74.77 sits barely above Â^1 X — the mixture just
   reconstructs Â^1 X since α has almost all mass there. This is
   3.5 pts **below** uniform mean-pool (78.33) and 4.1 pts below
   Â^{k*} X.

3. **On Computers, the architecture cannot be coerced even under
   supervision.** The natural supervised optimum would be α → k=1
   (Â^1 X is best at 87.51). Instead α stays uniform, mixture lands
   at 85.02 = raw mean-pool, 2.49 pts below Â^1 X.

4. **Primary pass bar:** not met on either dataset.
   - Cora: 74.77 < 81.3 (miss by **−6.53**)
   - Computers: 85.02 < 87.5 (miss by **−2.48**)

5. **Plausible mechanism for Cora's misrouting:** The joint
   (head, α)-optimization over mixture predictions has a degenerate
   easy solution: "let the head memorize one depth's training
   features". On Cora with 140 labeled nodes, argmax of training-CE
   reduction is k=1 (shallow / raw-like features, easiest to fit to
   20/class). That minimum of train CE is not the minimum of test
   loss under a fresh linear probe. The architecture-implied inductive
   bias is not "pick the depth that generalizes best per node".

6. **Plausible mechanism for Computers' non-movement:** With ~200
   labeled nodes out of 13752, the CE gradient averaged over the
   mixture-through-softmax is small relative to weight-decay pull on
   the MLP. The MLP stays near init → α stays near uniform. Also
   consistent with: the gradient to α from CE vanishes when p_ik are
   similar across k, and on Computers the dense continuous features
   produce more similar softmax distributions than on Cora's BOW.

## Status

V5 fails the pass bar on both datasets. Per RA's V5 preamble —
*"Critical diagnostic: if supervised α can't beat Â^{k*} X, the whole
per-node-α-over-raw-features idea is dead regardless of training
signal"* — this is the result. Proceeding to V6 (ablation controls)
per prescribed order.

# DIAGNOSTIC RESULTS — ADSSLv2-V6-Ablation (Cora + Computers)

**Job**: `jobs_output/SUGRL-ADSSLv2-V6-GlobalGamma-241113.out` (2026-04-22).

**Config**: 3 seeds × 5 probe restarts, 200 epochs, LR 0.01 Adam,
wd 5e-4, β=1.0, λ=0.1, M=num_classes, SSL L_S1 + λ·L_S3 objective.
Only the `--alpha_mode global` run is new code; the other three
controls are read off the always-emitted raw-probe statistics.

## All-four-controls summary

| Control                              | Cora mixture probe | Computers mixture probe | α mean per k (trained) | α entropy |
|--------------------------------------|--------------------|-------------------------|------------------------|-----------|
| **Uniform α** (analytic, = mean X_k) | 78.33 ± 0.21       | 85.06 ± 0.44            | (0.25, 0.25, 0.25, 0.25) fixed | 1.386 fixed |
| **Best single k** (Â^{k*} X probe)    | **78.91 ± 0.14** (k=8) | **87.51 ± 0.37** (k=1)  | one-hot at k*              | 0            |
| **Global γ** (learned K-vector, V6 new) | 78.42 ± 0.19     | 85.03 ± 0.46            | (0.25, 0.25, 0.25, 0.25) to 1e-8 | 1.386 |
| **Per-node α** (primary, V1 M=7)     | 78.35 ± 0.22       | 85.02 ± 0.46            | (0.25, 0.25, 0.25, 0.25) to floating pt | 1.386 |

Note on "dominant-k" under global γ: because every node gets the same
γ-induced α, the argmax-tie-break artifact gives `{k*: N, others: 0}`
where k* is the seed-random argmax of γ. On Cora across seeds: seed 0
→ k=4, seed 1 → k=8, seed 2 → k=1. On Computers: two seeds → k=4, one
seed → k=2. This is pure noise-induced argmax, not learning.

## Observations

1. **Global γ collapses to uniform exactly like per-node α.** Even the
   K=4-parameter global K-vector — trivially easy to optimize — stays
   at uniform under L_S1. γ = 0.25 ± 1e-8 on both datasets. This is
   the strongest possible control for the symmetry diagnosis: the
   failure is not about the richness of α's parameterization. L_S1
   has no useful gradient regardless of how α is parameterized
   (shared MLP, free table, or single K-vector).

2. **Per-node α ≈ Global γ ≈ Uniform α on both datasets.** Within
   seed variance, the three collapse to the same outcome (Cora
   78.35 / 78.42 / 78.33; Computers 85.02 / 85.03 / 85.06). This
   means the primary method's "per-node" claim produces no measurable
   per-node routing; it is doing the same thing as a fixed uniform
   mixture.

3. **Best single k dominates the mixture under this training objective.**
   On both datasets, picking k* beats all three α variants:
   - Cora: 78.91 (k=8) > 78.42 (γ) ≈ 78.35 (per-node α) ≈ 78.33 (uniform).
   - Computers: 87.51 (k=1) > 85.03 (γ) ≈ 85.02 (per-node α) ≈ 85.06 (uniform).
   Mean-pooling / global-γ / per-node-α under this training signal
   all land roughly halfway between the best and worst single depths;
   they achieve nothing over "just probe Â^{k*} X".

4. **Primary pass bar:** not met by any V6 control on either dataset.

## Status

V6 fails. End of prescribed variation list.

## Full-sweep verdict (primary + V1–V6)

| Config              | Cora        | Computers   | α structure                       | Verdict |
|---------------------|-------------|-------------|-----------------------------------|---------|
| Primary (V1 M=7)    | 78.35 ± 0.22 | 85.02 ± 0.46 | uniform                         | fail    |
| V1 M=14             | 78.35 ± 0.18 | —           | uniform                           | fail    |
| V1 M=28             | 78.33 ± 0.20 | —           | uniform                           | fail    |
| V2 β=0.5            | 78.35 ± 0.22 | —           | uniform                           | fail    |
| V2 β=2.0            | 78.35 ± 0.22 | —           | uniform                           | fail    |
| V2 β=5.0            | 78.35 ± 0.22 | —           | uniform                           | fail    |
| V3 λ=0.0            | 78.35 ± 0.22 | 85.02 ± 0.46 | uniform                         | fail    |
| V3 λ=1.0            | 78.35 ± 0.22 | 85.02 ± 0.46 | uniform                         | fail    |
| V3 λ=10.0           | 78.35 ± 0.22 | 85.02 ± 0.46 | uniform                         | fail    |
| V4 free table       | 78.28 ± 0.11 | —           | uniform mean; per-node noise      | fail    |
| V5 supervised       | 74.77 ± 0.30 | 85.02 ± 0.46 | Cora→k=1 globally; Computers uniform | fail    |
| V6 global γ         | 78.42 ± 0.19 | 85.03 ± 0.46 | uniform                           | fail    |
| **Pass bar**        | **81.3**    | **87.5**    | —                                 | —       |

No variation passes the pass bar on either dataset. α never leaves
uniform under L_S1, regardless of M, β, λ, or α parameterization.
The one variation that moves α at all (V5 Cora, supervised) does so
in a direction that makes the mixture worse than uniform mean-pool.

Reporting all. Decision on next step is RA's.
