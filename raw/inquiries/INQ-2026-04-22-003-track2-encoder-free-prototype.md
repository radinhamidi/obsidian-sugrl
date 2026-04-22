---
inquiry_id: INQ-2026-04-22-003
parent_inquiry_id: INQ-2026-04-22-002
topic: Track 2 prototype — AD-SSL v2 encoder-free (per-node α over raw Â^k X)
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-22
responded:
priority: blocking
status: open
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

<!-- awaiting response -->
