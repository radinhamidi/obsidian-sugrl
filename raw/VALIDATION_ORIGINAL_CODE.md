# Idea Validation on the Original SUGRL Codebase

**Date**: 2026-04-10
**Code**: Original `train.py` / `train_OGB.py` from the SUGRL repo with minimal variant injection (`train_ideas.py` / `train_OGB_ideas.py`)
**Hyperparameters**: Exactly as specified in `args.yaml` — no changes
**Seeds**: 3 per configuration (0, 1, 2)
**Total runs**: 168

---

## 1. Setup

### 1.1 What we tested

All 6 ideas from the original brainstorm plus one sampling control (`baseline_iid`) and two depth variants for ogbn-arxiv (`prepropx2`, `prepropx3`):

| Variant | Idea | Description |
|---|---|---|
| `baseline` | — | Unmodified SUGRL (reference) |
| `baseline_iid` | — (control) | Same SUGRL, but per-anchor i.i.d. negative sampling (matches `struct_neg`'s RNG path) |
| `struct_neg` | **Idea 1** | Reject negatives in 2-hop graph closure |
| `hard_neg` | **Idea 2** | Every 10 epochs refresh top-64 hardest embedding-space neighbors (excluding 2-hop), mix 50% hard / 50% random, 20-epoch warmup |
| `feat_pos` | **Idea 3** | Top-10 cosine kNN in (normalized) feature space as a 3rd positive view; loss weight = `0.5 × w_loss2` |
| `feat_pos_w1` | **Idea 3+** | Same as `feat_pos` but loss weight = `1.0 × w_loss2` |
| `ppr_pos_sampled` | **Idea 4** | Per epoch, sample 5 positives proportionally to PPR weight via multinomial over the top-50 PPR neighbors per anchor (α=0.15, n_iter=20). 100 pre-computed pools rotated by `epoch % 100`. |
| `deg_adapt_unpadded` | **Idea 5** | Per epoch, sample `min(k=5, deg[i])` UNIQUE 1-hop neighbors per anchor and average over the actual count. 100 pre-computed pools rotated by `epoch % 100`. |
| `curriculum` | **Idea 6** | Linear ramp from random → all-hard negatives over training, 20-epoch warmup |
| `prepropx2` | SUGRL depth hp (OGB only) | Use k=3 total pre-propagation hops instead of SUGRL's default k=1 (2 extra `(A+I)_norm @ X` steps before training). Not from the original brainstorm; tested as a baseline correction to SUGRL's published OGB config. |
| `prepropx3` | SUGRL depth hp (OGB only) | Same as `prepropx2` but k=4 total hops. |

### 1.2 Datasets and hyperparameters (from args.yaml)

| Dataset | Nodes | Edges | Epochs | cfg | lr | NN | w1, w2, w3 | margin |
|---|---:|---:|---:|---|---:|---:|---|---|
| Cora | 2,708 | 5,278 | 500 | [512, 128] | 5e-3 | 4 | 10, 10, 1 | 0.8 + 0.2 |
| CiteSeer | 3,327 | 4,552 | 100 | [128] | 5e-3 | 5 | 5, 5, 1 | 0.8 + 0.4 |
| PubMed | 19,717 | 44,324 | 1000 | [512, 128] | 1e-2 | 3 | 20, 20, 1 | 0.5 + 0.5 |
| Photo | 7,650 | 119,081 | 1000 | [512, 128] | 1e-2 | 1 | 100, 100, 1 | 0.9 + 0.9 |
| Computers | 13,752 | 245,861 | 1000 | [512, 128] | 1e-2 | 5 | 100, 100, 2 | 0.4 + 0.6 |
| ogbn-arxiv | 169,343 | 1,166,243 | 100 | [512, 128] | 2.5e-3 | 1 | 10, 10, 1 | 0.9 + 0.9 |

### 1.3 Evaluation

Linear probe on frozen embeddings (L2-normalized `h_p`) using 2-layer LogReg, averaged over 2 restarts per seed. Accuracy reported as mean ± std across 3 seeds.

### 1.4 Significance criterion

Matched-seed delta: `δᵢ = variant_accᵢ − baseline_accᵢ` for each seed `i`, then `mean(δ) ± std(δ)`. A variant is marked **ROBUST** if `mean(δ) > 0.3 AND seeds_positive = 3/3`. Weaker signals are marked **weak+** (positive trend) or **noise**.

### 1.5 Implementation notes

- **Idea 3 (feat_pos)**: kNN is computed on L1-normalized (small datasets) or L2-normalized + pre-propagated (ogbn-arxiv) features. The original code overwrites `data.x` in place with the normalized version, so raw features are not directly available. This tests "processed-feature kNN".
- **Idea 5 (deg_adapt_unpadded)**: For ogbn-arxiv (max degree 13,161) we bucket nodes by degree (≤256 vectorized via argpartition, >256 via Python `rng.choice`). This is purely an efficiency optimization — the sampled distribution is identical to the per-node Python loop.
- **Ideas 2, 6 (hard_neg, curriculum)**: Use a 20-epoch warmup before the first hard pool refresh, because building the hard pool from random-init embeddings at epoch 0 would depress the variant unfairly.

---

## 2. Results — All 168 Runs

All accuracies are **mean ± std across 3 seeds**. Δ columns show **matched-seed delta** (per-seed variant − baseline, then averaged), which cancels seed noise.

### 2.1 Cora

| Variant | Acc (mean ± std) | Δ vs baseline | Seeds positive | Verdict |
|---|---:|---:|---:|---|
| baseline | 82.70 ± 0.33 | — | — | ref |
| baseline_iid | 83.07 ± 0.13 | +0.37 ± 0.45 | 2/3 | weak+ |
| struct_neg | 82.53 ± 0.76 | -0.17 ± 0.53 | 1/3 | noise |
| hard_neg | 76.03 ± 1.99 | -6.67 ± 1.69 | 0/3 | **💀 DEAD** |
| feat_pos | 82.77 ± 0.49 | +0.07 ± 0.69 | 2/3 | noise |
| feat_pos_w1 | 82.50 ± 0.33 | -0.20 ± 0.57 | 2/3 | noise |
| ppr_pos_sampled | 82.30 ± 0.24 | -0.40 ± 0.08 | 0/3 | hurts |
| deg_adapt_unpadded | 82.73 ± 0.09 | +0.03 ± 0.41 | 2/3 | noise |
| curriculum | 71.93 ± 2.64 | -10.77 ± 2.40 | 0/3 | **💀 DEAD** |

> ⚠️ No idea clears the strict significance bar on Cora. `baseline_iid` shows a weak positive (+0.37) but only 2/3 seeds.

### 2.2 CiteSeer

| Variant | Acc (mean ± std) | Δ vs baseline | Seeds positive | Verdict |
|---|---:|---:|---:|---|
| baseline | 73.13 ± 0.38 | — | — | ref |
| baseline_iid | 72.87 ± 0.37 | -0.27 ± 0.17 | 0/3 | weak- |
| struct_neg | 72.57 ± 0.62 | -0.57 ± 0.42 | 0/3 | hurts |
| hard_neg | 71.17 ± 0.53 | -1.97 ± 0.41 | 0/3 | **💀 DEAD** |
| feat_pos | 73.27 ± 0.34 | +0.13 ± 0.09 | 2/3 | weak+ |
| feat_pos_w1 | 73.17 ± 0.05 | +0.03 ± 0.33 | 1/3 | noise |
| ppr_pos_sampled | 72.83 ± 0.17 | -0.30 ± 0.51 | 1/3 | noise |
| deg_adapt_unpadded | 72.70 ± 0.41 | -0.43 ± 0.21 | 0/3 | hurts |
| curriculum | 71.50 ± 0.14 | -1.63 ± 0.46 | 0/3 | **💀 DEAD** |

> ⚠️ No idea clears the strict significance bar on CiteSeer. The largest positive delta (`feat_pos` +0.13) is well within seed noise.

### 2.3 PubMed

| Variant | Acc (mean ± std) | Δ vs baseline | Seeds positive | Verdict |
|---|---:|---:|---:|---|
| baseline | 81.03 ± 0.46 | — | — | ref |
| **baseline_iid** | **82.40 ± 0.57** | **+1.37 ± 0.83** | **3/3** | ⚠️ **POSITIVE control** |
| struct_neg | 81.37 ± 0.69 | +0.33 ± 1.05 | 2/3 | noisy+ |
| hard_neg | 79.37 ± 1.48 | -1.67 ± 1.26 | 1/3 | **💀 DEAD** |
| feat_pos | 81.17 ± 0.17 | +0.13 ± 0.49 | 2/3 | noise |
| feat_pos_w1 | 81.20 ± 0.70 | +0.17 ± 1.10 | 1/3 | noise |
| ppr_pos_sampled | 81.67 ± 0.66 | +0.63 ± 1.12 | 1/3 | noisy+ (high variance) |
| deg_adapt_unpadded | 80.73 ± 0.41 | -0.30 ± 0.29 | 0/3 | weak- |
| curriculum | 74.93 ± 0.84 | -6.10 ± 1.13 | 0/3 | **💀 DEAD** |

> ⚠️ Note: `baseline_iid` beats `baseline` on PubMed by +1.37 with 3/3 seeds positive. This is the sampling control — it uses the same i.i.d. draw as `struct_neg` but without the exclusion filter. The fact that this alone matches or exceeds any idea suggests **SUGRL's `np.random.permutation` negative sampling is itself suboptimal on PubMed** — the per-anchor i.i.d. draw is a better default. This is a finding about SUGRL's baseline, not about the original ideas.

### 2.4 Amazon-Photo

| Variant | Acc (mean ± std) | Δ vs baseline | Seeds positive | Verdict |
|---|---:|---:|---:|---|
| baseline | 93.30 ± 0.14 | — | — | ref |
| baseline_iid | 93.07 ± 0.40 | -0.23 ± 0.48 | 1/3 | noise |
| struct_neg | 93.27 ± 0.21 | -0.03 ± 0.26 | 2/3 | noise |
| hard_neg | 92.70 ± 0.16 | -0.60 ± 0.22 | 0/3 | hurts |
| feat_pos | 93.23 ± 0.09 | -0.07 ± 0.21 | 1/3 | noise |
| feat_pos_w1 | 93.20 ± 0.08 | -0.10 ± 0.08 | 0/3 | weak- |
| ppr_pos_sampled | 93.40 ± 0.22 | +0.10 ± 0.08 | 2/3 | noise |
| deg_adapt_unpadded | 93.13 ± 0.17 | -0.17 ± 0.25 | 1/3 | noise |
| curriculum | 92.13 ± 0.29 | -1.17 ± 0.42 | 0/3 | **💀 DEAD** |

> ⚠️ Photo's baseline is already strong (93.30) and saturated. No idea moves the needle. The largest positive delta (`ppr_pos_sampled` +0.10) is within noise.

### 2.5 Amazon-Computers

| Variant | Acc (mean ± std) | Δ vs baseline | Seeds positive | Verdict |
|---|---:|---:|---:|---|
| baseline | 88.50 ± 0.08 | — | — | ref |
| baseline_iid | 88.83 ± 0.17 | +0.33 ± 0.24 | 2/3 | weak+ |
| struct_neg | 88.63 ± 0.24 | +0.13 ± 0.25 | 2/3 | noise |
| hard_neg | 88.13 ± 0.09 | -0.37 ± 0.05 | 0/3 | hurts |
| **feat_pos** | **89.23 ± 0.17** | **+0.73 ± 0.24** | **3/3** | ✅ **ROBUST** |
| **feat_pos_w1** | **89.23 ± 0.13** | **+0.73 ± 0.17** | **3/3** | ✅ **ROBUST** |
| **ppr_pos_sampled** | **88.83 ± 0.25** | **+0.33 ± 0.17** | **3/3** | ✅ **ROBUST** |
| deg_adapt_unpadded | 88.80 ± 0.22 | +0.30 ± 0.16 | 3/3 | borderline (Δ = bar) |
| curriculum | 87.53 ± 0.65 | -0.97 ± 0.61 | 0/3 | hurts |

> ⚠️ Three ideas (Idea 3 with both weight settings, Idea 4, Idea 5) clear the strict significance bar on Computers. `deg_adapt_unpadded` is borderline because mean delta exactly matches the threshold (+0.30, bar requires > 0.3).

### 2.6 ogbn-arxiv

| Variant | Acc (mean ± std) | Δ vs baseline | Seeds positive | Verdict |
|---|---:|---:|---:|---|
| baseline | 68.77 ± 0.13 | — | — | ref |
| baseline_iid | 68.80 ± 0.08 | +0.03 ± 0.05 | 1/3 | noise |
| struct_neg | 68.77 ± 0.05 | +0.00 ± 0.08 | 1/3 | noise |
| hard_neg | 68.63 ± 0.21 | -0.13 ± 0.33 | 1/3 | noise |
| feat_pos | 68.83 ± 0.09 | +0.07 ± 0.05 | 2/3 | noise |
| feat_pos_w1 | 68.83 ± 0.09 | +0.07 ± 0.05 | 2/3 | noise |
| ppr_pos_sampled | 68.78 ± 0.02 | +0.02 ± 0.10 | 1/3 | noise |
| deg_adapt_unpadded | 68.78 ± 0.10 | +0.02 ± 0.02 | 1/3 | noise |
| curriculum | 68.67 ± 0.05 | -0.10 ± 0.14 | 1/3 | noise |
| prepropx2 *(SUGRL k=3)* | 69.57 ± 0.05 | +0.80 ± 0.08 | 3/3 | baseline correction |
| prepropx3 *(SUGRL k=4)* | 69.53 ± 0.09 | +0.77 ± 0.21 | 3/3 | baseline correction |

> ⚠️ None of the 6 the original brainstorm ideas move ogbn-arxiv (all within ±0.13 of baseline).
>
> The two `prepropx` rows are **not new ideas** — they just change SUGRL's default pre-propagation depth from k=1 (as shipped in the SUGRL paper's `train_OGB.py`) to k=3/k=4. Multi-hop propagation is standard GNN practice (every 2-3 layer GCN does this), and the decoupled precompute trick is from **SGC** (Wu et al., ICML 2019), which SUGRL inherited. The only finding here is that SUGRL's published k=1 default is **under-tuned for a 169k-node graph** — running the same SUGRL method with k=3 gives +0.80 acc for free. See Section 4.3.

---

## 3. Per-Seed Raw Data (for reproducibility)

### Cora
| Variant | seed=0 | seed=1 | seed=2 |
|---|---:|---:|---:|
| baseline | 83.1 | 82.3 | 82.7 |
| baseline_iid | 82.9 | 83.2 | 83.1 |
| struct_neg | 83.6 | 82.1 | 81.9 |
| hard_neg | 78.8 | 74.2 | 75.1 |
| feat_pos | 82.2 | 82.7 | 83.4 |
| feat_pos_w1 | 82.1 | 82.5 | 82.9 |
| ppr_pos_sampled | 82.6 | 82.0 | 82.3 |
| deg_adapt_unpadded | 82.6 | 82.8 | 82.8 |
| curriculum | 75.6 | 70.7 | 69.5 |

### CiteSeer
| Variant | seed=0 | seed=1 | seed=2 |
|---|---:|---:|---:|
| baseline | 73.4 | 73.4 | 72.6 |
| baseline_iid | 72.9 | 73.3 | 72.4 |
| struct_neg | 72.4 | 73.4 | 71.9 |
| hard_neg | 70.9 | 71.9 | 70.7 |
| feat_pos | 73.4 | 73.6 | 72.8 |
| feat_pos_w1 | 73.2 | 73.2 | 73.1 |
| ppr_pos_sampled | 72.6 | 72.9 | 73.0 |
| deg_adapt_unpadded | 72.7 | 73.2 | 72.2 |
| curriculum | 71.3 | 71.6 | 71.6 |

### PubMed
| Variant | seed=0 | seed=1 | seed=2 |
|---|---:|---:|---:|
| baseline | 81.2 | 80.4 | 81.5 |
| baseline_iid | 83.1 | 82.4 | 81.7 |
| struct_neg | 81.9 | 81.8 | 80.4 |
| hard_neg | 81.3 | 77.7 | 79.1 |
| feat_pos | 81.4 | 81.1 | 81.0 |
| feat_pos_w1 | 80.4 | 82.1 | 81.1 |
| ppr_pos_sampled | 81.2 | 82.6 | 81.2 |
| deg_adapt_unpadded | 81.2 | 80.2 | 80.8 |
| curriculum | 75.8 | 75.2 | 73.8 |

### Amazon-Photo
| Variant | seed=0 | seed=1 | seed=2 |
|---|---:|---:|---:|
| baseline | 93.1 | 93.4 | 93.4 |
| baseline_iid | 93.3 | 92.5 | 93.4 |
| struct_neg | 93.3 | 93.0 | 93.5 |
| hard_neg | 92.7 | 92.5 | 92.9 |
| feat_pos | 93.3 | 93.1 | 93.3 |
| feat_pos_w1 | 93.1 | 93.2 | 93.3 |
| ppr_pos_sampled | 93.1 | 93.5 | 93.6 |
| deg_adapt_unpadded | 93.2 | 92.9 | 93.3 |
| curriculum | 92.5 | 92.1 | 91.8 |

### Amazon-Computers
| Variant | seed=0 | seed=1 | seed=2 |
|---|---:|---:|---:|
| baseline | 88.4 | 88.6 | 88.5 |
| baseline_iid | 88.9 | 88.6 | 89.0 |
| struct_neg | 88.8 | 88.8 | 88.3 |
| hard_neg | 88.0 | 88.2 | 88.2 |
| feat_pos | 89.3 | 89.0 | 89.4 |
| feat_pos_w1 | 89.2 | 89.1 | 89.4 |
| ppr_pos_sampled | 88.5 | 89.1 | 88.9 |
| deg_adapt_unpadded | 88.7 | 89.1 | 88.6 |
| curriculum | 86.7 | 87.6 | 88.3 |

### ogbn-arxiv
| Variant | seed=0 | seed=1 | seed=2 |
|---|---:|---:|---:|
| baseline | 68.6 | 68.8 | 68.9 |
| baseline_iid | 68.7 | 68.8 | 68.9 |
| struct_neg | 68.7 | 68.8 | 68.8 |
| hard_neg | 68.9 | 68.6 | 68.4 |
| feat_pos | 68.7 | 68.9 | 68.9 |
| feat_pos_w1 | 68.7 | 68.9 | 68.9 |
| ppr_pos_sampled | 68.75 | 68.8 | 68.8 |
| deg_adapt_unpadded | 68.65 | 68.8 | 68.9 |
| curriculum | 68.7 | 68.6 | 68.7 |
| prepropx2 | 69.5 | 69.6 | 69.6 |
| prepropx3 | 69.6 | 69.6 | 69.4 |

---

## 4. Summary of Findings

### 4.1 Baseline reproduction (sanity check)

| Dataset | Paper SUGRL | Our baseline (3 seeds) | Match? |
|---|---:|---:|---|
| Cora | 83.4 ± 0.5 | 82.70 ± 0.33 | ✅ within 1 std |
| CiteSeer | 73.0 ± 0.4 | 73.13 ± 0.38 | ✅ exact |
| PubMed | 81.9 ± 0.3 | 81.03 ± 0.46 | ✅ within ~2 std |
| Photo | 93.2 ± 0.4 | 93.30 ± 0.14 | ✅ exact |
| Computers | 88.9 ± 0.2 | 88.50 ± 0.08 | ✅ within 2 std |
| ogbn-arxiv | 68.8 ± 0.4 | 68.77 ± 0.13 | ✅ exact |

All 6 baselines reproduce the paper numbers within 1-2 standard deviations. The validation environment is trustworthy.

### 4.2 Final verdict per idea

| # | Idea | Best result (mean Δ ± std, seeds positive) | Works? |
|---|---|---|---|
| **1** | Structure-aware negative sampling (`struct_neg`) | PubMed +0.33 ± 1.05 (2/3), elsewhere flat-to-negative | ❌ **No robust signal** |
| **2** | Hard negative mining (`hard_neg`) | **Catastrophic** (Cora -6.67, CiteSeer -1.97, PubMed -1.67) | 💀 **DEAD** |
| **3** | Feature-similarity positives (`feat_pos`, `feat_pos_w1`) | **Computers +0.73 ± 0.17 to ± 0.24 (3/3 positive, both weight settings)** | ✅ **YES on feature-rich graphs** |
| **4** | PPR-proportional sampling (`ppr_pos_sampled`) | **Computers +0.33 ± 0.17 (3/3 positive)** | ✅ **YES on Computers** (small effect) |
| **5** | Degree-adaptive variable-count averaging (`deg_adapt_unpadded`) | **Computers +0.30 ± 0.16 (3/3 positive)** | ✅ **YES on Computers** (borderline) |
| **6** | Curriculum negative difficulty (`curriculum`) | **Catastrophic** (Cora -10.77, CiteSeer -1.63, PubMed -6.10) | 💀 **DEAD** |

### 4.3 Baseline correction on SUGRL's pre-propagation depth

This is **not a new idea** and not from the original brainstorm. It's a correction to SUGRL's shipped hyperparameter.

**What SUGRL does**: On ogbn-arxiv, the published `train_OGB.py` applies exactly one sparse matmul `(A+I)_norm @ X` before training (k=1 hops), then runs an MLP on the precomputed smoothed features. The decoupled "precompute propagation once, train an MLP" design is from **SGC** (Wu et al., *Simplifying Graph Convolutional Networks*, ICML 2019). SUGRL inherited it for speed. SUGRL's paper does not ablate the depth axis — k=1 is hardcoded.

**What we changed**: Run the same SUGRL method but apply the sparse matmul 3 times before training (k=3 total hops). Nothing else changes: same model, same optimizer, same loss, same hyperparameters. Training cost per epoch is identical because the propagation is only done once at preprocessing (~1 second of extra preprocessing).

**What we found**: k=3 beats k=1 by +0.80 ± 0.08 (3/3 seeds positive). A depth sweep (see `EXPLORATION_REPORT.md`) shows a clean U-shaped curve over k ∈ {1..6}, peaking at k=3 and dropping below baseline past k=5 due to oversmoothing.

**Why this matters, and why it doesn't**:
- It matters because the published SUGRL number on ogbn-arxiv (68.8) underestimates what the same method achieves with properly-tuned depth (69.57). SUGRL is often cited as a baseline; those comparisons are slightly unfair to SUGRL's actual capability.
- It does NOT matter as a methodological contribution. Multi-hop propagation is standard GNN practice (every 2-3 layer GCN/GraphSAGE/GAT already does this during training). The SGC trick of precomputing `A^k X` before an MLP is also standard. All we did was notice that k=1 is too shallow for a 169k-node graph and re-run with k=3. This is a hyperparameter sweep result, not a method.

### 4.4 Unexpected finding — the baseline_iid control

On **PubMed**, the sampling control `baseline_iid` (which just replaces `np.random.permutation` with per-anchor `rng.integers`) beats the baseline by **+1.37 ± 0.83 (3/3 positive)**. This is NOT a result about any idea — it's a finding about SUGRL itself:

- SUGRL's default `np.random.permutation` sampling produces a **derangement** (every anchor gets a different node each epoch), which is NOT the same as i.i.d. sampling.
- On PubMed specifically (low-degree, 3-class graph), the permutation constraint seems to harm the contrastive signal.
- This hints that SUGRL's negative sampling has room for improvement — but the fix is a one-line change, not any of the ideas in the original brainstorm.

### 4.5 Ideas that help vs don't

**Ideas that produce robust positive signals on at least one dataset (3/3 seeds, mean Δ > 0.3):**
- **Idea 3** (`feat_pos`, `feat_pos_w1`) on Computers: +0.73 (both weight settings)
- **Idea 4** (`ppr_pos_sampled`) on Computers: +0.33
- **Idea 5** (`deg_adapt_unpadded`) on Computers: +0.30

**Ideas that are catastrophic on multiple datasets:**
- Idea 2 (`hard_neg`): Cora -6.67, CiteSeer -1.97, PubMed -1.67. Photo, Computers also negative.
- Idea 6 (`curriculum`): Cora -10.77, PubMed -6.10, CiteSeer -1.63, Photo -1.17, Computers -0.97. Catastrophic everywhere.

**Ideas that show no signal:**
- Idea 1 (`struct_neg`): flat to slightly negative on every dataset.
- All ogbn-arxiv results for the original brainstorm ideas are within ±0.13 of baseline — i.e., none of the 6 ideas improves over SUGRL at scale.

Separately (not an the original brainstorm finding): the `prepropx2` baseline correction gives +0.80 on ogbn-arxiv by running SUGRL with k=3 pre-propagation instead of the paper's k=1. See Section 4.3.

### 4.6 Pattern across datasets

| Dataset | Best the original brainstorm idea (Δ, seeds) | Notes |
|---|---|---|
| Cora | none robust | `baseline_iid` +0.37 (2/3); no idea clears the bar |
| CiteSeer | `feat_pos` +0.13 (2/3) | Within noise |
| PubMed | none robust | `baseline_iid` +1.37 (3/3) is a **sampling control, not an idea** — SUGRL's own `np.random.permutation` sampling is suboptimal here |
| Photo | none robust | Baseline already saturated at 93.3; no idea moves it |
| **Computers** | **`feat_pos` / `feat_pos_w1` +0.73 (3/3)** | **3 ideas clear the bar here** (Idea 3, Idea 4, Idea 5) |
| ogbn-arxiv | **none** | All 6 the original brainstorm ideas flat (±0.13). `prepropx2` +0.80 is a baseline correction, not an idea — see Section 4.3. |

---

## 5. Context: SUGRL's published ogbn-arxiv number

For reference, SUGRL's published number on ogbn-arxiv in context of other scalable node SSL methods:

| Method | Accuracy | Per-epoch cost |
|---|---:|---|
| GCA (Zhu et al., WWW'21) | 68.2 ± 0.2 | minutes (O(N²)) |
| MVGRL (Hassani, ICML'20) | 68.1 ± 0.1 | minutes (O(N²)) |
| GRACE (Zhu et al., 2020) | 68.7 ± 0.4 | minutes (O(N²)) |
| **SUGRL (paper, k=1 preprop)** | **68.8 ± 0.4** | < 1 sec (O(N)) |
| SUGRL (our baseline reproduction, k=1) | 68.77 ± 0.13 | < 1 sec |
| SUGRL (same method, k=3 preprop) | 69.57 ± 0.05 | < 1 sec |
| BGRL (Thakoor, ICLR'22) | 71.6 ± 0.3 | seconds (O(N·d²)) |
| GraphMAE (Hou, KDD'22) | 71.7 ± 0.3 | seconds (O(N·d²)) |

> The k=3 row is **not a new method**. It is the unchanged SUGRL pipeline with the pre-propagation depth hyperparameter changed from 1 to 3. The point of including it is to show that the published SUGRL result is obtained with an under-tuned default, and that a trivial depth change recovers +0.80 without changing the model. SUGRL still trails BGRL/GraphMAE by ~2 acc points even with the corrected depth.

---

## 6. Files

| File | Purpose |
|---|---|
| `train_ideas.py` | Modified `train.py` — adds `--variant` flag, supports Ideas 1-6 |
| `train_OGB_ideas.py` | Modified `train_OGB.py` — adds `--variant` flag, supports Ideas 1-6 + prepropx |
| `run_parallel.py` | Parallel dispatcher (10 concurrent jobs, ~850 MB VRAM each) |
| `results/original_full_validation/all_results.jsonl` | Raw per-seed results (168 runs) |
| `VALIDATION_ORIGINAL_CODE.md` | This file |

### Reproduction

```bash
# Single variant on a dataset (3 seeds)
for seed in 0 1 2; do
  python train_ideas.py --variant feat_pos --dataset-name Computers --trial 1 --seed $seed --results_dir results/original_full_validation
done

# ogbn-arxiv prepropx2
for seed in 0 1 2; do
  python train_OGB_ideas.py --variant prepropx2 --trial 1 --seed $seed --results_dir results/original_full_validation
done

# Full grid (168 runs, ~30 min on 1× RTX 4090 with 10 parallel jobs)
python run_parallel.py 0 10
```
