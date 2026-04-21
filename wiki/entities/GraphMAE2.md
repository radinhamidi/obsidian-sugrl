---
title: GraphMAE2
type: entity
kind: method
venue: WWW 2023
url: https://arxiv.org/abs/2304.04779
tags: [method, baseline, generative, accuracy-ceiling, mask-predict]
created: 2026-04-21
updated: 2026-04-21
---

# GraphMAE2 — A Decoding-Enhanced Masked Self-Supervised Graph Learner

Hou, He, Cen, Liu, Dong, Kharlamov, Tang (Tsinghua + BIT + Bosch). WWW 2023, arXiv:2304.04779 (2023-04-10). Successor to [[GraphMAE]], explicitly targeting scalability to OGB-LSC (ogbn-Papers100M, 111M nodes / 1.6B edges). Current **accuracy champion** among SSL baselines at OGB scale.

## Motivation (§2)

Two identified weaknesses of masked feature reconstruction in the graph setting:
1. **Feature-discriminability trap.** With continuous/bag-of-word features, a GNN decoder can over-rely on raw inputs, memorising easy patterns instead of learning useful structure.
2. **Scale.** Prior graph SSL work mostly ran on small graphs; existing scaling attempts just bolt on neighborhood sampling or ClusterGCN, which are not designed for masked reconstruction.

GraphMAE2's answer: **regularise the decoding stage** (not the encoding) via two orthogonal mechanisms, and use **local clustering subgraphs** (PPR-Nibble) for scalable training.

## Mechanism (exact)

Encoder `f_E(·; θ)` (GAT by default), projector `g`, feature decoder `f_D`, latent decoder `g'`. Two losses:

### 1. Multi-view random re-mask decoding  (Eq. 2)

After encoding `H = f_E(A, X̃)` where `X̃` is the feature-masked input, **re-mask** a random subset `V⁽ʲ⁾ ⊂ V` of encoded node representations with a learnable `[DMASK]` token, then run the decoder. Repeat K times (paper uses **K=3**) with independent random re-masks and sum the reconstruction errors:

```
L_input = (1/|Ṽ|) Σ_{j=1..K} Σ_{v_i ∈ Ṽ} (1 − (x_i · z_i⁽ʲ⁾) / (‖x_i‖·‖z_i⁽ʲ⁾‖))^γ
```

i.e. the GraphMAE Scaled Cosine Error, but summed over K re-masked views. The randomness acts as regularisation preventing the decoder from memorising input patterns.

### 2. Latent representation prediction  (Eq. 4–5)

A **target generator** `f_E'(·; ξ), g'(·; ξ)` runs on the **unmasked** graph to produce target `X̄ = g'(f_E'(A, X; ξ))`. The online branch predicts this target in latent space with its own projector:

```
L_latent = (1/N) Σ_i (1 − (z̄_i · x̄_i) / (‖z̄_i‖·‖x̄_i‖))^γ
ξ ← τ·ξ + (1−τ)·θ    (EMA)
```

Note: this is **BYOL-style** (online-target EMA), but their stated framing is "self-distillation without heavy augmentations" — the target is the unmasked graph's embedding, not a stochastic augmentation.

### Total loss  (Eq. 6)

```
L = L_input + λ · L_latent
```

`λ` varies per dataset: **Arxiv 10.0, Products 5.0, MAG 0.1, Papers100M 10.0** (Table 10).

## Scalability: PPR-Nibble local clustering (§2 + Thm A.1)

Instead of GraphSAINT/Cluster-GCN, they use **personalized PageRank** to extract dense local subgraphs per seed node. Theorem A.1 (from Zhu et al. 2013 / Yin et al. 2017) guarantees the conductance of the returned cluster is small → **dense, well-connected** subgraphs, suited to masked reconstruction which needs good neighborhood aggregation. Linear-time implementation.

Ablation (Table 7): on ogbn-Products, PPR-Nibble local clustering beats GraphSAINT by **+0.63** and Cluster-GCN by **+2.24**.

## Numbers on ogbn-arxiv

| Setting | Method | Test acc |
|---|---|---:|
| Mini-batch linear probe (Tbl 3) | GraphMAE2 | **71.89 ± 0.03** |
| | GraphMAE | 71.03 ± 0.02 |
| | BGRL | 70.51 ± 0.03 |
| Full-graph linear probe (Tbl 9) | **GraphMAE2** | **71.95 ± 0.08** |
| | GraphMAE | 71.75 ± 0.17 |
| | BGRL | 71.64 ± 0.12 |
| | Supervised GCN | 71.74 ± 0.29 |
| Fine-tune all labels (Tbl 8) | GraphMAE2 | **72.69** |
| | GraphMAE | 72.38 |

**Our ogbn-arxiv accuracy ceiling: 71.95** (full-graph linear probe, consistent with our evaluation protocol). Fine-tuning goes to 72.69.

## Numbers on ogbn-Products (Table 3, linear probe)

GraphMAE2 **81.59** vs GraphMAE 78.89 vs BGRL 78.59 vs GGD 75.70. **+2.70 over GraphMAE**, the largest gap of any row.

## Numbers on ogbn-Papers100M (Table 3, linear probe)

| Method | Test |
|---|---:|
| SGC | 63.29 ± 0.19 |
| Random-Init | 61.55 ± 0.12 |
| BGRL | 62.18 ± 0.15 |
| GGD (paper value) | 63.50 ± 0.50 |
| GraphMAE | 62.54 ± 0.09 |
| **GraphMAE2** | **64.89 ± 0.04** |

All contrastive baselines (GRACE, BGRL, CCA-SSG) **underperform Random-Init** on Papers100M — a damning observation for contrastive methods at that scale. Only masked-feature methods (GraphMAE, GraphMAE2) consistently beat Random-Init.

## Numbers on small graphs (Table 5)

Cora 84.5 / CiteSeer 73.4 / PubMed 81.4 — marginally above GraphMAE and GGD. Paper notes: improvements are smaller on small graphs because bag-of-word features are more discrete/less noisy, and their multi-view re-mask regulariser has less to fix.

## Hyperparameters (Table 10)

All large datasets share:
- **hidden_size 1024, num_layer 4, masking 0.5, re-masking 0.5, num_re-masks = 3**
- GAT encoder + single-layer GAT feature decoder + MLP latent projector `g`
- AdamW with cosine decay (no warmup)

Per-dataset:
| | Arxiv | Products | MAG | Papers100M |
|---|---:|---:|---:|---:|
| λ | 10.0 | 5.0 | 0.1 | 10.0 |
| lr | 0.0025 | 0.002 | 0.001 | 0.001 |
| wd | 0.06 | 0.06 | 0.04 | 0.05 |
| max epoch | 60 | 20 | 10 | 10 |

## Ablations (§3.3)

**Component ablation on Products / MAG / Papers100M (Table 6):**
- w/o random re-mask: −0.55, −0.23, −0.73
- w/o latent rep pred: −1.58, −0.37, −1.91  ← the bigger one
- **w/o input recon** (latent only): −4.71, −4.04, −5.69 → collapses below GraphMAE. Pure latent prediction without feature reconstruction = near-trivial solution.

Takeaway: **feature reconstruction is the load-bearing signal; the latent BYOL-style target is regularisation.** This is opposite to BGRL's framing (BYOL is the main signal). Important for AD-SSL's loss design.

**Model capacity (Fig. 3):** width matters more than depth. Doubling hidden-size to 1024 gives ~2% accuracy; going deeper than 4 layers plateaus.

## Complexity

Linear in N (both time and space). At Papers100M: 8× A100-80GB, Python/PyTorch, 10 epochs max. No wall-clock numbers given in the paper itself — this is a flag for our Pareto analysis (we can only compare FLOPs/params, not measured seconds unless we reproduce).

## Role in AD-SSL

- **Right anchor of the Pareto figure** at OGB scale. Accuracy ceiling: 71.95 (full-graph arxiv), 64.89 (papers100M linear probe).
- **Do not claim to beat GraphMAE2 on accuracy.** Claim: AD-SSL matches GraphMAE/BGRL-range accuracy at GGD-range cost. GraphMAE2 is the "expensive accuracy upper bound" in our figure.
- **Design cues:**
  1. **K re-masked views, summed SCE loss** (Eq. 2) — structurally analogous to our K depth-pair bootstrap losses, summed. The K=3 default is a useful starting point.
  2. **Input reconstruction is load-bearing; latent target is auxiliary.** Our AD-SSL bootstrap-across-depths should be analogous to their *latent* loss (auxiliary); we may want a feature-reconstruction anchor as well to prevent collapse. Open question — test in ablation.
  3. **γ in SCE** also shows up here, confirming GraphMAE's design choice. Candidate knob for AD-SSL.
  4. **Local clustering for scaling** — if we go beyond ogbn-arxiv, PPR-Nibble beats GraphSAINT/Cluster-GCN for mask-style SSL. But: AD-SSL's whole point is precomputed `Â^k X` at full-graph scale — we may not need any subgraph sampling on ogbn-arxiv.

## Differences from AD-SSL (for related-work paragraph)

| | GraphMAE2 | AD-SSL |
|---|---|---|
| Views | K random re-masks on post-encoder `H` | K deterministic depth-propagations `Â^k X` |
| Encoder | 4-layer GAT, hidden 1024 | MLP on precomputed features |
| Loss | Feature reconstruction + latent BYOL | Bootstrap across depth pairs |
| Cost bottleneck | Encoder + K decoder forwards per step | Zero GNN forwards after precompute |
| Augmentation | Feature masking (input + re-mask) | None (different depths *are* the views) |

**Reviewer attack to expect:** "AD-SSL is GraphMAE2 minus the encoder." Defence: (a) views differ by *spectral depth*, not by random masking; (b) no decoder — cost is O(MLP) per step; (c) per-node α across depths is a learned mixture, which GraphMAE2 has no analogue of; (d) AD-SSL matches accuracy at a fraction of the cost, shown empirically.

## Contradiction flag for our narrative

GraphMAE2's Papers100M result (64.89) is **higher than GGD's 63.50**, and their framing claims contrastive methods *fail* at Papers100M scale (all GRACE/BGRL/CCA-SSG come in below Random-Init 61.55). If AD-SSL is classified as "contrastive-style bootstrap," reviewers may argue we should not extrapolate to Papers100M — masked reconstruction dominates at that scale, not bootstrap. We should either (a) scope claims to arxiv/Products, (b) empirically test AD-SSL at Papers100M ourselves, or (c) frame AD-SSL as orthogonal to the contrastive/generative dichotomy (per-node depth mixture is its own axis).

## Reproduction note

Official code: `https://github.com/THUDM/GraphMAE2`. Phase 1 reproduction target for our harness: 71.95 ± 0.3 on ogbn-arxiv full-graph linear probe. Hyperparameters in Table 10. Fine-tuning 72.69 is nice-to-have but not needed for our Pareto figure.
