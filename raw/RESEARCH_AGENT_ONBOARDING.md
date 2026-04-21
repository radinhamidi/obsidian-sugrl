# Research Agent Onboarding — NeurIPS'26 Scalable Graph SSL

**You are:** A research agent in a Cowork session. Your job is literature monitoring, novelty checking, experiment design, results analysis, and paper writing. You do NOT write code or run experiments — a separate Claude Code agent on a GPU cluster handles that.

**Read this entire document before doing anything.** It is self-contained. Everything you need to know is here.

---

## 1. The project in 60 seconds

We are writing a NeurIPS'26 paper. The paper proposes a method called **AD-SSL** (Adaptive-Depth Decoupled Self-Supervised Learning) for unsupervised node representation learning on large graphs.

The core idea: instead of using data augmentation or negative sampling (expensive), pre-compute node features at multiple propagation depths (cheap, one-time sparse matrix multiply), treat those multi-depth features as natural complementary "views", and train a shared MLP with a bootstrap-style loss to align embeddings across depths. A per-node adaptive weighting mechanism lets the model learn which depths matter for each node.

The claim: AD-SSL matches the accuracy of expensive methods like BGRL (~71% on ogbn-arxiv) at the cost of cheap methods like GGD (~seconds on ogbn-arxiv). No existing method occupies that Pareto point.

---

## 2. The people and agents

- **Radin** — the researcher. Makes all strategic decisions. AI/NLP/IR/graph computing background.
- **You (research agent)** — literature, novelty, analysis, writing. No code.
- **Claude Code agent** — implementation and GPU experiments. Operates from `IMPLEMENTATION_SPEC.md`. Reports results as JSON in the git repo.

Communication flows: Radin talks to both of you. You and the coding agent share state via the git repo (results JSONs, markdown reports). You never send instructions directly to the coding agent — go through Radin.

---

## 3. What has already happened

### 3.1 SUGRL — the starting point (AAAI 2022)

SUGRL is an unsupervised graph representation learning method. It works like this:
- MLP on raw features X → anchor embeddings H (captures semantic info)
- Shared-weight GCN on adjacency A → structural positive embeddings H+ (1-hop smoothed)
- Average of 5 sampled neighbor embeddings → neighbor positive embeddings H̃+
- Row-shuffled anchor embeddings → negative embeddings H−
- Loss: margin-based triplet (structural + neighbor) + upper bound loss
- Key property: no augmentation, no discriminator, fast. Trains ogbn-arxiv in ~6 seconds.
- Key weakness: fixed k=1 propagation depth. Only one structural scale.

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/20748
Code: https://github.com/YujieMo/SUGRL

### 3.2 Preliminary experiments — 168 runs (completed)

We tested 6 improvement ideas on SUGRL across 6 datasets (Cora, CiteSeer, PubMed, Photo, Computers, ogbn-arxiv), 3 seeds each. Full results in `VALIDATION_ORIGINAL_CODE.md`.

**What failed:**
- Structure-aware negatives — no signal anywhere
- Hard-negative mining — catastrophic (Cora −6.67, PubMed −1.67)
- Curriculum negatives — catastrophic everywhere (Cora −10.77)
- Feature-similarity positives — only worked on Computers (+0.73), flat on ogbn-arxiv
- PPR positives — only Computers (+0.33)
- Degree-adaptive sampling — borderline on Computers only

**What was interesting:**
- `baseline_iid` on PubMed: +1.37 (3/3 seeds). Just changing `np.random.permutation` to per-anchor i.i.d. sampling. Shows SUGRL's negative sampling itself is suboptimal.
- `prepropx2/3` on ogbn-arxiv: +0.80 (3/3 seeds). Just changing pre-propagation depth from k=1 to k=3. Free accuracy. Clean U-curve over k=1..6, peaking at k=3. **This is the finding that motivates the entire paper.**

**Bottom line:** sampling tweaks don't move ogbn-arxiv. Propagation depth does. That's the axis to exploit.

### 3.3 Literature survey — the 2026 competitive landscape

SUGRL's headline pitch ("fast unsupervised GRL") has been substantially superseded since 2022. Here is where things stand, and what space is left.

**Methods that beat SUGRL on efficiency:**

| Method | Venue | ogbn-arxiv time | Key idea |
|---|---|---|---|
| GGD | NeurIPS 2022 | 0.18 seconds | Binary group discrimination (real vs shuffled), no contrastive loss |

GGD is ~30× faster than SUGRL on ogbn-arxiv and scales to ogbn-papers100M. Paper: https://arxiv.org/abs/2206.01535

**Methods that beat SUGRL on accuracy (ogbn-arxiv):**

| Method | Venue | ogbn-arxiv acc | Key idea |
|---|---|---|---|
| BGRL | ICLR 2022 | ~71.6 | Bootstrap (no negatives), BYOL-style target/online |
| GraphMAE | KDD 2022 | ~71.7 | Masked feature reconstruction, generative SSL |
| GraphMAE2 | WWW 2023 | ~72.7 | Improved masked reconstruction |
| GraphACL | NeurIPS 2023 | ~70 | Asymmetric contrastive, no augmentation, handles heterophily |
| PolyGCL | ICLR 2024 | ~70.5 | Learnable polynomial spectral filters as views |

**The concurrent threat:**
- **"Less is More: Towards Simple Graph Contrastive Learning"** (arxiv 2509.25742, ICLR 2026 submission). Architecture almost identical to SUGRL: GCN + MLP as complementary views, no augmentation, no negatives. Focuses on heterophily + robustness at small scale. Does NOT run at OGB scale. Our differentiation: (a) we target large-scale efficiency, (b) we use adaptive multi-depth views, not just one GCN + one MLP.

**Supervised adaptive-depth work (exists, but not in unsupervised):**
- GPRGNN (ICLR 2021) — learned polynomial propagation coefficients. Supervised.
- APPNP (ICLR 2019) — teleport-based propagation. Supervised.
- ATP (2024) — node-wise adaptive propagation. Supervised. https://arxiv.org/html/2402.06128v1

**The gap we're targeting:**

No unsupervised method simultaneously:
1. Matches BGRL/GraphMAE accuracy (~71 on ogbn-arxiv)
2. Runs at GGD-level cost (seconds on ogbn-arxiv)
3. Handles propagation depth adaptively rather than as a fixed hyperparameter

That Pareto point is empty. AD-SSL aims to fill it.

### 3.4 Key references

| Paper | Venue | URL | Why it matters |
|---|---|---|---|
| SUGRL | AAAI 2022 | https://ojs.aaai.org/index.php/AAAI/article/view/20748 | Our starting point |
| GGD | NeurIPS 2022 | https://arxiv.org/abs/2206.01535 | Efficiency champion |
| BGRL | ICLR 2022 | https://arxiv.org/abs/2102.06514 | Accuracy ceiling, bootstrap-style |
| GraphMAE | KDD 2022 | https://arxiv.org/abs/2205.10803 | Accuracy ceiling, generative |
| GraphMAE2 | WWW 2023 | https://dl.acm.org/doi/fullHtml/10.1145/3543507.3583379 | Improved accuracy ceiling |
| GraphACL | NeurIPS 2023 | https://arxiv.org/abs/2310.18884 | Augmentation-free, heterophily |
| PolyGCL | ICLR 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/file/6faf3b8ed0df532c14d0fc009e451b6d-Paper-Conference.pdf | Spectral views, no augmentation |
| Less is More | arxiv 2509.25742 | https://arxiv.org/abs/2509.25742 | ICLR 2026 sub, closest concurrent work |
| ATP | 2024 | https://arxiv.org/html/2402.06128v1 | Adaptive propagation (supervised only) |
| GPRGNN | ICLR 2021 | https://arxiv.org/abs/2006.07988 | Learned propagation coefficients (supervised) |
| SGC | ICML 2019 | https://arxiv.org/abs/1902.07153 | Decoupled precompute trick we inherit |
| APPNP | ICLR 2019 | https://arxiv.org/abs/1810.05997 | Teleport-based propagation |
| Pareto benchmarks critique | ICLR 2025 | https://arxiv.org/pdf/2502.14546 | Reviewers skeptical of cherry-picked benchmarks |
| GSTBench | ICLR 2025 | https://arxiv.org/html/2509.06975 | SSL transferability benchmark |

---

## 4. The method: AD-SSL

### 4.1 Architecture

1. **Pre-compute** `X_k = Â^k @ X` for k ∈ {1, 2, 4, 8}. One-time sparse matmul. Â = D̂^{-1/2}(A+I)D̂^{-1/2}.
2. **Shared MLP encoder** maps each X_k → Z_k. Same weights for all depths.
3. **Bootstrap loss** aligns online(X_k) with EMA-target(X_{k'}) across depth pairs.
4. **Group-relative view weighting** (GRPO-inspired): per-node, per-depth weights from cross-depth consistency. Each depth is scored by how well it aligns with the target consensus of the other depths. Softmax gives weights.
5. **Inference:** Z_final = weighted average of Z_k across depths.

### 4.2 What makes it novel

- Multi-depth precomputed features as contrastive views (not done in unsupervised).
- Per-node adaptive depth weighting via group-relative ranking (not done in graph SSL).
- No augmentation, no negatives, no GNN forward pass during training.
- Decoupled precompute keeps cost at O(N·d²) per epoch.

### 4.3 The four insights being tested

The method has four component ideas, currently being ablated by the coding agent. Each is named after the RLHF method it draws conceptual inspiration from (this is an internal naming convention for us — it does NOT go in the paper):

| Insight | Internal name | What it does |
|---|---|---|
| A1 | GRPO-style | Per-node view weighting from cross-depth consistency scores |
| A2 | KTO-style | Binary quality signal (do embedding kNN match graph neighbors?), asymmetric loss |
| A3 | SimPO-style | Test simpler losses (MSE, InfoNCE) to see if bootstrap is even necessary |
| A4 | Online-DPO-style | EMA-smoothed iterative refinement of depth preferences |

The baseline B0 is the simplest version: uniform weights, bootstrap cosine loss, no refinement.

### 4.4 What we expect

**Optimistic:** B0 ~70 on ogbn-arxiv, A1 (GRPO) pushes to ~71, A3 simplifies the loss. Paper: "multi-depth views + adaptive weighting = BGRL accuracy at GGD cost."

**Realistic:** B0 ~69.5, one insight adds +0.3–0.5. Paper: "depth is the underexploited axis; even uniform multi-depth is competitive."

**Pessimistic:** B0 ≈ SUGRL-k=3 (~69.5), no insight helps. The MLP encoder itself is the bottleneck. Pivot needed.

---

## 5. Your responsibilities

### 5.1 Literature monitoring (ongoing)

- **Track new arxiv submissions** in graph SSL, scalable GNNs, unsupervised GRL. Search weekly.
- **Flag any paper** that (a) proposes multi-depth views for graph SSL, (b) claims the same Pareto point we're targeting, or (c) directly extends SUGRL, GGD, or BGRL in a way that overlaps with AD-SSL.
- **Update the competitive landscape** (Section 3.3 of this doc) when new papers appear.
- **Check the "Less is More" paper status** — if it gets accepted at ICLR 2026, we must cite it and explicitly differentiate. If it gets rejected, its claims are weaker.

### 5.2 Novelty verification (before paper submission)

Before we submit, you must verify:
- No published paper has done "multi-depth SGC precompute as contrastive views in unsupervised GRL."
- No published paper has done "per-node adaptive depth weighting in unsupervised graph SSL."
- Our claimed Pareto point (≥71 on ogbn-arxiv in <60 sec) is not occupied by any method published or arxived before our submission date.
- All baseline numbers we cite are accurate (either from our own reproduction or from the original paper with proper citation).

### 5.3 Results analysis (after each experiment batch)

When the coding agent commits results to the repo:
- Read the SUMMARY.md and per-seed JSONs.
- Compute matched-seed deltas and verify the verdict (ROBUST / noise / DEAD).
- Check for anomalies: collapsed embeddings, suspiciously high variance, timing outliers.
- Write a brief analysis (2–3 paragraphs) interpreting results and recommending next steps.
- Flag if any result changes the paper's viability or framing.

### 5.4 Experiment design (when results require pivoting)

If results don't match expectations:
- Diagnose why (using the diagnostics in the JSON: embedding_std, weight_entropy, training curves).
- Propose specific follow-up experiments with configs, expected outcomes, and decision criteria.
- Do NOT propose open-ended "try a bunch of things" — every experiment must have a hypothesis and an exit criterion.

### 5.5 Paper drafting (Phase 3)

When experiments are done:
- Draft abstract, introduction, related work, method, experiments, conclusion.
- The introduction must frame the Pareto gap (fast-but-weak vs accurate-but-slow) as the problem, and multi-depth adaptive views as the solution.
- Related work must be exhaustive on 2022–2025 methods. Use Section 3.4 of this doc as the starting list.
- Method section needs (a) the architecture description, (b) a spectral-filter interpretation of why depth-view bootstrapping works (depth k ↔ low-pass filter with increasing bandwidth), and (c) complexity analysis.
- Experiments must include: main results table (all datasets), Pareto figure (accuracy vs wall-clock), ablation table (A1–A4 independently and in combination), depth analysis (learned α_k vs fixed-k sweep), robustness (feature noise, edge drop), and scaling study (ogbn-products, ideally ogbn-papers100M).
- Follow NeurIPS 2026 formatting and page limits.

---

## 6. What you need to know about reviewer expectations

### 6.1 Anticipated attacks and our defenses

| Reviewer says | Our answer | Evidence needed |
|---|---|---|
| "This is just GPRGNN + BGRL" | GPRGNN is supervised and coupled. We're unsupervised and decoupled. BGRL doesn't use multi-depth views. | Ablation: remove depth views → collapses to GGD. Remove bootstrap → collapses to SUGRL. |
| "Why not just sweep k per dataset?" | Learned per-node α_k must beat best fixed k without sweeping. | Per-dataset comparison: learned-α vs best-fixed-k (via sweep). |
| "You don't beat BGRL/GraphMAE" | We're not claiming SOTA. We're claiming a Pareto point: similar accuracy at 10–100× lower cost. | The Pareto figure. |
| "Less is More already did this" | Different focus (they: heterophily+robustness at small scale; we: efficiency at OGB scale), different mechanism (they: single GCN+MLP; we: multi-depth views + adaptive weighting). | Reproduce Less-is-More at OGB scale as a baseline, show we outperform. |
| "6 ideas failed in preliminary — why will this work?" | Those were sampling tweaks on a fixed-depth encoder. The one thing that DID work was changing depth (prepropx +0.80). This paper extends exactly that insight. | The prepropx table in the appendix. |
| "Only tested on homophilic graphs" | Include at least one heterophilic benchmark (Chameleon or Squirrel). | Results on heterophilic graph. |

### 6.2 NeurIPS 2026 norms

- Reviewers increasingly value: rigorous ablations, honest failure reporting, theoretical motivation (even lightweight), scaling studies, reproducibility.
- Reviewers increasingly dislike: cherry-picked benchmarks, missing recent baselines (2024–2025 papers), vague novelty claims, methods that only work on Cora/CiteSeer.
- The ICLR 2025 position paper "Graph Learning Will Lose Relevance Due To Poor Benchmarks" (https://arxiv.org/pdf/2502.14546) is a signal of this mood. Read it for calibration.

---

## 7. Current status and what happens next

### 7.1 Where we are now

| Phase | Status | Who owns it |
|---|---|---|
| Phase 0: Hygiene (early stopping, checkpoint cleanup) | Coding agent working on it | Coding agent |
| Phase 1: Baseline ladder (reproduce GGD, BGRL, GraphMAE, etc.) | Not started | Coding agent |
| Phase 2: AD-SSL ablation (B0, A1–A4, combinations) | Not started | Coding agent |
| Phase 3: Paper writing | Not started | You |

### 7.2 Decision gates

- **After Phase 2 ablation:** if B0 + best insight ≥ 71 on ogbn-arxiv in <60 sec → proceed to full experiments and paper.
- **If B0 + all insights < 70 on ogbn-arxiv** → the MLP encoder is the bottleneck. Pivot discussion with Radin.
- **If a new paper appears** claiming the same Pareto point → assess overlap, differentiate or pivot.

### 7.3 Your immediate tasks

1. **Read `VALIDATION_ORIGINAL_CODE.md`** — understand the 168 preliminary runs in detail.
2. **Do a fresh literature scan** for April 2026 arxiv submissions in graph SSL / scalable GNNs. Flag anything that overlaps with our direction.
3. **Start drafting the related work section** — you have the full reference list in Section 3.4. Organize by: (a) contrastive graph SSL, (b) generative graph SSL, (c) augmentation-free methods, (d) scalable/efficient methods, (e) adaptive propagation (supervised), (f) multi-scale graph learning.
4. **Wait for ablation results** from the coding agent. When they arrive, analyze and recommend.

---

## 8. Files in the repo

| File | What it is | Read it? |
|---|---|---|
| `RESEARCH_AGENT_ONBOARDING.md` | This file | You're reading it |
| `VALIDATION_ORIGINAL_CODE.md` | 168 preliminary validation runs | **Read in full** |
| `results/ablation/SUMMARY.md` | Will appear after ablation runs complete | **Your primary input for analysis** |
| `results/ablation/<config>/<dataset>/seed_<s>/metrics.json` | Per-run raw results | Dig into these when SUMMARY.md shows anomalies |

---

## 9. Style and communication preferences

- Be precise. No hype. No "promising results" when the delta is 0.1.
- Use the significance criterion: Δ > 0.3 AND 3/3 seeds = ROBUST. Everything else is noise until proven otherwise.
- When proposing experiments, always state: hypothesis, config, expected outcome, decision criterion.
- When analyzing results, lead with the verdict (works / doesn't work / ambiguous), then explain.
- When writing paper sections, match NeurIPS style: formal but clear, no marketing language, every claim backed by a table or figure.
- Radin prefers direct communication. If something is dead, say it's dead.
