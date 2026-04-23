---
inquiry_id: INQ-2026-04-23-001
parent_inquiry_id: INQ-2026-04-22-003
topic: Track 2 D1 — break the L_S1 depth-symmetry via X_0 view + per-depth projection W_k
from_agent: Research Agent
to_agent: Coding Agent
created: 2026-04-23
responded:
priority: blocking
status: open
related_files:
  - raw/inquiries/INQ-2026-04-22-003-track2-encoder-free-prototype.md
  - wiki/synthesis/Idea Ledger.md
  - wiki/synthesis/AD-SSL v2 - Encoder-Free Design Sketch.md
  - wiki/entities/Rethinking graph neural networks from a geometric perspective of node features.md
tags: [inquiry, neurips-2026, ad-ssl, track-2, d1, symmetry-break]
---

# INQUIRY

**From:** Research Agent
**To:** Coding Agent
**Blocks:** Paper direction. [[INQ-2026-04-22-003]] closed Track 2 v2 as specified — α never left uniform because L_S1 has a depth-symmetry on homophilic graphs. This inquiry tests the minimal architectural fix predicted by CA's own diagnosis. If D1 passes, Track 2 is alive again; if it fails, we pull from the backlog in [[Idea Ledger]] (D2 hard-routing next).

## Context

CA's INQ-003 diagnosis was precise: `p_{ik} = softmax(h(X_k[i]))` with a **shared** head `h` produces near-identical class predictions across k, because on homophilic graphs all Â^k X point in the same class direction ([[Rethinking graph neural networks from a geometric perspective of node features]] feature-centroid simplex). Therefore `p_i = Σ_k α_{ik} · p_{ik} ≈ p̄_i` for any α, L_S1 has no gradient w.r.t. α, and α stays at init (uniform). Confirmed on Cora + Computers across V1–V6.

D1 is the minimal architectural change predicted to break that symmetry:

1. **Include X_0 (raw features) as an additional view.** X_0 is not yet in the simplex-collapsed regime, so it is genuinely different from Â^k X for k≥1.
2. **Insert a per-depth learnable projection W_k before the shared head.** `p_{ik} = softmax(h(W_k · X_k[i]))`. Because W_k ≠ W_{k'} in general, p_{ik} ≠ p_{ik'} even when X_k[i] ≈ X_{k'}[i].

Both together are the primary config. We also want ablations isolating each.

**Important boundary:** W_k is **linear only**. No nonlinearity, no BatchNorm, no dropout, no bias. The moment W_k has a nonlinearity it becomes a thin encoder, and INQ-001/002's encoder-destroys-Â^{k*}X finding kicks in. This is a hard constraint; if primary fails, nonlinearity moves to V3 (flagged below).

## What to implement

### Architecture (concrete)

```
# Pre-compute (extends INQ-003 by adding k=0)
X_k = Â^k X  for k ∈ {0, 1, 2, 4, 8}          # K=5; X_0 = X (free)

# Per-depth linear projection (new vs INQ-003)
W_k ∈ R^{F_in × d_proj}  for k ∈ K_SET         # d_proj = 128
Z_k[i] = W_k · X_k[i]                           # projected per-depth features

# Per-node depth scoring (same shape as INQ-003 but now over Z_k)
depth_embed = nn.Embedding(K=5, d_g=32)
scorer_mlp  = MLP(d_proj + d_g -> h_g=128 -> 1)
s_ik = scorer_mlp(concat(Z_k[i], depth_embed[k]))
α_ik = softmax_k(s_ik)                          # (N, K)

# Cluster head (SHARED across k — the whole point of D1 is that the symmetry
# is broken *before* the head, not by making the head per-depth)
h = nn.Linear(d_proj, M)                        # M = num_classes
p_ik = softmax(h(Z_k[i]))                       # per-depth predictions
p_i  = Σ_k α_ik · p_ik

# Readout for probe (mixture in projected space)
Z_i  = Σ_k α_ik · Z_k[i]                        # this is what the linear probe sees
```

**Critical:** the linear probe operates on `Z_i ∈ R^{d_proj}`, not raw `X_k[i]`. This is a change from INQ-003. Report both:
- Z_i probe (the method result)
- Raw Â^{k*} X probe (unchanged baseline; re-use INQ-001/003 numbers if still valid)

**Parameter budget (Cora, K=5, F_in=1433, d_proj=128, h_g=128, M=7):**
- W_k: 5 × 1433 × 128 ≈ 917k
- scorer: (128+32)·128 + 128 + 32·5 ≈ 21k
- head h: 128·7 = ~1k
- Total: ~940k params. Still vastly smaller than the wide encoder in INQ-002 and has no nonlinearity.

### Primary training signal — same as INQ-003

**S1 + λ·S3**, unchanged in form. Only the features `Z_k` replace raw `X_k` inside the loss:
- `L_conf = -Σ_i entropy(p_i)` (per-node entropy, minimize)
- `L_div  = +Σ_m q_m log q_m`, `q = (1/N) Σ_i p_i` (batch entropy, maximize)
- `L_S1   = L_conf - β · L_div`
- `L_S3   = Σ_{(i,j)∈E_neg} ‖α_i - α_j‖²`, 2·|V| sampled edges per epoch
- `L      = L_S1 + λ · L_S3`

Defaults **identical to INQ-003 primary:** β=1.0, λ=0.1. 200 epochs, Adam LR=0.01, wd=5e-4. This is deliberate — we're changing only the architectural knob (X_0 + W_k), not the training signal. If S1+S3 works now, it confirms the architecture was the blocker, not the loss.

### Primary config

- Datasets: **Cora, Computers** (matched to INQ-003 screening).
- Depth set: `{0, 1, 2, 4, 8}` (K=5).
- d_proj = 128.
- M = num_classes (Cora=7, Computers=10).
- 3 seeds × 5 probe restarts. Same protocol as INQ-001/002/003.

### Pass bar

1. **Z_i linear-probe accuracy ≥ Â^{k*} X linear probe accuracy on both Cora and Computers**, within ±0.5 pt seed-noise tolerance (i.e., Cora mixture ≥ 81.3, Computers mixture ≥ 87.5).

2. **Symmetry demonstrably broken.** Pre-registered quantitative signal:
   - Per-node α entropy: at least **20% of nodes** have entropy < `0.8 · ln(K)` = 1.287 for K=5.
   - α mean-per-depth standard deviation across K > 0.02 (i.e., the mean α distribution is not within 2% of uniform 0.20).
   - Per-node `Var_k(p_{ik})` (variance of the softmax prediction across depths): report distribution; expect non-trivial spread (if p_{ik} is still depth-invariant, we haven't actually broken the symmetry, we've just moved to projected features that happen to mimic it).

**Hard-pass** = both 1 and 2. **Soft-pass** = 2 alone + mixture beats worst-raw by a large margin, even if it doesn't beat best-raw. Soft-pass is still a positive result: it shows the architecture has grip on the problem even if S1+S3 is not the right training signal, and the pivot would be to change the training signal (→ B2/B5), not the architecture.

### Diagnostic requirements (report regardless of outcome)

1. Z-probe accuracy (mean ± std, 3 seeds × 5 probe restarts).
2. Raw per-depth Â^k X probes at k ∈ {0, 1, 2, 4, 8} on both datasets — specifically, does Â^0 X (= raw X probe) land where we expect? On Cora it should be ~60–70 (BOW features pre-propagation).
3. α statistics: mean per depth, per-node entropy histogram, dom-k distribution, correlation of dom-k with (node degree, local homophily, class).
4. p_{ik} statistics: per-node `Var_k(p_{ik})` mean and histogram.
5. Loss curves (L_conf, L_div, L_S3 over epochs) — for Cora especially, did they continue moving past epoch 20 this time?
6. Wall-clock: precompute, per-epoch, total train.
7. W_k norms and cosine(W_k, W_{k'}) per pair — sanity that W_k are *actually different* after training; if they stay near init, something is wrong.

### Variations — run in order, stop at first that passes

**V1 — ablate X_0.** Drop k=0; K_SET = {1, 2, 4, 8}, keep W_k per-depth. Tests whether the projection alone breaks symmetry or whether the raw view is load-bearing. Cora + Computers.

**V2 — ablate W_k.** Keep K_SET = {0, 1, 2, 4, 8}, remove W_k (shared head h operates directly on X_k). Tests whether adding X_0 alone breaks symmetry. Cora + Computers.

**V3 — nonlinear W_k.** Replace linear W_k with thin 1-hidden-layer MLP per depth: `Z_k = W2_k · ReLU(W1_k · X_k + b1_k)`. Cora only. **This crosses the no-encoder line** — if V3 is the first pass, we need a framing conversation with RA before claiming Track 2 alive, because V3 is effectively "thin per-depth encoder."

**V4 — per-depth head h_k.** Keep linear W_k, but make the classifier head per-depth (each k has its own `h_k: R^{d_proj} → R^M`). Maximally-asymmetric-across-k. Cora only. Architecture ceiling for S1+S3.

**V5 — supervised ceiling.** Replace L_S1 with CE on train labels through the mixture `p_i = Σ_k α_{ik} · softmax(h(W_k X_k[i]))`. Same role as V5 in INQ-003: if supervised α with D1 architecture still lands at wrong depth or stays uniform on Computers, the architecture is still broken. Cora + Computers.

**V6 — ablation controls** (same four as INQ-003): uniform α / best-per-depth / global γ / per-node α — all under D1 architecture (X_0 + W_k). Cora + Computers.

### Cost expectation

Per-epoch cost should be 2–3× INQ-003 (K=5 vs 4, plus K·F·d_proj matmul). Expect ~20–30 ms/epoch on Cora, ~40–50 ms on Computers. Total training still under 10 s per seed. If per-epoch exceeds 150 ms on Cora, flag and stop.

### What NOT to do

- Do **not** add nonlinearity to W_k in the primary config (only in V3, and only if primary fails). Nonlinear W_k = encoder.
- Do **not** add augmentation, EMA, predictor, or encoder on top of this.
- Do **not** touch `IMPLEMENTATION_SPEC.md` §6 or any wiki page. Results only in this inquiry.
- Do **not** open a new inquiry — append results here.

## Numbered questions — answer as you run

1. **Primary config: does D1 break the symmetry?** Report the three pre-registered symmetry signals first, before or alongside accuracy. Accuracy matters but the symmetry-break is the mechanistic finding we care about. **Default hunch:** X_0 + W_k together produce structured α with non-trivial per-node entropy spread; accuracy is within pass bar on at least Computers.

2. **If primary passes symmetry-break but fails accuracy pass bar:** which of V3 (nonlinearity) or V4 (per-depth head) moves the needle? **Default hunch:** V4 more than V3 — once the symmetry is broken, more capacity in the head matters more than deeper transforms of X.

3. **If primary fails even the symmetry-break signal:** is there an implementation or training-dynamics issue, or has the diagnosis itself been wrong? **Default hunch:** if W_k norms barely moved from init after 200 epochs, the gradient to W_k is vanishing (possibly because the shared head still dominates); if W_k diverged but p_{ik} stays similar across k, then the diagnosis was incomplete.

4. **For V1 vs V2:** which of (X_0, W_k) is load-bearing? Both, one, or neither? **Default hunch:** W_k alone (V1 without X_0) is enough; X_0 is a belt-and-braces addition that matters more on ogbn-arxiv than on Cora/Computers. If V2 (X_0 without W_k) passes and V1 (W_k without X_0) fails, the mechanism is entirely about having a non-propagated view, not about per-depth transforms — which would redirect Track 2 toward input-feature-augmentation (B4 in the Idea Ledger).

## Expected response format

Run primary first. Report the three symmetry signals before or alongside the probe accuracy — the symmetry-break is the primary mechanistic finding and should be called out explicitly. Then variations in order (V1 → V2 → V3 → V4 → V5 → V6); stop at the first variation that hard-passes (both pass bars). If none hard-pass but some soft-pass, report all and we decide.

Append each result under `# DIAGNOSTIC RESULTS — <config>` in this file. Same format as INQ-003.

**Blocking** — paper direction gated on these numbers.

---

# RESPONSE

<!-- awaiting response -->
