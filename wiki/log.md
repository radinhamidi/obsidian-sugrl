# SUGRL Wiki Log

Append-only chronological log of wiki activity. Each entry starts with
`## [YYYY-MM-DD] <event> | <title>` for easy parsing
(`grep "^## \[" log.md | tail -20`).

Event types: `ingest`, `query`, `lint`, `inquiry-open`, `inquiry-answer`,
`inquiry-close`, `experiment`, `note`.

---

## [2026-04-21] note | Vault initialized

Vault scaffolded. Directory structure, `CLAUDE.md` schema, inquiry template,
and this log created. Remote: `github.com/radinhamidi/obsidian-sugrl`. To be
mounted as a submodule inside the implementation repo where the Coding Agent
operates.

## [2026-04-21] ingest | RESEARCH_AGENT_ONBOARDING

Project brief ingested. Defines method **AD-SSL** (Adaptive-Depth Decoupled
SSL), NeurIPS 2026 target, researcher/agents roles, Pareto thesis (match
BGRL accuracy at GGD cost), and four-insight ablation plan (B0 + A1–A4).

Created: [[Thesis]], [[Competitive Landscape 2026]], [[Pareto Gap]],
[[Reviewer Attacks and Defenses]], [[Project Phases and Decision Gates]],
[[Ablation Plan - AD-SSL B0 A1-A4]], [[AD-SSL]], entity stubs for all 12
landscape methods + [[ogbn-arxiv]], concept pages for
[[Decoupled Precompute]], [[Multi-Depth Views]], [[Adaptive Depth Weighting]],
[[Bootstrap Loss]], [[Matched-Seed Delta]], [[Oversmoothing]].

## [2026-04-21] ingest | VALIDATION_ORIGINAL_CODE

168-run preliminary report ingested. All 6 brainstorm sampling-ideas fail on
ogbn-arxiv; only **prepropx (k=1→k=3) +0.80** is ROBUST and motivates the
entire paper. Feature-sim positives +0.73 on Computers only; curriculum/hard
negatives catastrophic. Baselines reproduce paper within 1–2σ.

Created: [[Preliminary Validation - 168 Runs]], [[Prepropx Depth Finding]].

## [2026-04-21] ingest | SUGRL (paper PDF)

Deepened [[SUGRL]] entity page with exact multiplet loss (ω1·L_S + ω2·L_N + L_U), shared MLP/GCN weights confirmation, and the **SUGRL-batch 69.3** number from Table 2 (overlooked in onboarding — flagged in page). No thesis-level changes.

## [2026-04-21] ingest | Less is More (paper PDF)

Major update. Paper is v3 (2026-03-20) with full author list (Zhao, Ji, Dai, Ma, Tay @ NTU). Mechanism is 2-view: 1 MLP + 1 k-layer GCN, global β, direct cosine loss. **They evaluate at 169k-node scale (Arxiv-year)** — softens our onboarding's "not at OGB scale" framing. Differentiation still solid: K vs 2 views, per-node α vs global β, bootstrap vs direct cosine. Created [[AD-SSL vs Less is More]] synthesis page; updated [[Reviewer Attacks and Defenses]] with honest framing and new evidence gaps (reproduce at standard ogbn-arxiv, 2-view ablation of AD-SSL).

## [2026-04-21] ingest | ATP (paper PDF)

Deepened [[ATP]] entity page. Mechanism is 2-part: High-Deg edge masking (HPC) + weight-free scalar r̃ per node via degree+eigenvector+clustering in the kernel `D̂^(r̃−1) Â D̂^(−r̃)`. **Supervised, single kernel, scalar-per-node.** Our K-view per-node-per-depth α mixture in SSL regime is distinct. Added dedicated reviewer-defense row for "this is just ATP in SSL." ATP's HPC flagged as possible preprocessing adoption (open question).

## [2026-04-21] ingest | GGD (paper PDF)

Deepened [[GGD]] entity page with exact numbers. **Critical finding**: GGD at hidden=1500 reaches **71.6 ± 0.5 in 0.95s** on ogbn-arxiv (Table 8) — our onboarding's framing of "BGRL accuracy at GGD cost" as an empty Pareto point is too loose. Updated [[Pareto Gap]] frontier sketch, [[ogbn-arxiv]] competitive table, and added a reviewer-defense row for "GGD-1500 already closed this gap." Also noted GGD's inference-time `H_final = H_θ + A^5 H_θ` trick — a weak, post-hoc single-depth version of our multi-depth views; strengthens our motivation but also raises "is multi-depth just GGD's power trick scaled up?" as a defense-needed question.

ogbn-papers100M numbers captured: GGD 63.5 test in 9h15m vs BGRL 62.1 in 26h28m — GGD is the right cost anchor at OGB-LSC scale.

## [2026-04-21] ingest | BGRL (paper PDF)

Deepened [[BGRL]] entity page with exact loss (`-cos(Z̃_1, H̃_2)`, symmetrized), complete ogbn-arxiv hyperparameters (3-layer GCN hidden 256, predictor MLP, **pf=0.0 pe=0.6** — edge-only aug, LayerNorm + weight standardisation, AdamW 1e-2, τ cosine 0.99→1.0, 10k steps). **No projector** — only a predictor, contra BYOL. 4 encoder forwards per step. Test: 71.64 ± 0.12 (val 72.53), essentially matching supervised GCN.

Actionable for AD-SSL: (1) our bootstrap loss should exactly mirror BGRL's form applied across depth pairs; (2) drop the projector, keep only predictor; (3) adopt their three-plot collapse-monitoring protocol (loss / embedding spread / norm); (4) use LayerNorm not BatchNorm on ogbn-arxiv; (5) cosine τ schedule. Also flagged: BGRL frozen-eval underperforms LabelProp on MAG240M — a cautionary note for our scale-study framing.

## [2026-04-21] ingest | GraphMAE (paper PDF)

Full PDF read (via `pdftotext`; poppler just installed). Rewrote [[GraphMAE]] with exact mechanism (4 design choices: mask input, GNN encoder, re-mask code, GNN decoder), exact Scaled Cosine Error `(1 − cos(x_i, z_i))^γ` Eq. 2, and ogbn-arxiv hyperparams (Tbl 7): **GAT encoder, hidden 1024, mask 0.5, γ=3, 1000 epochs Adam lr=1e-3 cosine, PReLU**. Key numeric correction: **GraphMAE-GCN on ogbn-arxiv is 71.87 ± 0.21** (Tbl 6 appendix, §A.2) — *higher than GraphMAE-GAT's 71.75 ± 0.17* which the main table uses. We should cite the GCN number for the ceiling. Scaled cosine error's γ > 1 sample-reweighting is flagged as a candidate knob for AD-SSL's bootstrap loss (down-weight trivially-aligned depth pairs).

## [2026-04-21] lint | Audit of GGD + BGRL pages against PDFs (post-poppler install)

After installing poppler, re-read both PDFs and audited the pages written earlier in the session. Two factual errors found and corrected:

1. **[[GGD]]** — claimed the inference-time graph power uses `n = 5` for all datasets. The paper defines `H_global = A^n H_θ` but does not pin n in the main experiments. Appendix A.6 uses n=10 for graph-power *timing*, and per-dataset hyperparameter tables do not list n. Fixed: treat n as an unfixed per-dataset hyperparameter; note the Appendix value.
2. **[[BGRL]]** — claimed "BGRL full-batch OOMs on 16GB V100 on ogbn-arxiv." Wrong: Table 5 of the BGRL paper runs BGRL full-graph with a 3-layer GCN on a 16GB V100 and only GRACE full-graph OOMs. The batched-BGRL setup is a detail of the *GGD* paper's reproduction (§5.2), not a BGRL-paper limitation. Fixed.

Other claims (exact losses, Table 5/8/Tbl-7 numbers, MAG240M + LabelProp Appendix J, complexity `6·C_enc·(M+N) + 4·C_pred·N + C_BGRL·N`, ATP's `R̃ = C·(R_dg + R_ev + R_cu)` Eq. 11, Less is More's GCN-MLP architecture + β aggregation, SUGRL Table 2 numbers) all verified against PDFs.

Going forward: all PDF ingests go through `pdftotext -layout <pdf> /tmp/<name>.txt` — Read tool's sandbox can't see `/opt/homebrew/bin`.

## [2026-04-21] ingest | GraphMAE2 (paper PDF)

Full rewrite of [[GraphMAE2]] from PDF (arXiv:2304.04779, WWW 2023). Two mechanisms: (1) **multi-view random re-mask decoding** — re-mask encoded `H` K=3 times with `[DMASK]` token, each forced to reconstruct input via SCE; (2) **latent representation prediction** — BYOL-style EMA target generator on unmasked graph, predicted in projector space. Total loss `L_input + λ·L_latent`, λ ∈ {0.1, 5, 10} per dataset. Scale via **PPR-Nibble local clustering** (Thm A.1, linear-time, dense subgraphs) — beats GraphSAINT +0.63 / Cluster-GCN +2.24 on Products.

Numbers captured: **ogbn-arxiv full-graph linear probe 71.95 ± 0.08** (Tbl 9; matches supervised GAT 72.10), mini-batch 71.89 (Tbl 3), fine-tune 72.69 (Tbl 8). **Products 81.59** (+2.70 over GraphMAE — biggest gap). **Papers100M 64.89** (beats GGD's 63.50; all contrastive methods GRACE/BGRL/CCA-SSG *underperform Random-Init 61.55* at that scale — a big flag for AD-SSL's scale-story).

Hyperparams (Tbl 10, all OGB): hidden 1024, 4-layer GAT, mask 0.5, re-mask 0.5, num_re-masks 3, AdamW cosine no-warmup. Key ablation (Tbl 6): removing latent pred costs more than removing re-mask (−1.91 vs −0.73 on Papers100M), but removing *input reconstruction* collapses everything — **feature reconstruction is load-bearing, latent BYOL is auxiliary**. Opposite framing from BGRL. Direct design guidance for AD-SSL's bootstrap loss placement.

Contradiction flag raised on [[Pareto Gap]]: contrastive methods fail at Papers100M per Tbl 3 — we should scope AD-SSL claims to arxiv/Products or empirically verify at Papers100M.

## [2026-04-21] note | Tier 1 ingest pass complete (7/7)

Wiki now has ~30 pages covering the thesis, competitive landscape,
preliminary results, and reviewer-defense map. All Tier 1 papers
(SUGRL, Less is More, ATP, GGD, BGRL, GraphMAE, GraphMAE2) ingested
from source PDFs with verified numbers. Ready for Coding Agent to
start Phase 1 (baseline ladder) and for literature monitoring to
begin.

Open follow-ups:
- Ingest the actual paper PDFs when they land in `raw/papers/` (currently
  only project docs are in `raw/`). Entity pages are stubs pending that.
- First literature scan (April 2026 arxiv).
- Draft a [[Novelty Verification Checklist]] page (referenced but not yet
  written).

## [2026-04-21] ingest | Tier 2 batch — SGC, APPNP, GPRGNN, GraphACL

Four Tier 2 entity pages rewritten from source PDFs.

- **[[SGC]]** (ICML 2019): Verified exact form `softmax(Sᴷ X Θ)` with `S = D̃^(−½) Ã D̃^(−½)`; spectral filter `ĝ(λ̃_i) = (1−λ̃_i)^K`. This is the direct precursor of AD-SSL's precompute. Community-reported ogbn-arxiv number: SGC ≈ 66.92 (from [[GraphMAE2]] Table 3); not in original paper.
- **[[APPNP]]** (ICLR 2019): teleport-α power iteration `Z⁽ᵏ⁺¹⁾ = (1−α)ÂZ⁽ᵏ⁾ + αH`, fixed point = personalized PageRank. **Geometric mixture over depths with weight `α(1−α)^k`** — the cleanest positioning ancestor for AD-SSL, which learns a per-node, non-monotone version of this mixture. 100-run bootstrap eval protocol worth adopting.
- **[[GPRGNN]]** (ICLR 2021): **Critical prior art.** Exact `Z = Σ_k γ_k H⁽ᵏ⁾` with **learned global γ_k**, allowing negative weights for heterophily. APPNP and SGC are special cases. The main reviewer threat — AD-SSL's differentiation is per-node α (vs global γ), SSL signal (vs supervised), and decoupled cost structure. Must implement "global-γ SSL" as a required ablation.
- **[[GraphACL]]** (NeurIPS 2023): asymmetric BYOL without augmentations, captures 2-hop monophily implicitly. **Matches BGRL on ogbn-arxiv (71.72 ± 0.26) without augmentation, and wins all heterophilic benchmarks** (Squirrel 54, Chameleon 69, Texas 71). Augmentation-free peer; differentiation must be cost (AD-SSL has zero GNN forwards).

Key downstream implications captured in pages:
1. Must run "global-γ SSL" (GPR-GNN-in-SSL-mode) as an ablation, per [[GPRGNN]] page.
2. Must compare against GraphACL at 71.72 on ogbn-arxiv in Phase 2 accuracy gate.
3. Heterophily framing: either scope out or run AD-SSL on Squirrel/Chameleon/Actor.
4. Eval protocol: 100-run bootstrap CIs, paired t-tests (APPNP style); statistical-significance-first reporting per [[Graph Learning Poor Benchmarks]].

## [2026-04-21] note | PolyGCL ingest BLOCKED — wrong PDF

`raw/papers/PolyGCL.pdf` is mis-labeled: the file is a quantum-physics paper (arXiv:2401.14853, "Dimensional gain in sensing through higher-dimensional quantum spin chain"). The intended paper is Chen, Gao, Wang, *PolyGCL* (ICLR 2024, arXiv:2402.15680). Entity page updated to reflect the blocker; claims it previously contained (~70.5 on ogbn-arxiv, spectral-filter framing) flagged as unverified. **Action needed: researcher should replace the PDF.**

## [2026-04-21] ingest | Tier 3 calibration papers

Created [[Graph Learning Poor Benchmarks]] and [[GSTBench]] as source pages (position/benchmark papers, not entities).

- **Poor Benchmarks** (Bechler-Speicher et al., 2025, arXiv:2502.14546): Position paper from a multi-institution group (Oxford/Technion/RWTH/Google). Condemns marginal-gain accuracy-only reporting. Our Pareto framing aligns; must report 95% CIs + paired t-tests and include unstructured-set baselines.
- **GSTBench** (Song et al., CIKM 2025): Empirical finding that **contrastive graph SSL does not transfer** across datasets when pretrained on ogbn-papers100M; only GraphMAE shows reliable transfer. Reinforces the [[GraphMAE2]] contradiction flag — AD-SSL (bootstrap-family) should not claim foundation-model-style transferability without explicit evidence. Suggests adding a reconstruction term to AD-SSL's loss is a hedge worth testing.

## [2026-04-21] note | All-tier ingest pass complete

Tier 1 (7/7) + Tier 2 (4/5, PolyGCL blocked) + Tier 3 (2/2) done. Wiki now has direct evidence-based characterisations of every method in the competitive landscape plus two calibration papers. Major downstream decisions to surface to researcher:
1. Fix `raw/papers/PolyGCL.pdf` (wrong file).
2. Decide heterophily scope (include Squirrel/Chameleon or scope out).
3. Decide transferability scope (single-dataset Pareto claim vs transfer study).
4. Adopt 100-run bootstrap protocol with statistical testing as default for all reported numbers.

Next: update [[index.md]], [[Pareto Gap]], [[Reviewer Attacks and Defenses]] to reflect new findings; write [[Novelty Verification Checklist]].

## [2026-04-21] ingest | PolyGCL (corrected PDF)

Researcher replaced the mislabeled PDF; re-ingested from `/tmp/polygcl.txt` (Chen, Lei, Wei, RUC — ICLR 2024, arXiv:2402.15680). Fully rewrote [[PolyGCL]] with PDF-verified mechanism (Chebyshev reparam + prefix sum/diff for low-pass/high-pass monotonicity, DGI-BCE on both views), numbers (Table 2 Chameleon 71.62 / Squirrel 56.49 / Texas 88.03; Table 3 Roman-empire 72.97; Table 4 arXiv-year 43.07 ± 0.23), and differentiation from AD-SSL. **Withdrew prior "~70.5 on ogbn-arxiv" memory claim — PolyGCL does not report ogbn-arxiv, only arXiv-year.**

## [2026-04-21] audit | Session pages checked against PDFs

Spot-checked key numerical claims on all pages written since Tier-2/3 batch started against the extracted PDF text in `/tmp/*.txt`.

- **SGC** [[SGC]]: Cora 81.0±0.0, Citeseer 71.9±0.1, Pubmed 78.9±0.0, Reddit 94.9 — verified in paper Table 2/3.
- **APPNP** [[APPNP]]: Citeseer 75.73±0.30, Cora-ML 85.09±0.25, PubMed 79.73±0.31 — verified. **Correction**: MS Academic APPNP stderr was ±0.17 (wrong) → **±0.08**; GCN on MS Academic ±0.15 (wrong) → **±0.09**. Both from paper Table 2. Fixed.
- **GPRGNN** [[GPRGNN]]: Eq. 1 formulation and APPNP/SGC as special cases verified (line 261 of PDF). γ_k learnable, end-to-end, verified.
- **GraphACL** [[GraphACL]]: ogbn-arxiv 71.72±0.26, Squirrel 54.05, Chameleon 69.12, Texas 71.08, Arxiv-year 47.21, Cora 84.20 — all verified in paper Table 2.
- **GSTBench** [[GSTBench]]: Spearman ρ = −0.394 (GCN) / −0.433 (GAT), ogbn-papers100M pretrain, 8 downstream (5 citation + 3 e-commerce) — verified.
- **Graph Learning Poor Benchmarks**: authors list, arXiv id, and 5 key claims inspected in title/abstract; position paper, no numerical results to audit.

Net: one numerical error found (APPNP MS Academic stderr), corrected in-place. No fabricated content elsewhere. Audit discipline = same pattern used for GGD/BGRL earlier session.

## [2026-04-21] synth | Propagated Tier 2/3 findings into synthesis

- Updated [[Pareto Gap]]: corrected PolyGCL description (2 views, global mix, no ogbn-arxiv number); added Transferability Scope section with the two framings (per-dataset Pareto vs papers100M foundation-model) and a recommendation to default to the safer #1.
- Updated [[Reviewer Attacks and Defenses]]: added 6 new attack rows (PolyGCL, GPRGNN, GraphMAE2, GSTBench contrastive-transfer, statistical-significance). Added 3 new defensive gaps (global-γ SSL ablation, 95% CI + paired t-test infra, optional transfer probe).
- Created [[Novelty Verification Checklist]]: 6 claims × experiment blocks with 🔴/🟡/🟢 priorities. Six 🔴-open items flagged for coding-agent inquiries.
- Updated [[index.md]] with new synthesis page and the two new source pages.

## [2026-04-21] decide | Scope locked: homophily-only + per-dataset training

Researcher confirmed both scope decisions after the baseline-scope audit showed homophily-only + per-dataset is the field default for our Pareto frontier ([[GGD]], [[SUGRL]], [[BGRL]], [[GraphMAE2]] all restrict to those). Added canonical **§ Scope** to [[Thesis]]; updated [[Pareto Gap]] to replace the open transferability question with a locked scope block; updated [[Reviewer Attacks and Defenses]] heterophily-row defence to cite the baseline-scope argument and checked off the two corresponding defensive gaps; removed Claim 5 ablations from [[Novelty Verification Checklist]] and marked the transfer-scoping row as done.

## [2026-04-21] scan | April 2026 arxiv landscape pass

Quick WebSearch pass for AD-SSL-adjacent work since last ingest. Three items flagged for low-priority triage (no 2026 preprint directly preempts AD-SSL's mechanism combination):

- **Bootstrap Latents of Nodes and Neighbors** (arXiv 2408.05087) — BGRL + neighbor positives.
- **Graph Homophily Booster** (ICLR 2026, arXiv 2602.07256) — multi-hop via constructed feature edges.
- **Rethinking Node-wise Propagation for Large-scale Graph Learning** (arXiv 2402.06128) — node-wise k-step propagation; likely supervised, joins [[ATP]] / [[GPRGNN]] lineage.

No ingest yet — 10-minute skim each when convenient. Landscape otherwise unchanged.

## [2026-04-21] ingest | Three arxiv-scan follow-ups

Downloaded and extracted the three papers flagged in the afternoon arxiv scan.

- **Rethinking Node-wise Propagation (arXiv 2402.06128)**: same paper as our existing [[ATP]] entity. Deleted duplicate PDF (`node-wise-propagation.pdf`). No new work.
- **[[BLNN]] — Bootstrap Latents of Nodes and Neighbors** (arXiv 2408.05087, Liu et al., Nanjing U). Full entity page written. BGRL + 1-hop neighbor-positive alignment with attention-computed supportiveness score. Evaluated on 5 small graphs only (WikiCS/Photo/Computer/CS/Physics); +0.3–0.7 over BGRL. **Not a preempt** — orthogonal axis (spatial 1-hop) from AD-SSL (multi-depth). Added to related-work positioning as "concurrent BGRL extension on a different axis."
- **[[GRAPHITE]] — Graph Homophily Booster** (arXiv 2602.07256, Qiu et al., UIUC, ICLR 2026). Brief entity page. Supervised graph preprocessor for heterophily. Out of AD-SSL scope per locked decision ([[Thesis]] § Scope); flagged for completeness only.

Updated [[index.md]] with both new entity pages. Net: one mechanism-relevant ingest (BLNN), one scope-out reference (GRAPHITE), one duplicate dropped.

## [2026-04-21] synth | BLNN/GRAPHITE propagated; stale numbers corrected in landscape

- Updated [[Competitive Landscape 2026]]: replaced approximate ogbn-arxiv accuracies with PDF-verified numbers from the earlier audit — BGRL 71.64 ± 0.24, GraphMAE 71.75 ± 0.17, GraphMAE2 71.95 ± 0.08, GraphACL 71.72 ± 0.26. PolyGCL explicitly marked "not reported on ogbn-arxiv" (corrects stale ~70.5 memory claim). Added BLNN and GRAPHITE to concurrent-work section; added GSTBench reference.
- Added BLNN-row attack to [[Reviewer Attacks and Defenses]]: clean spatial-vs-depth differentiation.

## [2026-04-21] sync | Thesis A1-A4 ↔ Novelty Checklist; entity audit pass

- Updated [[Thesis]] § Four insights to note the RL-alignment analogy origin ("GRPO/KTO/SimPO/Online-DPO" are brainstorming scaffolding, not paper-facing concepts) and add a column mapping each insight to the corresponding claim + 🔴 ablation in [[Novelty Verification Checklist]].
- Audited untouched entity pages ([[SUGRL]], [[BGRL]], [[GGD]], [[GraphMAE]], [[ATP]]) against source PDFs: **all numerical claims verified**. Two landscape-table corrections propagated back to [[Competitive Landscape 2026]]: BGRL stderr ±0.12 (canonical, not GraphACL's ±0.24 re-report); GraphMAE 71.87 ± 0.21 GCN variant flagged alongside 71.75 GAT.

## [2026-04-21] sync | Ablation Plan A1-A4 analogy-origin note

Extended the Thesis↔Checklist sync to [[Ablation Plan - AD-SSL B0 A1-A4]]: added a preamble noting the GRPO/KTO/SimPO/Online-DPO tags are RL-alignment brainstorming analogies (scaffolding only; do not surface in paper) and directing Coding Agent to treat the Weighting/Loss/Refinement columns as the authoritative spec. Matches [[Thesis]] § Four insights wording.

## [2026-04-21] note | CLAUDE.md updated

Added: (1) paper-vs-vault codename clarification (AD-SSL = paper, SUGRL = legacy vault name), (2) canonical synthesis landing pages list, (3) locked scope callout, (4) "Audit discipline (numerical claims)" section with the three past-incident examples (PolyGCL ogbn-arxiv, APPNP stderrs, BGRL stderr) as guardrails against future fabrication.

## [2026-04-21] inquiry-answer | INQ-2026-04-21-001 classification val splits

Answered CA's 6-question inquiry on split policy for classification early stopping. Decision: **Option B across the board** (public Planetoid splits, 10/10/80 seed-determined splits on Photo/Computers with 20 seeds, official OGB splits). Early stopping: eval_every=10, patience=20, min_delta=0.0, epochs as upper bound. ogbn-mag dropped from main table (out of scope per [[Thesis]] § Scope). Appendix sanity check: report final-epoch vs best-val-checkpoint test-acc on ogbn-arxiv to detect val-leakage drift. Created [[Splits and Protocol]] as the canonical wiki record.

## [2026-04-21] note | Propagate 5-trial protocol; add Splits-and-Protocol to CLAUDE.md

- [[Ablation Plan - AD-SSL B0 A1-A4]] § Seeds: dropped "3 for screening, 5 for headline" in favor of **5 trials everywhere**; updated ROBUST criterion to 5/5.
- [[Matched-Seed Delta]]: ROBUST bar now 5/5 under the locked protocol; prior 168-run results remain valid at 3/3 for the old protocol.
- CLAUDE.md: added [[Splits and Protocol]] to canonical landing pages list.

## [2026-04-21] ingest | Ji et al., "Rethinking GNNs from a Geometric Perspective of Node Features" (ICLR 2025)

Ingested PDF (raw/papers/rethinking-geometric-node-features.pdf). Replaced stub with full entity page covering:
- Feature centroid simplex construction (§2) — class-centroid convex hull, coarse geometry (regular vs degenerate simplex models).
- Theorems 1-2 linking aggregated features to simplex vertices; Corollary 1 establishing intrinsic feature-space limits on datasets like Actor.
- Geometric interpretation of oversmoothing (§4) as iterated Markov contraction of Δ_e.
- Practical tricks -AE (random intra-class edge insertion + very early stopping) and -AEN (L2 feature normalization); Table 2 numbers on heterophilic datasets captured.
- Confirmed: same last author (Wee Peng Tay, NTU) as [[Less is More]] → citation weight in related work.

Propagations:
- [[Less is More]] and [[AD-SSL vs Less is More]]: marked ingestion done, confirmed same-group attribution.
- [[Reviewer Attacks and Defenses]] "Only tested on homophilic graphs" row: added Corollary-1 (intrinsic feature-space limit on Actor) as theoretical backing for scope decision.
- [[index]]: added entry under baselines/prior methods section.

Not a baseline. Supervised, theoretical, heterophily-focused. Cite as theoretical prior + scope-justification support.

## [2026-04-21] note | Ji et al. 2025 propagation: Oversmoothing concept

Added geometric-framing subsection to [[Oversmoothing]] citing Ji et al.'s simplex-contraction interpretation alongside the spectral (Oono & Suzuki) framing. Gives AD-SSL's spectral-motivation paragraph a theoretical-pluralism hook without changing the mechanism claim.

## [2026-04-21] note | Ji et al. 2025 propagation: Multi-Depth Views + AD-SSL

- [[Multi-Depth Views]]: added "Theoretical backing" paragraph citing Ji et al. §4 (aggregation reduces feature variance) and Theorem 1a (successive aggregation concentrates toward class centroid) as geometric grounding for why depth views are a valid contrastive signal.
- [[AD-SSL]]: added "Theoretical grounding" pointer section linking Multi-Depth Views + Oversmoothing to the Ji et al. framework. Kept lean — full motivation stays in [[Thesis]].

## [2026-04-21] note | CLAUDE.md updated (ingest + inter-agent sections)

- Ingest section rewritten: explicit method-page-vs-source-page split (entities/ for methods like BGRL/PolyGCL/BLNN; sources/ for benchmarks/position/onboarding); added `pdftotext -layout` step; added post-ingest audit step pointing to the Audit-discipline section.
- New "Inter-agent protocol alignment" subsection: when an inquiry locks a protocol decision, propagate to every dependent page and grep for stale values before closing. Reference example: INQ-2026-04-21-001's 5-trial lock propagated to Ablation Plan + Matched-Seed Delta.

## [2026-04-21] lint | Entity audit completed: GPRGNN + SGC

Final entities audited against PDFs:
- [[GPRGNN]]: all mechanism/numbers/theoretical claims verified (APPNP/SGC special-case equivalences exact, K=10/hidden 64, 10-dataset coverage, Cora/Texas Fig 1 weight patterns, 95% CI reporting). Fixed Pan Li affiliation: Purdue, not UIUC.
- [[SGC]]: all mechanism/numbers verified (Cora 81.0 ± 0.0, CiteSeer 71.9 ± 0.1, PubMed 78.9 ± 0.0, Reddit 94.9, 2 L-BFGS steps on Reddit, 28× speedup on Pubmed from Fig 3). Fixed ogbn-arxiv reproduction number: 66.92 ± 0.08 (exact, per GraphMAE2 Table 3), not 66.9. Also tightened BGRL comparison to its canonical 71.64.

All entity pages in the wiki now traced line-by-line to source PDFs. No outstanding fabrication risk in the baseline/prior-art coverage.

## [2026-04-22] inquiry response | INQ-2026-04-22-001 B0 collapse
Picked Option B (InfoNCE as B0). Demanded three diagnostics from CA: per-depth inference (Z_k alone vs mean), per-depth collapse stats (embedding_std by k), InfoNCE τ sweep on Cora. Replaced Cora ≥82 gate with "B0 ≥ parameter-free Â¹X baseline per dataset" — right bar for a minimal baseline. Reframed Cora as report-for-convention, not headline dataset; Pareto story lives on Photo/Computers/ogbn-arxiv/ogbn-products. Wiki updates (AD-SSL loss field, Thesis loss framing, Ablation Plan B0 row, Novelty Checklist) to follow in a separate pass once CA results come in.

## [2026-04-22] propagate | INQ-2026-04-22-001 B0 = InfoNCE
Propagated B0 loss swap across wiki: [[AD-SSL]] entity (mechanism step 3 + one-line), [[Thesis]] (mechanism step 3 + B0 definition + negatives note), [[Ablation Plan - AD-SSL B0 A1-A4]] (B0 row + A3 re-scoped to bootstrap comparison), [[Novelty Verification Checklist]] (Claim 2 reframed to contrastive; old bootstrap-vs-BCE ablation now InfoNCE-vs-BCE). Added "Baseline gate — training adds value over not training" section to [[Splits and Protocol]] replacing old ≥82 Cora gate with per-dataset Â¹X linear-probe floor. All changes marked provisional pending CA's three diagnostics.

## [2026-04-22] inquiry response #2 | INQ-2026-04-22-001 diagnostic followup
CA's D1/D2/D3 + ogbn-arxiv diagnostic: trained encoder underperforms parameter-free Â^{k*} X linear probe on all three tested datasets (Cora −9.09, Computers −2.57, arxiv −8.03). Per-depth probe accuracy is flat on Cora (mean-pooling hypothesis refuted). Soft collapse to uniform unit-sphere across all depths. Told CA to run one sanity config (hidden=1024, dropout=0, τ=0.5) on Cora before calling the structural pivot (I/II/III). Radin flagged II (BGRL-pretrained encoder + test-time routing) as risky for the "scalable/fast on large graphs" claim — not dismissed but would force major reframing. Locked Â^{k*} X linear probe as a first-class main-table baseline (Q5 = YES). A1–A4 remain blocked until B0 is settled.

## [2026-04-22] inquiry response | INQ-2026-04-22-002 wide-encoder failed
CA's wide-encoder (hidden=1024, dropout=0) gave +1.8 pts on Cora — still 3 pts below Â¹X. Trained std 0.0537 < 1/√256 = 0.0625 (wider encoder collapses harder). Interpretation (i) refuted; interpretation (ii) confirmed: views too similar. Responded: run one more diagnostic — InfoNCE + edge-dropout p=0.3 per-view, narrow encoder, on Cora AND Computers. Pass bar unchanged (B0 ≥ Â¹X on both). Asked CA to measure per-epoch edge-dropout cost as % of full precompute (affects scalability claim independently of gate). Not picking Option I/III yet. Paper-level reframe around Â^{k*} X parameter-free probe result still on the table pending this diagnostic.

## [2026-04-22] inquiry response #2 | INQ-2026-04-22-002 edge-dropout split result
CA ran edge-dropout on Cora (+6.3 pts, PASSES Â¹X gate), Computers (−8.4 pts, fails −11 below gate), ogbn-arxiv (−1.1 pts, fails −9 below gate). Cost <10% on all. Option I dead as a universal recipe. CA's emerging hypothesis: InfoNCE instance-discrimination has a ceiling at/below parameter-free Â^{k*} X probe on already-strong features. Radin's decision: pursue two tracks in parallel — (1) ask CA for one more diagnostic (InfoNCE→bootstrap + predictor + EMA target + edge-dropout, BGRL-lite full kit on Cora+Computers), (2) research side starts designing the α-over-raw-Â^kX reframe (AD-SSL v2: no encoder, per-node α over raw propagated features, cheap unsup signal). No spec or wiki changes yet. A1–A4 still blocked.

## [2026-04-22] draft | AD-SSL v2 encoder-free reframe sketch
Created [[AD-SSL v2 - Encoder-Free Design Sketch]] under wiki/synthesis/. Drafts an alternative architecture triggered by three-dataset encoder-based B0 failure: no encoder, per-node α parameterized by small MLP, mixture directly over raw Â^k X. Ranks five candidate training signals (S1 confidence-max/IIC-style, S2 feature alignment, S3 graph-smoothness regularizer, S4 supervised, S5 no-training baseline). Lean: S1 + S3 as primary. Status: design draft, blocked on BGRL-lite diagnostic from [[INQ-2026-04-22-002]]. If that passes, v2 becomes secondary ablation; if it fails, v2 becomes the paper.

## [2026-04-22] inquiry-open | INQ-2026-04-22-003 Track 2 encoder-free prototype
BGRL-lite in INQ-002 fully collapsed (Cora 11.59, Computers 37.49, embedding std 0.004). Confirms encoder is the problem, not a parameter. Radin's call: commit to Track 2 prototype before deciding direction. Filed [[INQ-2026-04-22-003]] to CA: implement per-node α (shared MLP + depth embed) over raw Â^k X, train with S1 confidence-max (β=1.0) + λ·S3 graph smoothness (λ=0.1) on Cora+Computers. Pass bar: Z-probe ≥ Â^{k*}X on both. Variations V1–V6 queued if primary fails (M sweep, β sweep, λ sweep, free-table α, supervised α ceiling, ablation vs uniform/best-k/global-γ). Blocking; paper direction stalled until numbers land.
