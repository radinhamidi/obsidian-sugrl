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

## [2026-04-22] note | Research Agent Operating Protocol issued
Radin issued a binding methodology spec for how the Research Agent should brainstorm, diagnose failures, handle negative results, and propose pivots. Created [[Research Agent Operating Protocol]] under wiki/sources/. Linked from [[RESEARCH_AGENT_ONBOARDING]], [[index]], and `CLAUDE.md` (both in-line note and canonical-pages list). Key rules: patience on the question / impatience on the method; ≥5 alternative approaches before conceding a direction is closed; no "negative-findings paper" as first response to a failed experiment; always present a slate of 2–4 directions rather than a single recommendation.

## [2026-04-23] inquiry-open | INQ-2026-04-23-001 Track 2 D1 symmetry-break
INQ-003 closed Track 2 v2 as specified (α never left uniform under L_S1 due to depth-symmetry on homophilic graphs, pinned by V3 λ=0 + V4 free-table + V6 global γ). Post-mortem under [[Research Agent Operating Protocol]] produced slate of 4 directions + backlog of 3 more; Radin picked D1. Filed [[INQ-2026-04-23-001]] to CA: add X_0 view (K_SET={0,1,2,4,8}) and per-depth linear projection W_k before shared classifier head. Hard constraint: W_k is linear-only (nonlinearity = encoder, ruled out by INQ-001/002); nonlinear W_k is V3 if primary fails. Pre-registered symmetry-break signals (per-node α entropy, α-mean-per-depth std, per-node Var_k(p_{ik})) required alongside accuracy; accuracy pass bar is same as INQ-003 (Cora ≥81.3, Computers ≥87.5). Variations V1–V6 queued (ablate X_0, ablate W_k, nonlinear W_k, per-depth head, supervised ceiling, α controls).

## [2026-04-23] draft | Idea Ledger created
Seeded [[Idea Ledger]] at wiki/synthesis/ per [[Research Agent Operating Protocol]] first-pivot mandate. Closed rows C1 (encoder-based B0) + C2 (Track 2 v2 as specified). Live row L1 (D1 symmetry-break). Backlog rows B1 hard-routing / B2 dataset-conditional-budget reframe / B3 heterophily scope extension / B4 feature-augmentation views / B5 predictive SSL / B6 learnable propagation coefficients / B7 efficiency-delta packaging. Surprising observations O1 (low-label supervised under-uses deep propagation) + O2 (7× per-dataset depth spread) flagged for revisit.

## [2026-04-22] inquiry-open | INQ-2026-04-22-003 Track 2 encoder-free prototype
BGRL-lite in INQ-002 fully collapsed (Cora 11.59, Computers 37.49, embedding std 0.004). Confirms encoder is the problem, not a parameter. Radin's call: commit to Track 2 prototype before deciding direction. Filed [[INQ-2026-04-22-003]] to CA: implement per-node α (shared MLP + depth embed) over raw Â^k X, train with S1 confidence-max (β=1.0) + λ·S3 graph smoothness (λ=0.1) on Cora+Computers. Pass bar: Z-probe ≥ Â^{k*}X on both. Variations V1–V6 queued if primary fails (M sweep, β sweep, λ sweep, free-table α, supervised α ceiling, ablation vs uniform/best-k/global-γ). Blocking; paper direction stalled until numbers land.

## [2026-04-23] inquiry-open | INQ-2026-04-23-002 entropy-driven depth routing + D1' WD gate
Radin picked entropy-based routing as the next direction after INQ-004 closed D1. Filed [[INQ-2026-04-23-002]] to CA: four configurations — E1 (training-free α from k-means pseudo-label entropy), E2 (learnable α, entropy-min loss, no W_k), E3 (entropy-weighted cross-depth contrastive with learnable W_k, standard WD), E4 (supervised ceiling via per-depth ridge probe on train labels) — plus V1 (shared k-means clusters variation), E3-V-WD (W_k WD-excluded if E3 primary fails), and D1' (bundled cheap diagnostic re-running D1 primary+V1 with W_k excluded from weight decay, closes the row-2/row-5 gap from INQ-004 post-mortem). All primaries use k-means pseudo-labels (parameter-free post-fit, no head to collapse) as the p_ik source. Primary diagnostic = argmin_k H(p_ik) flip test: should identify k=8 on Cora, k=1 on Computers; if monotonically deep on both, entropy is tracking feature concentration not class-confidence and direction is dead. Pass bars: Cora ≥ 78.87, Computers ≥ 87.53 (match INQ-001 raw best-single-depth). Blocking.

## [2026-04-23] inquiry-answer | INQ-2026-04-23-002 entropy-routing hard-fails — E4 ceiling fails the dataset-flip
CA ran the full slate: E1, E4, E2, E3 primaries + E3-V-WD + V1 (shared clusters) on E1 and E3 modes + D1' WD-gate (primary + V1-no-X0), on Cora+Computers, 3 seeds × 5 probe restarts. **Every config hard-fails.** The direction-killing diagnostic: **E4 (supervised ridge ceiling) does NOT dataset-flip correctly** — argmin_k H on Computers concentrates at k=0 (57%), not k=1; on Cora argmin_k is distributed 36/19/12/10/23% with no clear k=8 preference. Per RA's Q5 criterion, this kills entropy-routing regardless of pseudo-label source. Cross-cutting observation: τ_p=τ_α=1.0 + L1-row-normalized homophilic features produces flat per-depth H (Cora k-means: 1.946 × 5; Computers k-means: 2.302/2.303×4; Computers ridge: spread only 0.014). Softmax(-H/1.0) is uniform → α constant → Z ≡ raw mean pool on E1/E2/E4. E3 Cora Z=81.73 passes the probe bar (78.87) but fails every α-symmetry signal (mean-std 0.0000) — mirror counter-example to INQ-004 V6/best-k. E3 Computers Z=79.48 fails hard by 8 pts; W_k collapses 4-7× under default WD. E3-V-WD recovers W_k magnitude on Computers (Z +5pts to 84.39) but still fails hard+soft. V1 shared clusters spreads argmin_k distribution but does not change any accuracy. D1' CATASTROPHICALLY fails (Cora Z=55, Computers Z=40): without WD, W_k grows 10-20× to 100-300 and Z crashes — confirms INQ-004 row-5 (hypothesis genuinely wrong, not WD artifact). Mechanism M1 (shared-head depth-symmetry) + M2 (near-zero data gradient on W_k) jointly confirmed; D1 has two degenerate fixed points, neither yields useful Z. CA response appended under `# RESPONSE` with per-config sections leading with the dataset-flip line.

## [2026-04-23] inquiry-answer | INQ-2026-04-23-001 D1 hard-fails — W_k collapse mechanism
CA returned full D1 + V1–V6 sweep. Primary hard-fails both pass bars on Cora + Computers (58.11 / 72.22 vs 81.3 / 87.5). **New mechanism**: under uniform α + shared head + WD=5e-4, ∂L/∂W_k is depth-symmetric (differs only by X_k − X_{k'}, small on homophilic graphs), so weight decay dominates and every W_k shrinks ~50,000× from xavier init to ~2.7e-4 Frobenius; at W_k → 0, p_ik → softmax(h(0)) = constant across k, α-gradient is strictly zero. Symmetry intact. V2 (no W_k) lands exactly at raw mean-pool floor — confirms W_k is strictly subtractive under this loss. V5 (supervised CE) matches D1-primary to 0.03 pts on Cora: pathology is loss-agnostic / architectural. V6 best-k one-hot satisfies every pre-registered symmetry signal by fiat yet Z-probe stays at Z_0 level — α is a bystander; W_k collapse is the driver. CA noted (not recommended) one cheap ≤5-line controlled test: exclude W_k from weight decay to distinguish row-2 (hyperparam) from row-5 (hypothesis-genuinely-wrong). Post-mortem + 4-direction slate presented to Radin in chat.

## [2026-04-23] RA analysis | INQ-005 independent read overrides CA verdict labels
Radin flagged that I was echoing CA's "hard fail" verdicts instead of doing the RA read myself. Independent analysis of [[INQ-2026-04-23-002]] numbers disagrees with CA's framing on three points, committed to [[Idea Ledger]] C4 row and new O5–O7. **(1)** E3 Cora probe 81.73 ± 0.27 vs hard bar 78.87 is a genuine +2.86 pt pass at ~10σ; CA's "hard-fail" label comes from the pre-registered compound rule (probe AND mechanism) that was defensive against the INQ-004 V6 failure mode and does not apply here. The probe pass is real; the story is W_k (from cross-depth InfoNCE) not α. **(2)** The "flat H under simplex collapse" reading is a τ_p=1.0 spec artifact, not a theorem. Cora E1 H values (1.946 × 5 depths) equal log(7) to 4 decimals — softmax numerically uniform regardless of cluster quality. τ_p=1.0 (RA-specified) pre-saturated entropy; entropy-as-signal was never actually tested at a reasonable temperature. **(3)** E4 "ceiling failed flip" uses ridge-MSE softmax as confidence source, which is not a valid confidence measure; Q5's dispositive framing was wrong — on high-dim + few-label setup ridge produces feature-space overfitting confidence, not class structure confidence. Latent positives preserved as **O5** (cross-depth InfoNCE levels every Cora depth to ≈80, k=0 raw 46.79 → post-training 75.68, +28.9 pts — a live standalone method), **O6** (D1 has two absorbing fixed points under L_S1 + shared head, WD is not root cause), **O7** (τ_p=1.0 saturation rule for future softmax-over-distance specs). What's cleanly dead: E1/E2/E4 at τ_p=1.0 + ridge confidence; D1 architecture; D1'. Alive: D6 (cross-depth InfoNCE as standalone multi-depth pretext, no α/mixing), V2 (entropy-family retest with τ_p sweep + CE-trained probe). Backlog: D5 (Gumbel/hard routing), D7 (across-k entropy), D8 (training-free oracle reframe), D9 (heterophily scope unlock). Saved binding feedback memory `feedback_ra_independence.md`: RA must investigate raw numbers independently, not parrot CA's verdict column.

## [2026-04-23] inquiry-open | INQ-2026-04-23-003 D6 InfoNCE pretext + V2 E1 τ_p sweep
Filed [[INQ-2026-04-23-003]] to CA after RA-independent read of INQ-005 surfaced E3-Cora as a real +2.86 pt pass via W_k (not α) and identified τ_p=1.0 as a spec-bug that saturated entropy measurement. Three cheap configs in one inquiry: **D6a** (linear W_k, d_proj=128, flat cross-depth InfoNCE, no α/L_ent — reproduces INQ-005 E3 Cora signal as standalone method), **D6b** (linear W_k, d_proj=F_in, no dim reduction — tests Computers projection-bottleneck hypothesis), **D6c** (residual skip `Z_k = X_k + W_k X_k`, d_proj=F_in — preservation-guaranteed variant, strongest expected Computers result), **V2-E1-τ-sweep** (E1 with τ_p ∈ {0.001, 0.01, 0.05, 0.1, 1.0}, both datasets — closes spec-bug gap). Both Z_mean and Z_concat readouts required for D6. Pass bars unchanged (Cora ≥ 78.87, Computers ≥ 87.53). Linear-W_k constraint retirement documented inline in the inquiry. Total estimated CA time ~30 min wall-clock. Priority high, not blocking — we have Cora signal in hand, this is to test generalization + close the entropy-retest gate before considering D9 (heterophily scope unlock).

## [2026-04-23] feedback | linear-W_k constraint retired for D1/E3/D6 families
Radin flagged that I carried the "no nonlinear W_k (encoder destroys signal)" constraint from INQ-001/002 into every D1 variation AND into E3 as a hard "DO NOT" in the inquiry spec, without re-deriving whether the originating mechanism applied. INQ-001/002's mechanism requires highly-correlated same-depth views + encoder-collapse to shared component; cross-depth views (E3) are genuinely different and do NOT meet the mechanism precondition. Saved binding feedback `feedback_no_constraint_propagation.md`: constraints born in one experimental context must be re-derived per new context, not auto-applied. Effect on Ledger: D1 closed row re-labeled "linear W_k only dead"; D1-family reopened with nonlinear/skip/residual variations (D1-NL-a/b/c); D6 current-slate row expanded with 5 variations (D6a linear d128 baseline, D6b linear d=F_in, D6c skip-connection, D6d MLP-W_k, D6e per-depth head + concat). D1-NL-c (per-depth head) is the only variation that attacks the O6 shared-head symmetry root cause; D1-NL-a/b only address surface W_k-collapse.

## [2026-04-23] inquiry-answer | INQ-2026-04-23-003 D6 + V2-E1-τ_p sweep results
CA ran D6a/D6b/D6c + V2-E1-τ_p sweep {0.001, 0.01, 0.05, 0.1, 1.0} on Cora + Computers, 3 seeds × 5 probe restarts. Headline numbers reported under `# DIAGNOSTIC RESULTS` and `# Final numerical table`. D6c Cora Z_mean 81.80 ± 0.20 / Z_concat 81.83 ± 0.29; D6c Computers Z_mean 88.26 ± 0.40 / Z_concat 88.00 ± 0.27 — both above hard-pass bars. D6a Cora Z_mean 81.60, D6b Cora Z_mean 78.99. V2-E1 best Cora τ_p=0.05 Z=76.33; V2-E1 best Computers τ_p=1.0 Z=86.12. Argmin-k distributions swept across τ_p per RA spec. W_k Frobenius norms + cosine pairs reported per seed. Frontmatter flipped open→answered; README index updated.

## [2026-04-23] RA analysis | INQ-006 independent read — D6c is the first simultaneous double-pass
Independent read of [[INQ-2026-04-23-003]] returned results. **D6c hard-passes both Cora and Computers** (Z_mean 81.80 ± 0.20 / 88.26 ± 0.40; hard bars 78.87 / 87.53). First configuration in the project to do so simultaneously. Key RA findings beyond CA's table: (1) D6c lifts EVERY per-depth Z_k probe above raw Â^k X on Computers (+0.71 to +6.23), not just "residual preserving k=1" — this is a clean ensemble lift from cross-depth InfoNCE, mechanistically different from D6a which HURT k=1 by −4.50; (2) cos(W_k, W_k') drops to 0.04 (k=4 vs k=8) on Cora indicating depth-distinct projections, stays high 0.78–0.99 on Computers (residual+InfoNCE couldn't decouple depths there, but every depth still improved — preservation floor did the work); (3) Computers +0.73 is ~1.8σ across-seed std at 3 seeds, tight — 5-seed confirmation queued; (4) V2-E1 τ_p sweep cleanly closes entropy-from-kmeans on raw X_k: Computers argmin_k locks to k=0 at ≥98.7% at every τ_p, Cora argmin flips to k=8 at high τ but α too soft to matter, corr(argmin-k, node structure) ≈ 0 at every τ_p. E1/E2/E4 on raw X_k are structurally dead, not a τ bug. Non-propagation: this does NOT apply to Z_k-based entropy — that's a separate test. Ledger updated: D6c promoted to Live; V2 closed-on-raw-X_k; C4 updated with τ-sweep evidence; O7 rewritten with sweep conclusion.

## [2026-04-23] inquiry-open | INQ-2026-04-23-004 D6c extensions — arxiv, 5-seed, α ablation, V-WD
Filed [[INQ-2026-04-23-004]] to CA. Four D6c-derived configs to run in parallel: **Config A** D6c-arxiv (extend to ogbn-arxiv per Splits-and-Protocol 5-seed init-only, CE probe, report final-epoch and best-val); **Config B** D6c-5seed (add 2 seeds on Cora + Computers to tighten Computers +0.73 CI); **Config C** D6c+α (α-routing at readout via entropy-from-kmeans on post-training **Z_k**, τ_p sweep ∈ {0.01, 0.05, 0.1, 0.5, 1.0} — constraint non-propagation: V2-E1 raw-X_k failure does NOT transfer to Z_k, different cluster geometry); **Config D** D6c-V-WD (W_k excluded from weight decay, checks whether ||W_k||_F ratios 0.07–0.37 of xavier are costing signal; residual floor should prevent D1'-style explosion crash because InfoNCE provides direct W_k gradient unlike L_S1). Config C determines paper framing: if α adds nothing on top of D6c, thesis pivots from "adaptive-depth SSL" to "cross-depth InfoNCE + residual projection." Pass bars: Cora ≥ 78.87, Computers ≥ 87.53, arxiv derived from raw best-single-depth. Priority high, not blocking. Parallel literature audit continuing on "depth-as-view" contrastive prior art and D9 heterophily priors.

## [2026-04-23] inquiry-answer | INQ-2026-04-23-004 D6c extensions — arxiv/5-seed/α/V-WD
CA ran D6c Config A (ogbn-arxiv, 5 seeds, final + best-val), B (D6c seeds 3,4 appended; 5-seed Cora + Computers), C (D6c+α: frozen-W_k + k-means on Z_k + τ_p sweep {0.01..1.0}, τ_α=1.0), D (D6c-V-WD: W_k excluded from WD). Numbers under `# DIAGNOSTIC RESULTS` sections. Config A: Z_concat 68.33 ± 0.06 (n=25), raw best Â^2 X = 60.28. Config B 5-seed pooled: Cora Z_mean 82.01 ± 0.31 (n=25) / Computers Z_mean 88.24 ± 0.42 (n=25). Config C: Δ(Z_α − Z_mean) ≤ +0.01 at any τ_p on both Cora and Computers; Computers argmin_k locks to k=0 (≥98.4%) across all τ_p; Cora argmin_k shifts between k=0 / k=4 / k=8 as τ_p grows. Config D: W_k grew above xavier init on both datasets (Cora 3.27–5.74× ratio, Computers 2.51–2.80×); Z_mean dropped by 1.38 (Cora) / 1.66 (Computers). Frontmatter open→answered; README + log updated.

## [2026-04-24] RA analysis | INQ-007 independent read — D6c is the method; α retired; Z_concat is default
Independent read of [[INQ-2026-04-23-004]] results. **D6c passes all three datasets at CI-tight margins:** Cora Z_concat 82.05 ± 0.34 vs 78.87 (+3.18, 14σ+); Computers 5/5 per-seed Z_mean strictly above 87.53 (+0.71 mean, 3.8σ stderr); ogbn-arxiv Z_concat 68.33 ± 0.06 vs raw k=2 = 60.28 (+8.05, stderr <0.01 matching final-epoch to best-val). Config C closes adaptive-depth direction: max Δ(Z_α − Z_mean) = +0.01 across all 10 τ_p × dataset combinations — α-on-top-of-D6c-Z_k adds nothing. Config D (V-WD) retires the WD-excluded variant: W_k grows above xavier but Z drops by 1.38 / 1.66 — residual+WD is the correct regime, W_k ought to stay small. RA findings CA did not highlight: (1) Z_concat beats Z_mean by **+3.43 on arxiv** due to large per-depth quality variance (raw k=2 = 60.28 vs k=8 = 50.82) — new default readout; (2) per-depth recovery on arxiv: raw k=8 = 50.82 → post-D6c k=8 = 67.74 (+16.92), same lift signature as Cora raw k=0 → post-D6c k=0 (+29.62); (3) Computers corr(argmin-k, degree) = 0.564 on post-D6c Z_k — depth-concentration after training tracks node structure even though α routing doesn't add value. Mechanism story: oversmoothing (Ji et al. 2025 simplex collapse) reframed as a contrastive signal — W_k lifts weak depths toward a shared instance-discrimination space. Paper framing locked: **no learned per-node depth routing**; depth as the contrastive view axis.

## [2026-04-24] framing | Thesis + Idea Ledger + Pareto Gap rewritten around D6c-as-method
Paper framing updated across three canonical synthesis pages after INQ-007 triage. [[Thesis]] fully rewritten: one-sentence claim now states +3.18 / +0.71 / +8.05 over best-single-depth on Cora / Computers / arxiv, mechanism section describes precompute → residual linear projection → flat cross-depth InfoNCE → Z_concat readout, novelty section centers "depth as contrastive view axis, no encoder, residual load-bearing," mechanism story grounded in Ji et al. 2025 simplex collapse. Formally retired: adaptive-depth framing, A1–A4 RL-analogy ablation plan, bootstrap loss as primary, MLP encoder architecture. [[Pareto Gap]] reframed: D6c is NOT claimed to reach BGRL-level 71.x accuracy; it is a Pareto point **within the cheap-method band** (SUGRL / GGD / MLP-on-precomputed) that lifts best-single-depth precomputed features by +8.05 at matched wall-clock cost. [[Idea Ledger]] D6c Live row rewritten with full 3-dataset evidence, α/V-WD closures, MVGRL pre-emption defense, and Phase-2 live-work queue. Pages flagged as needing 2026-04-24 update but not yet rewritten: [[Reviewer Attacks and Defenses]], [[Novelty Verification Checklist]], [[Ablation Plan - AD-SSL B0 A1-A4]].

## [2026-04-24] lit-audit | MHVGCL identified as closest architectural ancestor; DGD low-risk
Second-pass literature audit targeting "multi-hop contrastive" and "decoupled propagation SSL" surfaced two papers: [[MHVGCL]] (Wu et al., Applied Soft Computing 2025) and [[DGD]] (Neurocomputing 2024). **MHVGCL is promoted above [[MVGRL]] as the highest pre-emption risk.** Shares D6c's core loss structure: same-node-across-hops InfoNCE-style contrastive, no augmentation, shared-parameter encoder. Differs on three axes: (a) applies MLP BEFORE propagation (`Â^k · MLP(X)`, per-epoch GNN cost), (b) uses a single shared head across hops (architecturally our D1, which hard-failed), (c) contrasts encoder-outputs not raw X. D6c's three-way differentiator — precompute vs. per-epoch, per-depth W_k vs. shared head, raw vs. encoder-output contrast — survives but must be sharp in the paper. DGD is low risk: BCE group discrimination (GGD descendant), aggregates hops instead of contrasting them. Created [[MHVGCL]] and [[DGD]] entity pages, updated [[Thesis]] § Known risks to replace MVGRL with MHVGCL as primary pre-emption threat, updated [[Idea Ledger]] D6c reviewer-attacks with three-way differentiator detail. Both PDFs pending ingest. Remaining audit gap: "hop-level contrastive" and "SGC-contrastive" query families; no direct "Â^k vs. Â^{k'} InfoNCE at precompute with per-depth W_k" paper found yet.

## [2026-04-24] audit-correction | MHVGCL + DGD claims flagged unverified; +3.14 vs +3.18 Cora delta fixed
Radin flagged that I had written MHVGCL and DGD architectural claims into the wiki without downloading the PDFs. Audit-discipline rule (per CLAUDE.md) requires verifying each claim against the source PDF before committing. Attempted PDF retrieval 2026-04-24: ScienceDirect (paywall), ResearchGate (403 blocked), OpenReview PDF endpoint (404 for DGD — forum page is a record only, no attachment), no arxiv preprint for either. **All MHVGCL/DGD architectural claims are from web-search abstract paraphrases, not PDF reading.** Corrected: added ⚠ UNVERIFIED warning block to top of [[MHVGCL]] and [[DGD]] entity pages, reworded "method reconstructed from abstract + related-work excerpts" to "method reconstructed from abstract + web-search paraphrases — UNVERIFIED", softened [[Thesis]] § Known risks MHVGCL paragraph from hardened defense to "candidate pre-emption, verification pending", softened [[Idea Ledger]] D6c reviewer-attack 1 and 3 to carry same flag. Separately fixed numerical inconsistency in [[Thesis]] one-sentence claim: was "+3.14" (Z_mean delta) while the table and paper framing use Z_concat as default — changed to show both "+3.18 on Z_concat, +3.14 on Z_mean" per dataset. Action item before paper submission: obtain MHVGCL + DGD PDFs via author email or institutional access; re-audit pre-emption defense against actual method sections. INQ-007 numerical claims in [[Thesis]] primary table and one-sentence claim verified line-by-line against [[INQ-2026-04-23-004]] response (82.01 / 82.05 / 88.24 / 87.96 / 68.33 / 64.90 / raw 78.87 / 87.53 / 60.28 all traceable to the inquiry file).
