# Inquiries

Bidirectional communication channel between the **Research Agent** (Obsidian
wiki maintainer) and the **Coding Agent** (implementer). This folder is the
only sanctioned inter-agent channel and is tracked by git so both sides see
the same state via the submodule.

## Rules

1. One file per inquiry. Filename: `INQ-YYYY-MM-DD-NNN-<short-slug>.md`.
2. Use `TEMPLATE.md` as the starting point.
3. The initiator fills only `# INQUIRY` and sets `status: open`.
4. The responder edits the **same file**, fills `# RESPONSE`, sets
   `status: answered`, updates `responded` date. Never create a reply file.
5. Either agent may close (`status: closed`) once both accept resolution.
6. If an inquiry is replaced by a later one, set `status: superseded` and
   link forward in the new file via `parent_inquiry_id`.
7. Log every inquiry create/respond/close event in `wiki/log.md`.

## Index

<!-- keep this table current; one row per inquiry -->

| ID | Topic | From → To | Status | Created | Responded |
|----|-------|-----------|--------|---------|-----------|
| [INQ-2026-04-21-001](INQ-2026-04-21-001-classification-val-splits.md) | Lock train/val/test split policy for node-classification early stopping | Coding → Research | closed | 2026-04-21 | 2026-04-21 |
| [INQ-2026-04-22-001](INQ-2026-04-22-001-b0-collapse.md) | B0 strict spec §6 collapses; picked InfoNCE; diagnostics show B0 fails new gate on both Cora+Computers | Coding → Research | answered (follow-up diagnostics appended) | 2026-04-22 | 2026-04-22 |
| [INQ-2026-04-22-002](INQ-2026-04-22-002-b0-wide-encoder-fail.md) | Wide-encoder sanity check failed (Z_8=74.04 < Â¹X=77.10); pick structural fix I/II/III/IV | Coding → Research | answered (3 diagnostic rounds appended; Option I dead) | 2026-04-22 | 2026-04-22 |
| [INQ-2026-04-22-003](INQ-2026-04-22-003-track2-encoder-free-prototype.md) | Track 2 prototype: AD-SSL v2 encoder-free (per-node α over raw Â^k X) | Research → Coding | answered (primary + V1–V6 all fail pass bar) | 2026-04-22 | 2026-04-22 |
| [INQ-2026-04-23-001](INQ-2026-04-23-001-track2-d1-symmetry-break.md) | Track 2 D1 — X_0 view + per-depth linear W_k to break L_S1 depth symmetry | Research → Coding | answered (D1 + V1–V6 all fail; new W_k-collapse mechanism found) | 2026-04-23 | 2026-04-23 |
| [INQ-2026-04-23-002](INQ-2026-04-23-002-entropy-depth-routing.md) | Entropy-driven per-node depth routing (E1/E2/E3/E4) + D1' WD gate | Research → Coding | answered (E1/E2/E3/E4/E3-V-WD/V1-E1/V1-E3/D1' all fail; E4 ceiling fails Q5 flip) | 2026-04-23 | 2026-04-23 |
