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
| —  | —     | —         | —      | —       | —         |
