---
inquiry_id: INQ-YYYY-MM-DD-NNN
parent_inquiry_id:          # optional, link to prior inquiry this branches from
topic: <one-line topic>
from_agent: <Research Agent | Coding Agent>
to_agent: <Research Agent | Coding Agent>
created: YYYY-MM-DD
responded:                  # fill when answered
priority: low | medium | medium-high | high | blocking
status: open | answered | closed | superseded
related_files:
  - raw/...
  - wiki/...
tags: [inquiry, neurips-2026, ...]
---

# INQUIRY

**From:** <agent>
**To:** <agent>
**Blocks:** <what downstream work this gates, if anything>

## Context

<Background the other agent needs. Assume they have the wiki but not the
current conversation. Link to relevant wiki pages with [[wiki-links]] and
raw source paths.>

## The options / question

<If asking for a decision among N options, lay them out explicitly with
pros/cons. If asking an open question, state it precisely. If reporting a
finding that needs acknowledgment, say so.>

### Option A — <name>
Pros: ...
Cons: ...

### Option B — <name>
Pros: ...
Cons: ...

## Numbered questions

1. <specific question 1>  Default hunch: ...
2. <specific question 2>  Default hunch: ...

## Expected response format

<Pick A/B/C + one-paragraph justification / Answer questions 1-N inline /
Confirm-or-amend / Review-and-approve.>

<Optional: deadline or urgency note.>

---

# RESPONSE

**From:** <agent>
**Status:** <answered | needs-clarification | deferred>
**Date:** YYYY-MM-DD

## Decision / Answer

<Lead with the decision or direct answer.>

## Justification

<Reasoning that will still make sense to a future reader who doesn't have
this conversation's context.>

## Follow-up answers

1. <answer to question 1>
2. <answer to question 2>

## New inquiries spawned

<Optional: link any child inquiries this answer triggered.>
