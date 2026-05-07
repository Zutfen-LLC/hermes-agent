---
name: cca-lite-memory-update
description: Use when promoting durable inter-session facts into docs/cca-lite/hermes-memory.json. Classifies candidate memories across native memory, MemPalace, and repo-local cca-lite, then validates and commits the update.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [memory, cca-lite, inter-session, workflow, git, knowledge-management]
    related_skills: [writing-plans, systematic-debugging, requesting-code-review]
---

# CCA-Lite Memory Update

## Overview

Use this skill to turn session learnings into durable repo-local memory without conflating three different memory systems.

The target artifacts are `cca-repo/docs/cca-lite/hermes-memory.json` and `cca-repo/docs/cca-lite/hermes-memory-inbox.json`. The ledger file is the canonical version-controlled memory store; the inbox is the staging queue for promoted candidates.

This skill exists to keep the routing clean:

- Native memory = user preferences and stable personal/environment facts
- MemPalace = cross-session semantic recall and concept links
- cca-lite memory = repo-local durable state, decisions, invariants, and operational facts

## When to Use

Use when:
- A session produces a durable repo-local fact that should survive future conversations
- You need to promote an observation into a versioned memory record
- You want to update project memory and commit it like code
- You need to decide whether a fact belongs in native memory, MemPalace, or repo-local memory

Do not use when:
- The information is temporary session chatter
- The fact is only a user preference and belongs in native memory
- The fact is broad semantic knowledge better stored in MemPalace
- You are unsure whether the fact is stable enough to promote; keep it as an open question instead

## Routing Rules

### 1. Native memory
Use native memory for:
- user preferences
- correction about tone or style
- stable personal/environment facts about the user

### 2. MemPalace
Use MemPalace for:
- conceptual recall across projects
- search-friendly knowledge that is not authoritative repo state
- references to past discussions when a short pointer is enough

If MemPalace is unavailable or cannot import, treat that sink as skipped and continue routing native memory and cca-lite. A missing MemPalace backend should not block the durable-memory pass.

### 3. cca-lite memory
Use `hermes-memory.json` for:
- repo-local durable facts
- decisions
- invariants
- operational facts
- known pitfalls
- open questions that belong to the project record

## Workflow

### Step 1: Identify candidate facts
Read the current session and extract only facts that seem durable.

Ask:
- Would this still matter in a future session?
- Is this specific to the repo or workflow?
- Is this worth reviewing in git history?

If not, do not promote it.

### Step 2: Classify each fact
For every candidate, choose exactly one canonical home:
- native memory
- MemPalace
- cca-lite memory

If the fact is ambiguous, keep it out of the durable record and capture it as an open question.

### Step 3: Update `hermes-memory.json`
Queue the candidate into `docs/cca-lite/hermes-memory-inbox.json` first, then promote it into the ledger.

Add or refine entries in:
- `entries`
- `open_questions`
- `tombstones`

Prefer short, stable, reviewable statements over long narrative summaries.

### Step 4: Validate the file
Check that:
- the JSON parses cleanly
- required fields exist
- content hashes are present for promoted entries
- superseded or tombstoned items are marked explicitly

Suggested command:

```bash
python3 tools/hermes_memory_promote.py --repo /path/to/cca-repo validate
```

If the file format has project-specific validation scripts later, use those too.

### Step 5: Commit the update
Commit the memory update with a clear message such as:

```bash
python3 tools/hermes_memory_promote.py --repo /path/to/cca-repo sync
```

If you need to commit manually, use:

```bash
git add docs/cca-lite/hermes-memory.json docs/cca-lite/hermes-memory-inbox.json docs/cca-lite/hermes-memory-spec.md
git commit -m "docs: promote cca-lite memory"
```

## Common Pitfalls

1. Confusing the three stores
   - Native memory is not the repo ledger.
   - MemPalace is not the source of truth for project state.
   - `hermes-memory.json` should not be used for personal preferences.

2. Promoting too early
   - If a fact is speculative or likely to change, keep it as an open question.

3. Duplicating the same fact everywhere
   - Pick one canonical home.
   - Mirror only if there is a clear reason.

4. Writing long transcript-like entries
   - Use concise durable statements.
   - The file should behave like a ledger, not a chat export.

5. Forgetting to validate before commit
   - Parse the JSON first.
   - Confirm hashes and supersession markers are present where needed.

6. Treating a missing MemPalace backend as fatal
   - The router should degrade gracefully when the plugin is unavailable.
   - Native memory and cca-lite should still receive eligible facts.

## Verification Checklist

- [ ] Candidate facts were classified into exactly one canonical home
- [ ] Only durable repo-local facts were added to `hermes-memory.json`
- [ ] JSON validates cleanly
- [ ] Hashes are present for promoted entries
- [ ] Open questions are left unresolved instead of being forced into facts
- [ ] The change was committed to git with a clear message
- [ ] No user preference was misplaced into repo-local memory
