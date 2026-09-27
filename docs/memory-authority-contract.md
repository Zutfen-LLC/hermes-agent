# Memory Authority Contract — Engram as Durable Substrate (Issue #23)

Status: accepted-by-prototype (2026-09-27) — bounded prototype executed against the
live Engram deployment; see "Prototype evidence" below.
Supersedes the direction of PR #22 (closed, unmerged).
Builds on #19 (merged): the safety behavior there is retained per the issue's
disposition rules.

## 1. Layer authority

Three layers, one authoritative:

| Layer | Role | Authority |
|---|---|---|
| **Engram** | Durable learned context: lessons, facts, procedures, provenance, review state | **Authoritative for everything long-lived** |
| Local `MEMORY.md` / `USER.md` | Bootstrap/fallback/compat store: compact operator facts, offline/degraded operation | Non-authoritative once Engram holds the fact |
| Skills (`SKILL.md` + files) | Compact procedural instructions + a retrieval contract into Engram | Authoritative only for procedure, never for learned lessons |

Rules:

1. When Engram and a local file disagree, **Engram wins** for any fact that has
   been written to Engram. Local edits never promote themselves to authority.
2. A write that cannot reach Engram fails **visibly** (explicit error surface);
   it is never silently rerouted into a local file that would then claim
   authority. Degraded-mode local captures (if a future issue adds them) must be
   labeled `proposed-out-of-band` and enter the review queue on recovery — they
   are never equivalent to Engram-committed facts.
3. Engram carries the trust machinery (#19's local lifecycle machinery is not
   duplicated): proposed→active review, dispute, supersession, provenance,
   source-trust defaults, tenant scoping.

## 2. Bootstrap / degraded / offline behavior

| Mode | Behavior |
|---|---|
| Normal | Startup recall + semantic recall flow through the Engram bridge (`~/.hermes/plugins/engram_memory`); native durable adds are governed through the MemoryProvider face |
| Engram degraded/outage | Recollection fails **visibly** (breaker + rendered unavailability markers); retrieval helpers exit non-zero with one stderr line and empty stdout. The agent proceeds on skills' universal procedures + local MEMORY.md bootstrap facts and **says so in its report** |
| Offline / no Engram configured | Same as degraded; skills work (procedure survives), learned-lesson retrieval does not — by contract, and visibly |
| Fresh bootstrap | `MEMORY.md`/`USER.md` keep their current role: small operator-written facts load at session start; nothing in this contract changes first-run behavior |

The local files' minimal contract is deliberately unchanged from upstream:
direct writes, existing write-approval, existing drift/data-loss guards.
No new local lifecycle states are added.

## 3. Skill-associated memory representation (Engram schema)

A skill-associated lesson is one Engram memory item:

```
kind          = skill_lesson        (tenant-governed custom kind; registered once per tenant)
wing          = skills
room          = <skill-name>        (deterministic per-skill namespace filter)
subject_type  = skill               (added to the subject_type vocabulary; engram migration 048)
subject_id    = <skill-name>
subject_name  = <lesson group / trigger>
content       = "[<skill> / <group>] <lesson text>"
visibility    = tenant              (readable team-wide within the tenant; never cross-tenant)
source_type   = migration | manual  (migration: bulk import; manual: operator/agent writes)
external_source = hermes-skill-migration
external_id   = <skill-name>:<seq>:<slug>   (idempotent re-import key)
```

Governance (all enforced server-side, none by client discipline):

- The writer identity is a **dedicated agent principal** (read+write) created via
  `POST /v1/agents`; agent-type writes of `skill_lesson` land `review_status=proposed`.
- Activation (`proposed→active`) requires principal_type ∈ {user, admin} **and**
  `review` scope (Engram review policy) — an agent cannot self-activate a lesson.
- Dispute/withdrawal paths, supersession, and dedup (content hash / external id
  index) are Engram-native and reused unchanged.

## 4. Retrieval contract (how a skill reaches its lessons)

A skill that accumulates lessons carries, in its SKILL.md, a compact retrieval
directive instead of the lessons themselves:

1. **Activation-time automatic render (bounded index).** One inline-shell
   snippet renders the one-line-per-lesson index for the skill's room, hard-capped:

   `!`${HERMES_SKILL_DIR}/scripts/engram_skill_lessons.py --room <skill-name> --index --max-chars 3000``

   Hermes renders `` !`cmd` `` snippets at skill activation
   (`agent/skill_preprocessing.py`, opt-in via `skills.inline_shell: true`,
   output elided at 4000 chars — the cap is structural, not advisory).

2. **Task-scoped narrowed retrieval.** When the task matches a trigger in the
   index, the agent runs the same helper with `--query "<trigger>"`
   (`--mode semantic` optional) to pull the top-N full lessons.

3. **Helper contract** (`scripts/engram_skill_lessons.py`, shipped inside the
   skill): deterministic full-namespace listing (identical results across
   repeated calls), bounded render (`--max-chars` head+tail elision),
   failure = exit 3 + one stderr line + **empty stdout** (no local fallback
   copy exists to silently substitute).

4. **Add path.** New lessons are written with POST /v1/remember using the §3
   schema; they surface after operator promotion. The skill file itself never
   grows with lessons.

## 5. Minimum local-memory contract (post-convergence)

`MEMORY.md` / `USER.md` remain exactly what they are today in stock Hermes:

- safe direct writes through the existing store (locking, atomic replace);
- existing write approval/rejection gating;
- existing drift detection and data-loss guards protecting the actual files;
- char budgets unchanged.

Explicitly **dropped from future investment** (was PR #22's direction):
recoverable `blocked` lifecycle states, structured failure-kind taxonomy,
legacy-invalid migration, preflight/write-path parity harnesses for the
pending queue. The local pending-write queue stays at its current #19 shape
and is not developed into a second durable-memory platform.

## 6. #19 disposition matrix

| #19 mechanism | Disposition |
|---|---|
| Pinned matched-entry destructive ops; fail-closed apply (nothing on any failure) | **Stays: hard safety invariant** |
| Append-only audit ledger + `/memory undo`, owner-only custody | **Stays: hard safety invariant** |
| Write approval gate; `memory.allow_unattended_consolidation` default-false opt-in | **Stays: operator control** |
| Existing backlog non-auto-application; rejected-terminal evidence | **Stays: operator control** |
| Pending-queue status lifecycle (ready/stale/superseded/invalid/rejected) | **Compatibility-only**: kept working as-is; no new states; no new semantics |
| Background-review digest of the pending queue | **Compatibility-only**: bounded digest remains; no further expansion |
| Further lifecycle polish (blocked states, failure kinds, migration of legacy pending records) | **Abandoned** (PR #22 direction closed) |

## 7. Migration path (existing content → Engram)

Mechanics proven by the prototype script (committed under
`scripts/memory-engram-migration/`):

1. Parse the skill library (`references/*.md` lesson blocks) into discrete
   lessons; preserve group + numbering.
2. Import via `POST /v1/remember` as `source_type=migration`, executed by the
   dedicated **agent principal** of §3 (`hermes-skills-lessons`). Trust and
   confidence come from Engram's import/migration policy defaults (0.8/0.8
   absent tenant override), but an agent principal never auto-activates: the
   write lands `review_status='proposed'` and becomes retrievable only when a
   human review authority — a `user` or `admin` principal holding the required
   `review` scope — promotes it through the review queue. The agent cannot
   self-activate (server-enforced 403). `external_source=hermes-skill-migration`
   + deterministic `external_id` keep re-runs idempotent (probed before POST;
   dedup also enforced server-side). A differently authorized migration — one
   executed by a `user`, `admin`, or `system` principal — may carry different
   initial-state semantics under Engram's canonical trust policy
   (`resolve_trust_defaults()` in `engram/trust_policy.py` is authoritative);
   this contract does not hard-code a client-side policy for the source type.
3. Reduce the skill file to procedure + retrieval directive; freeze the local
   library file as a migration snapshot (never a retrieval source).
4. For `MEMORY.md`/`USER.md`: operator-approved export of durable facts into
   Engram (`source_type=migration`, kind=fact/preference/doctrine as
   appropriate); local files shrink but are not deleted (bootstrap role
   remains). **Not executed in this prototype slice** — follow-up issue.

## 8. Prototype evidence (executed 2026-09-27 against the live deployment)

- Skill: `review-gated-correction-passes` (35.8 KB SKILL.md + 76.4 KB
  correction-pattern-library ≈ 112 KB embedded) — 22 lessons migrated,
  reviewer-activated.
- Determinism: `GET /v1/items?kind=skill_lesson&wing=skills&room=<skill>`
  returned identical result lists across repeated calls (26–35 ms).
- Isolation (no flood): a different skill's room returns 0 lessons.
- Boundedness: activation renders a 27-line / 3.9 kB index (~1k tokens);
  full-text narrowing is task-scoped; Hermes' inline-shell 4000-char
  structural cap bounds any snippet regardless.
- Growth: new lesson via API → retrievable after promotion; `SKILL.md`
  hash unchanged before/after (md5 compared).
- Failure behavior: bad key → HTTP 401 → exit 3 + visible stderr marker;
  unreachable Engram (closed port) → instant exit 3 + visible marker;
  both through the real `skill_view(preprocess=True)` pipeline.
- Authority: agent-principal write lands `proposed` and is NOT retrievable
  until a human-type reviewer with `review` scope activates it (self-activation
  attempt returns 403 — server-enforced).
- E2E: `skill_view("review-gated-correction-passes", preprocess=True)` renders
  the live Engram index inline in the activation content (40.1 KB total,
  down from ~112 KB embedded).

## 9. Follow-up decomposition (filed as issues, out of scope here)

1. Wire `skills.inline_shell: true` as the documented default for profiles
   using Engram retrieval (+ setup UX surface).
2. `/memory`-adjacent operator UX for the `skill_lesson` review queue
   (list/promote/reject without raw curl).
3. MEMORY.md/USER.md durable-fact export into Engram (operator-approved,
   idempotent, provenance-preserving).
4. Profile-level deployment of the dedicated skills writer identity
   (provisioning docs + `hermes` setup hook), retiring direct owner-DB
   bootstrap.
5. Native `pre_llm_call`-path retrieval (plugin-side, no core change) for
   skills whose flow cannot use inline-shell rendering.

## 10. Non-goals honored

No Engram reimplementation inside Hermes; no co-equal authority for markdown;
no new vector store; no removal of offline capability (it is preserved, as a
visible degraded mode); no deletion of #19 safeguards.
