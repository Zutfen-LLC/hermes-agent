# Migration snapshot (issue #23)

`engram_skill_lessons.py` is the reference implementation of the
skill-lesson migration + retrieval contract
(`docs/memory-authority-contract.md` §3, §4, §7), executed 2026-09-27
against the live Engram deployment with the
`review-gated-correction-passes` skill.

It is committed as a **repo artifact** so the prototype is reproducible;
production skills should carry their own copy in
`scripts/engram_skill_lessons.py` inside the skill directory (the
activation snippet renders from `${HERMES_SKILL_DIR}`).

Usage:

```bash
export ENGRAM_BASE_URL=...      # deployed Engram
export ENGRAM_API_KEY=...       # read+write agent key (per-profile agent principal)

# one-time migration of a local lesson library (idempotent):
python scripts/memory-engram-migration/engram_skill_lessons.py \
    migrate ~/.hermes/skills/<cat>/<skill>/references/correction-pattern-library.md \
    --skill <skill-name>

# deterministic retrieval (what the skill's inline-shell snippet runs):
python scripts/memory-engram-migration/engram_skill_lessons.py \
    retrieve --room <skill-name> --index --max-chars 3000

# narrowed top-N for a matched trigger:
python scripts/memory-engram-migration/engram_skill_lessons.py \
    retrieve --room <skill-name> --query "<trigger>" --mode semantic --limit 3
```

Writes land `review_status=proposed` (agent principal); promote through
the Engram review queue (`POST /v1/items/<id>/review` with a user/admin
principal holding `review` scope).
