# CCA Three-Stage Memory Routing Implementation Plan

> **For Hermes:** Use the `subagent-driven-development` skill to implement this plan task-by-task.

**Goal:** Make Hermes automatically route durable memory through three canonical paths — native memory, MemPalace, and repo-local cca-lite — at manual trigger time, compression pre-flight, and normal session end.

**Architecture:** Introduce one shared memory-routing orchestration layer that fans out to the three canonical stores instead of treating them as separate ad hoc code paths. The orchestration layer should be called from the existing compression pre-flight hook, the session-end hook used by `/quit` and `/reset`, and a short manual trigger command (`cca-remember`). Because `MemoryManager` currently allows only one external provider, this plan assumes the router lives above the provider limit and coordinates the existing built-in memory, the MemPalace plugin, and the cca-lite ledger bridge as coordinated sinks.

**Tech Stack:** Python, Hermes memory-provider hooks, Hermes CLI slash commands, MemPalace plugin, repo-local cca-lite helper/script, pytest.

---

## Design Decisions to Lock Before Coding

1. Keep one canonical home per fact.
   - Native memory = user preferences and stable personal corrections.
   - MemPalace = cross-session semantic recall and relationships.
   - cca-lite = repo-local durable project/workflow memory.

2. Reuse the same router from all entry points.
   - Manual: `cca-remember`
   - Pre-compress: `on_pre_compress()`
   - End-of-session: `on_session_end()` / normal `/quit`

3. Preserve failure isolation.
   - If one store fails, the other two still run.
   - No single backend should block shutdown or compression.
   - If MemPalace is unavailable or cannot import, treat that sink as skipped and continue with native memory / cca-lite rather than failing the full routing pass.

4. Prevent duplicate promotion.
   - The router must dedupe facts across invocations and across destinations.
   - A fact should not be promoted twice just because both compression and session end fire in the same conversation.
   - Dedupe rule: canonicalize the candidate text, compute a stable SHA-256 fingerprint from the canonical payload, and suppress any candidate whose fingerprint already exists in the current-session dedupe cache or in the target store's recent/current ledger keys.
   - Canonicalization rules for text:
     - normalize Unicode to NFKC
     - convert CRLF/CR line endings to LF
     - trim leading/trailing whitespace
     - collapse runs of spaces and tabs to a single space
     - collapse 3+ consecutive blank lines to 2 blank lines
     - lowercase the final canonical text for comparison
   - Canonical payload should be serialized as stable-key-order JSON with exactly these fields in this order: destination, entry_kind, canonical_text, source_id.
   - `source_id` should be the stable upstream identifier when one exists; otherwise use an empty string.
   - Exclude ephemeral fields like timestamps, session-local ordering, transport metadata, prompt wording, and transient scores.
   - If the same fact is legitimately needed in two different stores, the fingerprints may differ by destination, but each store still dedupes internally.

5. Keep the cca-lite bridge configurable.
   - The repo path / helper command should not be hardcoded.
   - The manual trigger should use the same bridge logic as automatic hooks.

---

## Router Contract

The implementation should expose one internal helper that all three entry points call.

### Proposed signature

```python
route_memory_candidates(
    *,
    invocation_mode: Literal["manual", "pre_compress", "session_end"],
    session_id: str,
    messages: list[dict[str, Any]],
    hermes_home: str,
    source_event: str = "",
    candidate_window: int | None = None,
) -> dict[str, Any]
```

### Input rules

- `invocation_mode` is required and controls whether the call came from the manual command, compression pre-flight, or session teardown.
- `messages` must be the exact message slice relevant to the invocation.
- `candidate_window` is optional and, if provided, limits how much recent history is considered when extracting candidates.
- The helper must not mutate the input message list.

### Output rules

Return a compact dict with these top-level keys:

- `ok`: boolean
- `mode`: the invocation mode that ran
- `session_id`: the active session id
- `routed`: list of accepted candidates, each with:
  - `destination`
  - `entry_kind`
  - `fingerprint`
  - `status` (`written`, `updated`, or `no_op`)
- `suppressed`: list of suppressed duplicates, each with:
  - `destination`
  - `entry_kind`
  - `fingerprint`
  - `reason` (`session_duplicate`, `store_duplicate`, or `disabled`)
- `failed`: list of sink failures, each with:
  - `destination`
  - `error`
- `summary`: short human-readable status string

### Routing order

1. Extract candidate facts from the supplied messages.
2. Canonicalize each candidate using the agreed dedupe normalization.
3. Check the current-session dedupe cache before any sink writes.
4. Route accepted candidates to native memory, MemPalace, and cca-lite in that order.
5. Record store-specific fingerprints or ids so later invocations can suppress duplicates.
6. Return the compact status object even if one or more sinks failed.

### Failure policy

- A failure in one destination must not stop the others.
- `ok` should remain true if at least one destination succeeded and no unrecoverable error occurred.
- `ok` should be false only when the router itself cannot execute its extraction or canonicalization step.

---

## Task 1: Add a shared memory-routing orchestration module

**Objective:** Create a single internal entry point that classifies candidates and dispatches them to native memory, MemPalace, and cca-lite.

**Files:**
- Create: `agent/memory_router.py`
- Modify: `agent/memory_provider.py`
- Modify: `agent/memory_manager.py` if needed for a router-facing helper
- Modify: `references/memory-routing.md` if the policy needs one canonical operational spec in-tree

**Implementation shape:**
- Define a small dataclass / dict contract for routed candidates.
- Provide one public function that accepts session context, messages, and an invocation mode (`manual`, `pre_compress`, `session_end`).
- Route candidates to:
  - built-in memory write path
  - MemPalace sink
  - cca-lite sink/bridge
- Return a compact result summary with per-sink status.

**Verification:**
- Add unit tests for the router contract and dedupe behavior.
- Verify that a failure in one sink does not abort the others.

---

## Task 2: Wire the router into compression pre-flight

**Objective:** Ensure every compression event runs the 3-stage memory pass before tokens are discarded.

**Files:**
- Modify: `run_agent.py` at the existing `on_pre_compress` call site around `_compress_context()`
- Modify: `plugins/memory/mempalace/__init__.py` if the MemPalace plugin should delegate through the shared router instead of doing its own isolated write logic
- Modify: `agent/memory_provider.py` if hook signatures need richer context

**Implementation shape:**
- Keep the current pre-compress hook, but have it call the shared router.
- Preserve current compression behavior even if the router returns errors.
- Make sure the router sees the exact message slice about to be compressed.

**Verification:**
- Add a regression test that pre-compress calls the router exactly once per compression event.
- Add a test that compression still proceeds when the router throws.

---

## Task 3: Wire the router into session end / `/quit`

**Objective:** Guarantee normal session shutdown runs the same memory pass one last time.

**Files:**
- Modify: `run_agent.py` session shutdown flow around `on_session_end()` and `shutdown_all()`
- Modify: `cli.py` only if the `/quit` path bypasses the existing shutdown hook
- Modify: `gateway/run.py` only if gateway exit paths skip the existing end-of-session hook

**Implementation shape:**
- Reuse the existing session-end lifecycle hook instead of inventing a separate shutdown path.
- Ensure `/quit`, `/reset`, and clean gateway termination all run the final router pass.
- Do not make the shutdown path depend on network success.

**Verification:**
- Add tests proving the session-end hook fires on normal quit.
- Add tests proving the router runs once at shutdown even if compression already ran earlier.

---

## Task 4: Add the manual `cca-remember` trigger

**Objective:** Give the user a short explicit command that invokes the exact same router contract on demand.

**Files:**
- Modify: `hermes_cli/commands.py`
- Modify: `cli.py`
- Modify: `gateway/run.py` and any platform adapters needed for the command to work everywhere it should
- Add: tests for the new command path

**Implementation shape:**
- Register `cca-remember` as a thin alias/front door onto `route_memory_candidates(...)`.
- It should set `invocation_mode="manual"` and reuse the same extraction, canonicalization, dedupe, and sink dispatch logic as automatic hooks.
- The manual command should use the full session as its default collection window, since the working context is typically ~200k tokens before compression; keep that collection choice consistent across CLI and gateway, and do not give the command its own promotion semantics.
- Make the command best-effort and idempotent.

**Verification:**
- Add a CLI test that the command invokes the router.
- Add a gateway test that the command is dispatched correctly.

---

## Task 5: Add the cca-lite bridge/configuration layer

**Objective:** Connect the Hermes-side router to the repository-local cca-lite ledger without hardcoding machine-specific paths.

**Files:**
- Add or modify: a cca-lite config section in Hermes config handling
- Add: `references/memory-routing.md` or related docs explaining repo-path configuration
- Possibly modify: `plugins/memory` code if the bridge lives there

**Implementation shape:**
- Add config keys for the cca-lite repo root, ledger path, inbox path, and promotion command/script.
- Resolve the bridge command/path from config at runtime.
- Keep the bridge compatible with the existing `docs/cca-lite/hermes-memory.json` + `tools/hermes_memory_promote.py` flow.

**Verification:**
- Add a test that the bridge uses configured paths rather than a hardcoded repo.
- Add a test that the bridge no-ops cleanly when cca-lite is disabled.

---

## Task 6: Build the test matrix and regression coverage

**Objective:** Prove the three entry points all land in the same durable routing path.

**Files:**
- Create: `tests/agent/test_memory_router.py`
- Modify: `tests/agent/test_memory_provider.py`
- Modify: `tests/run_agent/test_compression_*` or add a focused regression test file
- Add: `tests/cli/test_cca_remember_command.py` if needed

**Test cases:**
- manual trigger invokes the same helper as compression
- compression pre-flight runs all three destinations
- session end runs the same helper
- duplicate facts are not promoted twice
- one sink failing does not block the others
- `/quit` and `/reset` both finalize memory

**Verification:**
- Run the targeted pytest files first.
- Then run the relevant Hermes test slice for run_agent / CLI command dispatch.

---

## Task 7: Update docs and rollout notes

**Objective:** Make the behavior discoverable and safe to operate.

**Files:**
- Modify: `references/memory-routing.md`
- Modify: any Hermes skill docs that mention memory workflows
- Modify: release/notes docs if the repo uses them

**Implementation shape:**
- Document the three trigger points and the canonical home for each memory type.
- Document that `cca-remember` is optional/manual, while compression and session end are automatic.
- Document the best-effort nature of shutdown hooks.

**Verification:**
- Confirm docs match the actual hooks and file paths.
- Confirm the plan and implementation names are consistent across Hermes and cca-repo.

---

## Recommended Implementation Order

1. Router module + tests
2. Compression pre-flight hook
3. Session-end hook
4. Manual `cca-remember` command
5. Cca-lite bridge/configuration
6. Full regression test pass
7. Documentation updates

---

## Acceptance Criteria

- The user can invoke `cca-remember` manually.
- Compression pre-flight automatically runs native memory, MemPalace, and cca-lite routing.
- Normal session end / `/quit` also runs the same 3-stage routing pass.
- No duplicate promotion is introduced.
- A failure in one memory store does not block the others.
- The behavior is covered by tests and documented in-tree.

---

## Notes / Open Questions

- Decide whether the cca-lite bridge should shell out to the repo-local promotion script or call a Python module directly.
- Decide whether the manual trigger should operate on the whole session or only the latest message window.
- Decide whether the router should live in `agent/` or inside the MemPalace plugin as a shared helper.
- Confirm the final config keys before implementation so path handling stays stable.
