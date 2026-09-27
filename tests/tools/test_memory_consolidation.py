"""Autonomous unattended memory consolidation (#106919): the opt-in flag
(``memory.allow_unattended_consolidation``), the apply path in
``tools/memory_tool.py`` (pin -> audit snapshot -> atomic public store path),
and the audit ledger + undo in ``tools/memory_consolidation.py``.

Default (flag off) keeps the #105921 stage-for-approval behavior; the approval
gate stays authoritative even when the flag is on.
"""

import hashlib
import json
from contextlib import contextmanager

import pytest

from tools.memory_tool import MemoryStore, memory_tool
from tools.skill_provenance import (
    reset_current_write_origin,
    reset_review_attended,
    set_current_write_origin,
    set_review_attended,
)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes-home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture()
def store(home):
    s = MemoryStore(memory_char_limit=500, user_char_limit=300)
    s.load_from_disk()
    return s


def _set_flag(value):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault("memory", {})["allow_unattended_consolidation"] = value
    cfg.save_config(c)


def _set_approval(enabled):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault("memory", {})["write_approval"] = enabled
    cfg.save_config(c)


@contextmanager
def unattended_review():
    token = set_current_write_origin("background_review")
    try:
        yield
    finally:
        reset_current_write_origin(token)


def _disk_entries(store, target="memory"):
    store.load_from_disk()
    return list(store._entries_for(target))


# =========================================================================
# Existing pending backlog: enabling the option never auto-applies staged work
# =========================================================================

class TestExistingBacklogNotAutoApplied:
    def test_enabling_the_flag_leaves_a_staged_proposal_pending(self, store):
        """Work staged under the old consent model is never applied because the option
        was switched on afterwards: the flag is only consulted when a NEW destructive
        proposal is written, and merely listing/re-classifying the queue never applies."""
        from tools import write_approval as wa
        _set_flag(False)
        _set_approval(False)
        store.add("memory", "entry staged under the old policy")
        store.add("memory", "keeper entry")
        with unattended_review():
            staged = json.loads(memory_tool(action="remove", old_text="entry staged",
                                            store=store))
        assert staged["staged"] is True

        _set_flag(True)  # operator opts in AFTER the proposal was staged
        from tools.write_approval import reclassify_pending_memory
        counts = reclassify_pending_memory(store, wa.MEMORY)
        assert counts["ready"] == 1 and counts["changed"] == 0
        assert "entry staged under the old policy" in _disk_entries(store)  # untouched
        records = wa.list_pending(wa.MEMORY)
        assert len(records) == 1 and (records[0].get("status") or "ready") == "ready"


# =========================================================================
# Default safety: the flag is off unless explicitly opted in
# =========================================================================

class TestDefaultSafety:
    def test_flag_absent_unattended_remove_stages(self, store):
        store.add("memory", "never delete the standing rule")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="standing rule", store=store))
        assert r["success"] is True and r["staged"] is True and r["proposal_staged"] is True
        assert "never delete the standing rule" in _disk_entries(store)

    def test_flag_false_explicit_unattended_remove_stages(self, store):
        _set_flag(False)
        store.add("memory", "rule that must survive")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="must survive", store=store))
        assert r.get("proposal_staged") is True
        assert "rule that must survive" in _disk_entries(store)

    def test_write_approval_on_beats_flag_on(self, store):
        _set_flag(True)
        _set_approval(True)
        store.add("memory", "gate wins over the opt-in")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="gate wins", store=store))
        assert r.get("proposal_staged") is True
        assert "gate wins over the opt-in" in _disk_entries(store)


# =========================================================================
# Opt-in: unattended consolidation applies directly, with audit + undo hint
# =========================================================================

class TestOptInApplies:
    def test_unattended_remove_applies_with_audit(self, store):
        from tools import write_approval as wa
        _set_flag(True)
        store.add("memory", "stale fact to drop")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="stale fact", store=store))
        assert r["success"] is True and r["done"] is True
        assert r["autonomously_consolidated"] is True
        assert r["audit_id"]
        assert "/memory undo" in r["message"]
        assert "stale fact to drop" not in _disk_entries(store)
        assert wa.pending_count("memory") == 0

    def test_unattended_replace_applies_with_audit(self, store):
        _set_flag(True)
        store.add("memory", "old wording kept for ages")
        with unattended_review():
            r = json.loads(memory_tool(action="replace", old_text="old wording",
                                       content="new consolidated wording", store=store))
        assert r["success"] is True and r["autonomously_consolidated"] is True
        assert r["audit_id"]
        # Same contract as the approve path: the exact overwritten entry is surfaced.
        assert r["replaced_entry"] == "old wording kept for ages"
        entries = _disk_entries(store)
        assert entries == ["new consolidated wording"]

    def test_unattended_atomic_batch_applies(self, store):
        from tools import write_approval as wa
        _set_flag(True)
        store.add("memory", "superseded entry")
        with unattended_review():
            r = json.loads(memory_tool(operations=[
                {"action": "remove", "old_text": "superseded entry"},
                {"action": "add", "content": "the consolidated successor"},
            ], store=store))
        assert r["success"] is True and r["autonomously_consolidated"] is True
        assert wa.pending_count("memory") == 0
        assert _disk_entries(store) == ["the consolidated successor"]

    def test_foreground_with_flag_on_no_audit(self, store):
        from tools.memory_consolidation import list_records
        _set_flag(True)
        store.add("memory", "foreground entry")
        r = json.loads(memory_tool(action="replace", old_text="foreground entry",
                                   content="foreground rewrite", store=store))
        assert r["success"] is True and "autonomously_consolidated" not in r
        assert list_records() == []

    def test_attended_review_with_flag_on_no_audit(self, store):
        from tools.memory_consolidation import list_records
        _set_flag(True)
        store.add("memory", "entry an explicit refine may rewrite")
        token = set_current_write_origin("background_review")
        att = set_review_attended(True)
        try:
            r = json.loads(memory_tool(action="replace", old_text="explicit refine",
                                       content="refined wording", store=store))
        finally:
            reset_review_attended(att)
            reset_current_write_origin(token)
        assert r["success"] is True and "autonomously_consolidated" not in r
        assert list_records() == []


# =========================================================================
# Validation: fail closed, store unchanged, nothing staged
# =========================================================================

class TestValidationFailClosed:
    def test_missing_anchor_refused(self, store):
        from tools import write_approval as wa
        _set_flag(True)
        store.add("memory", "unrelated standing entry")
        before = _disk_entries(store)
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="no such entry", store=store))
        assert r["success"] is False
        assert _disk_entries(store) == before
        assert wa.pending_count("memory") == 0

    def test_ambiguous_anchor_refused(self, store):
        _set_flag(True)
        store.add("memory", "alpha shared token")
        store.add("memory", "beta shared token")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="shared token", store=store))
        assert r["success"] is False
        assert len(_disk_entries(store)) == 2

    def test_stale_target_refused(self, store):
        _set_flag(True)
        store.add("memory", "entry that vanishes first")
        # An external writer rewrites MEMORY.md, dropping the entry.
        store._path_for("memory").write_text("only a replacement fact", encoding="utf-8")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="vanishes first", store=store))
        assert r["success"] is False
        assert _disk_entries(store) == ["only a replacement fact"]

    def test_concurrent_target_drift_applies_nothing(self, store, monkeypatch):
        """A writer that changes the pinned entry AFTER the fresh load/pin and BEFORE the
        commit makes the whole consolidation a no-op: the commit carries a full-store
        precondition checked under the store lock, and the pinned replay re-reads disk —
        either refuses rather than remove a different entry."""
        from tools import memory_consolidation as mc
        real_begin = mc.record_begin

        def drift_then_begin(target, ops, before_raw, before_entries, **kwargs):
            # Simulate a concurrent writer landing between the audit snapshot and the apply.
            store._path_for(target).write_text(
                "the entry was rewritten by someone else", encoding="utf-8")
            return real_begin(target, ops, before_raw, before_entries, **kwargs)

        monkeypatch.setattr(mc, "record_begin", drift_then_begin)
        _set_flag(True)
        store.add("memory", "entry pinned for consolidation")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="entry pinned", store=store))
        assert r["success"] is False
        assert _disk_entries(store) == ["the entry was rewritten by someone else"]

    def test_unrelated_entry_drift_between_audit_and_commit_applies_nothing(self, store, monkeypatch):
        """THE audit-before-state invariant (correction round): an unrelated writer adds
        an entry AFTER the audit snapshot and BEFORE the commit. The commit legitimately
        would preserve it — but the audit 'before' does not contain it, so a later undo
        would erase it. The full-store precondition refuses the WHOLE commit under the
        lock: nothing is applied, the concurrent write stays."""
        from tools import memory_consolidation as mc
        real_begin = mc.record_begin

        def unrelated_write_then_begin(target, ops, before_raw, before_entries, **kwargs):
            # A concurrent session adds an entry the consolidation never touches.
            path = store._path_for(target)
            path.write_text(before_raw + "\n§\nwritten by an unrelated session", encoding="utf-8")
            return real_begin(target, ops, before_raw, before_entries, **kwargs)

        monkeypatch.setattr(mc, "record_begin", unrelated_write_then_begin)
        _set_flag(True)
        store.add("memory", "consolidation target entry")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="consolidation target", store=store))
        assert r["success"] is False
        entries = _disk_entries(store)
        assert "consolidation target entry" in entries  # the remove did NOT apply
        assert "written by an unrelated session" in entries  # the concurrent write survives

    def test_targeted_entry_drift_applies_nothing(self, store, monkeypatch):
        """Targeted drift (the pinned entry itself changes) is equally refused by the
        same full-store precondition — covered separately from the unrelated-entry case."""
        from tools import memory_consolidation as mc
        real_begin = mc.record_begin

        def rewrite_target_then_begin(target, ops, before_raw, before_entries, **kwargs):
            store._path_for(target).write_text(
                before_raw.replace("drifting entry", "drifting entry (edited concurrently)"),
                encoding="utf-8")
            return real_begin(target, ops, before_raw, before_entries, **kwargs)

        monkeypatch.setattr(mc, "record_begin", rewrite_target_then_begin)
        _set_flag(True)
        store.add("memory", "drifting entry")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="drifting", store=store))
        assert r["success"] is False
        assert _disk_entries(store) == ["drifting entry (edited concurrently)"]

    def test_concurrent_batch_drift_applies_nothing(self, store, monkeypatch):
        """Batch variant: one drifted pinned entry voids the entire batch (all-or-nothing)."""
        from tools import memory_consolidation as mc
        from tools.memory_tool_store import ENTRY_DELIMITER
        real_begin = mc.record_begin
        drifted_raw = ENTRY_DELIMITER.join(
            ["drifted pinned entry", "a second untouched entry"])

        def drift_then_begin(target, ops, before_raw, before_entries, **kwargs):
            store._path_for(target).write_text(drifted_raw, encoding="utf-8")
            return real_begin(target, ops, before_raw, before_entries, **kwargs)

        monkeypatch.setattr(mc, "record_begin", drift_then_begin)
        _set_flag(True)
        store.add("memory", "first entry to consolidate")
        store.add("memory", "second entry to consolidate")
        with unattended_review():
            r = json.loads(memory_tool(operations=[
                {"action": "remove", "old_text": "first entry to consolidate"},
                {"action": "add", "content": "an addition that must not land"},
            ], store=store))
        assert r["success"] is False
        # The concurrent writer's content is intact: not the remove, not the add.
        assert store._path_for("memory").read_text(encoding="utf-8") == drifted_raw
        assert _disk_entries(store) == ["drifted pinned entry", "a second untouched entry"]

    def test_over_budget_replace_refused(self, home):
        _set_flag(True)
        s = MemoryStore(memory_char_limit=100, user_char_limit=100)
        s.load_from_disk()
        s.add("memory", "short seed entry")
        with unattended_review():
            r = json.loads(memory_tool(action="replace", old_text="short seed",
                                       content="X" * 150, store=s))
        assert r["success"] is False
        assert _disk_entries(s) == ["short seed entry"]

    def test_one_bad_op_in_batch_mutates_nothing(self, store):
        _set_flag(True)
        store.add("memory", "the good entry")
        with unattended_review():
            r = json.loads(memory_tool(operations=[
                {"action": "add", "content": "a good addition"},
                {"action": "remove", "old_text": "entry that does not exist"},
            ], store=store))
        assert r["success"] is False
        assert _disk_entries(store) == ["the good entry"]  # the good add is absent too

    def test_intermediate_over_budget_final_ok_batch_succeeds(self, home):
        _set_flag(True)
        s = MemoryStore(memory_char_limit=150, user_char_limit=100)
        s.load_from_disk()
        s.add("memory", "alpha entry one")
        s.add("memory", "beta entry two")
        # Intermediate state (140 + 14 + delimiters) would exceed 150; the final
        # state is just the 140-char replacement, so the atomic batch must pass.
        with unattended_review():
            r = json.loads(memory_tool(operations=[
                {"action": "replace", "old_text": "alpha entry one", "content": "C" * 140},
                {"action": "remove", "old_text": "beta entry two"},
            ], store=s))
        assert r["success"] is True and r["autonomously_consolidated"] is True
        assert _disk_entries(s) == ["C" * 140]

    def test_audit_failure_prevents_mutation(self, store, monkeypatch):
        from tools import memory_consolidation as mc
        from tools import write_approval as wa
        _set_flag(True)
        store.add("memory", "entry guarded by the audit gate")
        def _boom(*a, **kw):
            raise mc.MemoryConsolidationAuditError("ledger unwritable")
        monkeypatch.setattr(mc, "record_begin", _boom)
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="audit gate", store=store))
        assert r["success"] is False
        assert "audit" in r["error"].lower()
        assert "entry guarded by the audit gate" in _disk_entries(store)
        assert wa.pending_count("memory") == 0


# =========================================================================
# Recovery: the ledger and restore()
# =========================================================================

class TestRecovery:
    def _unattended_remove(self, store, old_text):
        _set_flag(True)
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text=old_text, store=store))
        assert r["success"] is True, r
        return r

    def test_ledger_has_begin_and_applied(self, store):
        from tools.memory_consolidation import list_records
        store.add("memory", "first fact to consolidate")
        store.add("memory", "second fact that stays")
        r = self._unattended_remove(store, "first fact")
        records = list_records()
        assert [rec["event"] for rec in records] == ["begin", "applied"]
        begin, applied = records
        assert begin["id"] == applied["id"] == r["audit_id"]
        assert begin["target"] == "memory" and begin["origin"] == "background_review"
        assert "first fact to consolidate" in begin["before_raw"]
        assert begin["before_entries"] == ["first fact to consolidate", "second fact that stays"]
        assert begin["ops"][0]["matched_entry"] == "first fact to consolidate"
        assert applied["counts"] == {"replaced": 0, "removed": 1, "added": 0}
        assert applied["undo_hint"] == f"/memory undo {r['audit_id']}"
        after_raw = store._read_raw_checked(store._path_for("memory"))[0]
        assert applied["after_sha256"] == hashlib.sha256(after_raw.encode("utf-8")).hexdigest()

    def test_restore_returns_exact_before_entries(self, store):
        from tools.memory_consolidation import restore
        store.add("memory", "alpha will be replaced")
        store.add("memory", "beta stays put")
        before = _disk_entries(store)
        _set_flag(True)
        with unattended_review():
            r = json.loads(memory_tool(action="replace", old_text="alpha will",
                                       content="alpha was rewritten by the fork", store=store))
        assert r["success"] is True
        assert _disk_entries(store) == ["alpha was rewritten by the fork", "beta stays put"]
        result = restore(r["audit_id"], store)
        assert result["success"] is True, result
        assert _disk_entries(store) == before  # exact, order included

    def test_restore_idempotent(self, store):
        from tools.memory_consolidation import restore
        store.add("memory", "sole entry removed then restored")
        r = self._unattended_remove(store, "sole entry")
        assert restore(r["audit_id"], store)["success"] is True
        again = restore(r["audit_id"], store)
        assert again["success"] is True
        assert "Already at the recorded before-state" in again["message"]
        assert _disk_entries(store) == ["sole entry removed then restored"]

    def test_restore_refuses_foreign_profile(self, store):
        from tools.memory_consolidation import list_records, restore
        store.add("memory", "entry in this profile only")
        r = self._unattended_remove(store, "this profile")
        # Rewrite the ledger's begin record as if it came from another Hermes home.
        from tools.memory_consolidation import _ledger_path
        lines = _ledger_path().read_text(encoding="utf-8").splitlines()
        assert lines
        rec = json.loads(lines[0])
        rec["hermes_home"] = "/somewhere/else/.hermes"
        lines[0] = json.dumps(rec, ensure_ascii=False)
        _ledger_path().write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = restore(r["audit_id"], store)
        assert result["success"] is False
        assert "different Hermes home" in result["error"]
        assert _disk_entries(store) == []  # nothing restored, nothing clobbered

    def test_undo_after_multi_op_batch(self, store):
        from tools.memory_consolidation import restore
        store.add("memory", "first original entry")
        store.add("memory", "second original entry")
        before = _disk_entries(store)
        _set_flag(True)
        with unattended_review():
            r = json.loads(memory_tool(operations=[
                {"action": "remove", "old_text": "first original"},
                {"action": "add", "content": "fresh fork conclusion"},
            ], store=store))
        assert r["success"] is True
        assert _disk_entries(store) == ["second original entry", "fresh fork conclusion"]
        assert restore(r["audit_id"], store)["success"] is True
        assert _disk_entries(store) == before  # exact, order included


# =========================================================================
# Correction round: the undo current-state precondition (transaction journal)
# =========================================================================

class TestUndoCurrentStatePrecondition:
    """restore() must PROVE the consolidation committed and that memory has not drifted
    since: begin-only (never committed / mutation failed) is a no-op; committed+drifted is
    REFUSED with current memory untouched; only current==expected-after restores."""

    def _applied_removal(self, store, victim="victim entry for undo"):
        _set_flag(True)
        store.add("memory", victim)
        store.add("memory", "keeper entry for undo")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text=victim[:14], store=store))
        assert r["success"] is True, r
        return r["audit_id"]

    def test_begin_only_record_undo_changes_nothing(self, store, monkeypatch):
        """Begin record + mutation failed => /memory undo <id> changes nothing. The
        reviewed implementation restored the before-state from the begin record alone,
        mutating memory for a transaction that never committed."""
        from tools import memory_consolidation as mc
        real_begin = mc.record_begin
        real_apply_batch = MemoryStore.apply_batch

        def refuse(*a, **kw):
            return {"success": False, "error": "commit failed (simulated)"}

        def begin_then_fail_apply(target, ops, before_raw, before_entries, **kwargs):
            audit_id = real_begin(target, ops, before_raw, before_entries, **kwargs)
            # Simulate the commit failing after the audit begin was durably recorded.
            # Patched on the CLASS (memory_tool imports the symbol into the module
            # under test) so both the consolidation path and restore() see it.
            monkeypatch.setattr(MemoryStore, "apply_batch", refuse)
            return audit_id

        monkeypatch.setattr(mc, "record_begin", begin_then_fail_apply)
        _set_flag(True)
        store.add("memory", "entry a begin only")
        store.add("memory", "entry b begin only")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="entry a", store=store))
        assert r["success"] is False
        # Find the begin-only id from the ledger (read while the patches are still
        # active; monkeypatch.undo() would ALSO undo the home fixture's HERMES_HOME).
        ledger_ids = [rec["id"] for rec in mc.list_records() if rec.get("event") == "begin"]
        assert ledger_ids
        monkeypatch.setattr(MemoryStore, "apply_batch", real_apply_batch)
        result = mc.restore(ledger_ids[0], store)
        assert result["success"] is True
        assert "nothing to restore" in result["message"]
        # Nothing was committed, nothing was rewritten.
        assert _disk_entries(store) == ["entry a begin only", "entry b begin only"]

    def test_begin_only_current_equals_before_is_noop_not_restore(self, store):
        """A failed consolidation intentionally leaves its begin record behind; current
        memory equals before, so undo must be an idempotent no-op — no rewrite at all."""
        from tools import memory_consolidation as mc
        _set_flag(True)
        store.add("memory", "survivor of failed consolidation")
        store.add("memory", "doomed by nothing")
        before_raw = store._read_raw_checked(store._path_for("memory"))[0]
        audit_id = mc.record_begin("memory", [{"action": "remove", "old_text": "doomed",
                                               "matched_entry": "doomed by nothing"}],
                                   before_raw, list(store._entries_for("memory")))
        # No apply ever happened; current raw is still the before raw.
        result = mc.restore(audit_id, store)
        assert result["success"] is True
        assert "nothing to restore" in result["message"]
        assert store._read_raw_checked(store._path_for("memory"))[0] == before_raw

    def test_committed_current_equals_after_restores_exact_before(self, store):
        from tools.memory_consolidation import restore
        audit_id = self._applied_removal(store, victim="alpha undo target")
        assert _disk_entries(store) == ["keeper entry for undo"]
        result = restore(audit_id, store)
        assert result["success"] is True, result
        assert _disk_entries(store) == ["alpha undo target", "keeper entry for undo"]  # exact, order included

    def test_committed_then_later_unrelated_add_undo_refused(self, store):
        from tools.memory_consolidation import restore
        audit_id = self._applied_removal(store)
        assert _disk_entries(store) == ["keeper entry for undo"]
        # A later legitimate write lands after the consolidation.
        assert store.add("memory", "later unrelated add")["success"]
        drifted = _disk_entries(store)
        result = restore(audit_id, store)
        assert result["success"] is False
        assert "changed since" in result["error"]
        assert audit_id in result["error"]  # audit id surfaced for manual recovery
        assert _disk_entries(store) == drifted  # the later add is preserved, untouched

    def test_committed_then_later_unrelated_replace_undo_refused(self, store):
        from tools.memory_consolidation import restore
        audit_id = self._applied_removal(store)
        assert store.replace("memory", "keeper entry", "keeper entry (later edit)")["success"]
        drifted = _disk_entries(store)
        result = restore(audit_id, store)
        assert result["success"] is False
        assert "changed since" in result["error"]
        assert _disk_entries(store) == drifted  # the later replace is preserved

    def test_applied_append_failure_planned_after_recovery(self, store, monkeypatch):
        """Post-commit ledger-write failure: the 'applied' event never lands, but the
        begin carries planned_after_sha256; when current memory EXACTLY equals that
        planned after-state, safe undo remains possible (recovery evidence adopted)."""
        from tools import memory_consolidation as mc
        real_applied = mc.record_applied

        def failing_applied(*a, **kw):
            raise mc.MemoryConsolidationAuditError("ledger append failed (simulated)")

        monkeypatch.setattr(mc, "record_applied", failing_applied)
        _set_flag(True)
        store.add("memory", "planned after keeper")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="planned after", store=store))
        assert r["success"] is True, r  # the commit is durable; only the ledger append failed
        # Restore the real record_applied WITHOUT monkeypatch.undo() (it would also undo
        # the home fixture's HERMES_HOME and point the ledger read at the real ~/.hermes).
        mc.record_applied = real_applied
        assert _disk_entries(store) == []
        result = mc.restore(r["audit_id"], store)
        assert result["success"] is True, result
        assert _disk_entries(store) == ["planned after keeper"]  # exactly the before-state

    def test_wrong_profile_refused(self, store):
        from tools import memory_consolidation as _mc
        audit_id = self._applied_removal(store)
        from tools.memory_consolidation import _ledger_path
        lines = _ledger_path().read_text(encoding="utf-8").splitlines()
        rewrote = False
        for i, line in enumerate(lines):
            rec = json.loads(line)
            if rec.get("id") == audit_id:
                rec["hermes_home"] = "/somewhere/else/.hermes"
                lines[i] = json.dumps(rec, ensure_ascii=False)
                rewrote = True
        assert rewrote
        _ledger_path().write_text("\n".join(lines) + "\n", encoding="utf-8")
        after = _disk_entries(store)
        result = _mc.restore(audit_id, store)
        assert result["success"] is False
        assert "different Hermes home" in result["error"]
        assert _disk_entries(store) == after

    def test_empty_before_state_refusal_stays_explicit(self, store):
        """An empty recorded before-state keeps its explicit refusal (restoring would
        empty the store) rather than a generic undo error."""
        from tools import memory_consolidation as mc
        store.add("memory", "only entry now")
        audit_id = mc.record_begin("memory", [], "", [])  # empty before-state record
        result = mc.restore(audit_id, store)
        assert result["success"] is False
        assert "empty recorded state" in result["error"]
        assert _disk_entries(store) == ["only entry now"]

    def test_repeated_undo_noop_no_corruption(self, store):
        from tools.memory_consolidation import list_records, restore
        audit_id = self._applied_removal(store)
        first = restore(audit_id, store)
        assert first["success"] is True
        expected = _disk_entries(store)
        for _ in range(3):
            again = restore(audit_id, store)
            assert again["success"] is True
            assert "nothing to restore" in again["message"]
        assert _disk_entries(store) == expected
        # The journal recorded the undo.
        assert any(rec.get("event") == "undone" and rec.get("id") == audit_id
                   for rec in list_records())


# =========================================================================
# Ledger custody: the audit ledger holds the full raw memory file, so it must
# never be readable by anyone MEMORY.md itself is not readable by.
# =========================================================================

class TestLedgerPermissions:
    def _modes(self):
        import os
        import stat
        from tools.memory_consolidation import _ledger_path
        path = _ledger_path()
        return (stat.S_IMODE(os.stat(path).st_mode),
                stat.S_IMODE(os.stat(path.parent).st_mode))

    @pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX permission bits")
    def test_ledger_and_dir_are_owner_only(self, store):
        self._unattended_remove_inline(store)
        ledger_mode, dir_mode = self._modes()
        assert ledger_mode == 0o600, oct(ledger_mode)
        assert dir_mode == 0o700, oct(dir_mode)

    @pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX permission bits")
    def test_loose_bits_from_an_older_run_are_repaired(self, store):
        import os
        from tools.memory_consolidation import _ledger_path
        self._unattended_remove_inline(store)
        path = _ledger_path()
        os.chmod(path, 0o664)
        os.chmod(path.parent, 0o775)
        # A second consolidation must tighten the existing inode again.
        store.add("memory", "another entry to consolidate")
        self._unattended_remove_inline(store, "another entry")
        assert self._modes() == (0o600, 0o700)

    def _unattended_remove_inline(self, store, text="ledger custody fact"):
        # Seed a keeper (the store refuses to empty a non-empty memory file) plus the
        # target entry, then consolidate unattended.
        if "ledger keeper fact" not in store._entries_for("memory"):
            store.add("memory", "ledger keeper fact")
        if text not in store._entries_for("memory"):
            store.add("memory", text)
        _set_flag(True)
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text=text, store=store))
        assert r["success"] is True, r
        return r
