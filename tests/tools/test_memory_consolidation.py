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

    def test_applied_append_failure_undo_refused_manual_recovery(self, store, monkeypatch):
        """CORRECTION ROUND 3 FLIP (was: planned_after fallback made this undo succeed).
        Post-commit ledger-write failure: the 'applied' event never lands. The begin's
        planned_after_sha256 is NOT commit evidence — current memory exactly equaling the
        planned after-state must NOT enable an automatic restore. Undo refuses safely,
        names the manual-recovery path, and memory is preserved exactly as committed."""
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
        assert "UNAVAILABLE" in r["message"]  # the tool result itself reports degraded undo
        # Restore the real record_applied WITHOUT monkeypatch.undo() (it would also undo
        # the home fixture's HERMES_HOME and point the ledger read at the real ~/.hermes).
        mc.record_applied = real_applied
        assert _disk_entries(store) == []
        result = mc.restore(r["audit_id"], store)
        assert result["success"] is False
        assert "no durable 'applied' event" in result["error"]
        assert "manual recovery" in result["error"] or "by hand" in result["error"]
        assert _disk_entries(store) == []  # the committed state is preserved untouched

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


# =========================================================================
# Correction round 2: one self-consistent BEFORE snapshot; canonical committed-after
# digest; atomic undo precondition
# =========================================================================

class TestSingleSourceBeforeSnapshot:
    """The begin record must describe exactly ONE store state: before_entries parsed from
    the same raw as before_sha256. A writer changing the file between the old independent
    reads used to produce a begin mixing entries from S0 with the digest of S1."""

    def test_begin_record_entries_parse_from_the_same_raw_as_its_digest(self, store):
        """Mechanical invariant on EVERY begin record: parse(before_raw) == before_entries
        under the store's canonical normalization (strip, drop empties, dedupe
        order-preserving) — the same rules production applies."""
        from tools.memory_consolidation import list_records
        _set_flag(True)
        store.add("memory", "snapshot witness entry")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="snapshot witness", store=store))
        assert r["success"] is True, r
        begins = [rec for rec in list_records() if rec.get("event") == "begin"]
        assert begins
        for rec in begins:
            normalized = list(dict.fromkeys(
                e for e in (x.strip() for x in rec["before_raw"].split("\n§\n")) if e))
            assert normalized == rec["before_entries"], rec["id"]

    def test_writer_between_old_read_points_cannot_split_the_begin_record(self, store, monkeypatch):
        """Deterministic race (correction round 2): the old code took before_entries from
        the store's in-memory view (loaded S0) and before_raw from a SECOND, independent
        disk read — a writer landing between them recorded begin entries from S0 with a
        digest of S1, and a later undo could erase the concurrent change. Replaying that
        exact interleaving (the concurrent write lands right after the load's read, before
        the snapshot read), the corrected single-snapshot path derives BOTH from one raw
        read, so the begin record is always self-consistent — proven mechanically by the
        parse(before_raw) == before_entries invariant, which FAILS on the split record the
        reviewed implementation produced for this same interleaving."""
        from tools import memory_consolidation as mc
        from tools.memory_tool_store import ENTRY_DELIMITER

        _set_flag(True)
        store.add("memory", "first race witness")
        store.add("memory", "second race witness")

        raw_entries = MemoryStore._read_raw_checked
        state = {"fired": False}

        def racing_read(path):
            result = raw_entries(path)
            # Fire on the load-time MEMORY.md read only: the concurrent writer lands
            # AFTER the store's in-memory view (S0) is populated and BEFORE the
            # consolidation's own snapshot read — the old two-read interleaving.
            if path.name == "MEMORY.md" and not state["fired"]:
                state["fired"] = True
                store._path_for("memory").write_text(
                    result[0] + ENTRY_DELIMITER + "concurrent late write",
                    encoding="utf-8")
            return result

        monkeypatch.setattr(MemoryStore, "_read_raw_checked", staticmethod(racing_read))

        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="first race witness", store=store))
        monkeypatch.setattr(MemoryStore, "_read_raw_checked", staticmethod(raw_entries))

        # The one-snapshot path read the post-write state S1, so the consolidation
        # validates, commits and audits against exactly S1 — and succeeds.
        assert r["success"] is True, r
        begins = [rec for rec in mc.list_records() if rec.get("event") == "begin"]
        assert begins
        for rec in begins:
            normalized = list(dict.fromkeys(
                e for e in (x.strip() for x in rec["before_raw"].split("\n§\n")) if e))
            assert normalized == rec["before_entries"], rec["id"]
            assert "concurrent late write" in rec["before_raw"]  # one snapshot: S1 whole
        # And undo restores exactly that recorded pre-commit state (S1), erasing nothing.
        entries_before_undo = _disk_entries(store)
        assert entries_before_undo == ["second race witness", "concurrent late write"]
        result = mc.restore(r["audit_id"], store)
        assert result["success"] is True, result
        assert _disk_entries(store) == ["first race witness", "second race witness",
                                        "concurrent late write"]

    def test_unreadable_memory_file_refused_before_anything(self, store, monkeypatch):
        """Read success is verified before the snapshot is used: an unreadable MEMORY.md
        refuses the consolidation outright instead of proceeding from a silently-empty
        (lossy) view."""
        from tools.memory_tool_store import MemoryStore as MS
        _set_flag(True)
        store.add("memory", "entry guarded against unreadable reads")  # file exists first
        real = MS._read_raw_checked

        def failing(path):
            if path.name == "MEMORY.md":
                return "", False
            return real(path)

        monkeypatch.setattr(MS, "_read_raw_checked", staticmethod(failing))
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="unreadable reads", store=store))
        monkeypatch.setattr(MS, "_read_raw_checked", staticmethod(real))

        assert r["success"] is False
        assert "could not be read" in r["error"]
        from tools.memory_consolidation import list_records
        assert list_records() == []  # no begin record, nothing applied


class TestCanonicalAppliedDigest:
    """applied.after_sha256 must be the planned-after digest of the state the validated
    commit wrote — never an unlocked reread that can absorb a post-commit writer."""

    def test_applied_after_sha_equals_planned_after_sha(self, store):
        from tools.memory_consolidation import list_records
        _set_flag(True)
        store.add("memory", "digest witness alpha")
        store.add("memory", "digest witness beta")
        with unattended_review():
            r = json.loads(memory_tool(action="replace", old_text="digest witness alpha",
                                       content="digest witness alpha (rewritten)", store=store))
        assert r["success"] is True, r
        begins = [rec for rec in list_records() if rec.get("event") == "begin"]
        applied = [rec for rec in list_records() if rec.get("event") == "applied"]
        assert begins and applied
        assert begins[-1]["id"] == applied[-1]["id"] == r["audit_id"]
        assert applied[-1]["after_sha256"] == begins[-1]["planned_after_sha256"]

    def test_post_commit_writer_not_in_applied_digest_and_undo_refuses(self, store, monkeypatch):
        """Race: a second writer lands immediately after apply_batch returns but BEFORE
        record_applied executes. The writer must NOT become part of applied.after_sha256
        (the digest stays the planned-after state), and undo must REFUSE while that later
        write exists (current memory matches neither before nor expected-after)."""
        from tools import memory_consolidation as mc

        real_apply_batch = MemoryStore.apply_batch
        state = {"committed": False}

        def apply_then_write(subject, target, operations, **kwargs):
            result = real_apply_batch(subject, target, operations, **kwargs)
            if result.get("success") and not state["committed"]:
                state["committed"] = True
                # The unrelated writer lands AFTER the lock is released, BEFORE
                # record_applied runs (which happens later in the caller).
                assert subject.add("memory", "unrelated post-commit write")["success"]
            return result

        monkeypatch.setattr(MemoryStore, "apply_batch", apply_then_write)
        _set_flag(True)
        store.add("memory", "post-commit race target")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="post-commit race", store=store))
        monkeypatch.setattr(MemoryStore, "apply_batch", real_apply_batch)

        assert r["success"] is True, r  # the consolidation itself committed
        begins = [rec for rec in mc.list_records() if rec.get("event") == "begin"]
        applied = [rec for rec in mc.list_records() if rec.get("event") == "applied"]
        audit_id = r["audit_id"]
        begin = next(rec for rec in begins if rec["id"] == audit_id)
        ap = next(rec for rec in applied if rec["id"] == audit_id)
        # 1. The unrelated write is NOT part of the transaction's committed-after identity.
        assert ap["after_sha256"] == begin["planned_after_sha256"]
        # 2. Undo refuses while the later write exists; it survives untouched.
        entries_after = _disk_entries(store)
        assert "unrelated post-commit write" in entries_after
        result = mc.restore(audit_id, store)
        assert result["success"] is False
        assert "changed since" in result["error"]
        assert _disk_entries(store) == entries_after


class TestAtomicUndoPrecondition:
    """restore() derives digest, current entries and the restore ops from ONE raw
    snapshot and commits with expected_before_raw: a writer landing between the
    decision and the restore commit fails the whole restore with zero mutation."""

    def _applied_removal(self, store, victim="atomic undo victim"):
        _set_flag(True)
        store.add("memory", victim)
        store.add("memory", "atomic undo keeper")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text=victim, store=store))
        assert r["success"] is True, r
        return r["audit_id"]

    def test_plain_restore_is_exact(self, store):
        from tools.memory_consolidation import restore
        store.add("memory", "plain restore alpha")
        store.add("memory", "plain restore beta")
        before = _disk_entries(store)
        _set_flag(True)
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="plain restore alpha",
                                       store=store))
        assert r["success"] is True, r
        assert _disk_entries(store) == ["plain restore beta"]
        result = restore(r["audit_id"], store)
        assert result["success"] is True, result
        assert _disk_entries(store) == before

    def test_unrelated_write_already_present_refuses(self, store):
        from tools.memory_consolidation import restore
        audit_id = self._applied_removal(store)
        assert store.add("memory", "later unrelated entry")["success"]
        drifted = _disk_entries(store)
        result = restore(audit_id, store)
        assert result["success"] is False
        assert "changed since" in result["error"]
        assert _disk_entries(store) == drifted

    def test_writer_adds_after_validation_before_commit_refused_and_survives(self, store, monkeypatch):
        """TOCTOU: a writer ADDS an entry after restore validated the digest but before
        the restore commits. The precondition makes the restore fail with zero mutation
        and the writer's entry survives."""
        from tools import memory_consolidation as mc
        real_apply_batch = MemoryStore.apply_batch
        audit_id = self._applied_removal(store)
        assert _disk_entries(store) == ["atomic undo keeper"]

        def write_then_apply(subject, target, operations, **kwargs):
            # Lands between restore's digest validation and its apply_batch commit.
            assert subject.add("memory", "concurrent add during undo")["success"]
            return real_apply_batch(subject, target, operations, **kwargs)

        monkeypatch.setattr(MemoryStore, "apply_batch", write_then_apply)
        result = mc.restore(audit_id, store)
        monkeypatch.setattr(MemoryStore, "apply_batch", real_apply_batch)

        assert result["success"] is False
        assert "changed on disk since" in result["error"]
        entries = _disk_entries(store)
        assert "atomic undo victim" not in entries  # the restore did NOT run
        assert "concurrent add during undo" in entries  # the writer survived

    def test_writer_edits_target_after_validation_before_commit_refused(self, store, monkeypatch):
        """TOCTOU: a writer EDITS the entry the restore would re-add, after validation.
        Same refusal, zero mutation, edit survives."""
        from tools import memory_consolidation as mc
        real_apply_batch = MemoryStore.apply_batch
        audit_id = self._applied_removal(store)

        def edit_then_apply(subject, target, operations, **kwargs):
            assert subject.replace("memory", "atomic undo keeper",
                                   "atomic undo keeper (edited mid-undo)")["success"]
            return real_apply_batch(subject, target, operations, **kwargs)

        monkeypatch.setattr(MemoryStore, "apply_batch", edit_then_apply)
        result = mc.restore(audit_id, store)
        monkeypatch.setattr(MemoryStore, "apply_batch", real_apply_batch)

        assert result["success"] is False
        assert "changed on disk since" in result["error"]
        assert _disk_entries(store) == ["atomic undo keeper (edited mid-undo)"]

    def test_repeated_undo_idempotent_after_precondition_restore(self, store):
        from tools.memory_consolidation import restore
        store.add("memory", "repeatable restore entry")
        audit_id = self._applied_removal(store, victim="repeatable restore entry")
        assert restore(audit_id, store)["success"] is True
        expected = _disk_entries(store)
        for _ in range(2):
            again = restore(audit_id, store)
            assert again["success"] is True
            assert "nothing to restore" in again["message"]
        assert _disk_entries(store) == expected

    def test_applied_ledger_failure_undo_refused_safely(self, store, monkeypatch):
        """CORRECTION ROUND 3 FLIP (was: planned_after recovery made this undo succeed).
        A successful commit whose 'applied' append failed leaves NO durable commit
        evidence; automatic undo refuses safely and identifies the manual recovery path
        (the begin record + its before snapshot), instead of treating the begin's
        planned-after digest as proof the mutation happened."""
        from tools import memory_consolidation as mc
        real_applied = mc.record_applied

        def failing_applied(*a, **kw):
            raise mc.MemoryConsolidationAuditError("ledger append failed (simulated)")

        monkeypatch.setattr(mc, "record_applied", failing_applied)
        _set_flag(True)
        store.add("memory", "planned after recovery witness")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="recovery witness", store=store))
        mc.record_applied = real_applied

        assert r["success"] is True, r
        assert "UNAVAILABLE" in r["message"]
        assert _disk_entries(store) == []
        result = mc.restore(r["audit_id"], store)
        assert result["success"] is False
        assert "no durable 'applied' event" in result["error"]
        # The refusal surfaces the manual recovery ingredients...
        assert "before_sha256" in result["error"] or "begin record" in result["error"]
        assert _disk_entries(store) == []


# =========================================================================
# CORRECTION ROUND 3 — planned-after is NOT commit evidence
#
# A begin/prepare record proves intent, not commit. current == before stays
# the idempotent no-op; automatic undo REQUIRES a durable 'applied' event
# (validated against the begin's plan); begin-only + current != before is
# REFUSED with the manual-recovery path surfaced. record_applied() reports
# success so the caller can advertise (or refuse to advertise) undo.
# =========================================================================


class TestUndoRequiresCommitEvidence:
    """The defect-3 reproduction matrix from the correction instruction."""

    def test_1_begin_exists_commit_failed_current_before_noop(self, store):
        """Required test 1: begin exists, commit failed, current still before => no-op."""
        from tools import memory_consolidation as mc
        _set_flag(True)
        store.add("memory", "keep one")
        store.add("memory", "keep two")
        before_raw = store._read_raw_checked(store._path_for("memory"))[0]
        before_entries = list(store._entries_for("memory"))
        planned_entries = ["keep one"]  # what the failed consolidation intended
        from tools.memory_tool_store import ENTRY_DELIMITER
        audit_id = mc.record_begin(
            "memory", [{"action": "remove", "old_text": "keep two",
                        "matched_entry": "keep two"}],
            before_raw, before_entries,
            planned_after_raw=ENTRY_DELIMITER.join(planned_entries))
        # The commit NEVER happened (simulated failure); memory still equals before.
        result = mc.restore(audit_id, store)
        assert result["success"] is True
        assert "nothing to restore" in result["message"]
        assert _disk_entries(store) == ["keep one", "keep two"]

    def test_2_begin_only_current_equals_planned_after_refused(self, store):
        """Required test 2 — THE defect reproduction. Begin journal: before=[A,B],
        planned after=[B]. The consolidation commit FAILS (no applied event). Later an
        INDEPENDENT manual removal of A lands memory exactly on the planned after-state.
        The reviewed implementation saw current == planned_after and restored [A,B],
        undoing the independent change. Correction: AUTO UNDO REFUSED; the independent
        state is preserved."""
        from tools import memory_consolidation as mc
        from tools.memory_tool_store import ENTRY_DELIMITER
        _set_flag(True)
        store.add("memory", "A independent change witness")
        store.add("memory", "B untouched witness")
        before_raw = store._read_raw_checked(store._path_for("memory"))[0]
        before_entries = ["A independent change witness", "B untouched witness"]
        planned_after_raw = ENTRY_DELIMITER.join(["B untouched witness"])
        audit_id = mc.record_begin(
            "memory", [{"action": "remove", "old_text": "A independent",
                        "matched_entry": "A independent change witness"}],
            before_raw, before_entries, planned_after_raw=planned_after_raw)
        # The consolidation commit fails; later, an independent actor removes A.
        store.remove("memory", "A independent change witness")
        independent_state = _disk_entries(store)
        assert independent_state == ["B untouched witness"]
        result = mc.restore(audit_id, store)
        assert result["success"] is False
        assert "no durable 'applied' event" in result["error"]
        # The independent state is PRESERVED — undo did not resurrect A.
        assert _disk_entries(store) == independent_state == ["B untouched witness"]

    def test_3_normal_commit_applied_event_undo_works(self, store):
        """Required test 3: normal successful commit + applied event + current == after
        => undo works."""
        from tools.memory_consolidation import restore
        _set_flag(True)
        store.add("memory", "alpha commit evidence target")
        store.add("memory", "commit evidence keeper")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="alpha commit",
                                       store=store))
        assert r["success"] is True, r
        assert "Recovery: /memory undo" in r["message"]
        result = restore(r["audit_id"], store)
        assert result["success"] is True, result
        assert _disk_entries(store) == ["alpha commit evidence target",
                                        "commit evidence keeper"]

    def test_5_applied_after_must_equal_planned_after(self, store, monkeypatch):
        """Required test 5: applied.after_sha256 must equal planned_after_sha256; an
        inconsistent journal (applied digest != planned digest) => refuse."""
        from tools import memory_consolidation as mc
        _set_flag(True)
        store.add("memory", "consistency check witness")
        store.add("memory", "consistency keeper")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="consistency check",
                                       store=store))
        assert r["success"] is True, r
        # Tamper with the ledger's applied event: rewrite after_sha256 to an unrelated
        # digest, leaving the begin's planned_after_sha256 intact.
        path = mc._ledger_path()
        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = []
        for line in lines:
            rec = json.loads(line)
            if rec.get("event") == "applied" and rec.get("id") == r["audit_id"]:
                rec["after_sha256"] = "f" * 64  # unrelated digest
            tampered.append(json.dumps(rec))
        path.write_text("\n".join(tampered) + "\n", encoding="utf-8")
        result = mc.restore(r["audit_id"], store)
        assert result["success"] is False
        assert "planned_after_sha256" in result["error"] and "inconsistent" in result["error"]
        assert _disk_entries(store) == ["consistency keeper"]


    def test_record_applied_reports_success_and_failure(self, store):
        """record_applied() returns True on a good append and False on a failed one, so
        the caller can report the degraded recovery state."""
        from tools import memory_consolidation as mc
        _set_flag(True)
        store.add("memory", "report witness")
        with unattended_review():
            r = json.loads(memory_tool(action="remove", old_text="report", store=store))
        assert r["success"] is True, r
        assert mc.record_applied("deadbeef0000", "memory", "x", {"removed": 1}) is True
        def boom(*a, **kw):
            raise mc.MemoryConsolidationAuditError("nope")
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mc, "_append_record", boom)
        try:
            assert mc.record_applied("deadbeef0001", "memory", "x", {}) is False
        finally:
            monkeypatch.undo()


    def test_undo_list_only_shows_applied(self, store):
        """``undo list`` shows only applied (commit-evidenced) ids: a begin-only failed
        consolidation is not advertised as undoable."""
        from hermes_cli.write_approval_commands import _memory_undo_list
        from tools import memory_consolidation as mc
        _set_flag(True)
        store.add("memory", "list alpha")
        with unattended_review():
            applied_r = json.loads(memory_tool(action="remove", old_text="list alpha",
                                               store=store))
        store.add("memory", "list beta")
        before_raw = store._read_raw_checked(store._path_for("memory"))[0]
        mc.record_begin("memory", [{"action": "remove", "old_text": "list beta"}],
                        before_raw, ["list beta"])
        out = _memory_undo_list()
        assert applied_r["audit_id"] in out
        begin_only = [rec["id"] for rec in mc.list_records()
                      if rec.get("event") == "begin" and rec["id"] != applied_r["audit_id"]]
        assert begin_only and all(b not in out for b in begin_only)
