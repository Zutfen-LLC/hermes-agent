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
        assert "Already at the recorded state" in again["message"]
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
