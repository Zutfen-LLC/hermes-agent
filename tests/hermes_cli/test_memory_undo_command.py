"""The /memory undo command surface: dispatch through handle_pending_subcommand,
usage/list rendering, restore via the shared handler against the real audit ledger,
and the registry wiring (subcommands/args_hint) that feeds help + autocomplete."""

import json

import pytest


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes-home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _store():
    from tools.memory_tool import MemoryStore
    s = MemoryStore(memory_char_limit=500, user_char_limit=300)
    s.load_from_disk()
    return s


def _disk_entries(store, target="memory"):
    store.load_from_disk()
    return list(store._entries_for(target))


def _record_consolidation(store, target="memory"):
    """Mirror the landed _autonomous_consolidation apply path: begin snapshot ->
    real public batch mutation -> applied record. Returns the audit id."""
    from tools.memory_consolidation import record_applied, record_begin
    before_entries = list(store._entries_for(target))
    before_raw = store._read_raw_checked(store._path_for(target))[0]
    victim = before_entries[0]
    audit_id = record_begin(target, [
        {"action": "remove", "old_text": victim[:10], "matched_entry": victim}],
        before_raw, before_entries)
    result = store.apply_batch(target, [{"action": "remove", "old_text": victim}])
    assert result.get("success"), result
    after_raw = store._read_raw_checked(store._path_for(target))[0]
    record_applied(audit_id, target, after_raw, {"replaced": 0, "removed": 1, "added": 0})
    return audit_id


def _undo(args, store):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    return handle_pending_subcommand(wa.MEMORY, args, memory_store=store)


# ---------------------------------------------------------------------------
# Usage / list rendering
# ---------------------------------------------------------------------------

def test_undo_no_args_shows_usage(home):
    assert _undo(["undo"], store=None) == \
        "Usage: /memory undo <audit_id>  (ids: /memory undo list)"


def test_undo_list_empty(home):
    assert _undo(["undo", "list"], store=None) == \
        "No autonomous consolidations recorded yet."


def test_undo_list_renders_recent_applied_records(home):
    store = _store()
    store.add("memory", "first consolidated entry")
    store.add("memory", "second surviving entry")
    audit_id = _record_consolidation(store)

    out = _undo(["undo", "list"], store=store)
    assert audit_id in out
    assert "replaced 0, removed 1, added 0" in out
    assert "  memory  replaced 0, removed 1, added 0" in out  # target column
    assert "/memory undo <audit_id>" in out


# ---------------------------------------------------------------------------
# Restore through the shared handler
# ---------------------------------------------------------------------------

def test_undo_restores_exact_before_entries(home):
    store = _store()
    store.add("memory", "alpha original entry")
    store.add("memory", "beta stays put")
    before = _disk_entries(store)
    audit_id = _record_consolidation(store)
    assert _disk_entries(store) == ["beta stays put"]  # the remove really applied

    out = _undo(["undo", audit_id], store=store)

    assert out.startswith(f"Undo {audit_id}:")
    assert "Restored memory to the state recorded before consolidation" in out
    assert _disk_entries(store) == before  # exact, order included


def test_undo_unknown_id_reports_error_and_changes_nothing(home):
    store = _store()
    store.add("memory", "entry that must not move")
    snapshot = _disk_entries(store)

    out = _undo(["undo", "deadbeef1234"], store=store)

    assert "deadbeef1234" in out
    assert "No consolidation audit record found" in out
    assert _disk_entries(store) == snapshot


def test_undo_refuses_cross_profile_record(home):
    from tools.memory_consolidation import _ledger_path
    store = _store()
    store.add("memory", "foreign profile entry")
    store.add("memory", "second entry kept on refusal")
    audit_id = _record_consolidation(store)
    after = _disk_entries(store)
    assert after == ["second entry kept on refusal"]  # the pinned entry was removed

    # Rewrite the begin record as if it came from another Hermes home.
    lines = _ledger_path().read_text(encoding="utf-8").splitlines()
    rewrote = False
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec.get("event") == "begin" and rec.get("id") == audit_id:
            rec["hermes_home"] = "/somewhere/else/.hermes"
            lines[i] = json.dumps(rec, ensure_ascii=False)
            rewrote = True
    assert rewrote
    _ledger_path().write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = _undo(["undo", audit_id], store=store)

    assert "different Hermes home" in out
    assert _disk_entries(store) == after  # nothing restored, nothing clobbered


def test_undo_without_store_says_unavailable(home):
    store = _store()
    store.add("memory", "some entry")
    store.add("memory", "another entry")
    audit_id = _record_consolidation(store)
    out = _undo(["undo", audit_id], store=None)
    assert out == "memory store unavailable"


# ---------------------------------------------------------------------------
# Dispatch + registry wiring
# ---------------------------------------------------------------------------

def test_unknown_subcommand_still_falls_through(home):
    """Non-undo unknown words return None so BOTH callers (cli mixin + gateway) print
    their 'Unknown /memory subcommand' fallback — which now mentions undo."""
    assert _undo(["frobnicate"], store=_store()) is None


def test_registry_exposes_undo_subcommand():
    from hermes_cli.commands import SUBCOMMANDS, resolve_command
    cmd = resolve_command("memory")
    assert cmd is not None
    assert cmd.args_hint == "[pending|approve|reject|diff|undo|approval] [id|on|off]"
    assert SUBCOMMANDS["/memory"] == \
        ["pending", "approve", "reject", "diff", "undo", "approval"]
