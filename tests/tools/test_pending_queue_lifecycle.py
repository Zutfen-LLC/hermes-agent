"""Pending-queue lifecycle for MEMORY records: classification (ready/stale/superseded/
invalid/rejected), staging dedup, approve/reject/bulk semantics, list rendering, and the
/memory diff renderer.

Records are never deleted once statused — they stay on disk as audit evidence — and a
record the operator rejected is terminal. Classification fails closed: malformed or
legacy unpinned records are 'invalid' forever.
"""

import json
import os
import shutil
import tempfile

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_pql_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _store():
    from tools.memory_tool import MemoryStore
    s = MemoryStore(memory_char_limit=4000, user_char_limit=4000)
    s.load_from_disk()
    return s


def _stage(payload, summary="proposal", origin="background_review", created_at=None):
    """Stage a record through the real path; optionally backdate created_at (superseded
    comparisons are timestamp-based)."""
    from tools import write_approval as wa
    record = wa.stage_write(wa.MEMORY, payload, summary=summary, origin=origin)
    if created_at is not None:
        record["created_at"] = created_at
        wa._pending_path(wa.MEMORY, record["id"]).write_text(
            json.dumps(record), encoding="utf-8")
    return record


def _stage_remove(matched_entry, old_text, *, origin="background_review", created_at=None,
                  target="memory", summary="proposal"):
    payload = {"action": "remove", "target": target, "old_text": old_text,
               "matched_entry": matched_entry}
    return _stage(payload, summary=summary, origin=origin, created_at=created_at)


def _status(record_id):
    from tools import write_approval as wa
    rec = wa.get_pending(wa.MEMORY, record_id)
    return rec.get("status"), rec


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_classify_ready_when_pinned_entry_present(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "alpha rule entry")
    rec = _stage_remove("alpha rule entry", "alpha rule")
    assert wa.classify_pending_memory(rec, store) == "ready"


def test_classify_stale_when_entry_removed(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "alpha rule entry")
    rec = _stage_remove("alpha rule entry", "alpha rule")
    store.remove("memory", "alpha rule entry")
    assert wa.classify_pending_memory(rec, store) == "stale"


def test_classify_stale_when_entry_text_changed_on_disk(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "beta rule entry")
    rec = _stage_remove("beta rule entry", "beta rule")
    # External hand edit rewrites MEMORY.md with a changed wording of the same entry.
    store._path_for("memory").write_text("- beta rule entry (edited)", encoding="utf-8")
    fresh = _store()
    assert wa.classify_pending_memory(rec, fresh) == "stale"


def test_classify_stale_persisted_by_reclassify_and_counts(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "gamma rule entry")
    rec = _stage_remove("gamma rule entry", "gamma rule")
    store.remove("memory", "gamma rule entry")
    counts = wa.reclassify_pending_memory(store)
    assert counts == {"ready": 0, "stale": 1, "superseded": 0, "invalid": 0,
                      "rejected": 0, "changed": 1}
    assert _status(rec["id"])[0] == "stale"


def test_classify_superseded_by_newer_active_same_pin(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "delta rule entry")
    old = _stage_remove("delta rule entry", "delta rule", created_at=1000.0)
    _stage({"action": "replace", "target": "memory", "old_text": "delta rule",
            "content": "delta rule v2", "matched_entry": "delta rule entry"},
           created_at=2000.0)
    assert wa.classify_pending_memory(old, store) == "superseded"


def test_not_superseded_when_newer_record_is_stale(hermes_home):
    """Each record is judged by its own pins: a newer record that is itself stale
    (already status-marked) must not supersede the older one."""
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "eps rule entry")
    old = _stage_remove("eps rule entry", "eps rule", created_at=1000.0)
    newer = _stage({"action": "replace", "target": "memory", "old_text": "eps rule",
                    "content": "eps rule v2", "matched_entry": "eps rule entry"},
                   created_at=2000.0)
    wa.update_pending_status(wa.MEMORY, newer["id"], wa.STATUS_STALE)
    assert wa.classify_pending_memory(old, store) == "ready"


def test_classify_invalid_unpinned_legacy_destructive(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "zeta rule entry")
    rec = _stage({"action": "remove", "target": "memory", "old_text": "zeta rule"})
    assert wa.classify_pending_memory(rec, store) == "invalid"


def test_classify_invalid_malformed_payload(hermes_home):
    from tools import write_approval as wa
    store = _store()
    rec = _stage({"action": "purge", "target": "memory"})
    assert wa.classify_pending_memory(rec, store) == "invalid"
    rec = _stage({"action": "remove", "target": "skills", "old_text": "x",
                  "matched_entry": "y"})
    assert wa.classify_pending_memory(rec, store) == "invalid"  # bad target


def test_classify_store_error_fails_closed(hermes_home, monkeypatch):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "eta rule entry")
    rec = _stage_remove("eta rule entry", "eta rule")

    class Boom:
        def _entries_for(self, target):
            raise RuntimeError("store exploded")

    assert wa.classify_pending_memory(rec, Boom()) == "invalid"


def test_classify_rejected_terminal(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "theta rule entry")
    rec = _stage_remove("theta rule entry", "theta rule")
    wa.update_pending_status(wa.MEMORY, rec["id"], wa.STATUS_REJECTED)
    rec = wa.get_pending(wa.MEMORY, rec["id"])  # re-read: status lives on disk
    assert wa.classify_pending_memory(rec, store) == "rejected"
    # reclassify skips it entirely: no counts churn, no rewrite.
    counts = wa.reclassify_pending_memory(store)
    assert counts == {"ready": 0, "stale": 0, "superseded": 0, "invalid": 0,
                      "rejected": 1, "changed": 0}


def test_reclassify_idempotent(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "iota rule entry")
    store.add("memory", "kappa rule entry")
    _stage_remove("iota rule entry", "iota rule")            # ready
    _stage({"action": "remove", "target": "memory", "old_text": "kappa"})  # invalid (unpinned)
    first = wa.reclassify_pending_memory(store)
    assert first["changed"] == 1  # only the invalid record gains a status
    second = wa.reclassify_pending_memory(store)
    assert second["changed"] == 0
    assert second["ready"] == 1 and second["invalid"] == 1


def test_reclassify_invalid_persists_for_audit(hermes_home):
    from tools import write_approval as wa
    store = _store()
    rec = _stage({"action": "remove", "target": "memory", "old_text": "x"})
    wa.reclassify_pending_memory(store)
    assert _status(rec["id"])[0] == "invalid"
    assert wa._pending_path(wa.MEMORY, rec["id"]).exists()


# ---------------------------------------------------------------------------
# Staging dedup
# ---------------------------------------------------------------------------

def test_stage_dedup_supersedes_older_same_pin(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "mu rule entry")
    first = _stage_remove("mu rule entry", "mu rule", created_at=1000.0)
    second = _stage({"action": "replace", "target": "memory", "old_text": "mu rule",
                     "content": "mu rule v2", "matched_entry": "mu rule entry"},
                    created_at=2000.0)
    status, rec = _status(first["id"])
    assert status == "superseded"
    assert rec["superseded_by"] == second["id"]
    # Old file still on disk (audit evidence); new record stays active.
    assert wa._pending_path(wa.MEMORY, first["id"]).exists()
    assert _status(second["id"])[0] is None


def test_stage_dedup_different_pins_both_active(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "nu rule entry")
    store.add("memory", "xi rule entry")
    first = _stage_remove("nu rule entry", "nu rule")
    _stage_remove("xi rule entry", "xi rule")
    assert _status(first["id"])[0] is None


def test_stage_dedup_foreground_never_supersedes(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "omicron rule entry")
    first = _stage_remove("omicron rule entry", "omicron rule", origin="foreground")
    _stage({"action": "replace", "target": "memory", "old_text": "omicron rule",
            "content": "omicron v2", "matched_entry": "omicron rule entry"},
           origin="foreground")
    assert _status(first["id"])[0] is None


def test_stage_dedup_only_supersedes_active_records(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "pi rule entry")
    stale_rec = _stage_remove("pi rule entry", "pi rule")
    wa.update_pending_status(wa.MEMORY, stale_rec["id"], wa.STATUS_STALE)
    newer = _stage({"action": "replace", "target": "memory", "old_text": "pi rule",
                    "content": "pi v2", "matched_entry": "pi rule entry"})
    # The already-stale record keeps its (more specific) status.
    assert _status(stale_rec["id"])[0] == "stale"
    assert _status(newer["id"])[0] is None


# ---------------------------------------------------------------------------
# reject semantics
# ---------------------------------------------------------------------------

def test_reject_bulk_by_flag_marks_rejected_files_stay(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "rho rule entry")
    stale1 = _stage_remove("rho rule entry", "rho rule")
    store.remove("memory", "rho rule entry")
    wa.reclassify_pending_memory(store)
    bad = _stage({"action": "purge", "target": "memory"})
    wa.reclassify_pending_memory(store)

    from hermes_cli.write_approval_commands import handle_pending_subcommand
    out = handle_pending_subcommand(wa.MEMORY, ["reject", "--stale"], memory_store=store)
    assert "Rejected 1" in out
    path = wa._pending_path(wa.MEMORY, stale1["id"])
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "rejected"
    # The invalid record was untouched by --stale.
    assert _status(bad["id"])[0] == "invalid"

    out = handle_pending_subcommand(wa.MEMORY, ["reject", "--invalid"], memory_store=store)
    assert "Rejected 1" in out
    assert _status(bad["id"])[0] == "rejected"


def test_reject_rejected_records_invisible_in_pending(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "sigma rule entry")
    rec = _stage_remove("sigma rule entry", "sigma rule")
    out = handle_pending_subcommand(wa.MEMORY, ["reject", rec["id"]], memory_store=store)
    assert "Rejected pending memory write" in out
    assert rec["id"] not in handle_pending_subcommand(wa.MEMORY, ["pending"], memory_store=store)
    listing = handle_pending_subcommand(wa.MEMORY, ["pending"], memory_store=store)
    assert "sigma" not in listing  # not even in the archived footer
    assert wa._pending_path(wa.MEMORY, rec["id"]).exists()  # still on disk for audit


def test_reject_all_marks_every_memory_record(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "tau rule entry")
    a = _stage_remove("tau rule entry", "tau rule")
    b = _stage({"action": "add", "target": "user", "content": "prefers dark mode"})
    out = handle_pending_subcommand(wa.MEMORY, ["reject", "all"], memory_store=store)
    assert "Rejected 2" in out
    for r in (a, b):
        assert _status(r["id"])[0] == "rejected"
        assert wa._pending_path(wa.MEMORY, r["id"]).exists()
    # Terminal: a second reject all finds nothing left to mark.
    out2 = handle_pending_subcommand(wa.MEMORY, ["reject", "all"], memory_store=store)
    assert "Rejected 0" in out2


def test_reject_multiple_ids(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "upsilon rule entry")
    a = _stage_remove("upsilon rule entry", "upsilon rule")
    b = _stage({"action": "add", "target": "memory", "content": "phi note"})
    out = handle_pending_subcommand(
        wa.MEMORY, ["reject", a["id"], b["id"], "nosuch"], memory_store=store)
    assert "Rejected 2" in out
    assert "nosuch" in out
    assert _status(a["id"])[0] == "rejected" and _status(b["id"])[0] == "rejected"


# ---------------------------------------------------------------------------
# approve semantics
# ---------------------------------------------------------------------------

def _mixed_queue(store):
    """Seed a mixed queue. The superseding newer replace is itself ready (its pin is
    live), so the queue holds TWO ready records, one stale, one superseded, one invalid."""
    from tools import write_approval as wa
    store.add("memory", "chi rule entry")
    ready = _stage({"action": "add", "target": "memory", "content": "psi note"})
    stale = _stage_remove("chi rule entry", "chi rule")
    store.remove("memory", "chi rule entry")  # pins drift after staging
    store.add("memory", "omega rule entry")
    superseded = _stage({"action": "replace", "target": "memory", "old_text": "omega rule",
                         "content": "omega v2", "matched_entry": "omega rule entry"},
                        created_at=1000.0)
    superseder = _stage({"action": "replace", "target": "memory", "old_text": "omega rule",
                         "content": "omega v3", "matched_entry": "omega rule entry"},
                        created_at=2000.0)
    invalid = _stage({"action": "purge", "target": "memory"})
    return ready, stale, superseded, invalid, superseder


def test_approve_all_applies_only_ready_records(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    ready, stale, superseded, invalid, superseder = _mixed_queue(store)

    out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    # Both READY records apply (the add + the newer replace that superseded the old one).
    assert "Approved 2" in out
    entries = store._entries_for("memory")
    assert "psi note" in entries and "omega v3" in entries
    assert "chi rule entry" not in entries  # the stale remove did NOT apply
    assert "skipped 3 non-ready" in out
    assert "1 stale" in out and "1 superseded" in out and "1 invalid" in out
    assert "remain archived" in out
    # Non-ready records stay on disk with statuses persisted; ready records are consumed.
    assert wa.get_pending(wa.MEMORY, ready["id"]) is None
    assert wa.get_pending(wa.MEMORY, superseder["id"]) is None
    assert _status(stale["id"])[0] == "stale"
    assert _status(superseded["id"])[0] == "superseded"
    assert _status(invalid["id"])[0] == "invalid"
    for r in (stale, superseded, invalid):
        assert wa._pending_path(wa.MEMORY, r["id"]).exists()


def test_approve_all_no_infinite_replay(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    _mixed_queue(store)
    handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    assert "No pending" in out and "Approved" not in out
    # A third run is still stable (classification idempotent, nothing new applied).
    out3 = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    assert "No pending" in out3


def test_approve_single_failed_attempt_persists_classification(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "stale single entry")
    rec = _stage_remove("stale single entry", "stale single")
    store.remove("memory", "stale single entry")

    out = handle_pending_subcommand(wa.MEMORY, ["approve", rec["id"]], memory_store=store)
    assert "Failed" in out
    assert _status(rec["id"])[0] == "stale"  # stops showing as ready
    # It no longer counts as an approval candidate.
    listing = handle_pending_subcommand(wa.MEMORY, ["pending"], memory_store=store)
    assert "Pending memory writes (1 stale):" in listing  # zero categories omitted
    assert rec["id"] not in listing  # not listed; counts-only footer


def test_approve_single_ready_still_applies(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "apply me entry")
    rec = _stage({"action": "replace", "target": "memory", "old_text": "apply me",
                  "content": "applied entry", "matched_entry": "apply me entry"})
    out = handle_pending_subcommand(wa.MEMORY, ["approve", rec["id"]], memory_store=store)
    assert "Approved 1" in out
    assert store._entries_for("memory") == ["applied entry"]


# ---------------------------------------------------------------------------
# /memory rendering
# ---------------------------------------------------------------------------

def test_pending_list_renders_counts_and_footer(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    ready, _stale, _superseded, _invalid, superseder = _mixed_queue(store)

    out = handle_pending_subcommand(wa.MEMORY, ["pending"], memory_store=store)
    assert "Pending memory writes (2 ready, 1 stale, 1 superseded, 1 invalid)" in out
    assert "Archived: 1 invalid, 1 stale, 1 superseded" in out
    assert "--stale" in out and "--superseded" in out and "--invalid" in out
    # The ready records are listed as approval candidates...
    assert ready["id"] in out and superseder["id"] in out
    # ...while archived ones appear only via the footer counts (no stale/invalid text
    # beyond the counts and the reject flags).
    assert "chi rule entry" not in out


def test_pending_list_only_nonzero_categories(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "solo entry")
    _stage({"action": "add", "target": "memory", "content": "lone note"})
    out = handle_pending_subcommand(wa.MEMORY, ["pending"], memory_store=store)
    assert "(1 ready)" in out
    assert "stale" not in out and "Archived" not in out


def test_bare_memory_shows_both_gate_lines(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    out = handle_pending_subcommand(wa.MEMORY, [], memory_store=store)
    assert "memory.write_approval = off" in out
    assert "unattended consolidation: off" in out


def test_bare_memory_shows_unattended_on_when_opted_in(hermes_home):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault("memory", {})["allow_unattended_consolidation"] = True
    cfg.save_config(c)

    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    out = handle_pending_subcommand(wa.MEMORY, [], memory_store=store)
    assert "unattended consolidation: on" in out


# ---------------------------------------------------------------------------
# /memory diff
# ---------------------------------------------------------------------------

def test_diff_without_id_shows_usage(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    assert handle_pending_subcommand(wa.MEMORY, ["diff"]) == "Usage: /memory diff <id>"
    assert "No pending memory write" in handle_pending_subcommand(wa.MEMORY, ["diff", "zzz"])


def test_diff_single_replace_full_before_after(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    entry = "Before " + "x" * 300 + " tail-unique-end " + "y" * 300
    after = "After " + "z" * 300 + " after-unique-end " + "w" * 300
    rec = _stage({"action": "replace", "target": "memory", "old_text": "Before",
                  "content": after, "matched_entry": entry})
    out = handle_pending_subcommand(wa.MEMORY, ["diff", rec["id"]], memory_store=store)
    assert rec["id"] in out
    # Head 300 chars of the pinned entry and the tail 150+ chars both survive truncation.
    assert entry[:200] in out
    assert entry[-150:] in out
    assert after[:200] in out
    assert after[-150:] in out
    assert "BEFORE" in out and "AFTER" in out
    # The middle was dropped (bounded rendering).
    assert "x" * 300 not in out


def test_diff_batch_enumerates_operations(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    rec = _stage({"action": "batch", "target": "memory", "operations": [
        {"action": "replace", "old_text": "a", "content": "new a",
         "matched_entry": "old a entry"},
        {"action": "remove", "old_text": "b", "matched_entry": "old b entry"},
        {"action": "add", "content": "fresh note"},
    ]})
    out = handle_pending_subcommand(wa.MEMORY, ["diff", rec["id"]], memory_store=store)
    assert "Operation 1 (replace)" in out
    assert "Operation 2 (remove)" in out
    assert "Operation 3 (add)" in out
    assert "old a entry" in out and "old b entry" in out
    assert "(entry removed)" in out
