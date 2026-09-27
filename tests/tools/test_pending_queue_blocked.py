"""Correction follow-up to the pending-queue lifecycle: the BLOCKED status.

A structurally valid proposal whose target/store is CURRENTLY unavailable (disabled
target, unreadable backing file, external drift a real commit would refuse) must not be
READY (approve-all retry loop) nor terminal INVALID (permanently stranded evidence). It
classifies ``blocked`` — non-terminal, re-evaluated every pass, skipped by approve-all,
invisible to the active digest, visible to the operator, explicitly rejectable, and
automatically recovered to ready when the condition clears.

Also pinned here: structured failure kinds replace error-text lifecycle matching
(re-wording an error can never flip a verdict), lifecycle inspection stays read-only
(no store write, no .bak, no failure-budget movement, no audit append), and legacy
PR #19 ``invalid`` records migrate exactly once to the corrected verdicts.
"""

import json
import os
import shutil
import stat
import tempfile

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_blocked_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _store(memory_enabled=True, user_profile_enabled=True, limit=4000):
    from tools.memory_tool import MemoryStore
    s = MemoryStore(memory_char_limit=limit, user_char_limit=limit,
                    memory_enabled=memory_enabled, user_profile_enabled=user_profile_enabled)
    s.load_from_disk()
    return s


def _stage_remove(entry, old_text, *, target="memory", created_at=None):
    from tools import write_approval as wa
    payload = {"action": "remove", "target": target, "old_text": old_text,
               "matched_entry": entry}
    return _stage_payload(payload, created_at=created_at)


def _stage_payload(payload, *, created_at=None):
    from tools import write_approval as wa
    rec = wa.stage_write(wa.MEMORY, payload, summary="blocked probe", origin="background_review")
    if created_at is not None:
        rec["created_at"] = created_at
        wa._pending_path(wa.MEMORY, rec["id"]).write_text(json.dumps(rec), encoding="utf-8")
    return rec


def _status(record_id):
    from tools import write_approval as wa
    rec = wa.get_pending(wa.MEMORY, record_id)
    return rec.get("status") if rec else None


def _apply_drift(path):
    """Non-roundtrippable external drift that keeps every pin intact; returns the
    original raw text so a test can repair the drift exactly."""
    raw = path.read_text(encoding="utf-8")
    if "\n§\n" in raw:
        drifted = raw.replace("\n§\n", "\n\n§\n", 1)  # stray blank line before a delimiter
    else:
        drifted = raw + "\n§\n"  # trailing empty segment: parse no longer round-trips
    path.write_text(drifted, encoding="utf-8")
    return raw


# ===========================================================================
# Requirements 1-8: disabled memory target => blocked, skipped, recovers
# ===========================================================================

def test_disabled_memory_target_classifies_blocked_not_ready_invalid(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "alpha rule entry")
    rec = _stage_remove("alpha rule entry", "alpha rule")
    disabled = _store(memory_enabled=False)
    assert wa.classify_pending_memory(rec, disabled) == "blocked"
    assert wa.classify_pending_memory(rec, disabled) not in ("ready", "invalid")


def test_disabled_user_target_classifies_blocked(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("user", "user fact entry")
    rec = _stage_remove("user fact entry", "user fact", target="user")
    disabled = _store(user_profile_enabled=False)
    assert wa.classify_pending_memory(rec, disabled) == "blocked"


def test_approve_all_skips_blocked_and_never_calls_apply(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    from tools.memory_tool import apply_memory_pending
    store = _store()
    store.add("memory", "alpha rule entry")
    rec = _stage_remove("alpha rule entry", "alpha rule")
    disabled = _store(memory_enabled=False)
    calls = {"n": 0}
    real = apply_memory_pending

    def counting(payload, s):
        calls["n"] += 1
        return real(payload, s)

    import tools.memory_tool as mt
    mt.apply_memory_pending = counting
    try:
        out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=disabled)
        assert calls["n"] == 0  # apply_memory_pending was NEVER reached
        assert "skipped 1 non-ready (1 blocked)" in out
        assert _status(rec["id"]) == "blocked"
        # second approve-all does not retry it either
        out2 = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=disabled)
        assert calls["n"] == 0
        assert "skipped" in out2
    finally:
        mt.apply_memory_pending = real


def test_blocked_recovers_to_ready_when_memory_reenabled(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "alpha rule entry")
    rec = _stage_remove("alpha rule entry", "alpha rule")
    disabled = _store(memory_enabled=False)
    assert wa.classify_pending_memory(rec, disabled) == "blocked"
    # re-enabled: the SAME record, no recreation
    assert wa.classify_pending_memory(rec, store) == "ready"
    counts = wa.reclassify_pending_memory(store)
    assert counts["ready"] == 1 and counts["blocked"] == 0 and counts["changed"] == 0
    assert _status(rec["id"]) in (None, "ready")
    # recovery from a PERSISTED blocked status persists the ready transition
    wa.update_pending_status(wa.MEMORY, rec["id"], wa.STATUS_BLOCKED)
    counts2 = wa.reclassify_pending_memory(store)
    assert counts2["ready"] == 1 and counts2["changed"] == 1
    assert _status(rec["id"]) in (None, "ready")


def test_blocked_user_target_recovers_when_profile_reenabled(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("user", "user fact entry")
    rec = _stage_remove("user fact entry", "user fact", target="user")
    disabled = _store(user_profile_enabled=False)
    assert wa.classify_pending_memory(rec, disabled) == "blocked"
    assert wa.classify_pending_memory(rec, store) == "ready"


def test_blocked_persists_as_blocked_not_hidden(hermes_home):
    """/memory pending persists blocked and the record stays visible (not malformed data)."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "alpha rule entry")
    rec = _stage_remove("alpha rule entry", "alpha rule")
    disabled = _store(memory_enabled=False)
    out = handle_pending_subcommand(wa.MEMORY, ["pending"], memory_store=disabled)
    assert _status(rec["id"]) == "blocked"
    assert "1 blocked" in out
    assert "blocked" in out  # operator-visible wording


# ===========================================================================
# Requirements 9-15: transient unreadable store => blocked, never invalid
# ===========================================================================

def test_unreadable_store_classifies_blocked(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "theta rule entry")
    rec = _stage_remove("theta rule entry", "theta rule")
    path = store._path_for("memory")
    os.chmod(path, 0)
    try:
        fresh = _store()
        assert wa.classify_pending_memory(rec, fresh) == "blocked"
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def test_unreadable_blocked_recovers_to_ready_after_readability_restored(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "theta rule entry")
    rec = _stage_remove("theta rule entry", "theta rule")
    path = store._path_for("memory")
    os.chmod(path, 0)
    try:
        fresh = _store()
        assert wa.classify_pending_memory(rec, fresh) == "blocked"
        counts = wa.reclassify_pending_memory(fresh)
        assert counts["blocked"] == 1 and counts["invalid"] == 0
        assert _status(rec["id"]) == "blocked"
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    counts = wa.reclassify_pending_memory(_store())
    assert counts["ready"] == 1 and counts["blocked"] == 0
    assert _status(rec["id"]) in (None, "ready")
    rec_now = wa.get_pending(wa.MEMORY, rec["id"])
    assert rec_now.get("status") != "invalid"  # never converted to terminal invalid


# ===========================================================================
# Requirements 16-23: external drift => blocked, read-only, recovers;
# real write keeps backup/remediation
# ===========================================================================

def test_external_drift_with_pin_intact_classifies_blocked(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "gamma rule entry")
    rec = _stage_remove("gamma rule entry", "gamma rule")
    _apply_drift(store._path_for("memory"))
    fresh = _store()
    assert wa.classify_pending_memory(rec, fresh) == "blocked"


def test_classification_creates_no_bak_and_moves_no_failure_counter(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "gamma rule entry")
    rec = _stage_remove("gamma rule entry", "gamma rule")
    path = store._path_for("memory")
    _apply_drift(path)
    fresh = _store()
    files_before = sorted(os.listdir(path.parent))
    counters_before = fresh._consolidation_failures
    wa.classify_pending_memory(rec, fresh)
    wa.classify_pending_memory_queue(fresh)
    wa.reclassify_pending_memory(fresh)
    assert sorted(os.listdir(path.parent)) == files_before  # NO .bak.* created
    assert fresh._consolidation_failures == counters_before  # budget untouched
    assert _status(rec["id"]) == "blocked"  # reclassify persisted blocked


def test_approve_all_skips_drift_blocked(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    from tools.memory_tool import apply_memory_pending
    store = _store()
    store.add("memory", "gamma rule entry")
    rec = _stage_remove("gamma rule entry", "gamma rule")
    path = store._path_for("memory")
    _apply_drift(path)
    fresh = _store()
    calls = {"n": 0}
    real = apply_memory_pending

    def counting(payload, s):
        calls["n"] += 1
        return real(payload, s)

    import tools.memory_tool as mt
    mt.apply_memory_pending = counting
    try:
        out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=fresh)
        assert calls["n"] == 0  # never attempted
        assert "1 blocked" in out
    finally:
        mt.apply_memory_pending = real


def test_drift_blocked_recovers_to_ready_after_drift_repaired(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "gamma rule entry")
    rec = _stage_remove("gamma rule entry", "gamma rule")
    path = store._path_for("memory")
    original_raw = _apply_drift(path)
    assert wa.classify_pending_memory(rec, _store()) == "blocked"
    # Repair: restore the exact clean raw text (same entries, round-trippable).
    path.write_text(original_raw, encoding="utf-8")
    assert wa.classify_pending_memory(rec, _store()) == "ready"
    assert _status(rec["id"]) in (None, "ready")


def test_real_write_under_drift_still_backs_up_and_refuses(hermes_home):
    """The REAL write path keeps its existing backup/remediation behavior — untouched."""
    from tools.memory_tool import apply_memory_pending
    store = _store()
    store.add("memory", "gamma rule entry")
    rec = _stage_remove("gamma rule entry", "gamma rule")
    path = store._path_for("memory")
    _apply_drift(path)
    fresh = _store()
    result = apply_memory_pending(rec["payload"], fresh)
    assert result.get("success") is False
    assert "drift_backup" in result
    baks = [f for f in os.listdir(path.parent) if ".bak." in f]
    assert baks, "real commit must still create the remediation backup"


# ===========================================================================
# Requirements 24-31: structured failure kinds; no error-string lifecycle logic
# ===========================================================================

def test_stale_preflight_reports_failure_kind_stale(hermes_home):
    from tools.memory_tool import MemoryStore
    from tools.memory_tool_store import KIND_STALE
    store = _store()
    store.add("memory", "kappa rule entry")
    ops = [{"action": "remove", "old_text": "kappa rule", "matched_entry": "kappa rule entry"}]
    store.remove("memory", "kappa rule entry")
    result = store.resolve_batch_entries("memory", ops)
    assert result.get("failure_kind") == KIND_STALE


def test_unreadable_reports_failure_kind_blocked(hermes_home):
    from tools.memory_tool_store import KIND_BLOCKED
    store = _store()
    store.add("memory", "lambda entry")
    path = store._path_for("memory")
    os.chmod(path, 0)
    try:
        result = store.resolve_batch_entries(
            "memory", [{"action": "remove", "old_text": "lambda", "matched_entry": "lambda entry"}])
        assert result.get("failure_kind") == KIND_BLOCKED
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def test_drift_probe_reports_failure_kind_blocked_read_only(hermes_home):
    from tools.memory_tool_store import KIND_BLOCKED
    store = _store()
    store.add("memory", "mu entry")
    path = store._path_for("memory")
    _apply_drift(path)
    probe = store.detect_drift("memory")
    assert probe.get("success") is False
    assert probe.get("failure_kind") == KIND_BLOCKED
    assert "drift_backup" not in probe  # read-only variant names no backup
    assert not [f for f in os.listdir(path.parent) if ".bak." in f]


def test_malformed_reports_failure_kind_invalid(hermes_home):
    from tools.memory_tool_store import KIND_INVALID
    store = _store()
    result = store.resolve_batch_entries("memory", [])
    assert result.get("failure_kind") == KIND_INVALID


def test_final_budget_failure_reports_failure_kind_invalid(hermes_home):
    from tools.memory_tool_store import KIND_INVALID
    store = _store(limit=120)
    store.add("memory", "tiny store entry")
    ops = [{"action": "replace", "old_text": "tiny store", "matched_entry": "tiny store entry",
            "content": "B" * 300}]
    result = store.resolve_batch_entries("memory", ops)
    assert result.get("success") is False
    assert result.get("failure_kind") == KIND_INVALID


def test_reworded_error_text_keeps_lifecycle_verdict(hermes_home):
    """Requirement 30: altering the human-readable text must not move the lifecycle."""
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "nu rule entry")
    rec = _stage_remove("nu rule entry", "nu rule")
    store.remove("memory", "nu rule entry")
    assert wa.classify_pending_memory(rec, store) == "stale"

    import tools.memory_tool_store as mts
    original = mts._stale_entry_message
    mts._stale_entry_message = lambda entry: "completely different wording entirely"
    try:
        assert wa.classify_pending_memory(rec, _store()) == "stale"
    finally:
        mts._stale_entry_message = original


def test_error_text_never_reads_lifecycle_markers(hermes_home):
    """Requirement 24/31: the classifier module must not carry error-string lifecycle
    matching, and a store entry containing the OLD marker phrases cannot affect any
    verdict."""
    import tools.write_approval as wa
    assert not hasattr(wa, "_STALE_PREFLIGHT_MARKERS")
    source = open(wa.__file__, encoding="utf-8").read()
    assert "_STALE_PREFLIGHT_MARKERS" not in source
    assert "no entry matched" not in source
    assert "changed since" not in source

    from tools import write_approval as wam
    store = _store()
    poisoned = ("no entry matched changed since could not be read is no longer "
                "matched multiple distinct")
    store.add("memory", poisoned)
    rec = _stage_remove(poisoned, poisoned[:20])
    assert wam.classify_pending_memory(rec, store) == "ready"
    store.remove("memory", poisoned)
    assert wam.classify_pending_memory(rec, store) == "stale"


# ===========================================================================
# Requirements 32-42: lifecycle regressions
# ===========================================================================

def test_sequential_batch_stays_ready_and_applies_to_final_state(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "entry A")
    ops = [{"action": "replace", "old_text": "entry A", "matched_entry": "entry A",
            "content": "entry B"},
           {"action": "replace", "old_text": "entry B", "matched_entry": "entry B",
            "content": "entry C"}]
    rec = wa.stage_write(wa.MEMORY, {"action": "batch", "target": "memory", "operations": ops},
                         summary="A->B->C", origin="background_review")
    assert wa.classify_pending_memory(rec, store) == "ready"
    out = handle_pending_subcommand(wa.MEMORY, ["approve", "all"], memory_store=store)
    assert "Approved 1" in out
    assert "entry C" in store._entries_for("memory")
    assert "entry A" not in store._entries_for("memory")


def test_genuinely_disappeared_pin_is_stale_and_recovers(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "xi rule entry")
    rec = _stage_remove("xi rule entry", "xi rule")
    store.remove("memory", "xi rule entry")
    assert wa.classify_pending_memory(rec, store) == "stale"
    store.add("memory", "xi rule entry")  # the exact reviewed state returns
    assert wa.classify_pending_memory(rec, store) == "ready"


def test_over_budget_pinned_proposal_remains_invalid(hermes_home):
    from tools import write_approval as wa
    store = _store(limit=120)
    store.add("memory", "omicron witness entry")
    rec = wa.stage_write(wa.MEMORY, {"action": "replace", "target": "memory",
                                     "old_text": "omicron witness",
                                     "content": "P" * 300,
                                     "matched_entry": "omicron witness entry"},
                         summary="over budget", origin="background_review")
    assert wa.classify_pending_memory(rec, store) == "invalid"


def test_structured_invalid_is_terminal_no_churn(hermes_home):
    from tools import write_approval as wa
    store = _store(limit=120)
    store.add("memory", "pi witness entry")
    rec = wa.stage_write(wa.MEMORY, {"action": "replace", "target": "memory",
                                     "old_text": "pi witness", "content": "Q" * 300,
                                     "matched_entry": "pi witness entry"},
                         summary="over budget", origin="background_review")
    counts = wa.reclassify_pending_memory(store)
    assert counts["invalid"] == 1
    stored = wa.get_pending(wa.MEMORY, rec["id"])
    assert stored["status"] == "invalid"
    assert stored["lifecycle_version"] == wa.LIFECYCLE_VERSION  # structured: terminal
    assert stored["status_reason"]
    counts2 = wa.reclassify_pending_memory(store)
    assert counts2["changed"] == 0  # no churn on later passes


def test_malformed_and_legacy_unpinned_remain_invalid(hermes_home):
    from tools import write_approval as wa
    store = _store()
    bad_action = wa.stage_write(wa.MEMORY, {"action": "explode", "target": "memory"},
                                summary="bad", origin="background_review")
    assert wa.classify_pending_memory(bad_action, store) == "invalid"
    unpinned = wa.stage_write(wa.MEMORY, {"action": "remove", "target": "memory",
                                          "old_text": "no pin"},
                              summary="legacy", origin="background_review")
    assert wa.classify_pending_memory(unpinned, store) == "invalid"


def test_rejected_remains_terminal(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "rho entry")
    rec = _stage_remove("rho entry", "rho")
    wa.update_pending_status(wa.MEMORY, rec["id"], wa.STATUS_REJECTED)
    rec = wa.get_pending(wa.MEMORY, rec["id"])
    assert wa.classify_pending_memory(rec, store) == "rejected"
    counts = wa.reclassify_pending_memory(store)
    assert counts["changed"] == 0
    assert _status(rec["id"]) == "rejected"


def test_supersession_semantics_unchanged(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "sigma rule entry")
    old = _stage_remove("sigma rule entry", "sigma rule", created_at=1000.0)
    newer = _stage_remove("sigma rule entry", "sigma rule", created_at=2000.0)
    assert wa.classify_pending_memory(old, store) == "superseded"
    assert wa.classify_pending_memory(newer, store) == "ready"


def test_partial_overlap_batches_stay_independently_reviewable(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "tau entry")
    store.add("memory", "ups entry")
    old = _stage_payload({"action": "batch", "target": "memory", "operations": [
        {"action": "replace", "old_text": "tau", "matched_entry": "tau entry", "content": "tau v2"},
        {"action": "remove", "old_text": "ups", "matched_entry": "ups entry"}]},
        created_at=1000.0)
    new = _stage_payload({"action": "batch", "target": "memory", "operations": [
        {"action": "replace", "old_text": "tau", "matched_entry": "tau entry", "content": "tau v3"}]},
        created_at=2000.0)
    verdicts = dict((r["id"], v) for r, v in wa.classify_pending_memory_queue(store))
    assert verdicts[old["id"]] == "ready"  # NOT superseded: distinct work remains
    assert verdicts[new["id"]] == "ready"


def test_existing_backlog_never_auto_applied_by_classification(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "phi entry")
    rec = _stage_remove("phi entry", "phi")
    for _ in range(3):
        wa.reclassify_pending_memory(store)
    assert _status(rec["id"]) in (None, "ready", "stale", "blocked", "superseded")
    assert "phi entry" in store._entries_for("memory")  # nothing applied
    assert wa.get_pending(wa.MEMORY, rec["id"]) is not None  # nothing consumed


# ===========================================================================
# Legacy migration (merged PR #19 already produced terminal invalid records)
# ===========================================================================

def test_legacy_invalid_from_transient_read_failure_migrates_to_ready(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "chi entry")
    rec = _stage_remove("chi entry", "chi")
    # Simulate the legacy classifier's mistake: terminal invalid, no v2 metadata.
    legacy = wa.get_pending(wa.MEMORY, rec["id"])
    legacy.pop("lifecycle_version", None)  # staged fresh via update-free path: absent already
    legacy["status"] = "invalid"
    wa._pending_path(wa.MEMORY, rec["id"]).write_text(json.dumps(legacy), encoding="utf-8")
    counts = wa.reclassify_pending_memory(store)
    assert counts["ready"] == 1 and counts["invalid"] == 0
    migrated = wa.get_pending(wa.MEMORY, rec["id"])
    assert migrated.get("status") in (None, "ready")


def test_legacy_invalid_from_unreadable_store_migrates_to_blocked_then_ready(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "psi entry")
    rec = _stage_remove("psi entry", "psi")
    legacy = wa.get_pending(wa.MEMORY, rec["id"])
    legacy["status"] = "invalid"  # legacy terminal record, no metadata
    wa._pending_path(wa.MEMORY, rec["id"]).write_text(json.dumps(legacy), encoding="utf-8")
    path = store._path_for("memory")
    os.chmod(path, 0)
    try:
        counts = wa.reclassify_pending_memory(_store())
        assert counts["blocked"] == 1 and counts["invalid"] == 0
        assert _status(rec["id"]) == "blocked"
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    wa.reclassify_pending_memory(store)
    assert _status(rec["id"]) in (None, "ready")


def test_still_truly_invalid_legacy_persists_with_v2_metadata_and_stays_terminal(hermes_home):
    from tools import write_approval as wa
    store = _store(limit=120)
    store.add("memory", "omega witness entry")
    rec = wa.stage_write(wa.MEMORY, {"action": "replace", "target": "memory",
                                     "old_text": "omega witness", "content": "R" * 300,
                                     "matched_entry": "omega witness entry"},
                         summary="over budget", origin="background_review")
    legacy = wa.get_pending(wa.MEMORY, rec["id"])
    legacy["status"] = "invalid"  # archived terminally by the legacy classifier
    wa._pending_path(wa.MEMORY, rec["id"]).write_text(json.dumps(legacy), encoding="utf-8")
    counts = wa.reclassify_pending_memory(store)
    assert counts["invalid"] == 1
    stored = wa.get_pending(wa.MEMORY, rec["id"])
    assert stored["status"] == "invalid"
    assert stored["lifecycle_version"] == wa.LIFECYCLE_VERSION
    assert stored["status_reason"] == "invalid"
    # Terminal thereafter: a second pass does not re-migrate or churn.
    counts2 = wa.reclassify_pending_memory(store)
    assert counts2["changed"] == 0


def test_rejected_legacy_record_never_resurrected(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "final entry")
    rec = _stage_remove("final entry", "final")
    legacy = wa.get_pending(wa.MEMORY, rec["id"])
    legacy["status"] = "rejected"  # operator decision, no v2 metadata needed
    wa._pending_path(wa.MEMORY, rec["id"]).write_text(json.dumps(legacy), encoding="utf-8")
    counts = wa.reclassify_pending_memory(store)
    assert counts["rejected"] == 1 and counts["ready"] == 0
    assert _status(rec["id"]) == "rejected"


def test_new_verdicts_persist_version_and_reason(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "st entry")
    stale_rec = _stage_remove("st entry", "st")
    store.remove("memory", "st entry")
    store.add("memory", "bl entry")
    blocked_rec = _stage_remove("bl entry", "bl")
    disabled = _store(memory_enabled=False)
    wa.reclassify_pending_memory(disabled)  # classifies against the DISABLED store
    # Both records are blocked here (disabled target judges before pin freshness);
    # both persist the structured metadata with the target_disabled reason.
    stored = wa.get_pending(wa.MEMORY, stale_rec["id"])
    assert stored["status"] == "blocked"
    assert stored["lifecycle_version"] == 2
    assert stored["status_reason"] == "target_disabled"
    stored_b = wa.get_pending(wa.MEMORY, blocked_rec["id"])
    assert stored_b["status"] == "blocked"
    assert stored_b["lifecycle_version"] == 2
    assert stored_b["status_reason"] == "target_disabled"


def test_stale_verdict_persists_reason_when_target_enabled(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "sv entry")
    stale_rec = _stage_remove("sv entry", "sv")
    store.remove("memory", "sv entry")
    wa.reclassify_pending_memory(store)
    stored = wa.get_pending(wa.MEMORY, stale_rec["id"])
    assert stored["status"] == "stale"
    assert stored["lifecycle_version"] == 2
    assert stored["status_reason"] in ("stale", "pin_gone")


def test_stale_persisted_record_reclassified_blocked_when_target_disabled(hermes_home):
    """Availability is judged before recovery on every pass: a stale record whose target
    gets disabled classifies blocked, then recovers only to its true verdict."""
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "bt entry")
    rec = _stage_remove("bt entry", "bt")
    store.remove("memory", "bt entry")
    wa.update_pending_status(wa.MEMORY, rec["id"], wa.STATUS_STALE)
    disabled = _store(memory_enabled=False)
    assert wa.classify_pending_memory(rec, disabled) == "blocked"
    assert wa.classify_pending_memory(rec, store) == "stale"


# ===========================================================================
# Digest + bulk reject surfaces
# ===========================================================================

def test_digest_omits_blocked_and_recovers_immediately(hermes_home):
    from agent.background_review import pending_memory_proposals_context
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "dig entry")
    rec = _stage_remove("dig entry", "dig")
    # blocked via disabled target: must NOT appear in the digest
    disabled = _store(memory_enabled=False)
    import tools.memory_tool as mt
    real_loader = mt.load_on_disk_store
    mt.load_on_disk_store = lambda: disabled
    try:
        assert pending_memory_proposals_context() == ""
        assert wa.classify_pending_memory(rec, disabled) == "blocked"
        # recovery: FRESH verdict (no /memory pending persistence pass needed)
        mt.load_on_disk_store = lambda: _store()
        digest = pending_memory_proposals_context()
        assert rec["id"] in digest and "ready" in digest
    finally:
        mt.load_on_disk_store = real_loader


def test_digest_pure_classification_persists_nothing(hermes_home):
    from agent.background_review import pending_memory_proposals_context
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "pure entry")
    rec = _stage_remove("pure entry", "pure")
    store.remove("memory", "pure entry")  # stale verdict incoming
    import tools.memory_tool as mt
    real_loader = mt.load_on_disk_store
    mt.load_on_disk_store = lambda: _store()
    try:
        assert pending_memory_proposals_context() == ""  # stale: omitted
        assert _status(rec["id"]) is None  # digest must not persist statuses
    finally:
        mt.load_on_disk_store = real_loader


def test_reject_blocked_flag_and_usage_line(hermes_home):
    from hermes_cli.write_approval_commands import _BULK_REJECT_FLAGS, handle_pending_subcommand
    from tools import write_approval as wa
    assert _BULK_REJECT_FLAGS["--blocked"] == "blocked"
    store = _store()
    store.add("memory", "rj entry")
    rec = _stage_remove("rj entry", "rj")
    disabled = _store(memory_enabled=False)
    out = handle_pending_subcommand(wa.MEMORY, ["reject", "--blocked"], memory_store=disabled)
    assert "Rejected 1" in out
    assert _status(rec["id"]) == "rejected"


def test_blocked_not_auto_rejected_by_any_pass(hermes_home):
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "na entry")
    rec = _stage_remove("na entry", "na")
    disabled = _store(memory_enabled=False)
    wa.reclassify_pending_memory(disabled)
    # No operator action taken: blocked stays blocked (never auto-rejected).
    assert _status(rec["id"]) == "blocked"


def test_pending_list_blocked_footer_wording(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    store = _store()
    store.add("memory", "ft entry")
    rec = _stage_remove("ft entry", "ft")
    disabled = _store(memory_enabled=False)
    out = handle_pending_subcommand(wa.MEMORY, ["pending"], memory_store=disabled)
    assert "1 blocked" in out
    assert "--blocked" in out
    assert "current store/config prevents safe application" in out
