"""Audit ledger + undo for autonomous unattended memory consolidation.

When ``memory.allow_unattended_consolidation`` is opted in, an unattended
background-review fork may apply replace/remove consolidation directly. Every
such application is recorded here FIRST (fail-closed: no audit record, no
mutation) in an append-only JSONL ledger under
``<HERMES_HOME>/memory_backups/consolidations.jsonl``, capturing the FULL
before-state (raw file content, entry list, sha256) so ``restore()`` can undo
the whole consolidation as ONE atomic public batch against the live store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home, mkdir_under_hermes_home

logger = logging.getLogger(__name__)

_DESTRUCTIVE_ACTIONS = ("replace", "remove")


class MemoryConsolidationAuditError(Exception):
    """The audit snapshot could not be persisted; the caller must apply NOTHING."""


def _backups_dir() -> Path:
    return get_hermes_home() / "memory_backups"


def _ledger_path() -> Path:
    return _backups_dir() / "consolidations.jsonl"


def _ensure_backups_dir() -> Path:
    """Create the backups dir owner-only, repairing bits left loose by an older run.

    ``mkdir`` modes are filtered through the process umask (and are frequently
    looser than intended), so the mode is applied explicitly afterwards.
    """
    directory = _backups_dir()
    mkdir_under_hermes_home(directory)
    with suppress(Exception):
        os.chmod(directory, 0o700)
    return directory


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append_record(record: Dict[str, Any]) -> None:
    """Append one JSONL record, owner-only. Any failure raises
    MemoryConsolidationAuditError — callers must not proceed without the ledger."""
    try:
        _ensure_backups_dir()
        # The ledger holds the FULL raw memory file content, so it is exactly as
        # sensitive as MEMORY.md/USER.md (written 0o600) — never looser. Same
        # convention as the memory store's lock file: O_NOFOLLOW where available,
        # and fchmod on the fd (not the path) both to repair a record left loose by
        # an older Hermes and to avoid a path-swap window.
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        raw_fd = os.open(_ledger_path(), flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(raw_fd, 0o600)
            with os.fdopen(raw_fd, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            with suppress(Exception):
                os.close(raw_fd)
            raise
    except Exception as e:  # OSError, permission, disk-full, guardian refusal…
        raise MemoryConsolidationAuditError(f"failed to append audit record: {e}") from e


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_begin(target: str, ops: List[Dict[str, Any]], before_raw: str,
                 before_entries: List[str], origin: str = "background_review",
                 planned_after_raw: Optional[str] = None) -> str:
    """Snapshot the pre-mutation state BEFORE anything is applied. Fail-closed:
    raises MemoryConsolidationAuditError when the ledger cannot be written —
    the caller must then refuse the consolidation entirely. Returns the audit id.

    ``planned_after_raw`` is the deterministic planned final state derived from the
    validated plan against this exact before-state; its sha256 lands in the record as
    ``planned_after_sha256`` — plan-consistency evidence used to VALIDATE the later
    'applied' event and to detect inconsistent journal records. It is NOT by itself
    commit evidence: automatic undo requires a durable 'applied' event."""
    audit_id = uuid.uuid4().hex[:12]
    destructive = [
        {"action": op.get("action"),
         "old_text": op.get("old_text"),
         "content": op.get("content"),
         "matched_entry": op.get("matched_entry")}
        for op in (ops or []) if (op or {}).get("action") in _DESTRUCTIVE_ACTIONS
    ]
    record = {
        "event": "begin", "id": audit_id, "ts": _utc_now_iso(),
        "hermes_home": str(get_hermes_home()), "target": target,
        "origin": origin, "ops": destructive,
        "before_raw": before_raw or "",
        "before_entries": list(before_entries or []),
        "before_sha256": _sha256(before_raw or ""),
    }
    if planned_after_raw is not None:
        record["planned_after_sha256"] = _sha256(planned_after_raw)
    _append_record(record)
    return audit_id


def record_undone(audit_id: str, target: str, restored_sha256: str) -> None:
    """Best-effort 'undone' audit event appended after a successful restore — durable
    evidence that the transaction was reversed (an already-undone id restores as a
    no-op, but the journal keeps its own record). Never fatal: the restore is already
    durable in the store."""
    record = {
        "event": "undone", "id": audit_id, "ts": _utc_now_iso(), "target": target,
        "restored_sha256": restored_sha256,
    }
    try:
        _append_record(record)
    except Exception:
        logger.warning("Consolidation %s restored but its 'undone' audit event failed to "
                       "persist.", audit_id, exc_info=True)


def record_applied(audit_id: str, target: str, after_raw: str, counts: Dict[str, int]) -> bool:
    """Best-effort completion record — now REPORTS success (correction round 3): the
    commit is already durable in the store and a ledger failure never invalidates it,
    but the CALLER must know when the durable commit-evidence event is missing, because
    ``/memory undo`` refuses automatic undo without an ``applied`` event. Returns True
    when the event was appended; False when it failed (the caller surfaces the degraded
    recovery state instead of silently advertising an undo command that cannot be
    safely honored; manual recovery stays possible from the begin record)."""
    record = {
        "event": "applied", "id": audit_id, "ts": _utc_now_iso(), "target": target,
        "after_sha256": _sha256(after_raw or ""),
        "counts": dict(counts or {}),
        "undo_hint": f"/memory undo {audit_id}",
    }
    try:
        _append_record(record)
    except Exception:
        logger.warning("Consolidation %s applied but its audit 'applied' record failed to "
                       "persist; automatic undo is unavailable for it (manual recovery "
                       "from the begin record remains possible).", audit_id, exc_info=True)
        return False
    return True


def _read_ledger() -> List[Dict[str, Any]]:
    path = _ledger_path()
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with suppress(Exception):
        # utf-8-sig: same BOM tolerance as the memory store's own reads — Windows
        # tooling (PowerShell Set-Content/Out-File) BOMs files it touches, and
        # json.loads on the resulting first line fails without it.
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            with suppress(Exception):
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
    return records


def get_record(audit_id: str) -> Optional[Dict[str, Any]]:
    """The 'begin' record for *audit_id*, or None."""
    for record in _read_ledger():
        if record.get("event") == "begin" and record.get("id") == audit_id:
            return record
    return None


def list_records() -> List[Dict[str, Any]]:
    """Every parseable ledger record, oldest first."""
    return _read_ledger()


def restore(audit_id: str, store: "Any") -> Dict[str, Any]:
    """Undo one recorded consolidation, fail-closed on the CURRENT state.

    The transaction journal contract (correction round 3 — planned-after is NOT commit
    evidence): the begin record carries ``before_sha256`` (and the full before-state).
    An ``applied`` event is the ONLY durable proof that the mutation happened. A
    begin/prepare record proves intent, not commit — current state accidentally equaling
    ``planned_after_sha256`` (e.g. the commit FAILED and an independent manual change
    later produced that same state) must not make undo restore the before-state and
    erase that independent change.

    Decision table on the CURRENT raw store digest:

    - current == before  -> idempotent no-op (never committed, or already restored);
      memory is NOT rewritten.
    - an ``applied``/committed event exists AND its ``after_sha256`` equals the begin's
      ``planned_after_sha256`` (when the begin planned one — consistency check) AND
      current == applied.after_sha256
      -> the consolidation provably committed and nothing else changed since: restore the
      exact before-state as ONE atomic public batch, then append a best-effort 'undone'
      event. Repeated undo is the no-op above.
    - no applied/commit event exists AND current != before
      -> REFUSE automatic undo. Preserve memory. Surface the audit id and the recorded
      before snapshot for manual recovery (a begin record alone cannot prove the
      mutation happened; restoring on it alone could undo an independent change).
    - an applied event exists but is INCONSISTENT with the begin's plan
      (after_sha256 != planned_after_sha256) -> REFUSE: the journal itself is
      inconsistent; manual recovery only.

    ``planned_after_sha256`` remains recovery-evidence DIAGNOSTICS: it validates the
    applied event against the transaction plan and detects inconsistent journal records.
    It must never by itself prove that the mutation happened.

    Also refuses cross-profile records and an empty recorded before-state (the store
    refuses emptying a non-empty file; that stays explicit rather than generic)."""
    record = get_record(audit_id)
    if record is None:
        return {"success": False,
                "error": f"No consolidation audit record found for id '{audit_id}'."}
    home = record.get("hermes_home")
    if home != str(get_hermes_home()):
        return {"success": False,
                "error": (f"Audit record '{audit_id}' belongs to a different Hermes home "
                          f"({home}); refusing to restore into {get_hermes_home()}.")}
    target = record.get("target", "memory")
    before_sha = record.get("before_sha256") or _sha256(record.get("before_raw") or "")
    before_entries = [e for e in (record.get("before_entries") or []) if e]

    # ONE current raw snapshot is the authority for the digest AND the current entry
    # list: an independent entries read could straddle a concurrent writer and let the
    # restore decide on state S0 while mutating S1. The restore batch then carries this
    # exact raw as its expected-before precondition, checked INSIDE the store's lock —
    # any write that lands between this decision and the restore commit makes the whole
    # restore fail with zero mutation instead of silently preserving or overwriting the
    # concurrent change and calling the result an exact restore.
    store.load_from_disk()
    path = store._path_for(target)
    current_raw, read_ok = store._read_raw_checked(path)
    if not read_ok:
        return {"success": False,
                "error": (f"Refusing to undo '{audit_id}': {path.name} exists but could not "
                          f"be read right now; nothing was changed — retry in a moment.")}
    current_sha = _sha256(current_raw)
    current_entries = list(dict.fromkeys(store._parse_entries(current_raw)))
    store._set_entries(target, current_entries)

    if current_sha == before_sha:
        return {"success": True,
                "message": "Already at the recorded before-state; nothing to restore."}
    if not before_entries:
        # Reaching an empty recorded state means removing every current entry; the
        # store already refuses emptying a non-empty file, so say so up front instead
        # of letting the operator read a generic batch refusal.
        return {"success": False,
                "error": (f"Restoring '{audit_id}' would remove every current entry to reach an "
                          f"empty recorded state. Refusing — edit {store._path_for(target).name} "
                          f"manually if that is really intended.")}

    # Commit evidence (correction round 3): ONLY a durable 'applied' event proves the
    # mutation happened. Its after-digest must also match the begin's plan when the begin
    # recorded one — an 'applied' event that disagrees with planned_after_sha256 is an
    # inconsistent journal, refused outright rather than trusted. The begin's
    # planned_after_sha256 alone proves nothing about commit and is never a fallback.
    events = [r for r in _read_ledger() if r.get("id") == audit_id]
    applied = next((r for r in reversed(events) if r.get("event") == "applied"), None)
    if applied is None:
        # No durable commit evidence. current != before (checked above), so this is NOT
        # the idempotent no-op: either the commit failed and memory was later changed by
        # something else, or the post-commit 'applied' append failed. In both cases an
        # automatic restore could erase an independent change — refuse and hand the
        # operator the audit id + the recorded before snapshot for manual recovery.
        return {"success": False,
                "error": (f"Refusing to automatically undo '{audit_id}': the ledger holds "
                          f"no durable 'applied' event for it, so the consolidation's "
                          f"commit cannot be proven (its commit may have failed, or the "
                          f"post-commit audit append failed). Memory is preserved as-is. "
                          f"For manual recovery the begin record holds the full before-state "
                          f"(before_sha256={before_sha[:12]}…, "
                          f"{len(before_entries)} entries) — inspect "
                          f"'{_ledger_path().name}' and restore by hand if intended.")}
    planned_sha = record.get("planned_after_sha256")
    after_sha = applied.get("after_sha256")
    if planned_sha and after_sha != planned_sha:
        return {"success": False,
                "error": (f"Refusing to undo '{audit_id}': its 'applied' event's "
                          f"after-digest does not match the begin record's "
                          f"planned_after_sha256 — the journal is inconsistent. Memory "
                          f"is untouched; recover manually from the begin record.")}
    if not after_sha or current_sha != after_sha:
        return {"success": False,
                "error": (f"Refusing to undo '{audit_id}': memory has changed since that "
                          f"consolidation committed (current state matches neither the "
                          f"recorded before-state nor its applied after-state), so an "
                          f"exact restore would erase those later changes. Memory is "
                          f"untouched; inspect audit id '{audit_id}' and recover manually "
                          f"if intended.")}

    current = current_entries  # derived from the SAME raw snapshot as current_sha above
    before_set, current_set = set(before_entries), set(current)
    survivors = [e for e in current if e in before_set]
    missing = [e for e in before_entries if e not in current_set]
    if survivors + missing == before_entries:
        # Minimal diff provably lands on the recorded order (removed entries were
        # trailing, or nothing moved): touch only the changed entries.
        ops = ([{"action": "remove", "old_text": e} for e in current if e not in before_set] +
               [{"action": "add", "content": e} for e in missing])
    else:
        # A mid-list entry changed: removing/appending cannot reproduce the recorded
        # order. Swap the whole entry set in one batch — the FINAL state is
        # before_entries (non-empty), so the store's empty-guard does not fire.
        ops = ([{"action": "remove", "old_text": e} for e in current] +
               [{"action": "add", "content": e} for e in before_entries])

    # The precondition binds the restore to the exact state that was proven safe: any
    # intervening write makes apply_batch refuse under the lock with zero mutation.
    result = store.apply_batch(target, ops, expected_before_raw=current_raw)
    if isinstance(result, dict) and result.get("success"):
        with suppress(Exception):  # best-effort journal evidence; the restore is durable
            record_undone(audit_id, target, before_sha)
        result = {**result, "restored": audit_id,
                  "message": f"Restored memory to the state recorded before consolidation "
                             f"'{audit_id}'."}
    return result
