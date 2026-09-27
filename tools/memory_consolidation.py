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
    ``planned_after_sha256`` — recovery evidence that lets a later undo prove the
    consolidation committed even when the best-effort 'applied' append failed."""
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


def record_applied(audit_id: str, target: str, after_raw: str, counts: Dict[str, int]) -> None:
    """Best-effort completion record. The commit is already durable in the store;
    a ledger failure here is logged, never fatal, and never invalidates it."""
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
                       "persist; the begin record remains as evidence.", audit_id, exc_info=True)


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

    The transaction journal contract: the begin record carries ``before_sha256`` (and the
    full before-state), an ``applied`` event (when its append succeeded) carries
    ``after_sha256``, and the begin's ``planned_after_sha256`` (when planned deterministically)
    is recovery evidence for the applied-append-failure case.

    Decision table on the CURRENT raw store digest:

    - current == before  -> idempotent no-op (never committed, or already restored);
      memory is NOT rewritten.
    - current == expected-after (applied.after_sha256, else planned_after_sha256)
      -> the consolidation provably committed and nothing else changed since: restore the
      exact before-state as ONE atomic public batch, then append a best-effort 'undone'
      event. Repeated undo is the no-op above.
    - anything else      -> REFUSE: memory changed since the consolidation (a later
      legitimate write would be erased by an exact restore). Current memory stays
      untouched; the audit id is returned for manual inspection.

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

    store.load_from_disk()
    current_raw = store._read_raw_checked(store._path_for(target))[0]
    current_sha = _sha256(current_raw)

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

    # The expected committed after-state: the applied event when it landed, else the
    # deterministic planned_after recorded in the begin (post-commit ledger-write-failure
    # recovery — see the contract above).
    events = [r for r in _read_ledger() if r.get("id") == audit_id]
    applied = next((r for r in reversed(events) if r.get("event") == "applied"), None)
    expected_after = (applied or {}).get("after_sha256") or record.get("planned_after_sha256")
    if not expected_after or current_sha != expected_after:
        return {"success": False,
                "error": (f"Refusing to undo '{audit_id}': memory has changed since that "
                          f"consolidation (current state matches neither the recorded "
                          f"before-state nor its committed/planned after-state), so an exact "
                          f"restore would erase those later changes. Memory is untouched; "
                          f"inspect audit id '{audit_id}' and recover manually if intended.")}

    current = list(store._entries_for(target))
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

    result = store.apply_batch(target, ops)
    if isinstance(result, dict) and result.get("success"):
        with suppress(Exception):  # best-effort journal evidence; the restore is durable
            record_undone(audit_id, target, before_sha)
        result = {**result, "restored": audit_id,
                  "message": f"Restored memory to the state recorded before consolidation "
                             f"'{audit_id}'."}
    return result
