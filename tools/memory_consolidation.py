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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append_record(record: Dict[str, Any]) -> None:
    """Append one JSONL record atomically-ish. Any failure raises
    MemoryConsolidationAuditError — callers must not proceed without the ledger."""
    try:
        mkdir_under_hermes_home(_backups_dir())
        with _ledger_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # OSError, permission, disk-full, guardian refusal…
        raise MemoryConsolidationAuditError(f"failed to append audit record: {e}") from e


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_begin(target: str, ops: List[Dict[str, Any]], before_raw: str,
                 before_entries: List[str], origin: str = "background_review") -> str:
    """Snapshot the pre-mutation state BEFORE anything is applied. Fail-closed:
    raises MemoryConsolidationAuditError when the ledger cannot be written —
    the caller must then refuse the consolidation entirely. Returns the audit id."""
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
    _append_record(record)
    return audit_id


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
    """Undo one recorded consolidation as ONE atomic public batch (``apply_batch``)
    against the CURRENT state: remove every entry that is present now but was not
    in the recorded before-state, re-add every before entry that is missing now.
    Idempotent; refuses cross-profile records and a batch that would empty the store."""
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
    before_entries = [e for e in (record.get("before_entries") or []) if e]

    store.load_from_disk()
    current = list(store._entries_for(target))

    if current == before_entries:
        return {"success": True, "message": "Already at the recorded state; nothing to restore."}
    if not before_entries:
        # Reaching an empty recorded state means removing every current entry; the
        # store already refuses emptying a non-empty file, so say so up front instead
        # of letting the operator read a generic batch refusal.
        return {"success": False,
                "error": (f"Restoring '{audit_id}' would remove every current entry to reach an "
                          f"empty recorded state. Refusing — edit {store._path_for(target).name} "
                          f"manually if that is really intended.")}

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
        result = {**result, "restored": audit_id,
                  "message": f"Restored memory to the state recorded before consolidation "
                             f"'{audit_id}'."}
    return result
