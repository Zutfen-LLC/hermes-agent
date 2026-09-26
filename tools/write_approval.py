#!/usr/bin/env python3
"""Write-approval gate + pending store for memory and skill writes.

A per-subsystem boolean ``write_approval`` gates the agent's cross-session writes —
**memory** (MEMORY.md / USER.md) and **skills** (SKILL.md + files) — from either
origin (**foreground** turn or **background_review** fork). ``false`` (default)
writes freely; ``true`` never commits directly: it prompts inline (memory,
interactive CLI only) or **stages** the write under
``<HERMES_HOME>/pending/{memory,skills}/<id>.json`` for out-of-band review.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Subsystem identifiers
MEMORY = "memory"
SKILLS = "skills"
_SUBSYSTEMS = (MEMORY, SKILLS)

# --- Pending-record lifecycle statuses (MEMORY) ---
#
# A staged memory proposal can outlive the state it was pinned to: the entry it pins may
# have been edited or removed, a newer proposal may already cover the same entry, or the
# record may predate pinning entirely. Such records are never deleted — each carries a
# ``status`` field (absent = active/ready, so records written before this field existed
# stay valid): 'stale' (a pinned entry is gone/changed), 'superseded' (a newer ACTIVE
# proposal pins the same entry; ``superseded_by`` names it), 'invalid' (malformed payload
# or legacy unpinned destructive op — fail closed forever) and 'rejected' (operator
# -archived). Statused files stay on disk as audit evidence.

STATUS_ACTIVE: Optional[str] = None
STATUS_STALE = "stale"
STATUS_SUPERSEDED = "superseded"
STATUS_INVALID = "invalid"
STATUS_REJECTED = "rejected"

# Per-subsystem config key. Intentionally a single boolean with no "block all writes"
# state — to disable a subsystem use its own enable flag (e.g. ``memory.memory_enabled``).
CONFIG_KEY = "write_approval"
_TRUTHY_STRINGS = frozenset({"on", "true", "yes", "1", "approve", "enabled"})


# --- Config resolution ---

def write_approval_enabled(subsystem: str) -> bool:
    """Read ``<subsystem>.write_approval``; any unset/invalid value means gate off."""
    if subsystem not in _SUBSYSTEMS:
        return False
    try:
        from hermes_cli.config import load_config, cfg_get
        return _normalize_enabled(cfg_get(load_config(), subsystem, CONFIG_KEY, default=False))
    except Exception:
        return False


def unattended_memory_consolidation_enabled() -> bool:
    """True only when the operator explicitly opted in AND the ordinary approval gate is off
    (write_approval=true stays authoritative: staging then remains the only path)."""
    try:
        from hermes_cli.config import load_config, cfg_get
        opted_in = _normalize_enabled(cfg_get(load_config(), MEMORY, "allow_unattended_consolidation", default=False))
    except Exception:
        return False
    return opted_in and not write_approval_enabled(MEMORY)


def _normalize_enabled(value: Any) -> bool:
    """Coerce a config value to bool; unknown → False (gate off). The string branch
    covers hand-edited configs (YAML already parses bare on/off/yes/no)."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() in _TRUTHY_STRINGS


# --- Pending store (file-backed) ---

def _pending_path(subsystem: str, pending_id: str) -> Path:
    return get_hermes_home() / "pending" / subsystem / f"{pending_id}.json"


def _pending_files(subsystem: str) -> list:
    d = _pending_path(subsystem, "").parent
    return list(d.glob("*.json")) if d.exists() else []


def stage_write(subsystem: str, payload: Dict[str, Any], *, summary: str, origin: str) -> Dict[str, Any]:
    """Persist a pending write and return its record (``id`` + metadata). ``payload`` is the exact
    kwargs to replay the write on approval; ``origin`` is ``foreground`` or ``background_review``.
    Best-effort: on disk failure it logs and still returns a record — the write is lost, which is
    the safe failure for an approval gate (nothing silently committed).

    Memory dedup: a background-review destructive proposal that pins an entry some still-active
    pending record already pins (same target) supersedes that record first — the old file is kept
    on disk, marked ``superseded``/``superseded_by`` the new id — so one entry change never queues
    for review twice. Foreground staging and hygiene failures never block the proposal."""
    pid = uuid.uuid4().hex[:8]
    try:
        if subsystem == MEMORY and origin == "background_review":
            _supersede_dups_before_staging(pid, payload)
    except Exception as e:  # never block a proposal on queue hygiene
        logger.warning("Pending-dedup scan failed; staging anyway: %s", e, exc_info=True)
    record = {
        "id": pid, "subsystem": subsystem, "action": payload.get("action", ""),
        "summary": (summary or "").strip(), "origin": origin or "foreground",
        "created_at": time.time(), "payload": payload,
    }
    try:
        atomic_json_write(_pending_path(subsystem, pid), record)
    except Exception as e:  # pragma: no cover - disk failure path
        logger.error("Failed to stage pending %s write: %s", subsystem, e, exc_info=True)
    return record


def _pinned_entries_of(payload: Dict[str, Any]) -> set:
    """Exact pinned-entry strings of a memory payload's destructive ops (empty when unpinned
    or non-destructive)."""
    pinned = set()
    for op in _memory_destructive_ops(payload):
        entry = op.get("matched_entry")
        if isinstance(entry, str) and entry:
            pinned.add(entry)
    return pinned


def _memory_destructive_ops(payload: Any) -> List[Dict[str, Any]]:
    """``memory_tool.destructive_ops`` guarded: it can only fail on shapes the memory gate
    itself refuses (non-dict payloads); a helper must not import its failures."""
    if not isinstance(payload, dict):
        return []
    with suppress(Exception):
        from tools.memory_tool import destructive_ops
        return [op for op in (destructive_ops(payload) or []) if isinstance(op, dict)]
    return []


def _supersede_dups_before_staging(new_id: str, payload: Dict[str, Any]) -> None:
    """Mark still-active pending memory records that pin an entry the NEW payload also pins
    (same target) as superseded by ``new_id``. Old files stay on disk (audit evidence)."""
    target = payload.get("target", "memory")
    new_pins = _pinned_entries_of(payload)
    if not new_pins:
        return
    for old in list_pending(MEMORY):
        if old.get("status") not in (None, "", "ready"):
            continue  # already archived (stale/superseded/invalid/rejected)
        old_payload = old.get("payload") or {}
        if not isinstance(old_payload, dict) or old_payload.get("target", "memory") != target:
            continue
        if _pinned_entries_of(old_payload) & new_pins:
            update_pending_status(MEMORY, old["id"], STATUS_SUPERSEDED, superseded_by=new_id)


def update_pending_status(subsystem: str, pending_id: str, status: Optional[str], *,
                          superseded_by: Optional[str] = None) -> bool:
    """Rewrite one pending record's file atomically with ``status`` set (plus
    ``status_updated_at`` and ``superseded_by`` when given); every other field is preserved.
    The record is NEVER deleted — statused files stay as audit evidence. True when rewritten;
    False when the record is missing or unreadable."""
    record = get_pending(subsystem, pending_id)
    if not isinstance(record, dict):
        return False
    if status is None:  # canonical active form: field absent
        record.pop("status", None)
    else:
        record["status"] = status
    record["status_updated_at"] = time.time()
    if superseded_by is not None:
        record["superseded_by"] = superseded_by
    try:
        atomic_json_write(_pending_path(subsystem, pending_id), record)
    except Exception as e:
        logger.error("Failed to status pending %s/%s -> %s: %s", subsystem, pending_id, status, e)
        return False
    return True


def list_pending(subsystem: str) -> List[Dict[str, Any]]:
    """Return all pending records for ``subsystem``, oldest first."""
    records: List[Dict[str, Any]] = []
    for p in _pending_files(subsystem):
        try:
            record = json.loads(p.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object, got {type(record).__name__}")
            records.append(record)
        except Exception:
            logger.warning("Skipping unreadable pending record: %s", p)
    records.sort(key=lambda r: r.get("created_at", 0))
    return records


def get_pending(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pending record by id, or None."""
    path = _pending_path(subsystem, pending_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a pending record. Returns True if it existed."""
    try:
        path = _pending_path(subsystem, pending_id)
        if path.exists():
            path.unlink()
            return True
    except Exception as e:  # pragma: no cover
        logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, e)
    return False


def pending_count(subsystem: str) -> int:
    """Cheap count of pending records (for notification badges)."""
    d = _pending_path(subsystem, "").parent
    if not d.exists():
        return 0
    with suppress(Exception):
        return sum(1 for _ in d.glob("*.json"))
    return 0


# --- Memory pending-queue lifecycle (classification) ---

_MEMORY_ACTIONS = frozenset({"add", "replace", "remove", "batch"})


def classify_pending_memory(record: Dict[str, Any], store, records: Optional[List[Dict[str, Any]]] = None) -> str:
    """Lifecycle classification of one pending MEMORY record against a loaded ``store``:
    ``'rejected'`` (terminal), ``'invalid'`` (fail closed forever), ``'stale'``,
    ``'superseded'`` or ``'ready'``. ``records`` (all pending records, oldest first)
    enables the superseded check — newer ACTIVE/ready records on the same target pinning
    an identical ``matched_entry`` make this one redundant. Any unexpected store error
    also classifies 'invalid' (fail closed)."""
    if not isinstance(record, dict) or record.get("status") == STATUS_REJECTED:
        return STATUS_REJECTED
    payload = record.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    if payload.get("action") not in _MEMORY_ACTIONS:
        return STATUS_INVALID
    destructive = _memory_destructive_ops(payload)
    if any(not (op.get("matched_entry") if isinstance(op, dict) else None) for op in destructive):
        return STATUS_INVALID  # legacy pre-pinning record: fail closed forever
    target = payload.get("target", "memory")
    if target not in ("memory", "user"):
        return STATUS_INVALID
    try:
        entries = store._entries_for(target)
        if any(op.get("matched_entry") not in entries for op in destructive):
            return STATUS_STALE
        if _is_superseded(record, records, target):
            return STATUS_SUPERSEDED
    except Exception:
        logger.warning("Memory pending classification failed; failing closed", exc_info=True)
        return STATUS_INVALID
    return "ready"


def _is_superseded(record: Dict[str, Any], records: Optional[List[Dict[str, Any]]], target: str) -> bool:
    """True when a strictly NEWER record — still active or ready, same target — pins at least
    one identical ``matched_entry``. A stale/superseded newer record does NOT supersede: each
    record is judged by its own pins. ``records=None`` fetches the queue (best-effort)."""
    if records is None:
        with suppress(Exception):
            records = list_pending(MEMORY)
    if not records:
        return False
    pinned = {op.get("matched_entry") for op in _memory_destructive_ops(record.get("payload"))}
    if not pinned:
        return False
    for other in records:
        if other.get("id") == record.get("id"):
            continue
        if other.get("status") not in (None, "", "ready"):
            continue
        if other.get("created_at", 0) <= record.get("created_at", 0):
            continue
        payload = other.get("payload") or {}
        if not isinstance(payload, dict) or payload.get("target", "memory") != target:
            continue
        if pinned & {op.get("matched_entry") for op in _memory_destructive_ops(payload)}:
            return True
    return False


def reclassify_pending_memory(store, subsystem: str = MEMORY) -> Dict[str, int]:
    """Re-classify every non-terminal pending ``subsystem`` record and persist changed
    statuses. Returns counts: ``{'ready', 'stale', 'superseded', 'invalid', 'rejected',
    'changed'}`` where ``changed`` counts records whose status field actually changed.
    'invalid' persists too (audit evidence); records already 'invalid'/'rejected' stay as
    they are (no churn)."""
    counts = {STATUS_STALE: 0, STATUS_SUPERSEDED: 0, STATUS_INVALID: 0, STATUS_REJECTED: 0,
              "ready": 0, "changed": 0}
    records = [r for r in list_pending(subsystem) if isinstance(r, dict)]
    # NEWEST first: when an older record runs its superseded check, every newer record
    # already carries its freshly-persisted status on this shared snapshot — a newer
    # record that just classified 'stale' must not supersede it.
    for record in reversed(records):
        current = record.get("status") or ""
        if current in (STATUS_REJECTED, STATUS_INVALID):
            counts[current] += 1
            continue
        verdict = classify_pending_memory(record, store, records)
        counts[verdict] += 1
        # '' (absent) and 'ready' are the same lifecycle state: a ready record is not
        # rewritten (no churn), while a stale→ready recovery persists the new status.
        current_norm = "" if current in ("", "ready") else current
        verdict_norm = "" if verdict == "ready" else verdict
        if verdict_norm != current_norm:
            update_pending_status(subsystem, record["id"], verdict)
            record["status"] = verdict  # shared snapshot: later (older) records see it
            counts["changed"] += 1
    return counts


def reject_pending_bulk(subsystem: str, status_value: str) -> int:
    """Status-mark every pending record whose status == ``status_value`` as 'rejected'
    (never delete — the files remain on disk as audit evidence). Returns how many were marked."""
    if status_value == STATUS_REJECTED:
        return 0
    return sum(1 for record in list_pending(subsystem)
               if record.get("status") == status_value
               and update_pending_status(subsystem, record["id"], STATUS_REJECTED))


# --- Write origin ---

def current_origin() -> str:
    """``foreground`` or ``background_review`` — reuses the skill-provenance ContextVar
    the background review fork sets; foreground turns leave it at the default."""
    with suppress(Exception):
        from tools.skill_provenance import get_current_write_origin
        return get_current_write_origin()
    return "foreground"


# --- Gate decision ---

@dataclass(slots=True, kw_only=True)
class GateDecision:
    """Result of evaluating the write gate; exactly one flag is True. ``allow``: do the real write;
    ``blocked``: user denied the inline prompt (``message`` says why); ``stage``: caller must
    ``stage_write`` the payload (``message`` is the user-facing "staged for approval" note)."""

    allow: bool = False
    blocked: bool = False
    stage: bool = False
    message: str = ""


def _staged(subsystem: str) -> GateDecision:
    where = "/skills pending" if subsystem == SKILLS else "/memory pending"
    return GateDecision(stage=True, message=(f"Staged for approval ({subsystem}.write_approval is on). "
                                             f"Not yet saved — review with {where}."))


def evaluate_gate(subsystem: str, *, inline_summary: str = "", inline_detail: str = "") -> GateDecision:
    """Decide what to do with a pending write: gate off → allow; gate on + skills (any origin) or
    background → stage; gate on + memory + foreground → inline prompt when an interactive channel
    exists, else stage. The gate only ever delays a write, never silently refuses it; ``blocked``
    is produced only when the user actively denies the inline prompt."""
    if not write_approval_enabled(subsystem):
        return GateDecision(allow=True)
    # Skills are too big to review inline; a background write runs in a daemon thread with no user.
    if subsystem == SKILLS or current_origin() == "background_review":
        return _staged(subsystem)
    granted = _prompt_inline_memory_approval(inline_summary, inline_detail)
    if granted is None:
        return _staged(MEMORY)
    if granted:
        return GateDecision(allow=True)
    return GateDecision(blocked=True, message="Memory write denied by user. The change was not saved.")


def _prompt_inline_memory_approval(summary: str, detail: str) -> Optional[bool]:
    """Prompt inline for a memory write: True approved, False denied, None → stage. Uses the per-thread
    CLI approval callback (``tools.terminal_tool.set_approval_callback``) directly, not
    ``prompt_dangerous_approval``: that wrapper falls back to ``input()`` (deadlock-prone under
    prompt_toolkit; silent deny in gateway sessions) and turns callback errors into a deny, whereas
    here a missing channel or failed prompt must stage instead.

    See #15216.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
    except Exception:
        return None
    callback = _get_approval_callback()
    if callback is None:
        return None
    header = summary.strip() or "Save to memory?"
    try:
        from tools.approval_prompt import callback_accepts
        extra = {"title": "Save to memory?"} if callback_accepts(callback, "title") else {}
        choice = callback(detail.strip() or header, f"Save to memory: {header}", allow_permanent=False, **extra)
    except Exception as e:
        logger.error("Inline memory approval prompt failed: %s", e)
        return None
    # unknown outcome → stage rather than drop
    return {"once": True, "session": True, "deny": False}.get(choice)


# --- Skill-specific helpers (gist + diff for the review affordances) ---

_GIST_TEMPLATES = {"write_file": "write {file_path} in '{name}'", "remove_file": "remove {file_path} from '{name}'",
                   "delete": "delete skill '{name}'"}


def skill_gist(action: str, name: str, *, content: str = "", file_path: str = "",
               old_string: str = "", new_string: str = "") -> str:
    """One-line heuristic gist (no model call) for a pending skill write: create/edit use
    the frontmatter ``description:``; patch/write_file describe the size of the change."""
    if action in {"create", "edit"} and content:
        desc = _frontmatter_description(content)
        size = f"{len(content) // 1024 + 1} KB" if len(content) >= 1024 else f"{len(content)} chars"
        return f"{'create' if action == 'create' else 'rewrite'} '{name}'{f' — {desc}' if desc else ''} ({size})"
    if action == "patch":
        removed = old_string.count("\n") + 1 if old_string else 0
        added = new_string.count("\n") + 1 if new_string else 0
        return f"patch '{name}' {file_path or 'SKILL.md'} (+{added}/-{removed} lines)"
    return _GIST_TEMPLATES.get(action, "{action} '{name}'").format(action=action, name=name, file_path=file_path)


def _frontmatter_description(content: str) -> str:
    """Extract the ``description:`` value from SKILL.md YAML frontmatter (≤140 chars)."""
    m = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip().strip("'\"")[:140] if m else ""


def _find_skill_path(name: str) -> Optional[Path]:
    """Directory of an installed skill, or None if unknown / lookup unavailable."""
    try:
        from tools.skill_manager_tool import _find_skill
    except Exception:
        return None
    # Only the import is guarded (as on main); a lookup failure propagates.
    found = _find_skill(name)
    return found["path"] if found else None


def skill_pending_diff(record: Dict[str, Any]) -> str:
    """Full content (create) or unified diff vs. the on-disk skill (edit/patch/write_file),
    rendered by /skills diff <id> on surfaces that can show it."""
    payload = record.get("payload", {})
    action = payload.get("action", "")
    name = payload.get("name", "")
    if action == "create":
        return payload.get("content") or ""
    if action not in {"edit", "patch", "write_file"}:
        return {"remove_file": f"remove file: {payload.get('file_path')} from skill '{name}'",
                "delete": f"delete skill '{name}'"}.get(action, f"({action} on '{name}')")

    # patch/write_file target a file inside the skill; edit always targets SKILL.md.
    target_label, current = "SKILL.md", ""
    skill_dir = _find_skill_path(name)
    if skill_dir:
        if action != "edit":
            target_label = payload.get("file_path") or "SKILL.md"
        with suppress(Exception):
            p = skill_dir / target_label
            current = p.read_text(encoding="utf-8-sig") if p.exists() else ""

    if action == "patch":
        old_s, new_s = payload.get("old_string") or "", payload.get("new_string") or ""
        new = current.replace(old_s, new_s) if current else f"(patch {old_s!r} → {new_s!r})"
    else:
        new = payload.get("content" if action == "edit" else "file_content") or ""
    diff = difflib.unified_diff(current.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile=f"a/{target_label}", tofile=f"b/{target_label}")
    return "".join(diff) or "(no textual change)"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def is_background() -> bool:
    return current_origin() == "background_review"
# ---- END PLUGIN-COMPAT ----
