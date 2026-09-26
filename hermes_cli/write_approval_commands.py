#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import List, Optional, Tuple

from tools import write_approval as wa

# Memory reject flags -> the archived status they bulk-reject.
_BULK_REJECT_FLAGS = {"--stale": wa.STATUS_STALE, "--superseded": wa.STATUS_SUPERSEDED,
                      "--invalid": wa.STATUS_INVALID}

_READY_STATUSES = ("", "ready")


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    return f"{subsystem}.write_approval = {'on' if on else 'off'}"


def _fmt_unattended_line() -> str:
    on = wa.unattended_memory_consolidation_enabled()
    return f"unattended consolidation: {'on' if on else 'off'}"


def _fmt_pending_list(subsystem: str, memory_store=None) -> str:
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."

    # Memory: classify first (statuses persisted) so the queue reflects the live store;
    # a hygiene failure must never break the listing. Re-list afterwards — classification
    # rewrites the record files, so the pre-classification snapshot is stale.
    counts = None
    if subsystem == wa.MEMORY and memory_store is not None:
        with suppress(Exception):
            counts = wa.reclassify_pending_memory(memory_store, subsystem)
        records = wa.list_pending(subsystem)

    ready = [r for r in records if (r.get("status") or "") in _READY_STATUSES]
    # Rejected records are invisible everywhere (audit-only, on disk); the footer surfaces
    # only the still-reviewable archived categories.
    archived = [r for r in records if r.get("status") in
                (wa.STATUS_STALE, wa.STATUS_SUPERSEDED, wa.STATUS_INVALID)]

    header = f"Pending {subsystem} writes"
    if counts is not None:
        nonzero = [(label, counts[label]) for label in ("ready", "stale", "superseded", "invalid")
                   if counts.get(label)]
        if nonzero:
            header += f" ({', '.join(f'{n} {label}' for label, n in nonzero)})"
    if not ready and not archived:
        return f"No pending {subsystem} writes."
    lines = [f"{header}:"]
    for r in ready:
        origin = r.get("origin", "foreground")
        tag = " [auto]" if origin == "background_review" else ""
        lines.append(f"  {r['id']}{tag}  {r.get('summary', '')}")
        if subsystem == wa.MEMORY:
            lines.extend(f"      {line}" for line in _matched_entries(r.get("payload") or {}))
    lines.append("")
    lines.append(f"Apply: /{subsystem} approve <id>   Reject: /{subsystem} reject <id>")
    if subsystem == wa.SKILLS:
        lines.append("Review full diff: /skills diff <id>")
    elif subsystem == wa.MEMORY:
        lines.append("Review full diff: /memory diff <id>")
    if archived:
        by = {}
        for r in archived:
            status = r.get("status") or "ready"
            by[status] = by.get(status, 0) + 1
        parts = ", ".join(f"{n} {status}" for status, n in sorted(by.items()))
        flags = " / ".join(flag for flag, status in sorted(_BULK_REJECT_FLAGS.items())
                           if status in by)
        lines.append(f"Archived: {parts} — not listed for approval; "
                     f"/{subsystem} reject {flags} to clear")
    return "\n".join(lines)


def handle_pending_subcommand(
    subsystem: str, args: List[str], *, memory_store=None, set_mode_fn=None) -> Optional[str]:
    """Dispatch a /memory or /skills write-approval subcommand.

    ``memory_store`` applies approved memory writes (CLI passes its live store; gateway a freshly
    loaded one) AND drives pending-record lifecycle classification; ``set_mode_fn`` persists the
    write_approval boolean. Returns text for the user, or None when the args are not a
    write-approval subcommand so the caller falls through to its other handling (e.g. /skills
    search)."""
    if not args:
        state = _fmt_state(subsystem)
        if subsystem == wa.MEMORY:
            state += f"\n{_fmt_unattended_line()}"
        return f"{state}\n\n" + _fmt_pending_list(subsystem, memory_store)
    sub, rest = args[0].lower(), args[1:]
    if sub == "pending":
        return _fmt_pending_list(subsystem, memory_store)
    if sub in {"approve", "apply"}:
        return _approve(subsystem, rest, memory_store)
    if sub in {"reject", "deny", "drop"}:
        return _reject(subsystem, rest)
    if sub == "diff" and subsystem == wa.MEMORY:
        return _memory_diff(rest, memory_store)
    if sub == "diff" and subsystem == wa.SKILLS:
        return _diff(rest)
    if sub == "undo" and subsystem == wa.MEMORY:
        return _memory_undo(rest, memory_store)
    if sub in {"approval", "mode"}:  # 'mode' kept as a back-compat alias
        return _set_approval(subsystem, rest, set_mode_fn)
    return None  # not ours — caller handles


def _usage(subsystem: str) -> str:
    usage = f"Usage: /{subsystem} approve|reject <id>  (or 'all')"
    if subsystem == wa.MEMORY:
        usage += f"  — bulk-reject archived: /memory reject {' /'.join(sorted(_BULK_REJECT_FLAGS))}"
    return usage


def _ready_memory_records(subsystem: str, memory_store) -> Tuple[Optional[str], List[dict]]:
    """Reclassify the memory queue and split it: ``(skip_summary, ready_records)``.
    ``skip_summary`` (None when nothing was skipped, or when classification is unavailable —
    then EVERY record is returned ready, preserving the pre-lifecycle behavior) reads like
    'skipped 4 non-ready (2 stale, 1 superseded, 1 invalid) — they remain archived'."""
    if memory_store is None:
        # No store to classify against: fall back to the record statuses alone (rejected
        # records stay invisible; unaudited ones remain approvable, as before lifecycles).
        return None, [r for r in wa.list_pending(subsystem) if r.get("status") != wa.STATUS_REJECTED]
    with suppress(Exception):
        wa.reclassify_pending_memory(memory_store, subsystem)
    records = wa.list_pending(subsystem)
    ready = [r for r in records if (r.get("status") or "") in _READY_STATUSES]
    skipped = [r for r in records if r.get("status") in
               (wa.STATUS_STALE, wa.STATUS_SUPERSEDED, wa.STATUS_INVALID)]
    if not skipped:
        return None, ready
    by = {}
    for r in skipped:
        status = r.get("status") or "ready"
        by[status] = by.get(status, 0) + 1
    parts = ", ".join(f"{n} {status}" for status, n in sorted(by.items()))
    return f"skipped {len(skipped)} non-ready ({parts}) — they remain archived", ready


def _approve(subsystem: str, rest: List[str], memory_store) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
    skip_summary = None
    if target.lower() == "all":
        if subsystem == wa.MEMORY:
            skip_summary, targets = _ready_memory_records(subsystem, memory_store)
            if not targets:
                out = "No pending memory writes ready to approve."
                if skip_summary:
                    out += f"\n{skip_summary}."
                return out
        else:
            targets = list(records)
    else:
        rec = wa.get_pending(subsystem, target)
        if not rec:
            return f"No pending {subsystem} write with id '{target}'."
        if rec.get("status") == wa.STATUS_REJECTED:
            # Rejected records are terminal evidence: applying (then unlinking) one
            # here would destroy the audit trail the rejection decision rests on.
            return (f"Record '{target}' was rejected and is kept as evidence only; "
                    f"recreate the change instead if you still want it.")
        targets = [rec]

    applied, failed, overwritten, removed = 0, [], [], []
    for rec in targets:
        ok, msg, result = _apply_one(subsystem, rec, memory_store)
        if ok:
            wa.discard_pending(subsystem, rec["id"])
            applied += 1
            overwritten.extend(f"  {rec['id']}: {text}" for text in _changed_entries(result, "replaced"))
            removed.extend(f"  {rec['id']}: {text}" for text in _changed_entries(result, "removed"))
        else:
            failed.append(f"{rec['id']}: {msg}")
            if subsystem == wa.MEMORY and target.lower() != "all" and memory_store is not None:
                # Operator override attempted a record that failed: persist its classification
                # so a permanently-failing record stops showing as ready.
                with suppress(Exception):
                    wa.update_pending_status(
                        subsystem, rec["id"], wa.classify_pending_memory(rec, memory_store))

    out = [f"Approved {applied} {subsystem} write(s)."]
    if overwritten:
        # A memory 'replace' overwrites the WHOLE matched entry (#117952); the approver
        # is the last person who can notice a clause went missing, so show what was lost.
        out.append("Overwrote entire entry (re-add anything you still need):")
        out.extend(overwritten)
    if removed:
        out.append("Removed entry (re-add anything you still need):")
        out.extend(removed)
    if failed:
        out.append("Failed:")
        out.extend(f"  {f}" for f in failed)
    if skip_summary:
        out.append(skip_summary)
    return "\n".join(out)


def _changed_entries(result: dict, kind: str) -> List[str]:
    """Full text of every entry a memory replace overwrote (``kind="replaced"``) or remove
    deleted (``"removed"``), single-op or batch shape."""
    single = result.get(f"{kind}_entry")
    batch = result.get(f"{kind}_entries") or {}
    return ([single] if single else []) + [batch[k] for k in sorted(batch, key=int)]


def _matched_entries(payload) -> List[str]:
    """The full entry each staged memory replace/remove is pinned to: the summary shows only
    the old_text search string, and approval applies to this entry, not to that search."""
    from tools.memory_tool import destructive_ops
    return [f"{op['action']}s entry: {op['matched_entry']}" if op.get("matched_entry")
            else f"{op['action']}: unpinned legacy target \u2014 reject and recreate before approving"
            for op in destructive_ops(payload)]


def _apply_one(subsystem: str, rec, memory_store):
    """``(ok, error, result)`` — *result* is the applier's full payload (empty on exceptions)."""
    payload = rec.get("payload", {})
    try:
        if subsystem == wa.MEMORY:
            if memory_store is None:
                return False, "memory store unavailable", {}
            from tools.memory_tool import apply_memory_pending
            result = apply_memory_pending(payload, memory_store)
        else:
            from tools.skill_manager_tool import apply_skill_pending
            result = json.loads(apply_skill_pending(payload))
        return bool(result.get("success")), result.get("error", ""), result
    except Exception as e:
        return False, str(e), {}


def _reject(subsystem: str, rest: List[str]) -> str:
    if not rest:
        return _usage(subsystem)
    if subsystem != wa.MEMORY:
        # Skills keep the legacy delete semantics (no lifecycle statuses).
        target = rest[0]
        if target.lower() == "all":
            n = sum(1 for rec in wa.list_pending(subsystem) if wa.discard_pending(subsystem, rec["id"]))
            return f"Rejected {n} pending {subsystem} write(s)."
        if wa.discard_pending(subsystem, target):
            return f"Rejected pending {subsystem} write '{target}'."
        return f"No pending {subsystem} write with id '{target}'."

    flags = [a for a in rest if a.lower() in _BULK_REJECT_FLAGS]
    wants_all = any(a.lower() == "all" for a in rest)
    ids = [a for a in rest if a.lower() not in _BULK_REJECT_FLAGS and a.lower() != "all"]

    n, missing = 0, []
    for flag in flags:
        n += wa.reject_pending_bulk(subsystem, _BULK_REJECT_FLAGS[flag.lower()])
    if wants_all:
        for rec in wa.list_pending(subsystem):
            if rec.get("status") != wa.STATUS_REJECTED and wa.update_pending_status(
                    subsystem, rec["id"], wa.STATUS_REJECTED):
                n += 1
    for pid in ids:
        if wa.get_pending(subsystem, pid) is None:
            missing.append(pid)
        elif wa.update_pending_status(subsystem, pid, wa.STATUS_REJECTED):
            n += 1
    if not flags and not wants_all and not ids:
        return f"No pending {subsystem} write with id '{rest[0]}'."
    if not flags and not wants_all and len(ids) == 1:
        if missing:
            return f"No pending {subsystem} write with id '{ids[0]}'."
        # Same wording as the legacy single reject; the record is now status-marked
        # 'rejected' (kept on disk for audit) instead of deleted.
        return f"Rejected pending {subsystem} write '{ids[0]}'."
    out = f"Rejected {n} pending {subsystem} write(s)."
    if missing:
        out += "\nNot found: " + ", ".join(missing)
    return out


def _diff(rest: List[str]) -> str:
    if not rest:
        return "Usage: /skills diff <id>"
    rec = wa.get_pending(wa.SKILLS, rest[0])
    if not rec:
        return f"No pending skill write with id '{rest[0]}'."
    return f"# Pending skill write {rec['id']}: {rec.get('summary', '')}\n\n" + wa.skill_pending_diff(rec)


# --- /memory diff: bounded BEFORE/AFTER rendering ---

_DIFF_HEAD, _DIFF_TAIL = 300, 200


def _bounded_field(text: str, head: int = _DIFF_HEAD, tail: int = _DIFF_TAIL) -> str:
    """head + '…' + tail of a field once it exceeds head+tail chars (chat-safe but
    recognizable at both ends)."""
    text = text or ""
    return text if len(text) <= head + tail else f"{text[:head]}…{text[-tail:]}"


def _indented(text: str) -> str:
    return "\n".join(f"    {line}" for line in (text or "").splitlines() or [""])


def memory_pending_diff(record: dict) -> str:
    """Bounded per-op BEFORE/AFTER text for a pending memory record: the FULL pinned
    ``matched_entry`` is BEFORE (not the old_text search string), the proposed
    ``content``/``new_text`` (whole new entry, #117952) is AFTER; batch ops are enumerated
    'Operation N (action)'."""
    payload = record.get("payload") or {}
    payload = payload if isinstance(payload, dict) else {}
    ops = (payload.get("operations") or []) if payload.get("action") == "batch" else [payload]
    chunks = [f"target: {payload.get('target', 'memory')}"]
    for i, op in enumerate(ops, start=1):
        op = op if isinstance(op, dict) else {}
        action = op.get("action", "?")
        lines = [f"Operation {i} ({action})" if payload.get("action") == "batch" else action]
        before = op.get("matched_entry") or op.get("old_text") or ""
        if before:
            lines.append("  BEFORE (whole pinned entry):")
            lines.append(_indented(_bounded_field(before)))
        after = op.get("content") or op.get("new_text") or ""
        if action in {"replace", "add"} and after:
            lines.append("  AFTER (whole new entry):")
            lines.append(_indented(_bounded_field(after)))
        elif action == "remove":
            lines.append("  (entry removed)")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def _memory_diff(rest: List[str], memory_store=None) -> str:
    if not rest:
        return "Usage: /memory diff <id>"
    rec = wa.get_pending(wa.MEMORY, rest[0])
    if not rec:
        return f"No pending memory write with id '{rest[0]}'."
    return f"# Pending memory write {rec['id']}: {rec.get('summary', '')}\n\n" + memory_pending_diff(rec)


def _memory_undo(rest: List[str], memory_store=None) -> str:
    """/memory undo <audit_id>: restore the recorded before-state of one autonomous
    consolidation as a single atomic batch. ``undo list`` shows recent audit ids."""
    if not rest:
        return "Usage: /memory undo <audit_id>  (ids: /memory undo list)"
    if rest[0].lower() == "list":
        return _memory_undo_list()
    audit_id = rest[0]
    if memory_store is None:
        return "memory store unavailable"
    try:
        from tools.memory_consolidation import restore
        result = restore(audit_id, memory_store) or {}
    except Exception as e:
        return f"Undo failed: {e}"
    detail = result.get("message") or result.get("error") or "no result"
    return f"Undo {audit_id}: {detail}"


def _memory_undo_list(limit: int = 10) -> str:
    """Up to *limit* most-recent 'applied' ledger records, newest first."""
    from tools.memory_consolidation import list_records
    try:
        applied = [r for r in list_records() if r.get("event") == "applied"]
    except Exception:
        applied = []
    if not applied:
        return "No autonomous consolidations recorded yet."
    lines = ["Recent autonomous consolidations (most recent first):"]
    for r in reversed(applied[-limit:]):
        counts = r.get("counts") or {}
        lines.append(f"  {r.get('id', '?')}  {r.get('ts', '?')}  {r.get('target', 'memory')}  "
                     f"replaced {counts.get('replaced', 0)}, removed {counts.get('removed', 0)}, "
                     f"added {counts.get('added', 0)}")
    lines.append("Restore one: /memory undo <audit_id>")
    return "\n".join(lines)


_APPROVAL_VALUES = {
    **dict.fromkeys(("on", "true", "yes", "1", "enable", "enabled"), True),
    **dict.fromkeys(("off", "false", "no", "0", "disable", "disabled"), False)}


def _set_approval(subsystem: str, rest: List[str], set_mode_fn) -> str:
    """Turn the approval gate on/off for a subsystem."""
    if not rest:
        return (f"{_fmt_state(subsystem)}\n"
                f"Set with: /{subsystem} approval <on|off>")
    arg = rest[0].strip().lower()
    enabled = _APPROVAL_VALUES.get(arg)
    if enabled is None:
        return f"Invalid value '{arg}'. Use: on or off."
    if set_mode_fn is None:
        val = "true" if enabled else "false"
        return (f"To change the {subsystem} approval gate, run:\n"
                f"  hermes config set {subsystem}.write_approval {val}")
    try:
        set_mode_fn(enabled)
    except Exception as e:
        return f"Failed to set {subsystem}.write_approval: {e}"
    return f"{subsystem}.write_approval set to '{'on' if enabled else 'off'}'."
