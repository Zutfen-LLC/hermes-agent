from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# Session-scoped dedupe cache for the current process.
_ROUTING_CACHE: dict[str, set[str]] = {}
_ROUTING_CACHE_LOCK = threading.Lock()


InvocationMode = Literal["manual", "pre_compress", "session_end"]


def canonicalize_text(text: str) -> str:
    """Return a stable text form for dedupe comparisons."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.lower()


def canonical_payload(destination: str, entry_kind: str, canonical_text: str, source_id: str = "") -> dict[str, str]:
    """Build the stable payload used for fingerprinting."""
    return {
        "destination": destination,
        "entry_kind": entry_kind,
        "canonical_text": canonical_text,
        "source_id": source_id,
    }


def fingerprint_payload(payload: dict[str, str]) -> str:
    """Return a stable SHA-256 fingerprint for a canonical payload."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_candidate(
    destination: str,
    entry_kind: str,
    text: str,
    source_id: str = "",
) -> tuple[str, str]:
    canonical_text = canonicalize_text(text)
    payload = canonical_payload(destination, entry_kind, canonical_text, source_id)
    return canonical_text, fingerprint_payload(payload)


def has_seen(session_id: str, fingerprint: str) -> bool:
    with _ROUTING_CACHE_LOCK:
        return fingerprint in _ROUTING_CACHE.get(session_id or "__global__", set())


def mark_seen(session_id: str, fingerprint: str) -> None:
    with _ROUTING_CACHE_LOCK:
        _ROUTING_CACHE.setdefault(session_id or "__global__", set()).add(fingerprint)


def clear_seen(session_id: str) -> None:
    with _ROUTING_CACHE_LOCK:
        _ROUTING_CACHE.pop(session_id or "__global__", None)


def _collect_text(messages: list[dict[str, Any]], candidate_window: int | None = None) -> str:
    if not messages:
        return ""

    window = messages[-candidate_window:] if candidate_window else messages
    try:
        from plugins.memory.mempalace import _extract_pre_compress_content
    except Exception:
        _extract_pre_compress_content = None  # type: ignore[assignment]

    if _extract_pre_compress_content is not None:
        try:
            return _extract_pre_compress_content(window, max_chars=4000)
        except Exception as exc:
            logger.debug("MemPalace extraction unavailable: %s", exc)

    parts: list[str] = []
    for msg in window:
        role = str(msg.get("role", ""))
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if role == "system":
            continue
        if role == "user" and "[CONTEXT COMPACTION" in content:
            continue
        label = "USER" if role == "user" else "ASSISTANT" if role == "assistant" else role.upper()
        parts.append(f"[{label}] {content[:800]}")
    combined = "\n".join(parts)
    return combined[:4000]


def _classify_native_target(text: str) -> str:
    preference_markers = (
        r"\bi prefer\b",
        r"\bi like\b",
        r"\bi want\b",
        r"\bplease remember\b",
        r"\bremember that\b",
        r"\bmy preference\b",
        r"\bdo not\b",
        r"\bdon't\b",
        r"\balways\b",
        r"\bnever\b",
    )
    if any(re.search(pat, text, re.IGNORECASE) for pat in preference_markers):
        return "user"
    return "memory"


def _entry_kind_for_target(target: str, text: str) -> str:
    if target == "user":
        return "preference"
    if re.search(r"\bdecision\b|\bdecided\b|\bchose\b", text, re.IGNORECASE):
        return "decision"
    if re.search(r"\bopen question\b|\bquestion\b|\btodo\b", text, re.IGNORECASE):
        return "open_question"
    return "operational_fact"


def _resolve_mempalace_config(config: dict[str, Any] | None = None) -> tuple[str, list[tuple[list[re.Pattern], str, str]]]:
    config = config or {}
    mp_cfg = config.get("mempalace", {}) if isinstance(config.get("mempalace", {}), dict) else {}
    palace_path = os.environ.get("MEMPALACE_PALACE_PATH", "").strip()
    if not palace_path and isinstance(mp_cfg, dict):
        configured_path = str(mp_cfg.get("palace_path", "")).strip()
        if configured_path:
            palace_path = os.path.expanduser(configured_path)
    if not palace_path:
        palace_path = str(Path.home() / ".mempalace" / "palace")

    routing_rules: list[tuple[list[re.Pattern], str, str]] = []
    try:
        from plugins.memory.mempalace import _compile_user_rules
        if isinstance(mp_cfg, dict):
            routing_rules = _compile_user_rules(mp_cfg.get("routing_rules", []) or [])
    except Exception as exc:
        logger.debug("MemPalace routing rules unavailable: %s", exc)
    return palace_path, routing_rules


def _mempalace_route(content: str, invocation_mode: InvocationMode, config: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        from plugins.memory.mempalace import _add_drawer, _route_content
    except Exception as exc:
        return {"success": False, "error": f"MemPalace unavailable: {exc}"}

    palace_path, routing_rules = _resolve_mempalace_config(config)
    wing, room = _route_content(content, routing_rules)
    return _add_drawer(
        palace_path,
        wing=wing,
        room=room,
        content=content,
        added_by=f"hermes-router:{invocation_mode}",
    )


def _resolve_cca_lite_bridge(config: dict[str, Any] | None = None) -> tuple[str, Path | None]:
    config = config or {}
    cca_cfg = config.get("cca_lite", {}) if isinstance(config.get("cca_lite", {}), dict) else {}

    command = os.environ.get("HERMES_CCA_LITE_COMMAND", "").strip()
    if not command and isinstance(cca_cfg, dict):
        command = str(cca_cfg.get("command", "")).strip()

    inbox_path: Path | None = None
    raw_path = os.environ.get("HERMES_CCA_LITE_INBOX_PATH", "").strip()
    if not raw_path and isinstance(cca_cfg, dict):
        raw_path = str(cca_cfg.get("inbox_path", "")).strip()
    if not raw_path and isinstance(cca_cfg, dict):
        repo_root = str(cca_cfg.get("repo_root", "")).strip()
        if repo_root:
            raw_path = str(Path(repo_root).expanduser() / "docs/cca-lite/hermes-memory-inbox.json")
    if not raw_path:
        fallback_repo = Path(__file__).resolve().parents[2] / "cca-repo" / "docs/cca-lite/hermes-memory-inbox.json"
        if fallback_repo.exists() or fallback_repo.parent.exists():
            raw_path = str(fallback_repo)
    if raw_path:
        inbox_path = Path(os.path.expanduser(raw_path))
    return command, inbox_path


def _append_json_item(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Any = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = []

    appended = False
    if isinstance(payload, list):
        seen_ids = {str(entry.get("id", "")) for entry in payload if isinstance(entry, dict)}
        if item["id"] not in seen_ids:
            payload.append(item)
            appended = True
    elif isinstance(payload, dict):
        entries = payload.setdefault("entries", [])
        if isinstance(entries, list):
            seen_ids = {str(entry.get("id", "")) for entry in entries if isinstance(entry, dict)}
            if item["id"] not in seen_ids:
                entries.append(item)
                appended = True
    else:
        payload = [item]
        appended = True

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return {"success": True, "appended": appended, "path": str(path)}


def _route_to_cca_lite(
    *,
    session_id: str,
    invocation_mode: InvocationMode,
    source_event: str,
    entry_kind: str,
    text: str,
    canonical_text: str,
    fingerprint: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command, inbox_path = _resolve_cca_lite_bridge(config)
    payload = {
        "id": fingerprint,
        "kind": entry_kind,
        "text": text,
        "canonical_text": canonical_text,
        "captured_at": datetime.now().isoformat(),
        "source": source_event or invocation_mode,
        "session_id": session_id,
        "content_hash": f"sha256:{fingerprint}",
        "invocation_mode": invocation_mode,
    }

    if command:
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                shell=True,
                check=False,
                capture_output=True,
                timeout=30,
            )
            if completed.returncode == 0:
                return {"success": True, "mode": "command", "command": command}
            return {
                "success": False,
                "mode": "command",
                "command": command,
                "error": completed.stderr.decode("utf-8", errors="replace") or completed.stdout.decode("utf-8", errors="replace") or "cca-lite command failed",
            }
        except Exception as exc:
            return {"success": False, "mode": "command", "command": command, "error": str(exc)}

    if inbox_path is None:
        return {"success": True, "mode": "disabled", "reason": "cca-lite bridge not configured"}

    return _append_json_item(inbox_path, payload)


def route_memory_candidates(
    *,
    invocation_mode: InvocationMode,
    session_id: str,
    messages: list[dict[str, Any]],
    memory_store: Any = None,
    source_event: str = "",
    candidate_window: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a session slice into native memory, MemPalace, and cca-lite."""
    session_key = session_id or "__global__"
    config = config or {}
    content = _collect_text(messages, candidate_window=candidate_window)
    canonical_text = canonicalize_text(content)
    if not canonical_text:
        return {
            "ok": True,
            "mode": invocation_mode,
            "session_id": session_id,
            "routed": [],
            "suppressed": [],
            "failed": [],
            "summary": "No durable memory candidates found.",
        }

    native_target = _classify_native_target(canonical_text)
    native_kind = _entry_kind_for_target(native_target, canonical_text)
    mempalace_kind = _entry_kind_for_target("memory", canonical_text)
    cca_kind = native_kind

    routed: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    def _route_destination(destination: str, entry_kind: str, writer) -> None:
        _, fingerprint = fingerprint_candidate(destination, entry_kind, canonical_text, session_key)
        if has_seen(session_key, fingerprint):
            suppressed.append({
                "destination": destination,
                "entry_kind": entry_kind,
                "fingerprint": fingerprint,
                "reason": "session_duplicate",
            })
            return

        result = writer(fingerprint)
        if result.get("success"):
            status = "no_op" if result.get("mode") == "disabled" else ("written" if result.get("appended", True) else "no_op")
            routed.append({
                "destination": destination,
                "entry_kind": entry_kind,
                "fingerprint": fingerprint,
                "status": status,
            })
            mark_seen(session_key, fingerprint)
            return

        failed.append({
            "destination": destination,
            "error": result.get("error", "unknown error"),
        })

    if memory_store is not None:
        def _write_native(_: str) -> dict[str, Any]:
            try:
                return memory_store.add(native_target, content)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        _route_destination("native_memory", native_kind, _write_native)

    def _write_mempalace(_: str) -> dict[str, Any]:
        try:
            return _mempalace_route(content, invocation_mode, config=config)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    _route_destination("mempalace", mempalace_kind, _write_mempalace)

    def _write_cca_lite(fingerprint: str) -> dict[str, Any]:
        try:
            return _route_to_cca_lite(
                session_id=session_id,
                invocation_mode=invocation_mode,
                source_event=source_event,
                entry_kind=cca_kind,
                text=content,
                canonical_text=canonical_text,
                fingerprint=fingerprint,
                config=config,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    _route_destination("cca_lite", cca_kind, _write_cca_lite)

    written = len([item for item in routed if item.get("status") == "written"])
    skipped = len([item for item in routed if item.get("status") != "written"])
    summary_bits = [f"{written} written"]
    if skipped:
        summary_bits.append(f"{skipped} skipped")
    if suppressed:
        summary_bits.append(f"{len(suppressed)} suppressed")
    if failed:
        summary_bits.append(f"{len(failed)} failed")
    return {
        "ok": not failed,
        "mode": invocation_mode,
        "session_id": session_id,
        "routed": routed,
        "suppressed": suppressed,
        "failed": failed,
        "summary": "; ".join(summary_bits),
        "content": content,
    }
