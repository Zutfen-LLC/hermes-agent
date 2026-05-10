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
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

# Session-scoped dedupe cache for the current process.
_ROUTING_CACHE: dict[str, set[str]] = {}
_ROUTING_CACHE_LOCK = threading.Lock()


InvocationMode = Literal["manual", "pre_compress", "session_end"]
MemoryDestination = Literal["native_user", "native_memory", "mempalace", "cca_lite"]
MEMORY_DESTINATIONS: set[str] = {"native_user", "native_memory", "mempalace", "cca_lite"}


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


def _entry_kind_for_target(target: str, text: str) -> str:
    if target == "user":
        return "preference"
    if re.search(r"\bdecision\b|\bdecided\b|\bchose\b", text, re.IGNORECASE):
        return "decision"
    if re.search(r"\bopen question\b|\bquestion\b|\btodo\b", text, re.IGNORECASE):
        return "open_question"
    return "operational_fact"


def classify_memory_fact(content: str, requested_target: str = "memory") -> dict[str, Any]:
    """Choose the canonical sink for a single durable fact before writing it."""
    canonical_text = canonicalize_text(content)
    requested_target = requested_target if requested_target in ("memory", "user") else "memory"

    user_patterns = (
        r"\bi prefer\b",
        r"\bmy preference\b",
        r"\bmy timezone\b",
        r"\bmy role\b",
        r"\bmy coding style\b",
        r"\bmy name is\b",
        r"\bcall me\b",
        r"\bi am\b",
        r"\bi'm\b",
        r"\bthe user prefers\b",
        r"\bthe user is\b",
        r"\bthe user's\b",
        r"\bthe user likes\b",
        r"\bthe user wants\b",
        r"\bthe user expects\b",
        r"\buser correction\b",
        r"\bpersonal correction\b",
        r"\bdon't do that again\b",
        r"\bdo not do that again\b",
    )
    if requested_target == "user" or any(re.search(pat, canonical_text, re.IGNORECASE) for pat in user_patterns):
        return {
            "canonical_destination": "native_user",
            "native_target": "user",
            "entry_kind": "preference",
            "reason": "user_or_person_specific",
        }

    repo_patterns = (
        r"\bthis repo\b",
        r"\bthis repository\b",
        r"\bthis codebase\b",
        r"\bcurrent repo\b",
        r"\bcurrent project\b",
        r"\bproject-specific\b",
        r"\brepo-local\b",
        r"\bbelongs in git\b",
        r"\bdocs/cca-lite\b",
        r"\bhermes-memory\.json\b",
        r"\bdecision\b",
        r"\binvariant\b",
        r"\bconvention\b",
    )
    if any(re.search(pat, canonical_text, re.IGNORECASE) for pat in repo_patterns):
        return {
            "canonical_destination": "cca_lite",
            "native_target": None,
            "entry_kind": _entry_kind_for_target("memory", canonical_text),
            "reason": "repo_local_project_state",
        }

    cross_project_patterns = (
        r"\bcross-project\b",
        r"\bacross projects\b",
        r"\bacross repos\b",
        r"\breusable\b",
        r"\bsemantic recall\b",
        r"\boperational lesson\b",
        r"\bnot tied to (one|a|this) repo\b",
        r"\bacross sessions\b",
        r"\bgeneral lesson\b",
    )
    if any(re.search(pat, canonical_text, re.IGNORECASE) for pat in cross_project_patterns):
        return {
            "canonical_destination": "mempalace",
            "native_target": None,
            "entry_kind": "operational_fact",
            "reason": "cross_project_reusable_knowledge",
        }

    environment_patterns = (
        r"\benvironment\b",
        r"\bsetup\b",
        r"\binstalled\b",
        r"\bthis machine\b",
        r"\btoolchain\b",
        r"\bos\b",
        r"\bhermes home\b",
        r"\bapi key\b",
        r"\bconfig\b",
    )
    if any(re.search(pat, canonical_text, re.IGNORECASE) for pat in environment_patterns):
        return {
            "canonical_destination": "native_memory",
            "native_target": "memory",
            "entry_kind": "operational_fact",
            "reason": "stable_environment_or_setup_fact",
        }

    return {
        "canonical_destination": "native_memory",
        "native_target": "memory",
        "entry_kind": _entry_kind_for_target("memory", canonical_text),
        "reason": "default_native_memory",
    }


def _fallback_destinations(destination: MemoryDestination, routing: dict[str, Any]) -> list[MemoryDestination]:
    if destination in ("native_user", "native_memory"):
        if routing.get("entry_kind") in ("decision", "open_question"):
            return ["cca_lite", "mempalace"]
        if routing.get("reason") == "cross_project_reusable_knowledge":
            return ["mempalace", "cca_lite"]
        return ["mempalace", "cca_lite"]
    if destination == "mempalace":
        return ["native_memory", "cca_lite"]
    if destination == "cca_lite":
        return ["mempalace", "native_memory"]
    return ["native_memory"]


def _reviewed_routing(
    heuristic_routing: dict[str, Any],
    *,
    canonical_destination: str | None = None,
    classification_reason: str | None = None,
) -> dict[str, Any]:
    routing = dict(heuristic_routing)
    heuristic_destination = str(heuristic_routing.get("canonical_destination", ""))
    reviewed_destination = str(canonical_destination or "").strip()
    routing["classification_source"] = "heuristic"
    routing["heuristic_destination"] = heuristic_destination

    if not reviewed_destination:
        return routing

    review = {
        "requested_destination": reviewed_destination,
        "accepted": reviewed_destination in MEMORY_DESTINATIONS,
    }
    if classification_reason:
        review["reason"] = classification_reason.strip()
    routing["agent_review"] = review

    if reviewed_destination not in MEMORY_DESTINATIONS:
        return routing

    routing["classification_source"] = "agent_review"
    routing["canonical_destination"] = reviewed_destination
    routing["reason"] = classification_reason.strip() if classification_reason else "agent_review"
    if reviewed_destination == "native_user":
        routing["native_target"] = "user"
        routing["entry_kind"] = "preference"
    elif reviewed_destination == "native_memory":
        routing["native_target"] = "memory"
        routing["entry_kind"] = str(routing.get("entry_kind") or "operational_fact")
    else:
        routing["native_target"] = None
        routing["entry_kind"] = str(routing.get("entry_kind") or "operational_fact")
    return routing


def route_memory_tool_write(
    *,
    requested_target: str,
    content: str,
    native_add: Callable[[str, str], dict[str, Any]],
    session_id: str = "",
    source_event: str = "memory_tool",
    invocation_mode: InvocationMode = "manual",
    canonical_destination: str | None = None,
    classification_reason: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route one memory-tool add to its canonical sink, with explicit fallback metadata."""
    config = config or {}
    routing = _reviewed_routing(
        classify_memory_fact(content, requested_target),
        canonical_destination=canonical_destination,
        classification_reason=classification_reason,
    )
    destination: MemoryDestination = routing["canonical_destination"]
    canonical_text = canonicalize_text(content)
    entry_kind = str(routing.get("entry_kind") or "operational_fact")
    session_key = session_id or "__memory_tool__"
    fallback_attempted = False
    attempts: list[dict[str, Any]] = []

    def _metadata(
        *,
        actual_sink: str | None,
        fallback: bool = False,
        rerouted: bool = False,
        error: str | None = None,
    ) -> dict[str, Any]:
        meta = {
            "requested_target": requested_target,
            "canonical_destination": destination,
            "actual_sink": actual_sink,
            "rerouted": rerouted,
            "fallback": fallback,
            "routing_reason": routing.get("reason"),
            "classification_source": routing.get("classification_source", "heuristic"),
            "heuristic_destination": routing.get("heuristic_destination"),
            "agent_review": routing.get("agent_review"),
            "attempts": attempts,
        }
        if error:
            meta["error"] = error
        return meta

    def _write(dest: MemoryDestination) -> dict[str, Any]:
        if dest == "native_user":
            return native_add("user", content)
        if dest == "native_memory":
            return native_add("memory", content)
        if dest == "mempalace":
            return _mempalace_route(content, invocation_mode, config=config)
        if dest == "cca_lite":
            _, fingerprint = fingerprint_candidate("cca_lite", entry_kind, canonical_text, session_key)
            return _route_to_cca_lite(
                session_id=session_id,
                invocation_mode=invocation_mode,
                source_event=source_event,
                entry_kind=entry_kind,
                text=content,
                canonical_text=canonical_text,
                fingerprint=fingerprint,
                config=config,
            )
        return {"success": False, "error": f"unknown destination {dest}"}

    destinations = [destination] + _fallback_destinations(destination, routing)
    seen_destinations: set[str] = set()
    last_error = "no sink accepted the fact"

    for candidate in destinations:
        if candidate in seen_destinations:
            continue
        seen_destinations.add(candidate)
        try:
            result = _write(candidate)
        except Exception as exc:
            result = {"success": False, "error": str(exc)}

        attempts.append({
            "destination": candidate,
            "success": bool(result.get("success")),
            "mode": result.get("mode"),
            "reason": result.get("reason"),
            "error": result.get("error"),
        })

        disabled = result.get("success") and result.get("mode") == "disabled"
        if result.get("success") and not disabled:
            result = dict(result)
            result["routing"] = _metadata(
                actual_sink=candidate,
                fallback=fallback_attempted,
                rerouted=candidate != destination,
            )
            return result

        last_error = str(result.get("reason") or result.get("error") or "sink unavailable")
        fallback_attempted = True

    return {
        "success": False,
        "error": last_error,
        "routing": _metadata(actual_sink=None, fallback=fallback_attempted, error=last_error),
    }


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
        logger.debug("MemPalace unavailable; skipping sink: %s", exc)
        return {"success": True, "mode": "disabled", "reason": f"MemPalace unavailable: {exc}"}

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
    """Route a session slice to one canonical durable memory sink."""
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

    routing = classify_memory_fact(content, "memory")
    destination = routing["canonical_destination"]
    entry_kind = str(routing.get("entry_kind") or "operational_fact")
    _, fingerprint = fingerprint_candidate(destination, entry_kind, canonical_text, session_key)

    routed: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    if has_seen(session_key, fingerprint):
        suppressed.append({
            "destination": destination,
            "entry_kind": entry_kind,
            "fingerprint": fingerprint,
            "reason": "session_duplicate",
        })
    else:
        def _native_add(target: str, text: str) -> dict[str, Any]:
            if memory_store is None:
                return {"success": False, "mode": "disabled", "error": "native memory store not available"}
            try:
                return memory_store.add(target, text)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        result = route_memory_tool_write(
            requested_target=str(routing.get("native_target") or "memory"),
            content=content,
            native_add=_native_add,
            session_id=session_id,
            source_event=source_event,
            invocation_mode=invocation_mode,
            config=config,
        )
        result_routing = result.get("routing", {}) if isinstance(result, dict) else {}
        actual_sink = result_routing.get("actual_sink")
        if result.get("success"):
            routed.append({
                "destination": actual_sink,
                "canonical_destination": result_routing.get("canonical_destination"),
                "entry_kind": entry_kind,
                "fingerprint": fingerprint,
                "status": "written" if result.get("appended", True) else "no_op",
                "fallback": bool(result_routing.get("fallback")),
                "rerouted": bool(result_routing.get("rerouted")),
                "routing": result_routing,
            })
            mark_seen(session_key, fingerprint)
        else:
            failed.append({
                "destination": destination,
                "error": result.get("error", "unknown error"),
                "routing": result_routing,
            })

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
