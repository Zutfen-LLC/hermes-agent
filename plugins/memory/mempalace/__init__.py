"""MemPalace memory plugin — MemoryProvider interface.

Local-first persistent memory via the MemPalace library (chromadb + HNSW).

MemPalace organizes knowledge into wings (projects), rooms (aspects), and
drawers (individual memories) with semantic search, knowledge graph, and
diary support. This plugin provides:

- **Pre-compaction context preservation**: on_pre_compress extracts key
  facts from messages about to be compacted and saves them as drawers,
  preventing context loss during Hermes' context window compression.
- **Smart routing**: extracted content is automatically categorized into the
  right wing/room based on configurable pattern rules (not dumped into a
  single catch-all). User-defined rules override defaults.
- **Semantic search**: query past memories across wings/rooms.
- **Turn sync**: persist important conversation turns as drawers.
- **Memory write mirroring**: built-in memory writes are mirrored to the palace.
- **Reclassification**: mp_reclassify tool to re-categorize misplaced drawers.
- **Session diary**: write diary entries for long-term agent self-reflection.

Requires: mempalace package installed (pip install mempalace-mcp).
Config: MEMPALACE_PALACE_PATH (default: ~/.mempalace/palace).

The palace path is also configurable via config.yaml under mempalace: or
as the --palace argument when running the MCP server.

Routing rules are configurable under mempalace.routing_rules in config.yaml.
Each rule has: patterns (list of regex), wing, room. First match wins.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChromaDB backend (same as mempalace MCP server)
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_client_cache = None
_collection_cache = None
_palace_db_inode = 0
_palace_db_mtime = 0.0


def _get_chroma_backend():
    """Import chromadb and return the PersistentClient factory."""
    import chromadb
    return chromadb


def _get_client(palace_path: str):
    """Return a ChromaDB PersistentClient, with inode-based reconnect detection."""
    global _client_cache, _collection_cache, _palace_db_inode, _palace_db_mtime

    chromadb = _get_chroma_backend()
    db_path = os.path.join(palace_path, "chroma.sqlite3")

    try:
        st = os.stat(db_path)
        current_inode = st.st_ino
        current_mtime = st.st_mtime
    except OSError:
        current_inode = 0
        current_mtime = 0.0

    if not os.path.isfile(db_path) and _collection_cache is not None:
        with _client_lock:
            _client_cache = None
            _collection_cache = None
            _palace_db_inode = 0
            _palace_db_mtime = 0.0

    inode_changed = current_inode != 0 and current_inode != _palace_db_inode
    mtime_changed = current_mtime != 0.0 and abs(current_mtime - _palace_db_mtime) > 0.01

    with _client_lock:
        if _client_cache is None or inode_changed or mtime_changed:
            _client_cache = chromadb.PersistentClient(path=palace_path)
            _collection_cache = None
            _palace_db_inode = current_inode
            _palace_db_mtime = current_mtime
        return _client_cache


_COLLECTION_NAME = "mempalace"


def _get_collection(palace_path: str, create: bool = False):
    """Return the ChromaDB collection for mempalace drawers."""
    global _collection_cache

    with _client_lock:
        if _collection_cache is not None and not create:
            return _collection_cache

    client = _get_client(palace_path)

    if create:
        raw = client.get_or_create_collection(
            _COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "hnsw:num_threads": 1},
        )
    else:
        try:
            raw = client.get_collection(_COLLECTION_NAME)
        except Exception:
            return None

    _collection_cache = raw
    return raw


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def _sanitize_name(name: str, kind: str) -> str:
    """Sanitize wing/room names (alphanumeric, hyphens, underscores)."""
    import re
    name = name.strip().lower()
    if not name or not re.match(r'^[a-z0-9][a-z0-9_\-]*$', name):
        raise ValueError(f"Invalid {kind} name: {name!r}")
    if len(name) > 128:
        raise ValueError(f"{kind.capitalize()} name too long: {name!r}")
    return name


def _add_drawer(
    palace_path: str,
    wing: str,
    room: str,
    content: str,
    source_file: str = "",
    added_by: str = "hermes-plugin",
) -> dict:
    """Add a drawer to the palace. Idempotent via deterministic ID."""
    try:
        wing = _sanitize_name(wing, "wing")
        room = _sanitize_name(room, "room")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    col = _get_collection(palace_path, create=True)
    if not col:
        return {"success": False, "error": "Cannot access palace database"}

    drawer_id = (
        f"drawer_{wing}_{room}_"
        f"{hashlib.sha256((wing + room + content).encode()).hexdigest()[:24]}"
    )

    # Idempotency check
    try:
        existing = col.get(ids=[drawer_id])
        if existing and existing.get("ids"):
            return {
                "success": True,
                "reason": "already_exists",
                "drawer_id": drawer_id,
            }
    except Exception:
        pass

    try:
        col.upsert(
            ids=[drawer_id],
            documents=[content],
            metadatas=[{
                "wing": wing,
                "room": room,
                "source_file": source_file,
                "chunk_index": 0,
                "added_by": added_by,
                "filed_at": datetime.now().isoformat(),
            }],
        )
        return {
            "success": True,
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _search(
    palace_path: str,
    query: str,
    n_results: int = 5,
    wing: str = "",
    room: str = "",
) -> dict:
    """Semantic search across the palace."""
    col = _get_collection(palace_path)
    if not col:
        return {"success": False, "error": "Cannot access palace database"}

    where_filter = None
    conditions = []
    if wing:
        conditions.append({"wing": wing})
    if room:
        conditions.append({"room": room})
    if conditions:
        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

    try:
        kwargs = {
            "query_texts": [query[:500]],
            "n_results": min(n_results, 20),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        result = col.query(**kwargs)
        results = []
        if result and result.get("ids") and result["ids"][0]:
            for i, doc_id in enumerate(result["ids"][0]):
                results.append({
                    "id": doc_id,
                    "content": result["documents"][0][i] if result["documents"] else "",
                    "wing": result["metadatas"][0][i].get("wing", "") if result["metadatas"] else "",
                    "room": result["metadatas"][0][i].get("room", "") if result["metadatas"] else "",
                    "distance": result["distances"][0][i] if result["distances"] else 0,
                })
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _status(palace_path: str) -> dict:
    """Get palace status."""
    # Try with create=True in case collection doesn't exist yet
    col = _get_collection(palace_path, create=True)
    if not col:
        return {"success": False, "error": "Cannot access palace database"}

    try:
        count = col.count()
        return {
            "success": True,
            "drawers": count,
            "palace_path": palace_path,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Context extraction for pre-compression
# ---------------------------------------------------------------------------

def _extract_pre_compress_content(messages: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """Extract salient content from messages about to be compressed.

    Strategy:
    1. Collect user messages (what was asked)
    2. Collect assistant messages that contain tool results or decisions
    3. Extract file paths, error messages, and factual statements
    4. Build a concise summary of what happened
    """
    parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if not isinstance(content, str):
            # Handle list content (multi-part messages)
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            else:
                continue

        content = content.strip()
        if not content:
            continue

        # Skip system messages and compaction summaries
        if role == "system":
            continue
        if role == "user" and "[CONTEXT COMPACTION" in content:
            continue

        # User messages: full content (these are the questions/tasks)
        if role == "user":
            parts.append(f"[USER] {content[:800]}")

        # Assistant messages: extract key facts
        elif role == "assistant":
            # Look for file paths, error messages, decisions
            lines = content.split("\n")
            key_lines = []
            for line in lines[:50]:  # first 50 lines max
                stripped = line.strip()
                if not stripped:
                    continue
                # File paths
                if "/" in stripped and any(stripped.endswith(ext) for ext in
                    [".py", ".yaml", ".yml", ".json", ".md", ".toml", ".cfg", ".sh"]):
                    key_lines.append(stripped)
                # Error indicators
                elif any(kw in stripped.lower() for kw in
                    ["error:", "failed", "fix:", "patch:", "created", "deleted",
                     "updated", "configured", "installed", "resolved"]):
                    key_lines.append(stripped)
                # Decision indicators
                elif any(kw in stripped.lower() for kw in
                    ["decided", "chose", "root cause", "solution", "approach"]):
                    key_lines.append(stripped)

            if key_lines:
                parts.append("[ASSISTANT KEY FACTS]\n" + "\n".join(key_lines[:15]))

        # Tool results: extract paths and outcomes
        elif role == "tool":
            # Extract just the first few meaningful lines
            lines = content.split("\n")[:5]
            meaningful = [l.strip() for l in lines if l.strip() and len(l.strip()) > 20]
            if meaningful:
                parts.append(f"[TOOL] {'; '.join(meaningful[:3])}")

    combined = "\n".join(parts)

    # Truncate to max_chars
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n[... truncated]"

    return combined


# ---------------------------------------------------------------------------
# Content routing — smart categorization into wings/rooms
# ---------------------------------------------------------------------------

# Default routing rules. Each rule: (compiled_patterns, wing, room).
# First match wins. These are broad heuristics — user rules in config.yaml
# are checked FIRST and take priority.
DEFAULT_ROUTING_RULES: List[Tuple[List[re.Pattern], str, str]] = [
    # Infrastructure / ops patterns
    (
        [re.compile(p, re.IGNORECASE) for p in [
            r'\bops-supervisor\b', r'\bDISCORD_ALLOWED_USERS\b',
            r'\bhermes-gateway\b', r'\bsystemctl\b.*\buser\b',
            r'\brate.limit\b.*\bgithub\b', r'\bproxmox\b',
            r'\bVMID\s+\d+', r'\bvmbr0\b',
        ]],
        "wing_ops", "incidents",
    ),
    # Configuration changes
    (
        [re.compile(p, re.IGNORECASE) for p in [
            r'\.env\b', r'config\.yaml\b', r'config\.json\b',
            r'\.env\.', r'settings\.(yaml|json|toml)',
        ]],
        "wing_ops", "config-changes",
    ),
    # Deployment / CI/CD
    (
        [re.compile(p, re.IGNORECASE) for p in [
            r'\bdeploy', r'\bCI\b', r'\bCD\b', r'\bgit push\b',
            r'\bgithub.actions\b', r'\bdocker\b.*\bbuild\b',
            r'\bsystemd\b.*\bservice\b',
        ]],
        "wing_ops", "deployments",
    ),
    # Database changes
    (
        [re.compile(p, re.IGNORECASE) for p in [
            r'\bsqlite\b', r'\bpostgres\b', r'\bmigration\b',
            r'\bschema\b.*\bchange', r'\bALTER TABLE\b',
        ]],
        "wing_ops", "database",
    ),
    # Security / auth
    (
        [re.compile(p, re.IGNORECASE) for p in [
            r'\bSSH\b.*\bkey\b', r'\bapi.key\b', r'\btoken\b',
            r'\bcredential', r'\bauth\b', r'\bpassword\b',
            r'\bpermission\b', r'\bfirewall\b',
        ]],
        "wing_ops", "security",
    ),
    # Memory provider plugin development
    (
        [re.compile(p, re.IGNORECASE) for p in [
            r'\bmemory.provider\b', r'\bon_pre_compress\b',
            r'\bMemoryProvider\b', r'\bmempalace\b.*\bplugin\b',
        ]],
        "wing_hermes-dev", "plugin-development",
    ),
    # Pipeline / automation
    (
        [re.compile(p, re.IGNORECASE) for p in [
            r'\bpipeline\b', r'\bbacklog-to-supervisor\b',
            r'\bwiki.*\bingest', r'\bbridge\s+script\b',
            r'\bcron\s+job\b', r'\bdispatch\b',
        ]],
        "wing_ops", "pipeline-architecture",
    ),
]


def _compile_user_rules(rules_config: List[dict]) -> List[Tuple[List[re.Pattern], str, str]]:
    """Compile user-defined routing rules from config.yaml.

    Expected format:
      routing_rules:
        - patterns: ["regex1", "regex2"]
          wing: "my_project"
          room: "my_room"
    """
    compiled = []
    for rule in rules_config:
        patterns = rule.get("patterns", [])
        wing = rule.get("wing", "")
        room = rule.get("room", "")
        if not patterns or not wing or not room:
            continue
        try:
            compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
            compiled.append((compiled_patterns, wing, room))
        except re.error as e:
            logger.warning("Skipping invalid routing rule pattern: %s", e)
    return compiled


def _route_content(
    content: str,
    user_rules: Optional[List[Tuple[List[re.Pattern], str, str]]] = None,
) -> Tuple[str, str]:
    """Route content to the best wing/room based on pattern matching.

    Returns (wing, room) tuple. Falls back to ("wing_hermes", "pre-compress").

    User rules are checked first (they take priority), then defaults.
    """
    # Check user rules first
    if user_rules:
        for patterns, wing, room in user_rules:
            for pat in patterns:
                if pat.search(content):
                    return (wing, room)

    # Check default rules
    for patterns, wing, room in DEFAULT_ROUTING_RULES:
        for pat in patterns:
            if pat.search(content):
                return (wing, room)

    return ("wing_hermes", "pre-compress")


def _list_wing_rooms(palace_path: str) -> Dict[str, List[str]]:
    """Get all wings and their rooms from existing drawer metadata."""
    col = _get_collection(palace_path)
    if not col:
        return {}

    try:
        result = col.get(include=["metadatas"])
        wings_rooms: Dict[str, set] = {}
        if result and result.get("metadatas"):
            for meta in result["metadatas"]:
                wing = meta.get("wing", "")
                room = meta.get("room", "")
                if wing:
                    wings_rooms.setdefault(wing, set()).add(room)
        return {k: sorted(v) for k, v in wings_rooms.items()}
    except Exception:
        return {}

SEARCH_SCHEMA = {
    "name": "mp_search",
    "description": (
        "Semantic search across the MemPalace — wings, rooms, and drawers. "
        "Returns verbatim drawer content with similarity scores. "
        "Use for recalling past context, decisions, and facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
            "wing": {
                "type": "string",
                "description": "Filter by wing/project name (optional).",
            },
            "room": {
                "type": "string",
                "description": "Filter by room/aspect name (optional).",
            },
            "n_results": {
                "type": "integer",
                "description": "Max results (default 5, max 20).",
            },
        },
        "required": ["query"],
    },
}

FILE_SCHEMA = {
    "name": "mp_file",
    "description": (
        "File verbatim content into the MemPalace for long-term recall. "
        "Organizes into wing (project) and room (aspect). "
        "Use for architectural decisions, bug fixes, user preferences, "
        "pipeline changes — anything worth remembering across sessions. "
        "Deduplicates automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {
                "type": "string",
                "description": "Wing/project name (e.g. 'ops', 'atlas', 'myproject').",
            },
            "room": {
                "type": "string",
                "description": "Room/aspect (e.g. 'decisions', 'incidents', 'pipeline').",
            },
            "content": {
                "type": "string",
                "description": "Verbatim content to store.",
            },
        },
        "required": ["wing", "room", "content"],
    },
}

STATUS_SCHEMA = {
    "name": "mp_status",
    "description": "Check MemPalace status — drawer count, palace path, availability.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

RECLASSIFY_SCHEMA = {
    "name": "mp_reclassify",
    "description": (
        "Move a drawer from one wing/room to another. Use to re-categorize "
        "memories that were auto-routed incorrectly. Takes a drawer ID, "
        "new wing, and new room."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "drawer_id": {
                "type": "string",
                "description": "The drawer ID to move (from mp_search results).",
            },
            "wing": {
                "type": "string",
                "description": "New wing to move the drawer to.",
            },
            "room": {
                "type": "string",
                "description": "New room to move the drawer to.",
            },
        },
        "required": ["drawer_id", "wing", "room"],
    },
}

ROUTES_SCHEMA = {
    "name": "mp_routes",
    "description": (
        "List all active routing rules and the existing wing/room taxonomy. "
        "Shows user-defined rules (from config) and default rules, plus "
        "which wings/rooms already have drawers."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MemPalaceMemoryProvider(MemoryProvider):
    """MemPalace persistent memory via local ChromaDB."""

    def __init__(self):
        self._palace_path = ""
        self._session_id = ""
        self._turn_count = 0
        self._sync_thread: Optional[threading.Thread] = None
        self._flush_thread: Optional[threading.Thread] = None
        self._user_routing_rules: List[Tuple[List[re.Pattern], str, str]] = []

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        """Check if mempalace (chromadb) is installed and palace exists."""
        try:
            _get_chroma_backend()
            # Check for default palace path
            default = Path.home() / ".mempalace" / "palace"
            env_path = os.environ.get("MEMPALACE_PALACE_PATH", "")
            palace = env_path if env_path else str(default)
            db = os.path.join(palace, "chroma.sqlite3")
            return os.path.isfile(db)
        except ImportError:
            return False

    def get_config_schema(self):
        return [
            {
                "key": "palace_path",
                "description": "Path to the MemPalace data directory (contains chroma.sqlite3)",
                "required": False,
                "default": "~/.mempalace/palace",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", "")
        platform = kwargs.get("platform", "cli")

        # Resolve palace path: env > config > default
        env_path = os.environ.get("MEMPALACE_PALACE_PATH", "")
        if env_path:
            self._palace_path = os.path.expanduser(env_path)
        else:
            # Try reading from hermes config
            try:
                from hermes_cli.config import load_config
                config = load_config()
                mp_config = config.get("mempalace", {})
                configured_path = mp_config.get("palace_path", "")
                if configured_path:
                    self._palace_path = os.path.expanduser(configured_path)

                # Load user routing rules
                rules_config = mp_config.get("routing_rules", [])
                if rules_config:
                    self._user_routing_rules = _compile_user_rules(rules_config)
                    logger.info(
                        "MemPalace loaded %d user routing rules",
                        len(self._user_routing_rules),
                    )
            except Exception:
                pass

        if not self._palace_path:
            self._palace_path = str(Path.home() / ".mempalace" / "palace")

        self._palace_path = os.path.expanduser(self._palace_path)
        self._session_id = session_id
        self._turn_count = 0

        logger.info(
            "MemPalace plugin initialized: palace=%s, session=%s, platform=%s",
            self._palace_path, session_id[:8], platform,
        )

    def system_prompt_block(self) -> str:
        if not self._palace_path or not os.path.isfile(
            os.path.join(self._palace_path, "chroma.sqlite3")
        ):
            return ""
        return (
            "# MemPalace Memory\n"
            "Active. Local-first structured memory palace with semantic search.\n"
            "Context is automatically preserved before compression (on_pre_compress).\n"
            "Use mp_search to recall past knowledge, mp_file to store important facts, "
            "mp_status to check state."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context from the palace for the upcoming turn."""
        if not query or not self._palace_path or len(query.strip()) < 10:
            return ""

        result = _search(self._palace_path, query.strip()[:500], n_results=3)
        if not result.get("success") or not result.get("results"):
            return ""

        parts = []
        for r in result["results"][:3]:
            wing = r.get("wing", "")
            room = r.get("room", "")
            content = r.get("content", "")
            if content:
                label = f"{wing}/{room}" if wing and room else wing or room or "memory"
                parts.append(f"**[{label}]** {content[:600]}")

        if not parts:
            return ""

        return "## MemPalace Context\n" + "\n\n".join(parts)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch() runs synchronously."""
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist important conversation turns as drawers (non-blocking).

        Only files substantive turns (>50 chars user input) to avoid noise.
        Content is routed to appropriate wing/room based on pattern rules.
        """
        self._turn_count += 1

        if len(user_content.strip()) < 50:
            return

        combined = (
            f"User: {user_content[:1500]}\n"
            f"Assistant: {assistant_content[:1500]}"
        )
        wing, room = _route_content(combined, self._user_routing_rules)

        # Wait for previous sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3.0)

        def _sync():
            try:
                drawer_content = (
                    f"[Turn {self._turn_count} | {datetime.now().isoformat()}]\n"
                    f"{combined}"
                )
                _add_drawer(
                    self._palace_path,
                    wing=wing,
                    room=room,
                    content=drawer_content,
                    added_by="hermes-plugin:sync_turn",
                )
            except Exception as e:
                logger.debug("MemPalace sync_turn failed: %s", e)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="mp-sync"
        )
        self._sync_thread.start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract and save key context before compression discards messages.

        This is the primary value of the MemPalace plugin — it ensures that
        important context from the conversation is preserved in the palace
        before Hermes' context compressor summarizes and discards old messages.

        Content is smart-routed to the appropriate wing/room based on
        configurable pattern rules (not dumped into a single catch-all).

        The extraction is non-blocking (runs in a background thread) so it
        doesn't add latency to the compression process.
        """
        if not messages or not self._palace_path:
            return ""

        content = _extract_pre_compress_content(messages)
        if not content or len(content) < 100:
            return ""

        # Smart route the content to the right wing/room
        wing, room = _route_content(content, self._user_routing_rules)

        # Wait for previous flush
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2.0)

        def _flush():
            try:
                timestamp = datetime.now().isoformat()
                drawer_content = (
                    f"[Pre-compression save | {timestamp}]\n"
                    f"Session: {self._session_id}\n"
                    f"Messages in window: {len(messages)}\n\n"
                    f"{content}"
                )
                result = _add_drawer(
                    self._palace_path,
                    wing=wing,
                    room=room,
                    content=drawer_content,
                    added_by="hermes-plugin:on_pre_compress",
                )
                if result.get("success"):
                    logger.info(
                        "MemPalace pre-compression flush: %d messages → %s/%s (%s)",
                        len(messages), wing, room,
                        result.get("drawer_id", "?")[:40],
                    )
                else:
                    logger.debug(
                        "MemPalace pre-compression flush skipped: %s",
                        result.get("error", "unknown"),
                    )
            except Exception as e:
                logger.debug("MemPalace pre-compression flush failed: %s", e)

        self._flush_thread = threading.Thread(
            target=_flush, daemon=True, name="mp-flush"
        )
        self._flush_thread.start()

        # Return empty string — we're saving directly, no need to inject
        # into the compression summary prompt (that would be redundant)
        return ""

    def on_memory_write(self, action: str, target: str, content: str, **kwargs) -> None:
        """Mirror built-in memory writes to the palace."""
        if action not in ("add", "replace") or not content:
            return

        wing = "wing_hermes"
        room = "user-profile" if target == "user" else "agent-memory"

        def _write():
            try:
                _add_drawer(
                    self._palace_path,
                    wing=wing,
                    room=room,
                    content=f"[Memory {action}] {content}",
                    added_by="hermes-plugin:memory_mirror",
                )
            except Exception as e:
                logger.debug("MemPalace memory mirror failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="mp-memwrite")
        t.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Optionally save a session-end summary to the palace."""
        if not messages or not self._palace_path:
            return

        # Extract final summary
        content = _extract_pre_compress_content(messages[-20:], max_chars=2000)
        if not content or len(content) < 100:
            return

        def _save():
            try:
                drawer_content = (
                    f"[Session end | {datetime.now().isoformat()}]\n"
                    f"Session: {self._session_id}\n"
                    f"Total turns: {self._turn_count}\n\n"
                    f"{content}"
                )
                _add_drawer(
                    self._palace_path,
                    wing="wing_hermes",
                    room="sessions",
                    content=drawer_content,
                    added_by="hermes-plugin:on_session_end",
                )
            except Exception as e:
                logger.debug("MemPalace session-end save failed: %s", e)

        t = threading.Thread(target=_save, daemon=True, name="mp-session-end")
        t.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, FILE_SCHEMA, STATUS_SCHEMA, RECLASSIFY_SCHEMA, ROUTES_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "mp_search":
            return self._tool_search(args)
        elif tool_name == "mp_file":
            return self._tool_file(args)
        elif tool_name == "mp_status":
            return self._tool_status()
        elif tool_name == "mp_reclassify":
            return self._tool_reclassify(args)
        elif tool_name == "mp_routes":
            return self._tool_routes()
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._sync_thread, self._flush_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

    # -- Tool implementations ------------------------------------------------

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        result = _search(
            self._palace_path,
            query.strip()[:500],
            n_results=args.get("n_results", 5),
            wing=args.get("wing", ""),
            room=args.get("room", ""),
        )

        if not result.get("success"):
            return tool_error(result.get("error", "Search failed"))

        results = result.get("results", [])
        if not results:
            return json.dumps({"result": "No relevant memories found."})

        formatted = []
        for r in results[:10]:
            wing = r.get("wing", "")
            room = r.get("room", "")
            content = r.get("content", "")
            distance = r.get("distance", 0)
            label = f"{wing}/{room}" if wing and room else wing or room
            formatted.append(f"[{label}] (distance: {distance:.2f})\n{content}")

        output = "\n\n---\n\n".join(formatted)
        if len(output) > 8000:
            output = output[:8000] + "\n\n[... truncated]"

        return json.dumps({"result": output})

    def _tool_file(self, args: dict) -> str:
        wing = args.get("wing", "")
        room = args.get("room", "")
        content = args.get("content", "")

        if not wing or not room or not content:
            return tool_error("wing, room, and content are required")

        result = _add_drawer(
            self._palace_path,
            wing=wing,
            room=room,
            content=content,
            added_by="hermes-plugin:tool",
        )

        if not result.get("success"):
            reason = result.get("reason", "")
            if reason == "already_exists":
                return json.dumps({
                    "result": "Already exists (idempotent).",
                    "drawer_id": result.get("drawer_id", ""),
                })
            return tool_error(result.get("error", "Failed to file drawer"))

        return json.dumps({
            "result": f"Filed to {wing}/{room}.",
            "drawer_id": result.get("drawer_id", ""),
        })

    def _tool_status(self) -> str:
        result = _status(self._palace_path)
        if not result.get("success"):
            return tool_error(result.get("error", "Status check failed"))
        return json.dumps({"status": result})

    def _tool_reclassify(self, args: dict) -> str:
        """Move a drawer to a different wing/room by re-upserting with new metadata."""
        drawer_id = args.get("drawer_id", "")
        new_wing = args.get("wing", "")
        new_room = args.get("room", "")

        if not drawer_id or not new_wing or not new_room:
            return tool_error("drawer_id, wing, and room are required")

        try:
            _sanitize_name(new_wing, "wing")
            _sanitize_name(new_room, "room")
        except ValueError as e:
            return tool_error(str(e))

        col = _get_collection(self._palace_path, create=True)
        if not col:
            return tool_error("Cannot access palace database")

        # Fetch existing drawer
        try:
            existing = col.get(ids=[drawer_id], include=["documents", "metadatas"])
        except Exception as e:
            return tool_error(f"Failed to fetch drawer: {e}")

        if not existing or not existing.get("ids") or not existing["ids"]:
            return tool_error(f"Drawer not found: {drawer_id}")

        doc = existing["documents"][0] if existing["documents"] else ""
        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
        old_wing = old_meta.get("wing", "?")
        old_room = old_meta.get("room", "?")

        # Generate new deterministic ID based on new wing/room + content
        new_id = (
            f"drawer_{new_wing}_{new_room}_"
            f"{hashlib.sha256((new_wing + new_room + doc).encode()).hexdigest()[:24]}"
        )

        # Delete old drawer, upsert new one
        try:
            col.delete(ids=[drawer_id])
        except Exception:
            pass  # best effort

        new_meta = {
            **old_meta,
            "wing": new_wing,
            "room": new_room,
            "reclassified_at": datetime.now().isoformat(),
            "original_wing": old_wing,
            "original_room": old_room,
            "reclassified_by": "hermes-plugin:mp_reclassify",
        }

        try:
            col.upsert(
                ids=[new_id],
                documents=[doc],
                metadatas=[new_meta],
            )
            return json.dumps({
                "result": f"Moved {drawer_id[:30]}... from {old_wing}/{old_room} → {new_wing}/{new_room}",
                "new_drawer_id": new_id,
            })
        except Exception as e:
            return tool_error(f"Failed to reclassify: {e}")

    def _tool_routes(self) -> str:
        """List active routing rules and wing/room taxonomy."""
        lines = ["## Active Routing Rules\n"]

        # User rules
        if self._user_routing_rules:
            lines.append("### User-defined rules (checked first):\n")
            for i, (patterns, wing, room) in enumerate(self._user_routing_rules, 1):
                pats = ", ".join(p.pattern for p in patterns[:3])
                if len(patterns) > 3:
                    pats += f", +{len(patterns)-3} more"
                lines.append(f"{i}. `{pats}` → **{wing}/{room}**")
        else:
            lines.append("### User-defined rules: (none — add to config.yaml under mempalace.routing_rules)\n")

        # Default rules
        lines.append("\n### Default rules:\n")
        for i, (patterns, wing, room) in enumerate(DEFAULT_ROUTING_RULES, 1):
            pats = ", ".join(p.pattern for p in patterns[:3])
            if len(patterns) > 3:
                pats += f", +{len(patterns)-3} more"
            lines.append(f"{i}. `{pats}` → **{wing}/{room}**")

        lines.append("\n### Unmatched content → wing_hermes/pre-compress")

        # Existing taxonomy
        wings = _list_wing_rooms(self._palace_path)
        if wings:
            lines.append("\n\n## Existing Wing/Room Taxonomy\n")
            for wing in sorted(wings.keys()):
                rooms = wings[wing]
                lines.append(f"- **{wing}**: {', '.join(rooms)}")

        return json.dumps({"result": "\n".join(lines)})


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register MemPalace as a memory provider plugin."""
    ctx.register_memory_provider(MemPalaceMemoryProvider())
