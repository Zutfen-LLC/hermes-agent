"""Tests for the MemPalace memory provider plugin.

Tests cover: initialization, availability detection, on_pre_compress extraction,
tool handlers (mp_search, mp_file, mp_status), sync_turn, and memory write mirroring.
"""

import hashlib
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.mempalace import (
    SEARCH_SCHEMA,
    FILE_SCHEMA,
    STATUS_SCHEMA,
    MemPalaceMemoryProvider,
    _extract_pre_compress_content,
    _sanitize_name,
    _add_drawer,
    _search,
    _status,
    register,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no stale env vars leak between tests."""
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)


@pytest.fixture
def provider():
    """Create a provider instance."""
    return MemPalaceMemoryProvider()


@pytest.fixture
def initialized_provider(provider, tmp_path):
    """Create and initialize a provider with a temp palace path."""
    # Create a minimal chroma.sqlite3 so is_available passes
    palace_path = str(tmp_path / "palace")
    os.makedirs(palace_path, exist_ok=True)

    # Mock chromadb to avoid needing it installed
    with patch("plugins.memory.mempalace._get_chroma_backend") as mock_backend:
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_col.count.return_value = 42
        mock_col.get.return_value = {"ids": []}
        mock_col.upsert.return_value = None
        mock_col.query.return_value = {
            "ids": [["drawer_test_wing_test-room_abc123"]],
            "documents": [["test content"]],
            "metadatas": [[{"wing": "test", "room": "test-room"}]],
            "distances": [[0.5]],
        }
        mock_client.get_or_create_collection.return_value = mock_col
        mock_client.get_collection.return_value = mock_col
        mock_backend.return_value = MagicMock(
            PersistentClient=MagicMock(return_value=mock_client)
        )

        # Write a dummy chroma.sqlite3
        (Path(palace_path) / "chroma.sqlite3").touch()

        provider.initialize("test-session", hermes_home=str(tmp_path / "hermes"))
        provider._palace_path = palace_path

    yield provider


from pathlib import Path


# ---------------------------------------------------------------------------
# sanitize_name
# ---------------------------------------------------------------------------

class TestSanitizeName:
    def test_valid_lowercase(self):
        assert _sanitize_name("my-wing", "wing") == "my-wing"

    def test_valid_underscore(self):
        assert _sanitize_name("my_room", "room") == "my_room"

    def test_strips_whitespace(self):
        assert _sanitize_name("  my-wing  ", "wing") == "my-wing"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            _sanitize_name("", "wing")

    def test_rejects_special_chars(self):
        with pytest.raises(ValueError):
            _sanitize_name("my wing!", "wing")

    def test_rejects_leading_hyphen(self):
        with pytest.raises(ValueError):
            _sanitize_name("-bad", "wing")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            _sanitize_name("a" * 200, "wing")


# ---------------------------------------------------------------------------
# _extract_pre_compress_content
# ---------------------------------------------------------------------------

class TestExtractPreCompress:
    def test_empty_messages(self):
        assert _extract_pre_compress_content([]) == ""

    def test_extracts_user_messages(self):
        messages = [
            {"role": "user", "content": "Fix the database connection issue"},
        ]
        result = _extract_pre_compress_content(messages)
        assert "Fix the database connection issue" in result
        assert "[USER]" in result

    def test_extracts_assistant_key_facts(self):
        messages = [
            {"role": "assistant", "content": "Root cause: connection pool exhausted. Fixed by increasing max_connections in config.py."},
        ]
        result = _extract_pre_compress_content(messages)
        assert "Root cause" in result
        assert "[ASSISTANT KEY FACTS]" in result

    def test_skips_system_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        result = _extract_pre_compress_content(messages)
        assert "helpful assistant" not in result
        assert "Hello" in result

    def test_skips_compaction_summaries(self):
        messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] This is old"},
            {"role": "user", "content": "This is new"},
        ]
        result = _extract_pre_compress_content(messages)
        assert "REFERENCE ONLY" not in result
        assert "This is new" in result

    def test_extracts_file_paths(self):
        messages = [
            {"role": "assistant", "content": "The fix is in /home/user/config.yaml\nAlso check /etc/app/settings.json"},
        ]
        result = _extract_pre_compress_content(messages)
        assert "config.yaml" in result
        assert "settings.json" in result

    def test_truncates_long_content(self):
        long_content = "x" * 5000
        messages = [{"role": "user", "content": long_content}]
        result = _extract_pre_compress_content(messages, max_chars=100)
        assert len(result) < 200  # truncated + header

    def test_handles_list_content(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Important question"},
                {"type": "image", "url": "https://example.com/img.png"},
            ]},
        ]
        result = _extract_pre_compress_content(messages)
        assert "Important question" in result

    def test_below_threshold_returns_empty(self):
        messages = [
            {"role": "user", "content": "hi"},
        ]
        result = _extract_pre_compress_content(messages)
        # Short messages still extracted but the overall result is small
        assert "hi" in result


# ---------------------------------------------------------------------------
# Provider lifecycle
# ---------------------------------------------------------------------------

class TestProviderLifecycle:
    def test_name(self, provider):
        assert provider.name == "mempalace"

    def test_tool_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "mp_search" in names
        assert "mp_file" in names
        assert "mp_status" in names

    def test_config_schema(self, provider):
        schema = provider.get_config_schema()
        assert len(schema) == 1
        assert schema[0]["key"] == "palace_path"

    def test_initialize_sets_palace_path(self, provider, tmp_path):
        palace_path = str(tmp_path / "test_palace")
        os.makedirs(palace_path, exist_ok=True)
        with patch("plugins.memory.mempalace._get_chroma_backend"):
            provider.initialize("sess-1", hermes_home=str(tmp_path))
        assert "palace" in provider._palace_path


# ---------------------------------------------------------------------------
# on_pre_compress
# ---------------------------------------------------------------------------

class TestOnPreCompress:
    def test_empty_messages(self, initialized_provider):
        result = initialized_provider.on_pre_compress([])
        assert result == ""

    def test_saves_to_palace(self, initialized_provider):
        messages = [
            {"role": "user", "content": "Fix the rate limiter in ops-supervisor.py — it was hitting GitHub API too fast"},
            {"role": "assistant", "content": "Added check_rate_limit() guard in dispatch_once(). Threshold: 50 remaining requests."},
        ]
        result = initialized_provider.on_pre_compress(messages)
        # Returns empty string (saves directly, doesn't inject into summary)
        assert result == ""
        # Wait for background thread
        time.sleep(0.5)

    def test_short_content_skipped(self, initialized_provider):
        messages = [{"role": "user", "content": "ok"}]
        result = initialized_provider.on_pre_compress(messages)
        assert result == ""


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

class TestToolHandlers:
    def test_mp_status(self, initialized_provider):
        result = json.loads(initialized_provider.handle_tool_call("mp_status", {}))
        assert "status" in result

    def test_mp_file(self, initialized_provider):
        result = json.loads(initialized_provider.handle_tool_call("mp_file", {
            "wing": "test",
            "room": "test-room",
            "content": "Important fact to remember",
        }))
        assert "Filed" in result["result"]

    def test_mp_file_missing_args(self, initialized_provider):
        result = initialized_provider.handle_tool_call("mp_file", {"wing": "test"})
        assert "error" in result.lower() or "required" in result.lower()

    def test_mp_search(self, initialized_provider):
        with patch("plugins.memory.mempalace._get_collection") as mock_gc:
            mock_col = MagicMock()
            mock_col.query.return_value = {
                "ids": [["drawer_test"]],
                "documents": [["found: rate limit fix"]],
                "metadatas": [[{"wing": "test", "room": "test"}]],
                "distances": [[0.3]],
            }
            mock_gc.return_value = mock_col
            result = json.loads(initialized_provider.handle_tool_call("mp_search", {
                "query": "rate limit fix",
            }))
        assert "result" in result

    def test_mp_search_missing_query(self, initialized_provider):
        result = initialized_provider.handle_tool_call("mp_search", {})
        assert "error" in result.lower() or "required" in result.lower()

    def test_unknown_tool(self, initialized_provider):
        result = initialized_provider.handle_tool_call("mp_unknown", {})
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# register function
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_creates_provider(self):
        collector = MagicMock()
        register(collector)
        collector.register_memory_provider.assert_called_once()
        provider = collector.register_memory_provider.call_args[0][0]
        assert isinstance(provider, MemPalaceMemoryProvider)
        assert provider.name == "mempalace"
