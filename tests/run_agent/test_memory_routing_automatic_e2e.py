"""End-to-end tests for automatic durable-memory hooks in AIAgent.

These tests exercise the actual hook entry points rather than the shared
router helper directly:
- shutdown_memory_provider() for session-end routing
- _compress_context() for pre-compress routing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sess-auto-route"
    agent.model = "test-model"
    agent._memory_store = None
    agent._session_db = None
    agent._todo_store = MagicMock()
    agent._todo_store.format_for_injection.return_value = ""
    agent.tools = []
    agent.context_compressor = MagicMock()
    agent.context_compressor.context_length = 1000
    agent.context_compressor.last_prompt_tokens = 0
    agent.context_compressor.last_completion_tokens = 0
    agent.context_compressor.compression_count = 0
    agent.context_compressor._last_summary_error = None
    agent._cached_system_prompt = "cached system prompt"
    agent._last_compression_summary_warning = None
    agent.logs_dir = Path("/tmp")
    agent._vprint = lambda *args, **kwargs: None
    agent._emit_warning = MagicMock()
    agent._invalidate_system_prompt = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="new system prompt")
    return agent


def test_shutdown_memory_provider_routes_before_teardown() -> None:
    agent = _make_agent()
    events = []

    def fake_route(messages, *, invocation_mode, source_event, candidate_window=None):
        events.append(("route", list(messages), invocation_mode, source_event, candidate_window))
        return {"ok": True, "summary": "session end routed"}

    agent.route_memory_session = fake_route  # type: ignore[assignment]
    agent._memory_manager = MagicMock()
    agent._memory_manager.shutdown_all.side_effect = lambda: events.append(("shutdown_all",))
    agent.context_compressor.on_session_end.side_effect = lambda session_id, messages: events.append(
        ("compressor_end", session_id, list(messages))
    )

    messages = [
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": "got it"},
    ]

    agent.shutdown_memory_provider(messages)

    assert events == [
        (
            "route",
            messages,
            "session_end",
            "shutdown",
            None,
        ),
        ("shutdown_all",),
        ("compressor_end", "sess-auto-route", messages),
    ]
    agent._memory_manager.shutdown_all.assert_called_once()
    agent.context_compressor.on_session_end.assert_called_once_with(
        "sess-auto-route", messages
    )


def test_compress_context_routes_before_context_shrinks() -> None:
    agent = _make_agent()
    events = []

    def fake_route(messages, *, invocation_mode, source_event, candidate_window=None):
        events.append(("route", list(messages), invocation_mode, source_event, candidate_window))
        return {"ok": True, "summary": "pre-compress routed"}

    def fake_compress(messages, *, current_tokens=None, focus_topic=None):
        events.append(("compress", list(messages), current_tokens, focus_topic))
        return [{"role": "assistant", "content": "compressed"}]

    agent.route_memory_session = fake_route  # type: ignore[assignment]
    agent.context_compressor.compress.side_effect = fake_compress

    messages = [
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": "sure"},
    ]

    compressed, new_system_prompt = agent._compress_context(
        messages,
        "system prompt",
        approx_tokens=42,
    )

    assert events == [
        (
            "route",
            messages,
            "pre_compress",
            "compression",
            None,
        ),
        (
            "compress",
            messages,
            42,
            None,
        ),
    ]
    assert compressed == [{"role": "assistant", "content": "compressed"}]
    assert new_system_prompt == "new system prompt"
    agent.context_compressor.compress.assert_called_once()
    agent._invalidate_system_prompt.assert_called_once()
    agent._build_system_prompt.assert_called_once()
