"""Tests for the shared memory routing hook wiring in AIAgent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sess-route-test"
    agent.model = "test-model"
    agent._memory_store = None
    agent._memory_manager = None
    agent._todo_store = MagicMock()
    agent._todo_store.format_for_injection.return_value = ""
    agent.tools = []
    agent.context_compressor = MagicMock()
    agent.context_compressor.compress.return_value = [{"role": "assistant", "content": "compressed"}]
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 0
    agent.context_compressor.context_length = 1000
    agent.context_compressor.last_prompt_tokens = 0
    agent.context_compressor.last_completion_tokens = 0
    agent._cached_system_prompt = "cached system prompt"
    agent._last_compression_summary_warning = None
    agent._session_db = None
    agent.logs_dir = Path("/tmp")
    agent._vprint = lambda *args, **kwargs: None
    agent._emit_warning = MagicMock()
    agent._invalidate_system_prompt = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="new system prompt")
    return agent


def test_compress_context_routes_durable_memory_before_summarization(monkeypatch) -> None:
    agent = _make_agent()
    routed = []

    def fake_route(messages, *, invocation_mode, source_event, candidate_window=None):
        routed.append(
            {
                "messages": list(messages),
                "invocation_mode": invocation_mode,
                "source_event": source_event,
                "candidate_window": candidate_window,
            }
        )
        return {"ok": True, "summary": "routed"}

    monkeypatch.setattr(agent, "route_memory_session", fake_route)

    messages = [
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": "sure"},
    ]

    compressed, new_system_prompt = agent._compress_context(messages, "system prompt", approx_tokens=42)

    assert routed == [
        {
            "messages": messages,
            "invocation_mode": "pre_compress",
            "source_event": "compression",
            "candidate_window": None,
        }
    ]
    assert compressed == [{"role": "assistant", "content": "compressed"}]
    assert new_system_prompt == "new system prompt"
    agent.context_compressor.compress.assert_called_once()
