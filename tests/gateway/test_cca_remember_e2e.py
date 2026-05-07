"""End-to-end tests for the /cca-remember gateway command.

These tests exercise the full gateway dispatch path:
- slash command parsing
- canonical command resolution / aliasing
- command:<name> hook emission
- gateway handler invocation
- durable-memory routing handoff into AIAgent
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-remember-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": "sure"},
        {"role": "user", "content": "also this"},
    ]
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/cca-remember should not fall through to the agent")
    )
    runner._cleanup_agent_resources = MagicMock()
    return runner


class _FakeAIAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._print_fn = None
        self.route_memory_session = MagicMock(
            return_value={"summary": "3 routed", "failed": []}
        )


def test_gateway_new_command_routes_memory_before_reset(monkeypatch):
    runner = _make_runner()
    runner._handle_cca_remember_command = AsyncMock(return_value="🧠 3 routed")
    runner._invalidate_session_run_generation = lambda *args, **kwargs: None
    runner._evict_cached_agent = lambda *args, **kwargs: None
    runner._clear_session_boundary_security_state = lambda *args, **kwargs: None
    runner._format_session_info = lambda: ""
    runner._telegram_topic_root_lobby = lambda *_args, **_kwargs: False
    runner._telegram_topic_new_header = lambda *_args, **_kwargs: ""
    runner.session_store._entries = {
        build_session_key(_make_source()): SimpleNamespace(session_id="sess-old")
    }
    runner.session_store.reset_session = MagicMock(
        return_value=SessionEntry(
            session_key=build_session_key(_make_source()),
            session_id="sess-new",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
    )

    result = asyncio.get_event_loop().run_until_complete(
        runner._handle_reset_command(_make_event("/new"))
    )

    runner._handle_cca_remember_command.assert_awaited_once()
    runner.session_store.reset_session.assert_called_once_with(build_session_key(_make_source()))
    assert isinstance(result, str)


@pytest.mark.parametrize("command_text", ["/cca-remember", "/remember"])
def test_gateway_dispatch_routes_memory_through_the_full_command_path(
    monkeypatch, command_text
):
    import gateway.run as gateway_run
    import run_agent

    runner = _make_runner()
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("test-model", {"api_key": "test-key"})
    )

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAIAgent)

    result = asyncio.get_event_loop().run_until_complete(
        runner._handle_message(_make_event(command_text))
    )

    assert result == "🧠 3 routed"
    runner.hooks.emit_collect.assert_awaited_once()
    assert runner.hooks.emit_collect.await_args.args[0] == "command:cca-remember"
    runner.session_store.get_or_create_session.assert_called_once()
    runner.session_store.load_transcript.assert_called_once_with("sess-remember-1")
    runner._resolve_session_agent_runtime.assert_called_once()
    runner._cleanup_agent_resources.assert_called_once()
    runner._run_agent.assert_not_awaited()

    # The fake agent instance is created inside the handler; inspect the call via
    # the route_memory_session mock that the handler invoked.
    # pylint: disable=protected-access
    agent_instance = runner._cleanup_agent_resources.call_args.args[0]
    assert agent_instance.route_memory_session.call_count == 1
    call = agent_instance.route_memory_session.call_args
    assert call.args[0] == [
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": "sure"},
        {"role": "user", "content": "also this"},
    ]
    assert call.kwargs["invocation_mode"] == "manual"
    assert call.kwargs["source_event"] == "cca-remember"
