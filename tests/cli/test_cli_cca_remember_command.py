"""Tests for CLI /cca-remember command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = MagicMock()
    cli_obj.conversation_history = [
        {"role": "user", "content": "I prefer concise replies."},
        {"role": "assistant", "content": "Got it."},
    ]
    cli_obj.session_id = "sess-remember-test"
    cli_obj._pending_input = MagicMock()
    cli_obj._app = None
    return cli_obj


def test_cca_remember_routes_memory_from_current_conversation() -> None:
    cli_obj = _make_cli()
    cli_obj.agent.route_memory_session.return_value = {"summary": "3 routed"}

    with patch("builtins.print") as mock_print:
        result = cli_obj.process_command("/cca-remember")

    assert result is True
    cli_obj.agent.route_memory_session.assert_called_once()
    routed_messages = cli_obj.agent.route_memory_session.call_args.args[0]
    assert routed_messages == cli_obj.conversation_history
    assert cli_obj.agent.route_memory_session.call_args.kwargs["invocation_mode"] == "manual"
    assert any("3 routed" in str(call) for call in mock_print.call_args_list)
