"""A session pinned to one auth type never rotates onto a pool entry of another (#216 prereq: delegated OAuth
children must not adopt a stale API-key entry on rate-limit rotation)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.client_lifecycle import ClientLifecycleMixin
from agent.credential_pool import PooledCredential

CODEX_URL = "https://chatgpt.com/backend-api/codex"


@pytest.mark.parametrize("pinned, entry_auth_type, adopted", [
    ("oauth", "api_key", False), ("oauth", "oauth", True), (None, "api_key", True),
])
def test_swap_honours_pinned_auth_type(pinned, entry_auth_type, adopted):
    agent = SimpleNamespace(provider="openai-codex", model="gpt-test", base_url=CODEX_URL, api_key="current",
                            api_mode="codex_responses", _client_kwargs={}, _pinned_auth_type=pinned,
                            _reapply_route_client_config=MagicMock(), _replace_primary_openai_client=MagicMock())
    entry = PooledCredential(provider="openai-codex", id="e", label="e", auth_type=entry_auth_type, priority=0,
                             source="manual", access_token="candidate", base_url=CODEX_URL)

    assert ClientLifecycleMixin._swap_credential(agent, entry) is adopted
    assert agent.api_key == ("candidate" if adopted else "current")
