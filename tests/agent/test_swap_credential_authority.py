"""A session bound to an authentication authority never rotates outside it (ops-supervisor#216): rotation may move
between entries of the same authority, never onto another auth type or endpoint."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.auth_authority import AuthAuthority
from agent.client_lifecycle import ClientLifecycleMixin
from agent.credential_pool import PooledCredential

CODEX_URL = "https://chatgpt.com/backend-api/codex"
OTHER_URL = "https://codex-proxy.invalid/backend-api/codex"


@pytest.mark.parametrize("bound_type, entry_type, entry_url, adopted", [
    ("oauth", "api_key", CODEX_URL, False),  # never onto an API key
    ("oauth", "oauth", OTHER_URL, False),    # never onto another endpoint
    ("oauth", "oauth", CODEX_URL, True),     # a sibling entry of the same authority
    ("api_key", "oauth", CODEX_URL, False),
    (None, "api_key", CODEX_URL, True),      # unbound sessions (the parent) keep pool rotation as before
])
def test_rotation_stays_within_the_bound_authority(bound_type, entry_type, entry_url, adopted):
    authority = AuthAuthority("openai-codex", "chatgpt.com/backend-api/codex", bound_type, "device_code", "dc") \
        if bound_type else None
    agent = SimpleNamespace(provider="openai-codex", model="gpt-test", base_url=CODEX_URL, api_key="current",
                            api_mode="codex_responses", _client_kwargs={}, _auth_authority=authority,
                            _reapply_route_client_config=MagicMock(), _replace_primary_openai_client=MagicMock())
    entry = PooledCredential(provider="openai-codex", id="e", label="e", auth_type=entry_type, priority=0,
                             source="manual", access_token="candidate", base_url=entry_url)

    assert ClientLifecycleMixin._swap_credential(agent, entry) is adopted
    assert agent.api_key == ("candidate" if adopted else "current")
    assert agent._auth_authority is authority  # the authority itself is immutable across rotation
