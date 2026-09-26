"""An ``AuthAuthority``'s route is frozen and enforced, scheme included (ops-supervisor#216).

Reproducers (red at a345aeb): ``endpoint_identity`` dropped the scheme, so an ``http://`` spelling of an ``https://``
authority compared equal; and ``admits()`` only checked the entry against the agent's MUTABLE ``base_url``, so a
child whose ``base_url`` was re-pointed could rotate onto a credential for the new route. The frozen
``AuthAuthority.route`` (``normalize_route_base_url``) is now the one authority a candidate is compared against.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.auth_authority import AuthAuthority, endpoint_identity, route_identity
from agent.client_lifecycle import ClientLifecycleMixin
from agent.credential_pool import PooledCredential

CODEX_URL = "https://chatgpt.com/backend-api/codex"
CODEX_HTTP_URL = "http://chatgpt.com/backend-api/codex"
OTHER_URL = "https://codex-proxy.invalid/backend-api/codex"


def _agent(authority, base_url=CODEX_URL, provider="openai-codex"):
    return SimpleNamespace(provider=provider, model="gpt-test", base_url=base_url, api_key="current",
                           api_mode="codex_responses", _client_kwargs={}, _auth_authority=authority,
                           _reapply_route_client_config=MagicMock(), _replace_primary_openai_client=MagicMock())


def _entry(url, auth_type="oauth", provider="openai-codex"):
    return PooledCredential(provider=provider, id="candidate", label="candidate", auth_type=auth_type, priority=0,
                            source="manual", access_token="candidate-token", base_url=url)


def test_https_and_http_are_different_authorities():
    assert route_identity(CODEX_URL) != route_identity(CODEX_HTTP_URL)
    assert endpoint_identity(CODEX_URL) != endpoint_identity(CODEX_HTTP_URL)
    authority = AuthAuthority.for_route("openai-codex", CODEX_URL, "oauth", "device_code")
    assert not authority.admits(_entry(CODEX_HTTP_URL), CODEX_HTTP_URL)


def test_canonically_equivalent_spellings_stay_one_authority():
    authority = AuthAuthority.for_route("openai-codex", "https://CHATGPT.com:443/backend-api/codex/", "oauth", "dc")
    assert authority.admits(_entry(CODEX_URL), CODEX_URL)
    assert authority.endpoint == "https://chatgpt.com/backend-api/codex"


@pytest.mark.parametrize("mutated_url", [OTHER_URL, CODEX_HTTP_URL])
def test_rotation_compares_against_the_frozen_route_not_the_mutated_base_url(mutated_url):
    """Frozen at the https Codex route; the agent's base_url is later re-pointed and the candidate serves only the
    new route: refused, nothing changes."""
    authority = AuthAuthority.for_route("openai-codex", CODEX_URL, "oauth", "device_code", "dc")
    agent = _agent(authority, base_url=mutated_url)
    assert ClientLifecycleMixin._swap_credential(agent, _entry(mutated_url)) is False
    assert agent.api_key == "current" and agent.base_url == mutated_url


@pytest.mark.parametrize("auth_type, entry_url, entry_provider, adopted", [
    ("oauth", CODEX_URL, "openai-codex", True),     # same auth type + same frozen endpoint
    ("api_key", CODEX_URL, "openai-codex", False),  # different auth type, same endpoint
    ("oauth", OTHER_URL, "openai-codex", False),    # same auth type, different endpoint
    ("oauth", CODEX_URL, "openai", False),          # a mixed pool's entry of another provider on the same route
])
def test_admission_requires_auth_type_endpoint_and_provider_together(auth_type, entry_url, entry_provider, adopted):
    authority = AuthAuthority.for_route("openai-codex", CODEX_URL, "oauth", "device_code", "dc")
    agent = _agent(authority)
    assert ClientLifecycleMixin._swap_credential(agent, _entry(entry_url, auth_type, entry_provider)) is adopted
    assert agent.api_key == ("candidate-token" if adopted else "current")


def test_the_enforceable_route_never_reaches_diagnostics():
    """Userinfo is part of the enforceable route (compared) but never of the display endpoint, repr or metadata."""
    authority = AuthAuthority.for_route("custom", "https://user:hunter2-FIXTURE@llm.example/v1", "api_key", "explicit")
    assert "hunter2-FIXTURE" in authority.route
    for rendered in (authority.endpoint, authority.describe(), repr(authority), repr(authority.as_metadata())):
        assert "hunter2-FIXTURE" not in rendered
