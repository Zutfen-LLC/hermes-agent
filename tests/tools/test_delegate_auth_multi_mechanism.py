"""Delegation binds a credential by the authority that selected it, not by the provider's native login (#216).

Reproducers (red at a345aeb):
- an https-only Codex OAuth login authorized an ``http://`` child route (the scheme was not part of the check);
- Nous — native login OAuth, but its runtime also accepts an explicit inference API key — was treated as
  OAuth-only, so a configured Nous API key was refused (delegation_auth_type_mismatch).
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_pool import PooledCredential
from tools.delegate_tool import _build_child_agent
from tools.delegate_tool_auth import (
    AUTH_UNDETERMINED, KEY_EXPLICIT, KEY_RUNTIME, ROUTE_MISMATCH, DelegationAuthError,
)

CODEX_URL = "https://chatgpt.com/backend-api/codex"
CODEX_HTTP_URL = "http://chatgpt.com/backend-api/codex"
NOUS_URL = "https://inference-api.nousresearch.com/v1"
PROXY_URL = "https://llm-proxy.invalid/v1"
OAUTH_TOKEN = "oauth-access-token-FIXTURE"
NOUS_KEY = "nous-inference-key-FIXTURE"


def _parent(provider, url, api_key, pool=None, entry_id=None):
    return MagicMock(
        provider=provider, base_url=url, api_key=api_key, api_mode="chat_completions", model="parent-model",
        client=None, _client_kwargs={"base_url": url, "api_key": api_key}, _credential_pool=pool,
        _credential_pool_entry_id=entry_id, _auth_authority=None, acp_command=None, acp_args=[],
        requested_provider=provider, _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(),
        _session_db=None, _print_fn=None, tool_progress_callback=None, thinking_callback=None, request_overrides={},
        reasoning_config=None, _fallback_chain=None, capabilities=None, max_tokens=None, session_id="parent-sid")


def _fake_child(**kwargs):
    child = MagicMock(**{k: kwargs.get(k) for k in ("provider", "base_url", "api_key", "model")})
    child._session_init_model_config = {}
    return child


def _build(parent, pool=None, **overrides):
    with patch("tools.delegate_tool._resolve_child_credential_pool", return_value=pool), \
            patch("run_agent.AIAgent", side_effect=_fake_child) as agent_cls:
        child = _build_child_agent(task_index=0, goal="g", context=None, toolsets=None, model=None, max_iterations=3,
                                   task_count=1, parent_agent=parent, **overrides)
    return child, agent_cls


def test_https_oauth_login_never_authorizes_an_http_child_route():
    runtime = {"provider": "openai-codex", "base_url": CODEX_URL, "api_key": OAUTH_TOKEN, "source": "device_code"}
    with patch("hermes_cli.runtime_provider.resolve_oauth_store_runtime", return_value=runtime), \
            patch("run_agent.AIAgent") as agent_cls, \
            patch("tools.delegate_tool._resolve_child_credential_pool", return_value=None):
        with pytest.raises(DelegationAuthError) as failure:
            _build_child_agent(task_index=0, goal="g", context=None, toolsets=None, model=None, max_iterations=3,
                               task_count=1, parent_agent=_parent("openai-codex", CODEX_URL, OAUTH_TOKEN),
                               override_provider="openai-codex", override_base_url=CODEX_HTTP_URL)
    agent_cls.assert_not_called()
    assert failure.value.code == ROUTE_MISMATCH
    assert OAUTH_TOKEN not in str(failure.value)


def test_a_configured_nous_api_key_binds_verbatim_as_api_key():
    child, agent_cls = _build(_parent("openai-codex", CODEX_URL, OAUTH_TOKEN), override_provider="nous",
                              override_base_url=NOUS_URL, override_api_key=NOUS_KEY, override_key_origin=KEY_EXPLICIT)
    assert agent_cls.call_args.kwargs["api_key"] == NOUS_KEY
    assert (child._auth_authority.provider, child._auth_authority.auth_type) == ("nous", "api_key")


@pytest.mark.parametrize("stamped", ["oauth", "api_key"])
def test_a_runtime_resolved_nous_key_takes_the_resolving_rungs_auth_type(stamped):
    child, _ = _build(_parent("openai-codex", CODEX_URL, OAUTH_TOKEN), override_provider="nous",
                      override_base_url=NOUS_URL, override_api_key=NOUS_KEY, override_key_origin=KEY_RUNTIME,
                      override_key_source="portal", override_key_auth_type=stamped)
    assert child._auth_authority.auth_type == stamped


def test_an_undetermined_nous_parent_credential_stays_on_its_route_and_never_rotates():
    """Nothing canonical says whether an unpooled Nous parent runs on its OAuth login or an API key: same route, the
    child keeps it bound ``undetermined`` (admits no rotation); another route never receives it."""
    parent = _parent("nous", NOUS_URL, OAUTH_TOKEN)
    child, agent_cls = _build(parent)
    assert agent_cls.call_args.kwargs["api_key"] == OAUTH_TOKEN
    assert child._auth_authority.auth_type == AUTH_UNDETERMINED
    sibling = PooledCredential(provider="nous", id="o", label="o", auth_type="oauth", priority=0, source="manual",
                               access_token="other", base_url=NOUS_URL)
    assert not child._auth_authority.admits(sibling, NOUS_URL)

    child, agent_cls = _build(parent, override_base_url=PROXY_URL)
    assert agent_cls.call_args.kwargs["api_key"] != OAUTH_TOKEN
