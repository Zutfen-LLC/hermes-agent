"""``X-Hermes-Provider-API-Key`` stays API-key carriage only (PR #15 invariant, ops-supervisor#216).

A provider whose registered metadata mandates OAuth or an external process has no API-key rung; before this guard
openai-codex's explicit rung forwarded any header value as its bearer, so the header could smuggle an OAuth access
token. Delegated OAuth children resolve their authority natively and never touch the request-scoped channel.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms import api_server_provider_credentials as pc
from tools.delegate_tool import delegate_task

OAUTH_TOKEN = "oauth-access-token-FIXTURE"
CODEX_URL = "https://chatgpt.com/backend-api/codex"


@pytest.mark.parametrize("provider", ["openai-codex", "codex", "xai-oauth", "nous", "copilot-acp"])
def test_login_and_process_providers_reject_request_scoped_keys(provider):
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolve:
        with pytest.raises(pc.ProviderCredentialError) as failure:
            pc.resolve_credential_runtime(pc.ProviderCredentialOverride(api_key=OAUTH_TOKEN, provider=provider),
                                          target_model=None)
    resolve.assert_not_called()
    assert failure.value.code == "provider_credential_unsupported"
    assert OAUTH_TOKEN not in failure.value.message


def test_api_key_providers_keep_request_scoped_semantics():
    runtime = pc.resolve_credential_runtime(pc.ProviderCredentialOverride(api_key="glm-request-key", provider="zai"),
                                            target_model=None)
    assert (runtime["provider"], runtime["api_key"]) == ("zai", "glm-request-key")


def test_delegated_oauth_child_never_uses_the_request_scoped_channel():
    """An OAuth child binds its pooled authority; the request-scoped credential machinery is never involved and the
    child's client carries no provider-key header."""
    from agent.credential_pool import CredentialPool, PooledCredential

    pool = CredentialPool("openai-codex", [PooledCredential(
        provider="openai-codex", id="dc", label="dc", auth_type="oauth", priority=0, source="device_code",
        access_token=OAUTH_TOKEN, base_url=CODEX_URL)])
    parent = MagicMock(
        provider="openai-codex", base_url=CODEX_URL, api_key=OAUTH_TOKEN, api_mode="codex_responses", model="m",
        client=None, _client_kwargs={"base_url": CODEX_URL, "api_key": OAUTH_TOKEN}, _credential_pool=pool,
        _credential_pool_entry_id="dc", acp_command=None, acp_args=[], requested_provider="openai-codex",
        _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(), _session_db=None,
        _print_fn=None, tool_progress_callback=None, thinking_callback=None, request_overrides={},
        reasoning_config=None, _fallback_chain=None, capabilities=None, max_tokens=None)
    child = MagicMock(_session_init_model_config={})
    child.run_conversation.return_value = {"final_response": "ok", "completed": True, "api_calls": 1, "messages": []}
    channel = [patch.object(pc, name, side_effect=AssertionError(name)) for name in
               ("extract_provider_credential", "resolve_credential_runtime", "apply_credential_runtime")]
    for spy in channel:
        spy.start()
    try:
        with patch("run_agent.AIAgent", return_value=child) as agent_cls:
            delegate_task(goal="code it", parent_agent=parent)
    finally:
        for spy in channel:
            spy.stop()
    kwargs = agent_cls.call_args.kwargs
    assert kwargs["api_key"] == OAUTH_TOKEN
    assert pc.PROVIDER_API_KEY_HEADER not in repr(kwargs.get("request_overrides") or {})
