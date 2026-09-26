"""``X-Hermes-Provider-API-Key`` stays API-key carriage only (PR #15 invariant, ops-supervisor#216).

The header is admitted exactly when the provider's runtime accepts an API key
(``agent.auth_authority.provider_accepted_mechanisms``): OAuth-only and external-process providers reject it
before resolution — openai-codex's explicit rung would otherwise forward it as its OAuth bearer — while a provider
whose native login is OAuth but whose explicit rung takes an inference API key (Nous) keeps the PR #15 contract.
Delegated OAuth children resolve their authority natively and never touch the request-scoped channel.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms import api_server_provider_credentials as pc
from tools.delegate_tool import delegate_task

OAUTH_TOKEN = "oauth-access-token-FIXTURE"
REQUEST_KEY = "request-scoped-key-FIXTURE"
CODEX_URL = "https://chatgpt.com/backend-api/codex"


def _external_process_providers():
    from hermes_cli.auth import PROVIDER_REGISTRY
    return sorted(p for p, cfg in PROVIDER_REGISTRY.items() if cfg.auth_type == "external_process")


def _resolve(provider, key):
    return pc.resolve_credential_runtime(pc.ProviderCredentialOverride(api_key=key, provider=provider),
                                         target_model=None)


@pytest.mark.parametrize("provider", ["openai-codex", "codex", "xai-oauth", *_external_process_providers()])
def test_login_only_and_process_providers_reject_request_scoped_keys(provider):
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolve:
        with pytest.raises(pc.ProviderCredentialError) as failure:
            _resolve(provider, OAUTH_TOKEN)
    resolve.assert_not_called()
    assert failure.value.code == "provider_credential_unsupported"
    assert OAUTH_TOKEN not in failure.value.message


@pytest.mark.parametrize("provider", ["nous", "zai"])
def test_api_key_capable_providers_carry_the_request_key_verbatim(provider):
    """Nous' native login is OAuth, but its explicit rung takes an inference API key: PR #15 accepted it and the
    request key reached the resolved runtime unchanged. zai is a plain API-key provider."""
    runtime = _resolve(provider, REQUEST_KEY)
    assert (runtime["provider"], runtime["api_key"], runtime["source"]) == (provider, REQUEST_KEY, "explicit")
    assert runtime["auth_type"] == "api_key"


def test_an_explicit_resolver_alone_never_admits_an_oauth_route(monkeypatch):
    """Registering an explicit-credential rung for an OAuth provider does not make it an API-key provider: an
    undeclared rung carries the provider's registered mechanism, so the header is still refused."""
    from hermes_cli import runtime_provider as rp
    monkeypatch.setitem(rp._EXPLICIT_RESOLVERS, "xai-oauth",
                        lambda rq, mc, key, url, tm: rp._runtime("xai-oauth", "codex_responses", url, key))
    assert rp.explicit_credential_auth_type("xai-oauth") == "oauth"
    with pytest.raises(pc.ProviderCredentialError) as failure:
        _resolve("xai-oauth", OAUTH_TOKEN)
    assert failure.value.code == "provider_credential_unsupported"


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
