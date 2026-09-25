import pytest
from unittest.mock import MagicMock, patch
from gateway.platforms import api_server_provider_credentials as pc

KEY = "sk-prv...0001"
URL = "https://api.deepinfra.com/v1/openai"

def adapter():
    a = MagicMock()
    a._room_grant_token.return_value = None
    a._expected_api_key.return_value = "gateway-key"
    a._clean_runtime_id.side_effect = lambda value, **kw: value
    return a

def req(key=KEY):
    request = MagicMock()
    request.headers = {pc.PROVIDER_API_KEY_HEADER: key} if key else {}
    return request

@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_request_key_replaces_static_registry_key(provider):
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={"provider": provider, "api_key": KEY, "base_url": None, "api_mode": "chat_completions"}):
        runtime = pc.resolve_credential_runtime(pc.ProviderCredentialOverride(api_key=KEY, provider=provider), target_model=None)
    kwargs = {"provider": "static", "api_key": "STATIC", "base_url": "https://old.example", "credential_pool": object()}
    pc.apply_credential_runtime(kwargs, runtime)
    assert kwargs["api_key"] == KEY and kwargs["provider"] == provider and kwargs["credential_pool"] is None

@pytest.mark.parametrize("provider", ["deepinfra", "anthropic"])
def test_base_url_without_header_fails_closed_before_resolution(provider):
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolver:
        with pytest.raises(pc.ProviderCredentialError) as caught:
            pc.extract_provider_credential(adapter(), req(None),
                                           {"provider": provider, "provider_base_url": URL},
                                           scope_fn=lambda: "scope")
    assert caught.value.code == "provider_api_key_required"
    resolver.assert_not_called()

@pytest.mark.parametrize("url", [
    "https:///no-host", "https://user:pass@example.com", "https://user@example.com/v1",
    "https://example.com/a\n", "https://example.com/a\x00b", "https://example.com:99999",
    "https://bad..host/v1", "https://999.999.999.999/v1", "https://127.1/v1",
    "https://0x7f000001/v1", "https://example.com:/v1", "https://example.com./v1",
    "https://example.com/v1?token=x",
    "https://example.com/v1#fragment", "https://example.com/v1\\other",
    "https://example.com/" + "x" * 2048,
])
def test_structurally_invalid_base_urls_rejected(url):
    with pytest.raises(pc.ProviderCredentialError) as caught:
        pc.extract_provider_credential(adapter(), req(), {"provider": "openai", "provider_base_url": url}, scope_fn=lambda: "s")
    assert caught.value.code == "invalid_provider_base_url"

@pytest.mark.parametrize("url", [URL, URL + "/", "http://localhost:1234/v1", "http://127.0.0.1:1234/v1",
                                 "http://[::1]:1234/v1"])
def test_valid_base_urls_pass_verbatim(url):
    with patch.object(pc, "credential_fingerprint", return_value="fp"):
        credential = pc.extract_provider_credential(adapter(), req(),
             {"provider": "openai", "provider_base_url": url}, scope_fn=lambda: "s")
    assert credential.base_url == url

def test_direct_runtime_resolution_rejects_base_only_without_static_key_lookup():
    credential = pc.ProviderCredentialOverride(provider="deepinfra", base_url="https://fake-provider.invalid/v1")
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolver:
        with pytest.raises(pc.ProviderCredentialError) as caught:
            pc.resolve_credential_runtime(credential, target_model="m")
    resolver.assert_not_called()
    assert caught.value.code == "provider_api_key_required"


def test_resolver_arbitrary_key_echo_never_reaches_response_or_exception_chain():
    secret = "ARBITRARY-CREDENTIAL-7f19-not-a-known-key-pattern"
    credential = pc.ProviderCredentialOverride(api_key=secret, provider="deepinfra")
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=RuntimeError("resolver echoed " + secret)):
        with pytest.raises(pc.ProviderCredentialError) as caught:
            pc.resolve_credential_runtime(credential, target_model="m")
    assert caught.value.code == "provider_resolution_failed"
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None


def test_fingerprint_secret_fails_closed_without_gateway_secret(monkeypatch):
    from gateway import hosted_room_peer
    for error in (OSError("read-only"), ValueError("corrupt")):
        with patch.object(hosted_room_peer, "gateway_room_grant_secret", side_effect=error):
            with pytest.raises(pc.ProviderCredentialError) as caught:
                pc.extract_provider_credential(adapter(), req(),
                    {"provider": "deepinfra"}, scope_fn=lambda: "scope")
        assert caught.value.code == "provider_credential_fingerprint_unavailable"
        assert caught.value.status == 503
        assert KEY not in str(caught.value)


def test_fingerprint_unavailable_cannot_fall_back_to_api_server_key():
    from gateway import hosted_room_peer
    a = adapter()
    a._expected_api_key.return_value = "client-known-key-before-rotation"
    with patch.object(hosted_room_peer, "gateway_room_grant_secret", side_effect=OSError("read-only")):
        for gateway_key in ("client-known-key-before-rotation", "client-known-key-after-rotation"):
            a._expected_api_key.return_value = gateway_key
            with pytest.raises(pc.ProviderCredentialError) as caught:
                pc.extract_provider_credential(a, req(), {"provider": "deepinfra"},
                                               scope_fn=lambda: "scope")
            assert caught.value.code == "provider_credential_fingerprint_unavailable"


def test_gateway_only_fingerprint_stable_despite_api_server_key_rotation():
    from gateway import hosted_room_peer
    a = adapter()
    with patch.object(hosted_room_peer, "gateway_room_grant_secret", return_value=b"installation-only-key"):
        first = pc.extract_provider_credential(a, req(), {"provider": "deepinfra"}, scope_fn=lambda: "scope")
        a._expected_api_key.return_value = "rotated-client-known-key"
        second = pc.extract_provider_credential(a, req(), {"provider": "deepinfra"}, scope_fn=lambda: "scope")
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != KEY
