"""Adversarial initialization-path proof for the restricted tool-free envelope (PR #32 R2).

The NO-GO finding: ``_new_restricted_agent`` set ``_restricted_wire_binding`` only AFTER
``AIAgent(**kwargs)`` returned, but the constructor builds the client during init
(``agent_init._build_client`` → ``_init_openai_client`` → ``_create_openai_client`` →
``agent_runtime_helpers.create_openai_client``). On that initialization path the guard
consumer read ``_restricted_wire_binding`` before it existed, so provider-profile
client-kwargs extras ran and a provider-supplied client (``ProviderProfile.create_client``)
could be built and returned BEFORE any restricted check was active.

[RED] ``test_tool_free_init_path_never_runs_provider_client_hooks`` fails on the reviewed
head cb55914 (the injected hook fires and the provider's object becomes ``agent.client``)
and passes on the corrected head (the guard precedes any client construction; the actual
request rides the guarded, non-redirecting SDK client through the real production
client-construction path with only the transport stubbed).

[PIN] The remaining tests pin retained behavior: the ``input_only_v1`` envelope keeps its
exact v1 initialization semantics, and the tool-free binding drives the guard chokepoint.
Every positive expectation was probe-verified against the production functions on the
reviewed head before being written.
"""
import json

import httpx
import pytest

from gateway.platforms import api_server, api_server_restricted_runs as restricted
from tests.gateway.test_restricted_runs_transport import restricted_service

CREDS = {"provider": "deepinfra", "model": "Qwen/Qwen3.8-Flash", "api_key": "test-key",
         "base_url": "https://api.deepinfra.com/v1/openai", "api_mode": "chat_completions"}


def test_tool_free_init_path_never_runs_provider_client_hooks(restricted_service, monkeypatch):
    """[RED] The init-time client build must already run under the restricted guard.

    A hostile provider plugin replaces the ``deepinfra`` profile: its ``create_client``
    (the provider-supplied client hook) and ``build_client_kwargs_extras`` (the provider
    profile extras) both record invocations. The real ``AIAgent`` constructor builds the
    client during init through ``create_openai_client``; the restricted envelope requires
    that NEITHER hook executes, the provider's object never becomes ``agent.client``, and
    the request the turn actually sends rides the SDK client built by the guarded
    production path (non-redirecting, model-only) with a stubbed transport.
    """
    import providers
    from providers import ProviderProfile

    calls = {"create_client": 0, "extras": 0}
    hostile_client = object()  # sentinel: must never become the wire client

    class HostileProfile(ProviderProfile):
        def build_client_kwargs_extras(self, **context):
            calls["extras"] += 1
            return {"following_redirects": "pwned"}

        def create_client(self, **client_kwargs):
            calls["create_client"] += 1
            return hostile_client

    original = providers.get_provider_profile("deepinfra")
    hostile = HostileProfile(name="deepinfra", base_url=CREDS["base_url"], auth_type="api_key")
    monkeypatch.setitem(providers._REGISTRY, "deepinfra", hostile)

    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(adapter, dict(CREDS), None,
                                             _api_server=api_server, envelope="hermes_tool_free_v1")

    # The v2 guard must have been active DURING construction: neither provider hook ran.
    assert calls["create_client"] == 0, "provider-supplied client hook ran during restricted init"
    assert calls["extras"] == 0, "provider profile client-kwargs extras ran during restricted init"
    # And the provider's object is not the client the agent carries.
    assert agent.client is not hostile_client

    # The actual request must ride a guarded, non-redirecting SDK client. Stub ONLY the
    # transport factory: the real ``create_openai_client`` runs its restricted checks
    # (injected-client refusal, redirecting-keepalive refusal) and builds the real SDK
    # client over the mock transport; the turn's middleware bypass is asserted live.
    wire = []

    def respond(request):
        wire.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"id": "chatcmpl-init-proof", "object": "chat.completion",
            "model": CREDS["model"], "created": 1, "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "bounded"}, "finish_reason": "stop"}]})

    mock_http = httpx.Client(transport=httpx.MockTransport(respond))
    assert mock_http.follow_redirects is False  # httpx default; the guard requires exactly this
    monkeypatch.setattr(agent, "_build_keepalive_http_client",
                        lambda base_url="", verify=True, **kw: mock_http)
    monkeypatch.setattr("hermes_cli.middleware.apply_llm_request_middleware",
                        lambda *args, **kw: pytest.fail("request middleware ran"))
    monkeypatch.setattr("hermes_cli.middleware.run_llm_execution_middleware",
                        lambda *args, **kw: pytest.fail("execution middleware ran"))
    monkeypatch.setattr("agent.turn_api_request._fire_pre_api_request_hook",
                        lambda *args, **kw: pytest.fail("request hook ran"))
    agent._cached_system_prompt = "Supplied input only."
    result = restricted._restricted_agent_turn(agent, "Only this snapshot")

    assert result["final_response"] == "bounded"
    assert wire, "no request reached the transport"
    assert all(url == "https://api.deepinfra.com/v1/openai/chat/completions" for url, _ in wire)
    assert all(body["model"] == CREDS["model"] and not body.get("tools")
               and not body.get("extra_body") and not body.get("web_search_options")
               for _, body in wire)
    # The SDK client the turn built on the request path is non-redirecting.
    request_client = agent._create_request_openai_client(reason="restricted-init-proof")
    try:
        assert request_client._client.follow_redirects is False
    finally:
        agent._close_request_openai_client(request_client, reason="restricted-init-proof")


def test_init_time_client_on_tool_free_envelope_is_sdk_client_with_redirects_off(restricted_service):
    """[PIN] The client built DURING construction on the tool-free envelope is the real
    SDK client with redirects disabled — never a provider-supplied object. (Passes at the
    reviewed head too: pins that the correction does not degrade the init client.)"""
    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(adapter, dict(CREDS), None,
                                             _api_server=api_server, envelope="hermes_tool_free_v1")
    init_client = agent.client
    assert init_client is not None
    assert type(init_client).__name__ == "OpenAI"
    assert init_client._client.follow_redirects is False


def test_input_only_v1_envelope_keeps_v1_initialization_behavior(restricted_service):
    """[PIN] ``input_only_v1`` keeps its exact v1 initialization semantics: the native
    OpenAI route constructs the SDK client through the unguarded (profile-aware) path,
    and no tool-free wire binding is installed. (Positive control probe-verified at the
    reviewed head; the binding is the tool-free envelope's opt-in state.)"""
    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(adapter, {"provider": "openai", "model": "gpt-4o",
        "api_key": "test-key", "base_url": "https://api.openai.com/v1",
        "api_mode": "chat_completions"}, None, _api_server=api_server)
    assert type(agent.client).__name__ == "OpenAI"
    assert getattr(agent, "_restricted_wire_binding", None) is None


def test_tool_free_binding_is_the_resolved_route_and_drives_the_chokepoint(restricted_service):
    """[PIN] The binding installed by the constructor equals the resolved route, and the
    client chokepoint consumes it: a drifted base_url is rejected before any provider
    hook or client construction can run on a rebuild path."""
    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(adapter, dict(CREDS), None,
                                             _api_server=api_server, envelope="hermes_tool_free_v1")
    assert agent._restricted_wire_binding == (
        agent.provider, agent.model, agent.base_url, "chat_completions")
    drifted = dict(agent._client_kwargs)
    drifted["base_url"] = "https://evil.invalid/v1"
    with pytest.raises(RuntimeError, match="route drift"):
        agent._create_openai_client(drifted, reason="binding-pin", shared=True)
