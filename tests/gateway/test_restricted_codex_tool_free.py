"""Issue #35: restricted tool-free ``codex_responses`` wire contract.

RED at the reviewed starting head ``713c677695``: the valid ``openai-codex`` OAuth
route (provider ``openai-codex``, auth ``oauth``, wire ``codex_responses``, canonical
``https://chatgpt.com/backend-api/codex`` endpoint) is rejected by
``_validate_tool_free_route`` / ``_new_restricted_agent(envelope="hermes_tool_free_v1")``
with ``RuntimeError("restricted tool-free route is not enforceable")`` — the deliberate
chat-completions-only dialect boundary. Every security case below fails there (no
dialect admitted) or pins the fail-closed property the new dialect must keep.

The delegation chain is exercised through the REAL production functions; only the
credential boundary (``resolve_runtime_provider``) and the external HTTP boundary are
stubbed.
"""

import json
import threading

import pytest

from gateway.platforms import api_server_restricted_runs as restricted
from tests.gateway.test_restricted_runs_transport import restricted_service

CODEX_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_SECRET = "codex-oauth-store-secret-9042"

CODEX_CREDS = {
    "provider": "openai-codex", "model": "gpt-5-codex", "api_key": CODEX_SECRET,
    "base_url": CODEX_BASE, "api_mode": "codex_responses", "auth_type": "oauth",
}


def _codex_runtime(*, requested, target_model=None, base_url=CODEX_BASE, **_kw):
    """The exact runtime shape the openai-codex OAuth rung returns."""
    return {"provider": "openai-codex", "model": target_model or "gpt-5-codex",
            "base_url": base_url, "api_mode": "codex_responses",
            "api_key": CODEX_SECRET, "auth_type": "oauth",
            "source": "hermes-auth-store"}


def _write_profile(home, profile_yaml):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(
        "delegation:\n  profiles:\n" + profile_yaml, encoding="utf-8")


CODEX_PROFILE_YAML = """    codex-restricted:
      provider: openai-codex
      model: gpt-5-codex
      auth_type: oauth
      api_mode: codex_responses
      base_url: %s
""" % CODEX_BASE


def _resolve_codex_route(monkeypatch, tmp_profile_home=None):
    """The real _resolve_profile_execution -> bind_child_authority chain for the Codex profile.

    Only the credential boundary is stubbed: both rungs the real auth path consults
    (resolve_runtime_provider for the direct-endpoint branch's request_overrides and
    resolve_oauth_store_runtime for the OAuth store) return the fake openai-codex
    OAuth runtime. No real credential store is read.
    """
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _codex_runtime)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_oauth_store_runtime",
                        lambda *a, **k: _codex_runtime(requested="openai-codex"))
    from types import SimpleNamespace
    from tools.delegate_tool_config import _load_config, _resolve_profile_execution
    from tools.delegate_tool_auth import bind_child_authority
    parent = SimpleNamespace(request_overrides=None)
    cfg = _load_config()
    creds, reasoning = _resolve_profile_execution(cfg, "codex-restricted", parent)
    route = dict(creds)
    authority = bind_child_authority(
        route, parent_agent=parent, pool=None, key_origin=creds.get("key_origin"),
        key_source=creds.get("key_source"), key_auth_type=creds.get("key_auth_type"),
        same_route=False, expected_auth_type=creds.get("auth_type"), profile="codex-restricted")
    return {**creds, "api_key": route.get("api_key")}, reasoning, authority


# ── 1-4: the delegation chain resolves the real route; admission before/after ──

def test_codex_oauth_profile_resolves_through_real_delegation_chain(restricted_service, monkeypatch):
    _write_profile(None, CODEX_PROFILE_YAML)
    creds, reasoning, authority = _resolve_codex_route(monkeypatch)
    assert creds["provider"] == "openai-codex"
    assert creds["api_mode"] == "codex_responses"
    assert creds["base_url"] == CODEX_BASE
    assert creds["auth_type"] == "oauth"
    assert creds["api_key"] == CODEX_SECRET  # OAuth credential bound by the real auth path


def test_bound_authority_records_oauth(restricted_service, monkeypatch):
    _write_profile(None, CODEX_PROFILE_YAML)
    _, _, authority = _resolve_codex_route(monkeypatch)
    assert authority.auth_type == "oauth"
    assert authority.endpoint == CODEX_BASE


def test_codex_tool_free_route_admitted_after_fix(restricted_service, monkeypatch):
    """[RED at 713c677695] The exact trusted route tuple is admitted."""
    _write_profile(None, CODEX_PROFILE_YAML)
    creds, reasoning, authority = _resolve_codex_route(monkeypatch)
    restricted._validate_tool_free_route(creds)  # raises at the starting head
    agent = restricted._new_restricted_agent(
        None, creds, reasoning, authority, _api_server=None, envelope="hermes_tool_free_v1")
    # Retains codex_responses: no forced chat_completions conversion.
    assert agent.api_mode == "codex_responses"
    assert agent._restricted_wire_binding == (
        "openai-codex", agent.model, CODEX_BASE, "codex_responses")


# ── 6-10: endpoint pinning (canonical URL parsing, never substring) ──

@pytest.mark.parametrize("base_url", [
    "https://api.openai.com/v1",                       # foreign hostname
    "https://evil.invalid/backend-api/codex",          # foreign hostname, same path
    "https://chatgpt.com.evil.invalid/backend-api/codex",  # suffix confusion
    "https://notchatgpt.com/backend-api/codex",        # prefix confusion
    "http://chatgpt.com/backend-api/codex",            # scheme drift
    "https://chatgpt.com:8443/backend-api/codex",      # unexpected port
    "https://user:token@chatgpt.com/backend-api/codex",  # userinfo authority
    "https://chatgpt.com/backend-api/codex/extra",     # unexpected path
    "https://chatgpt.com/backend-api",                 # truncated path
    "https://chatgpt.com/backend-api/codex?x=1",       # query
    "https://chatgpt.com/backend-api/codex#f",         # fragment
])
def test_codex_route_rejects_endpoint_drift(restricted_service, base_url):
    with pytest.raises(RuntimeError, match="not enforceable"):
        restricted._validate_tool_free_route({**CODEX_CREDS, "base_url": base_url})


def test_codex_route_rejects_trailing_slash_variant_is_admitted(restricted_service):
    """Only proven-equivalent spellings: a trailing slash is the same path."""
    restricted._validate_tool_free_route({**CODEX_CREDS, "base_url": CODEX_BASE + "/"})


# ── 11-13: auth-type and provider confusion ──

def test_codex_route_rejects_api_key_auth_type(restricted_service):
    with pytest.raises(RuntimeError, match="not enforceable"):
        restricted._validate_tool_free_route({**CODEX_CREDS, "auth_type": "api_key"})


def test_codex_route_rejects_missing_auth_type(restricted_service):
    with pytest.raises(RuntimeError, match="not enforceable"):
        restricted._validate_tool_free_route(
            {k: v for k, v in CODEX_CREDS.items() if k != "auth_type"})


def test_codex_dialect_rejects_bearer_authority_at_construction(restricted_service, monkeypatch):
    """An api_key-resolved authority cannot satisfy the OAuth dialect at construction."""
    _write_profile(None, CODEX_PROFILE_YAML)
    creds, reasoning, _authority = _resolve_codex_route(monkeypatch)
    from types import SimpleNamespace
    api_key_authority = SimpleNamespace(auth_type="api_key")
    with pytest.raises(RuntimeError, match="not enforceable"):
        restricted._new_restricted_agent(
            None, creds, reasoning, api_key_authority, _api_server=None,
            envelope="hermes_tool_free_v1")


def test_oauth_credential_does_not_authorize_provider_openai(restricted_service):
    """The Codex OAuth route is not an openai route; openai stays the trusted-table route."""
    with pytest.raises(RuntimeError, match="restricted"):
        restricted._validate_tool_free_route(
            {**CODEX_CREDS, "provider": "openai", "base_url": "https://api.openai.com/v1"})


def test_missing_oauth_credential_fails_closed(restricted_service, monkeypatch):
    """bind_child_authority fails closed when no OAuth login exists for openai-codex."""
    from types import SimpleNamespace
    from hermes_cli.runtime_provider import AuthError
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "resolve_oauth_store_runtime",
                        lambda *a, **k: (_ for _ in ()).throw(AuthError("openai-codex_auth_missing")))
    from tools.delegate_tool_auth import bind_child_authority, DelegationAuthError
    with pytest.raises(DelegationAuthError):
        bind_child_authority(
            dict(CODEX_CREDS), parent_agent=SimpleNamespace(request_overrides=None),
            pool=None, key_origin=None, key_source=None, key_auth_type=None,
            same_route=False, expected_auth_type="oauth", profile="codex-restricted")


# ── provider/API-mode/model drift ──

@pytest.mark.parametrize("overrides", [
    {"provider": "custom"},
    {"provider": "openai"},
    {"api_mode": "chat_completions"},
    {"api_mode": "anthropic_messages"},
    {"request_overrides": {"model": "other"}},
    {"fallback_providers": ["openai"]},
    {"command": "agentic-provider"},
    {"model": "  "},
])
def test_codex_route_rejects_identity_and_capability_drift(restricted_service, overrides):
    with pytest.raises(RuntimeError, match="not enforceable"):
        restricted._validate_tool_free_route({**CODEX_CREDS, **overrides})


# ── 17-21: the outbound wire contract (dialect-native allowlist) ──

def _codex_agent(restricted_service):
    from types import SimpleNamespace
    agent = SimpleNamespace(
        provider="openai-codex", model="gpt-5-codex", base_url=CODEX_BASE,
        api_mode="codex_responses", tools=[], valid_tool_names=set(), enabled_toolsets=[])
    agent._restricted_wire_binding = (agent.provider, agent.model, agent.base_url, agent.api_mode)
    return agent


def _codex_wire_kwargs(agent):
    return {
        "model": agent.model,
        "instructions": "Bounded evidence only.",
        "input": [{"role": "user", "content": "snapshot", "type": "message"}],
        "store": False,
        "prompt_cache_key": "pck_0123456789abcdef01234567",
        "reasoning": {"effort": "medium", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "timeout": 600.0,
        "extra_headers": {"session_id": "sess-1", "x-client-request-id": "pck_0123456789"},
    }


def test_codex_wire_accepts_minimal_justified_shape(restricted_service):
    agent = _codex_agent(restricted_service)
    restricted._validate_codex_tool_free_wire(agent, _codex_wire_kwargs(agent))


@pytest.mark.parametrize("field,value", [
    ("tools", [{"type": "function", "name": "terminal"}]),
    ("tool_choice", "required"),
    ("tool_choice", "auto"),
    ("parallel_tool_calls", True),
    ("web_search_options", {}),
    ("hosted_tools", [{"type": "web_search"}]),
    ("computer_use", {}),
    ("extra_body", {"arbitrary": "field"}),
    ("extra_body", {"tools": []}),
    ("context_management", [{"type": "compaction"}]),
    ("metadata", {"trace": "x"}),
    ("service_tier", "priority"),
    ("service_tier", "auto"),
    ("fallbacks", [{"provider": "openai"}]),
    ("api_key", "injected"),
    ("authorization", "Bearer x"),
    ("stream", True),
    ("unknown_field", 1),
    ("temperature", 0.7),
    ("max_output_tokens", 4096),
    ("text", {"verbosity": "low"}),
])
def test_codex_wire_rejects_capability_and_unknown_fields(restricted_service, field, value):
    agent = _codex_agent(restricted_service)
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_codex_tool_free_wire(agent, {**_codex_wire_kwargs(agent), field: value})


def test_codex_wire_rejects_arbitrary_headers(restricted_service):
    agent = _codex_agent(restricted_service)
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_codex_tool_free_wire(agent, {**_codex_wire_kwargs(agent),
            "extra_headers": {"session_id": "s", "x-custom": "v"}})
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_codex_tool_free_wire(agent, {**_codex_wire_kwargs(agent),
            "extra_headers": {"Authorization": "Bearer injected"}})


def test_codex_wire_rejects_store_true_and_model_drift(restricted_service):
    agent = _codex_agent(restricted_service)
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_codex_tool_free_wire(agent, {**_codex_wire_kwargs(agent), "store": True})
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_codex_tool_free_wire(agent, {**_codex_wire_kwargs(agent), "model": "other"})


def test_codex_wire_rejects_route_drift_and_restored_tools(restricted_service):
    agent = _codex_agent(restricted_service)
    drifted = _codex_agent(restricted_service)
    drifted.model = "gpt-other"
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_tool_free_wire(drifted, _codex_wire_kwargs(agent))
    contaminated = _codex_agent(restricted_service)
    contaminated.tools = [{"type": "function"}]
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_tool_free_wire(contaminated, _codex_wire_kwargs(agent))
    fallback = _codex_agent(restricted_service)
    fallback._fallback_activated = True
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_tool_free_wire(fallback, _codex_wire_kwargs(agent))


def test_chat_completions_wire_policy_unchanged_and_not_unioned(restricted_service):
    """The chat allowlist is NOT widened by the Codex dialect and vice versa."""
    agent = _codex_agent(restricted_service)
    chat_agent = type(agent)(provider="deepinfra", model="Qwen/Qwen3.8-Flash",
                             base_url="https://api.deepinfra.com/v1/openai",
                             api_mode="chat_completions", tools=[], valid_tool_names=set(),
                             enabled_toolsets=[])
    chat_agent._restricted_wire_binding = (chat_agent.provider, chat_agent.model,
                                           chat_agent.base_url, chat_agent.api_mode)
    chat_safe = {"model": chat_agent.model,
                 "messages": [{"role": "user", "content": "snapshot"}]}
    restricted._validate_tool_free_wire(chat_agent, chat_safe)
    # Codex-dialect fields are NOT valid on the chat wire:
    for field in ("instructions", "input", "store", "reasoning", "include"):
        with pytest.raises(RuntimeError, match="outbound request"):
            restricted._validate_tool_free_wire(chat_agent, {**chat_safe, field: 1})
    # Chat fields are NOT valid on the Codex wire:
    for field in ("messages", "temperature", "max_tokens", "response_format"):
        with pytest.raises(RuntimeError, match="outbound request"):
            restricted._validate_codex_tool_free_wire(agent, {**_codex_wire_kwargs(agent), field: 1})


# ── 22-24: construction tool boundary on the new dialect ──

def test_codex_restricted_construction_zero_tools_and_binding(restricted_service, monkeypatch):
    _write_profile(None, CODEX_PROFILE_YAML)
    creds, reasoning, authority = _resolve_codex_route(monkeypatch)
    agent = restricted._new_restricted_agent(
        None, creds, reasoning, authority, _api_server=None, envelope="hermes_tool_free_v1")
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent.enabled_toolsets == []
    assert agent.disabled_toolsets  # every toolset disabled
    assert agent.api_mode == "codex_responses"
    assert getattr(agent, "_disable_streaming", False) is True


def test_codex_restricted_agent_turn_detects_late_tool_contamination(restricted_service, monkeypatch):
    _write_profile(None, CODEX_PROFILE_YAML)
    creds, reasoning, authority = _resolve_codex_route(monkeypatch)
    agent = restricted._new_restricted_agent(
        None, creds, reasoning, authority, _api_server=None, envelope="hermes_tool_free_v1")
    agent.tools = [{"type": "function", "function": {"name": "terminal"}}]
    import contextlib

    @contextlib.contextmanager
    def _profile_scope(_profile):
        yield

    with pytest.raises(RuntimeError, match="unexpectedly has tools"):
        restricted._restricted_agent_turn_scoped(
            type("S", (), {"_profile_scope": staticmethod(_profile_scope)})(),
            agent, "input", None)


def test_codex_init_guard_binds_codex_dialect_during_construction(restricted_service, monkeypatch):
    """The R3 construction-local guard carries the codex_responses wire identity."""
    import agent.agent_init as agent_init
    real_policy = agent_init._apply_openai_header_policy
    observed = {}

    def spy_policy(agent, kwargs):
        from agent.restricted_init_guard import _restricted_init_binding
        observed["carried"] = _restricted_init_binding.get()
        return real_policy(agent, kwargs)

    monkeypatch.setattr(agent_init, "_apply_openai_header_policy", spy_policy)
    _write_profile(None, CODEX_PROFILE_YAML)
    creds, reasoning, authority = _resolve_codex_route(monkeypatch)
    agent = restricted._new_restricted_agent(
        None, creds, reasoning, authority, _api_server=None, envelope="hermes_tool_free_v1")
    carried = observed.get("carried")
    assert carried is not None, "init-time guard was not observable during construction"
    assert carried[1][3] == "codex_responses"
    assert carried[1][0] == "openai-codex"


def test_codex_request_time_base_url_override_rejected_by_client_chokepoint(
        restricted_service, monkeypatch):
    _write_profile(None, CODEX_PROFILE_YAML)
    creds, reasoning, authority = _resolve_codex_route(monkeypatch)
    agent = restricted._new_restricted_agent(
        None, creds, reasoning, authority, _api_server=None, envelope="hermes_tool_free_v1")
    drifted = dict(agent._client_kwargs)
    drifted["base_url"] = "https://evil.invalid/v1"
    with pytest.raises(RuntimeError, match="route drift"):
        agent._create_openai_client(drifted, reason="binding-pin", shared=True)


# ── 28: credential secrecy ──

def test_codex_safe_identity_contains_no_credential(restricted_service, monkeypatch):
    from gateway.platforms import api_server
    _write_profile(None, CODEX_PROFILE_YAML)
    creds, reasoning, authority = _resolve_codex_route(monkeypatch)
    agent = restricted._new_restricted_agent(
        None, creds, reasoning, authority, _api_server=None, envelope="hermes_tool_free_v1")
    identity = restricted._identity(None, "codex-restricted", creds,
                                    {"provider": "openai-codex", "model": "gpt-5-codex",
                                     "base_url": CODEX_BASE, "api_mode": "codex_responses",
                                     "auth_type": "oauth"}, authority, agent=agent,
                                    _api_server=api_server)
    assert identity["auth_type"] == "oauth"
    assert identity["endpoint_identity"] == "https://chatgpt.com"
    assert CODEX_SECRET not in json.dumps(identity)


# ── 25-27 regressions: ordinary Codex / chat tool-free / DeepInfra tool-free ──
# (existing suites cover these; explicit pins here for the issue matrix.)

def test_ordinary_codex_transport_still_permits_tools_and_extra_body():
    """Ordinary (non-restricted) Codex preflight behavior is unchanged."""
    from agent.codex_responses_adapter import _preflight_codex_api_kwargs
    normalized = _preflight_codex_api_kwargs({
        "model": "gpt-5-codex", "instructions": "sys", "input": [],
        "tools": [{"type": "function", "name": "terminal", "parameters": {}}],
        "tool_choice": "auto", "extra_body": {"custom": 1},
    })
    assert normalized["tools"] and normalized["tool_choice"] == "auto"
    assert normalized["extra_body"] == {"custom": 1}


def test_chat_completions_tool_free_regression(restricted_service):
    from gateway.platforms import api_server
    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(adapter, {
        "provider": "deepinfra", "model": "Qwen/Qwen3.8-Flash", "api_key": "test-key",
        "base_url": "https://api.deepinfra.com/v1/openai", "api_mode": "chat_completions",
    }, None, _api_server=api_server, envelope="hermes_tool_free_v1")
    assert agent._restricted_wire_binding[3] == "chat_completions"


def test_deepinfra_tool_free_regression_via_validate(restricted_service):
    creds = {"provider": "deepinfra", "model": "Qwen/Qwen3.8-Flash", "api_key": "k",
             "base_url": "https://api.deepinfra.com/v1/openai", "api_mode": "chat_completions"}
    restricted._validate_tool_free_route(creds)  # still admitted, unchanged
    with pytest.raises(RuntimeError, match="not enforceable"):
        restricted._validate_tool_free_route({**creds, "api_mode": "codex_responses"})


# ── End-to-end gateway test: real HTTP surface, only the external boundary faked ──

@pytest.mark.asyncio
async def test_codex_restricted_gateway_end_to_end(restricted_service, monkeypatch):
    """Full POST /v1/restricted-runs -> real agent -> real Codex transport -> fake wire.

    Only the external HTTP boundary is stubbed (a loopback Responses SSE server from
    tests/fakes/providers) and the credential rungs; the gateway handler, profile
    resolution, authority binding, restricted construction, the R3 guard, the
    conversation loop, the Codex transport and the restricted wire gate all run for
    real. The trusted endpoint table is repointed at the loopback origin the same way
    the gateway pins the canonical ChatGPT/Codex origin in production.
    """
    import asyncio
    from tests.fakes.providers.openai_responses import FakeResponsesServer
    from hermes_constants import get_hermes_home

    client, adapter = restricted_service
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _codex_runtime)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_oauth_store_runtime",
                        lambda *a, **k: _codex_runtime(requested="openai-codex"))

    with FakeResponsesServer(api_key=CODEX_SECRET) as srv:
        origin = srv.base_url.rstrip("/")
        from urllib.parse import urlsplit as _us
        _o = _us(origin)
        # Repoint the trusted-origin table AND the runtime's endpoint at the loopback
        # fake (canonical parsing of the fake's own origin; production stays pinned to
        # the canonical ChatGPT/Codex origin in the module constants).
        monkeypatch.setattr(restricted, "_CODEX_TRUSTED_ORIGIN",
                            (_o.scheme, _o.hostname, _o.port, _o.path.rstrip("/")))
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                            lambda **kw: _codex_runtime(base_url=origin, **kw))
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_oauth_store_runtime",
                            lambda *a, **k: _codex_runtime(requested="openai-codex", base_url=origin))
        # No pinned base_url: the endpoint comes from runtime resolution, exactly the
        # production shape (a base_url in the profile would take the direct-endpoint
        # branch and reclassify the provider).
        _write_profile(get_hermes_home(), """    codex-restricted:
      provider: openai-codex
      model: gpt-5-codex
      auth_type: oauth
      api_mode: codex_responses
      restricted_tool_free: true
""")

        payload = {"delegation_profile_id": "codex-restricted", "work_class": "context_gather",
                   "input": "Analyze this supplied snapshot only.",
                   "capability_envelope": "hermes_tool_free_v1"}
        headers = {"Authorization": "Bearer restricted-scope-key-0123456789",
                   "Idempotency-Key": "codex-e2e-1"}
        response = await client.post("/v1/restricted-runs", json=payload, headers=headers)
        assert response.status == 202, await response.text()
        accepted = await response.json()
        assert accepted["replayed"] is False
        identity = accepted["resolved_identity"]
        assert identity["resolved_provider"] == "openai-codex"
        assert identity["auth_type"] == "oauth"
        assert CODEX_SECRET not in json.dumps(identity)

        status = {}
        deadline = asyncio.get_event_loop().time() + 30
        while asyncio.get_event_loop().time() < deadline:
            polled = await client.get(f"/v1/runs/{accepted['run_id']}",
                                      headers={"Authorization": "Bearer test-gateway-key"})
            status = await polled.json()
            if status.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
                break
            await asyncio.sleep(0.05)
        assert status.get("status") == "completed", status
        assert status.get("output") == "Fake title"
        assert CODEX_SECRET not in json.dumps(status)

        bodies = [r["body"] for r in srv.requests if r["path"].rstrip("/").endswith("/responses")]
        assert bodies, "no request reached the Responses wire"
        allowed = {"model", "instructions", "input", "store", "reasoning", "include",
                   "prompt_cache_key", "stream"}
        for body in bodies:
            assert set(body) <= allowed, sorted(set(body) - allowed)
            assert body.get("store") is False
            assert body.get("model") == "gpt-5-codex"
            assert not body.get("tools") and not body.get("tool_choice")
        assert srv.invalid_requests() == []
        assert all(r["headers"].get("authorization") == f"Bearer {CODEX_SECRET}"
                   for r in srv.requests if r["path"].rstrip("/").endswith("/responses"))

        # Idempotent replay preserves the exact route identity (no new run).
        replay = await client.post("/v1/restricted-runs", json=payload, headers=headers)
        assert replay.status == 202, await replay.text()
        replayed = await replay.json()
        assert replayed["replayed"] is True
        assert replayed["resolved_identity"] == identity
