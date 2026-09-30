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

R3 (process-global guard race): the reviewed head activates the init-time guard by
temporarily mutating the ``AIAgent`` CLASS (``AIAgent._restricted_wire_binding = ...``).
That state is process-global, so an unrelated construction overlapping a parked
restricted construction inherits (or is suppressed by) another request's binding, and two
concurrent restricted constructions overwrite/delete each other's guard.

[RED] ``test_ordinary_construction_is_isolated_from_active_restricted_guard`` — while a
restricted construction is parked inside the client chokepoint with its guard active, an
ordinary deepinfra agent must still run its own provider profile hooks; at the reviewed
head the class-level binding suppresses them (counters stay 0).

[RED] ``test_two_restricted_constructions_with_distinct_bindings_do_not_cross_contaminate``
— two ``hermes_tool_free_v1`` constructions with different model/base_url bindings,
deterministically overlapped with a ``threading.Barrier`` inside the chokepoint: each
must observe only its own binding, neither may overwrite or remove the other's guard, and
both final per-instance bindings and non-redirecting clients must be intact. At the
reviewed head the last writer's class attribute is what the chokepoint resolves.

R4 (nested-construction inheritance): the R3 independent adversarial review found that
a NESTED ordinary construction running inside a guarded constructor's dynamic extent
(e.g. triggered from a provider-policy seam) still read the same ContextVar and
inherited the parent's binding — its provider hooks were suppressed and a differing
route would route-drift-fail. Commit e8e1370e paired the ContextVar with a single-use
construction token (the chokepoint accepts the carried binding only from the
token-holding instance), but shipped with NO committed regression: the reproducer
lived only in the reviewer's scratch probe.

[RED] ``test_nested_ordinary_construction_inside_guarded_extent_builds_unguarded`` —
while a ``hermes_tool_free_v1`` construction's init guard is active, a nested ordinary
deepinfra construction through the real ``AIAgent`` constructor must run its own
provider profile hooks, receive no ``_restricted_wire_binding``, claim/consume no
construction token, and build its normal SDK client; the parent restricted agent must
retain its exact binding, run no provider hook, and keep its guarded non-redirecting
client. At 337a7cb (ContextVar without the construction token) the nested hooks are
suppressed — the inheritance defect.

[PIN] The remaining tests pin retained behavior: the ``input_only_v1`` envelope keeps
its exact v1 initialization semantics, and the tool-free binding drives the guard
chokepoint. Every positive expectation was probe-verified against the production
functions on the reviewed head before being written.
"""
import json
import threading

import httpx
import pytest

import agent.agent_runtime_helpers as agent_runtime_helpers
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

    # Force plugin discovery FIRST so the registry is not rewritten under the hostile
    # profile, then replace the deepinfra entry (last-writer-wins, same seam a
    # $HERMES_HOME plugin uses).
    providers.get_provider_profile("deepinfra")
    hostile = HostileProfile(name="deepinfra", base_url=CREDS["base_url"], auth_type="api_key")
    monkeypatch.setitem(providers._REGISTRY, "deepinfra", hostile)
    assert providers.get_provider_profile("deepinfra") is hostile

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

_RESTRICTED_CREDS_A = dict(CREDS, model="Qwen/Qwen3.8-Flash",
                           base_url="https://api.deepinfra.com/v1/openai")
_RESTRICTED_CREDS_B = dict(CREDS, model="Qwen/Qwen3.8-Flash-B",
                           base_url="https://deepinfra-b.example/v1")


def test_nested_ordinary_construction_inside_guarded_extent_builds_unguarded(
        restricted_service, monkeypatch):
    """[RED] A nested ORDINARY construction inside an ACTIVE restricted construction's
    dynamic extent builds unguarded — and cannot disturb the parent's guard.

    The ``hermes_tool_free_v1`` factory has already minted its single-use construction
    token and set the ContextVar on the constructor's own ``Context``; the constructor is
    running on the same thread. From a production init seam
    (``agent_init._apply_openai_header_policy``, invoked from ``_build_client`` before the
    client chokepoint), an ordinary deepinfra agent is constructed through the real
    ``AIAgent`` constructor, with an observable provider profile recording both hook
    invocations. The nested agent must run its OWN hooks, carry no restricted binding,
    claim no construction token, and build the normal SDK client over the real
    ``create_openai_client`` chokepoint; the parent restricted agent must retain its exact
    binding, run no provider hook itself, and keep its guarded non-redirecting SDK client
    after the nested construction completes. No sleeps: the nesting rides the guarded
    constructor's own dynamic extent. At 337a7cb (ContextVar without the construction
    token) the nested construction reads the parent's carried binding, its hooks are
    suppressed, and it inherits the binding — the reviewer's MEDIUM finding.
    """
    import providers
    from providers import ProviderProfile

    nested_calls = {"extras": 0, "create_client": 0}

    class NestedCountingProfile(ProviderProfile):
        """Counting stand-in for the deepinfra profile: same base-class semantics,
        observable invocations (the stock ``create_client`` returns None and the normal
        SDK client is the correct outcome)."""

        def build_client_kwargs_extras(self, **context):
            nested_calls["extras"] += 1
            return {}

        def create_client(self, **client_kwargs):
            nested_calls["create_client"] += 1
            return None

    # Same plugin-discovery-first discipline as the hostile-profile proof above:
    # force discovery BEFORE replacing the registry entry, then swap the deepinfra
    # profile for the counting stand-in (the same seam a $HERMES_HOME plugin uses).
    providers.get_provider_profile("deepinfra")
    nested_profile = NestedCountingProfile(name="deepinfra", base_url=CREDS["base_url"],
                                           auth_type="api_key")
    monkeypatch.setitem(providers._REGISTRY, "deepinfra", nested_profile)
    assert providers.get_provider_profile("deepinfra") is nested_profile

    _, adapter = restricted_service
    nested: list = []
    inside_nested = [False]

    import agent.agent_init as agent_init
    from run_agent import AIAgent

    real_policy = agent_init._apply_openai_header_policy

    def nesting_policy(agent, kwargs):
        # Fire exactly once: during the PARENT restricted constructor's client build,
        # nest an ordinary construction on the same thread/context. The parent's own
        # client build happens on the restricted wire, where this policy seam is
        # reachable but the nested construction is not re-entered.
        if not inside_nested[0]:
            inside_nested[0] = True
            try:
                nested.append(AIAgent(
                    provider="deepinfra", model="Qwen/Qwen3.8-Flash", api_key="nested-key",
                    base_url=CREDS["base_url"], api_mode="chat_completions",
                    enabled_toolsets=[], disabled_toolsets=[], max_iterations=2,
                    platform="api_server", quiet_mode=True, verbose_logging=False,
                    skip_context_files=True, skip_memory=True))
            finally:
                inside_nested[0] = False
        return real_policy(agent, kwargs)

    monkeypatch.setattr(agent_init, "_apply_openai_header_policy", nesting_policy)

    parent = restricted._new_restricted_agent(
        adapter, dict(_RESTRICTED_CREDS_A), None, _api_server=api_server,
        envelope="hermes_tool_free_v1")

    # The nested construction ran (the reproducer is engaged, not vacuous).
    assert nested, "nested construction never ran"
    nested_agent = nested[0]
    # The nested ordinary agent executed its OWN provider profile hooks through the
    # real create_openai_client chokepoint despite the parent's active guard.
    assert nested_calls["extras"] > 0, (
        "nested ordinary profile build_client_kwargs_extras suppressed by the parent's "
        "active restricted guard (nested construction inherited the binding)")
    assert nested_calls["create_client"] == 1, (
        "nested ordinary profile create_client not consulted; the provider hook ladder "
        "must run normally (the stock hook returns None and the SDK client is the "
        "correct outcome)")
    # The nested agent is an ordinary agent: no restricted binding, no construction
    # token (it did not consume or claim the parent's single-use token), normal client.
    assert getattr(nested_agent, "_restricted_wire_binding", None) is None, (
        "nested ordinary agent inherited the restricted binding")
    assert getattr(nested_agent, "_restricted_construction_token", None) is None, (
        "nested ordinary agent claims a construction token")
    assert type(nested_agent.client).__name__ == "OpenAI", (
        "nested agent lost its normal SDK client")

    # The PARENT restricted agent is intact after the nested construction completed:
    # exact binding retained, no provider hook ran for its own builds, and its
    # initialization-time client is the guarded non-redirecting SDK client.
    assert parent._restricted_wire_binding == (
        _RESTRICTED_CREDS_A["provider"], _RESTRICTED_CREDS_A["model"],
        _RESTRICTED_CREDS_A["base_url"], "chat_completions")
    assert type(parent.client).__name__ == "OpenAI"
    assert parent.client._client.follow_redirects is False


def _install_restricted_construction_parking(monkeypatch, parked: dict) -> None:
    """Park every registered restricted builder thread deterministically INSIDE
    ``AIAgent.__new__`` — after ``_new_restricted_agent`` has activated its guard and
    before the constructor reaches the client chokepoint — so a sibling construction
    deterministically overlaps an ACTIVE restricted guard.

    Membership rides the builder thread's identity (registered at thread start in
    ``parked["owners"]``), never shared security state; ordinary constructions on other
    threads pass straight through.
    """
    import run_agent

    real_agent_new = run_agent.AIAgent.__new__

    def marked_new(cls, *args, **kwargs):
        if threading.current_thread().ident in parked["owners"]:
            parked["parked_ids"].add(threading.current_thread().ident)
            parked["barrier"].wait(timeout=15)
        return real_agent_new(cls)

    monkeypatch.setattr(run_agent.AIAgent, "__new__", marked_new)


def test_ordinary_construction_is_isolated_from_active_restricted_guard(restricted_service, monkeypatch):
    """[RED] An ordinary agent constructed while a restricted guard is ACTIVE keeps its
    own provider profile hooks, route, and client.

    A ``hermes_tool_free_v1`` construction is parked deterministically inside
    ``AIAgent.__new__`` — its guard is already active (class binding set) and the
    constructor has not yet reached the client chokepoint. While parked, an ordinary
    deepinfra agent is constructed through the real ``AIAgent`` constructor. The
    ordinary agent must execute its own (observable) profile extras hook, build its
    normal SDK client, and carry no restricted binding; the parked restricted agent
    must still complete with its own binding intact. At the reviewed head the
    process-global ``AIAgent._restricted_wire_binding`` class attribute is visible to
    the ordinary construction (instance lookup falls through to the class), so the
    profile extras are suppressed and the ordinary agent inherits the binding — the
    class-state defect.
    """
    import providers
    from providers import ProviderProfile

    calls = {"ordinary_extras": 0, "ordinary_create_client": 0}

    class OrdinaryProfile(ProviderProfile):
        """Counting stand-in for the deepinfra profile: same base-class semantics,
        observable invocations."""

        def build_client_kwargs_extras(self, **context):
            calls["ordinary_extras"] += 1
            return {}

        def create_client(self, **client_kwargs):
            calls["ordinary_create_client"] += 1
            return None  # base-class semantics: the SDK client is the correct outcome

    ordinary_profile = OrdinaryProfile(name="deepinfra", base_url=CREDS["base_url"],
                                       auth_type="api_key")

    # Same plugin-discovery-first discipline as the hostile-profile proof above.
    providers.get_provider_profile("deepinfra")
    monkeypatch.setitem(providers._REGISTRY, "deepinfra", ordinary_profile)
    assert providers.get_provider_profile("deepinfra") is ordinary_profile

    _, adapter = restricted_service
    parked: dict = {"owners": {}, "parked_ids": set(), "barrier": threading.Barrier(2)}
    _install_restricted_construction_parking(monkeypatch, parked)

    restricted_error = []
    restricted_agent = []

    def build_restricted():
        parked["owners"][threading.current_thread().ident] = "restricted"
        try:
            restricted_agent.append(restricted._new_restricted_agent(
                adapter, dict(_RESTRICTED_CREDS_A), None,
                _api_server=api_server, envelope="hermes_tool_free_v1"))
        except Exception as error:  # pragma: no cover - reported below
            restricted_error.append(error)

    worker = threading.Thread(target=build_restricted, daemon=True)
    worker.start()
    # Pair with the restricted builder: when this wait returns, the restricted guard is
    # ACTIVE and the construction is parked inside __new__.
    parked["barrier"].wait(timeout=15)
    assert parked["parked_ids"], "restricted construction never parked"

    # Overlapping ORDINARY construction through the real production constructor while
    # the restricted guard is active.
    from run_agent import AIAgent
    ordinary = AIAgent(provider="deepinfra", model="Qwen/Qwen3.8-Flash", api_key="ordinary-key",
                       base_url=CREDS["base_url"], api_mode="chat_completions",
                       enabled_toolsets=[], disabled_toolsets=[],
                       max_iterations=3, platform="api_server", quiet_mode=True,
                       verbose_logging=False, skip_context_files=True, skip_memory=True)

    # The ordinary agent ran its OWN provider profile hooks despite the active guard.
    assert calls["ordinary_extras"] > 0, (
        "ordinary profile build_client_kwargs_extras suppressed by the active restricted guard")
    assert calls["ordinary_create_client"] == 1, (
        "ordinary profile create_client not consulted; the provider hook ladder must run "
        "normally (the stock hook returns None and the SDK client is the correct outcome)")
    assert type(ordinary.client).__name__ == "OpenAI", "ordinary agent lost its normal client"
    assert getattr(ordinary, "_restricted_wire_binding", None) is None, (
        "ordinary agent inherited the restricted binding")
    assert ordinary.base_url == CREDS["base_url"], "ordinary agent route drifted"

    worker.join(timeout=60)
    assert not restricted_error, f"restricted construction failed: {restricted_error}"
    assert restricted_agent, "restricted construction never completed"
    expected = (_RESTRICTED_CREDS_A["provider"], _RESTRICTED_CREDS_A["model"],
                _RESTRICTED_CREDS_A["base_url"], "chat_completions")
    assert restricted_agent[0]._restricted_wire_binding == expected
    assert type(restricted_agent[0].client).__name__ == "OpenAI"
    assert restricted_agent[0].client._client.follow_redirects is False


def test_two_restricted_constructions_with_distinct_bindings_do_not_cross_contaminate(
        restricted_service, monkeypatch):
    """[RED] Two overlapping ``hermes_tool_free_v1`` constructions with distinct model /
    base_url bindings each observe ONLY their own binding at the init-time client build;
    neither can overwrite or remove the other's guard.

    Deterministic overlap: both builder threads park inside ``AIAgent.__new__`` on a
    ``threading.Barrier`` (each AFTER its guard is active), so both constructions are
    open simultaneously before either client chokepoint runs. At the reviewed head the
    process-global class attribute holds only ONE binding at a time: the second writer's
    value is what both chokepoints read (route drift for the first), and the first
    constructor's cleanup ``del`` removes the other's active guard mid-construction.
    Both constructions must complete with their own per-instance binding and a
    non-redirecting SDK client, and no provider profile hook may run for either.
    """
    import providers
    from providers import ProviderProfile

    hostile_calls = {"extras": 0, "create_client": 0}

    class CountingHostileProfile(ProviderProfile):
        """Counting hostile stand-in: both hooks recorded, base-class semantics."""

        def build_client_kwargs_extras(self, **context):
            hostile_calls["extras"] += 1
            return {}

        def create_client(self, **client_kwargs):
            hostile_calls["create_client"] += 1
            return None

    hostile_profile = CountingHostileProfile(name="deepinfra", base_url=CREDS["base_url"],
                                             auth_type="api_key")
    providers.get_provider_profile("deepinfra")
    monkeypatch.setitem(providers._REGISTRY, "deepinfra", hostile_profile)
    assert providers.get_provider_profile("deepinfra") is hostile_profile

    _, adapter = restricted_service
    parked: dict = {"owners": {}, "parked_ids": set(), "barrier": threading.Barrier(3)}
    _install_restricted_construction_parking(monkeypatch, parked)

    errors = []
    agents = {}

    def build(key, creds):
        parked["owners"][threading.current_thread().ident] = key
        try:
            agents[key] = restricted._new_restricted_agent(
                adapter, dict(creds), None, _api_server=api_server,
                envelope="hermes_tool_free_v1")
        except Exception as error:
            errors.append((key, error))

    threads = [threading.Thread(target=build, args=(key, creds), daemon=True)
               for key, creds in (("a", _RESTRICTED_CREDS_A), ("b", _RESTRICTED_CREDS_B))]
    for thread in threads:
        thread.start()
    parked["barrier"].wait(timeout=15)  # third party: release only when BOTH are parked
    assert len(parked["parked_ids"]) == 2, (
        f"expected both constructions parked, saw {parked['parked_ids']}")
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, f"restricted constructions failed: {errors}"
    assert hostile_calls == {"extras": 0, "create_client": 0}, (
        f"a restricted construction lost its guard and ran provider hooks: {hostile_calls}")
    assert len(agents) == 2

    expected = {
        "a": (_RESTRICTED_CREDS_A["provider"], _RESTRICTED_CREDS_A["model"],
              _RESTRICTED_CREDS_A["base_url"], "chat_completions"),
        "b": (_RESTRICTED_CREDS_B["provider"], _RESTRICTED_CREDS_B["model"],
              _RESTRICTED_CREDS_B["base_url"], "chat_completions"),
    }
    for key, agent in agents.items():
        assert agent._restricted_wire_binding == expected[key], (
            f"agent {key} carries binding {agent._restricted_wire_binding}")
        assert (agent.provider, agent.model, agent.base_url, agent.api_mode) == expected[key]
        assert type(agent.client).__name__ == "OpenAI"
        assert agent.client._client.follow_redirects is False
