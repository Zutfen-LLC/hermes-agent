"""Delegated children bind to one authentication authority, never to an opaque inherited string (ops-supervisor#216).

Reproducer (red at b5201c3): an ``openai-codex`` child whose inherited token no OAuth pool entry carries, with no
OAuth login store, was constructed and dispatched on that token. Every child route now resolves to an
``AuthAuthority`` (provider, endpoint, auth type, credential-source category, pool entry) before construction and
fails closed — parent untouched — when an OAuth-only provider has no canonical OAuth authority.
"""

import json
import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from tools.delegate_tool import _build_child_agent, delegate_task
from tools.delegate_tool_auth import AUTH_TYPE_MISMATCH, AUTH_UNAVAILABLE, DelegationAuthError
from tools.delegate_tool_child_run import _lease_child_credential

CODEX_URL = "https://chatgpt.com/backend-api/codex"
ZAI_URL = "https://api.z.ai/api/paas/v4"
OPENAI_URL = "https://api.openai.com/v1"
OAUTH_TOKEN = "oauth-access-token-FIXTURE"
STALE = "stale-opaque-inherited-token-FIXTURE"
SVCACCT = "sk-svcacct-FIXTURE-NOT-A-REAL-KEY"
SECRETS = (OAUTH_TOKEN, STALE, SVCACCT)


def _entry(eid, *, auth_type, token, provider="openai-codex", url=CODEX_URL, source="manual", priority=0, **kw):
    return PooledCredential(provider=provider, id=eid, label=eid, auth_type=auth_type, priority=priority,
                            source=source, access_token=token, base_url=url, **kw)


def _parent(provider, url, api_key, pool=None, entry_id=None):
    return MagicMock(
        provider=provider, base_url=url, api_key=api_key, api_mode="chat_completions", model="parent-model",
        client=None, _client_kwargs={"base_url": url, "api_key": api_key}, _credential_pool=pool,
        _credential_pool_entry_id=entry_id, acp_command=None, acp_args=[], requested_provider=provider,
        _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(), _session_db=None,
        _print_fn=None, tool_progress_callback=None, thinking_callback=None, request_overrides={},
        reasoning_config=None, _fallback_chain=None, capabilities=None, max_tokens=None, session_id="parent-sid")


def _fake_child(**kwargs):
    child = MagicMock(**{k: kwargs.get(k) for k in ("provider", "base_url", "api_key", "model")})
    child._session_init_model_config = {}
    return child


def _build(parent, **overrides):
    """Build a child through the real delegation path (AIAgent stubbed); ``(child, AIAgent mock)``."""
    with patch("run_agent.AIAgent", side_effect=_fake_child) as agent_cls:
        child = _build_child_agent(task_index=0, goal="g", context=None, toolsets=None,
                                   model=overrides.pop("model", None), max_iterations=3, task_count=1,
                                   parent_agent=parent, **overrides)
    return child, agent_cls


def _bound_key(child):
    swap = child._swap_credential
    return swap.call_args[0][0].runtime_api_key if swap.called else child.api_key


def test_oauth_child_without_canonical_authority_fails_closed_before_execution():
    """Reproducer: openai-codex child, no OAuth pool entry, no login store, inherited opaque token → refused before
    the child is constructed, with a stable non-secret code; never downgraded onto the pooled API key."""
    pool = CredentialPool("openai-codex", [_entry("svc", auth_type="api_key", token=SVCACCT)])
    with pytest.raises(DelegationAuthError) as failure:
        child, agent_cls = _build(_parent("openai-codex", CODEX_URL, STALE, pool))
        _lease_child_credential(child)  # dispatch: the last point before the child calls the provider
    assert failure.value.code == AUTH_UNAVAILABLE
    assert not any(secret in str(failure.value) for secret in SECRETS)
    assert pool._active_leases == {}

    # Through the tool: the parent gets a tool error and keeps running; no child is ever constructed.
    with patch("run_agent.AIAgent") as agent_cls:
        result = json.loads(delegate_task(goal="code it", parent_agent=_parent("openai-codex", CODEX_URL, STALE, pool)))
    agent_cls.assert_not_called()
    assert AUTH_UNAVAILABLE in result["error"]


def test_oauth_children_bind_the_parents_oauth_entry_not_an_api_key(caplog):
    """Pooled OAuth: every sibling binds the parent's concrete OAuth entry (one refresh owner), never the API-key
    entry that wins least-leased selection; the bind log names the authority without any secret."""
    pool = CredentialPool("openai-codex", [
        _entry("stale-key", auth_type="api_key", token=SVCACCT, priority=0),
        _entry("dc", auth_type="oauth", token=OAUTH_TOKEN, priority=1, source="device_code"),
    ])
    parent = _parent("openai-codex", CODEX_URL, OAUTH_TOKEN, pool, entry_id="dc")
    with caplog.at_level(logging.INFO, logger="tools.delegate_tool"):
        children = [_build(parent)[0] for _ in range(3)]
        leases = [_lease_child_credential(c)[1] for c in children]
    assert leases == ["dc"] * 3 and [_bound_key(c) for c in children] == [OAUTH_TOKEN] * 3
    authority = children[0]._auth_authority
    assert (authority.provider, authority.endpoint, authority.auth_type, authority.auth_source, authority.entry_id) == \
        ("openai-codex", "https://chatgpt.com/backend-api/codex", "oauth", "device_code", "dc")
    assert "auth_type=oauth auth_source=device_code entry=dc" in caplog.text
    assert not any(secret in caplog.text for secret in SECRETS)


def test_non_pooled_oauth_binds_the_login_store_not_the_inherited_string():
    """Singleton OAuth: with no OAuth pool entry, the child adopts the credential the provider's login store
    resolves (refresh/write-through included) — the inherited string is never used."""
    store = {"api_key": OAUTH_TOKEN, "base_url": CODEX_URL, "source": "hermes-auth-store"}
    pool = CredentialPool("openai-codex", [_entry("svc", auth_type="api_key", token=SVCACCT)])
    with patch("hermes_cli.runtime_provider.resolve_oauth_store_runtime", return_value=store) as resolve:
        child, _ = _build(_parent("openai-codex", CODEX_URL, STALE, pool))
    resolve.assert_called_once()
    assert child.api_key == OAUTH_TOKEN
    assert (child._auth_authority.auth_type, child._auth_authority.auth_source) == ("oauth", "hermes-auth-store")
    assert _lease_child_credential(child)[1] is None and child._swap_credential.call_count == 0  # never the API key


def test_parent_api_key_never_reaches_a_codex_child():
    """A parent on an API key (e.g. a request-scoped OpenAI key, pool dropped) routing a child to the Codex endpoint:
    the key is not inherited across the route change; the child uses Codex's own OAuth authority or fails closed."""
    parent = _parent("openai-api", OPENAI_URL, SVCACCT)
    route = dict(override_provider="openai-codex", override_base_url=CODEX_URL)
    with pytest.raises(DelegationAuthError) as failure:
        _build(parent, **route)
    assert failure.value.code == AUTH_UNAVAILABLE and SVCACCT not in str(failure.value)
    store = {"api_key": OAUTH_TOKEN, "base_url": CODEX_URL, "source": "hermes-auth-store"}
    with patch("hermes_cli.runtime_provider.resolve_oauth_store_runtime", return_value=store):
        child, agent_cls = _build(parent, **route)
    assert agent_cls.call_args.kwargs["api_key"] == OAUTH_TOKEN and child._auth_authority.auth_type == "oauth"


def test_explicit_routes_never_inherit_unrelated_parent_credentials():
    """Provider override: the runtime-resolved key, never the parent's. Endpoint override under an OAuth parent: the
    OAuth token is not forwarded (no-auth placeholder instead). Explicit delegation.api_key: kept as configured,
    never replaced by the parent's authority, and refused outright for an OAuth-only provider."""
    oauth_parent = _parent("openai-codex", CODEX_URL, OAUTH_TOKEN, CredentialPool("openai-codex", [
        _entry("dc", auth_type="oauth", token=OAUTH_TOKEN, source="device_code")]), entry_id="dc")

    child, agent_cls = _build(oauth_parent, override_provider="zai", override_base_url=ZAI_URL,
                              override_api_key="glm-runtime-key", override_key_origin="runtime",
                              override_key_source="env:GLM_API_KEY")
    assert agent_cls.call_args.kwargs["api_key"] == "glm-runtime-key"
    assert (child._auth_authority.auth_type, child._auth_authority.auth_source) == ("api_key", "env:GLM_API_KEY")

    child, agent_cls = _build(oauth_parent, override_provider="custom", override_base_url="https://gateway.invalid/v1")
    assert agent_cls.call_args.kwargs["api_key"] == "no-key-required" and child._auth_authority.auth_type == "none"

    key_parent = _parent("zai", ZAI_URL, "k1", CredentialPool("zai", [
        _entry("k1", auth_type="api_key", token="k1", provider="zai", url=ZAI_URL, source="env:GLM_API_KEY")]),
        entry_id="k1")
    child, agent_cls = _build(key_parent, override_base_url=ZAI_URL, override_api_key="explicit-key")
    assert agent_cls.call_args.kwargs["api_key"] == "explicit-key" and child._auth_authority.auth_source == "explicit"
    assert not isinstance(child._credential_pool, CredentialPool)  # no pool: the lease can never swap it out

    with pytest.raises(DelegationAuthError) as failure:
        _build(oauth_parent, override_provider="openai-codex", override_base_url=CODEX_URL,
               override_api_key="explicit-key")
    assert failure.value.code == AUTH_TYPE_MISMATCH and "explicit-key" not in str(failure.value)


def test_api_key_child_inherits_and_spreads_over_api_key_entries_only():
    """API-key parent → child: the parent's entry backs the child; least-leased spreading stays on compatible
    API-key entries for the child's endpoint (never an OAuth entry, never another host)."""
    pool = CredentialPool("zai", [
        _entry("k1", auth_type="api_key", token="k1", provider="zai", url=ZAI_URL, source="env:GLM_API_KEY"),
        _entry("oauth", auth_type="oauth", token="oauth-z", provider="zai", url=ZAI_URL),
        _entry("other-host", auth_type="api_key", token="k3", provider="zai", url="https://open.bigmodel.cn/api"),
        _entry("k2", auth_type="api_key", token="k2", provider="zai", url=ZAI_URL, priority=1),
    ])
    pool.acquire_lease("k1")  # the parent holds k1
    child, agent_cls = _build(_parent("zai", ZAI_URL, "k1", pool, entry_id="k1"))
    assert agent_cls.call_args.kwargs["api_key"] == "k1"
    assert (child._auth_authority.auth_type, child._auth_authority.auth_source) == ("api_key", "env:GLM_API_KEY")

    assert _lease_child_credential(child)[1] == "k2" and _bound_key(child) == "k2"


def test_external_process_and_local_routes_carry_no_pooled_credential(monkeypatch):
    """External-process providers keep their process authentication; a local/no-auth profile route gets no
    manufactured credential (not the parent's key, not an env key) and no pooled entry is ever admitted."""
    monkeypatch.setenv("OPENAI_API_KEY", SVCACCT)
    parent = _parent("zai", ZAI_URL, "k1")
    child, _ = _build(parent, override_provider="copilot-acp", override_base_url="acp://copilot")
    assert child._auth_authority.auth_type == "external_process"

    child, agent_cls = _build(parent, override_provider="custom", override_base_url="http://127.0.0.1:11434/v1",
                              override_profile="local_light", override_auth_type="none")
    assert agent_cls.call_args.kwargs["api_key"] == "no-key-required"
    assert (child._auth_authority.auth_type, child._auth_authority.profile) == ("none", "local_light")
    probe = _entry("any", auth_type="api_key", token="k", provider="custom", url="http://127.0.0.1:11434/v1")
    assert not child._auth_authority.admits(probe, "http://127.0.0.1:11434/v1")


def test_child_secrets_stay_out_of_logs_errors_and_persisted_state(caplog):
    """No child credential reaches logs, tool results/errors, session metadata, or anything delegation persists
    under HERMES_HOME; the only secret-bearing hand-off is the child's own client credential."""
    from hermes_constants import get_hermes_home

    built = []

    def _completing_child(**kwargs):
        child = _fake_child(**kwargs)
        child.run_conversation.return_value = {"final_response": "done", "completed": True, "api_calls": 1,
                                               "messages": []}
        built.append(child)
        return child

    pool = CredentialPool("openai-codex", [_entry("dc", auth_type="oauth", token=OAUTH_TOKEN, source="device_code"),
                                           _entry("svc", auth_type="api_key", token=SVCACCT)])
    orphan_pool = CredentialPool("openai-codex", [_entry("svc", auth_type="api_key", token=SVCACCT)])
    with caplog.at_level(logging.DEBUG), patch("run_agent.AIAgent", side_effect=_completing_child):
        result = delegate_task(goal="code it", parent_agent=_parent("openai-codex", CODEX_URL, OAUTH_TOKEN, pool,
                                                                   entry_id="dc"))
        failed = delegate_task(goal="code it", parent_agent=_parent("openai-codex", CODEX_URL, STALE, orphan_pool))
    assert len(built) == 1 and built[0].api_key == OAUTH_TOKEN
    assert built[0]._session_init_model_config["_delegate_auth"] == {
        "provider": "openai-codex", "endpoint": "https://chatgpt.com/backend-api/codex", "auth_type": "oauth",
        "auth_source": "device_code", "entry_id": "dc"}
    metadata = json.dumps(built[0]._session_init_model_config)
    persisted = "".join(p.read_text(errors="ignore") for p in get_hermes_home().rglob("*") if p.is_file())
    for text in (caplog.text, result, failed, metadata, persisted):
        assert not any(secret in text for secret in SECRETS)


PARENT_KEY = "PARENT-GLM-SECRET-FIXTURE"
TARGET_KEY = "TARGET-KEY-FIXTURE"


@pytest.mark.parametrize("route, expected", [
    ({"override_provider": "custom", "override_base_url": "https://foreign.example/v1"}, "no-key-required"),
    ({"override_provider": "custom", "override_base_url": "http://127.0.0.1:11434/v1"}, "no-key-required"),
    ({"override_base_url": "https://foreign.example/v1"}, None),
    ({"override_base_url": ZAI_URL}, None),
    ({"override_provider": "deepseek", "override_base_url": "https://api.deepseek.com/v1"}, None),
    ({"override_provider": "deepseek", "override_base_url": "https://api.deepseek.com/v1",
      "override_api_key": TARGET_KEY, "override_key_origin": "runtime", "override_key_source": "env:DEEPSEEK_API_KEY"}, TARGET_KEY),
    ({"override_base_url": "https://foreign.example/v1", "override_api_key": TARGET_KEY}, TARGET_KEY),
    ({"override_profile": "isolated"}, None),
    ({}, PARENT_KEY),
])
def test_parent_credential_requires_unchanged_authenticated_route(route, expected, caplog):
    from tools.delegate_tool_config import _resolve_child_runtime
    from hermes_constants import get_hermes_home

    parent = _parent("zai", ZAI_URL, PARENT_KEY)
    with caplog.at_level(logging.DEBUG), patch("run_agent.AIAgent", side_effect=_fake_child) as constructor:
        if expected is None:
            with pytest.raises(DelegationAuthError) as failure:
                _build_child_agent(0, "g", None, None, None, 3, 1, parent, **route)
            constructor.assert_not_called()
            assert failure.value.code == AUTH_UNAVAILABLE
            assert PARENT_KEY not in str(failure.value)
        else:
            child = _build_child_agent(0, "g", None, None, None, 3, 1, parent, **route)
            assert constructor.call_args.kwargs["api_key"] == expected
            assert PARENT_KEY not in json.dumps(child._session_init_model_config)
            assert PARENT_KEY not in json.dumps(child._auth_authority.as_metadata())
    assert PARENT_KEY not in caplog.text
    persisted = "".join(p.read_text(errors="ignore") for p in get_hermes_home().rglob("*") if p.is_file())
    assert PARENT_KEY not in persisted
    runtime = _resolve_child_runtime(
        parent, {}, PARENT_KEY, model=None, override_provider=route.get("override_provider"),
        override_base_url=route.get("override_base_url"), override_api_key=route.get("override_api_key"),
        override_api_mode=None, override_acp_command=None, override_acp_args=None,
        override_profile=route.get("override_profile"))
    assert runtime["api_key"] == (PARENT_KEY if not route else route.get("override_api_key"))


@pytest.mark.parametrize("case", ["pool", "frozen-scheme", "frozen-provider", "oauth", "external", "cloud", "profile"])
def test_target_authority_is_independent_of_parent_key(case, caplog, monkeypatch, tmp_path):
    from agent.auth_authority import AuthAuthority
    from tools.delegate_tool_config import _resolve_profile_execution

    parent = _parent("zai", ZAI_URL, PARENT_KEY)
    route = {"override_provider": "deepseek", "override_base_url": "https://api.deepseek.com/v1"}
    expected = TARGET_KEY
    if case == "pool":
        pool = CredentialPool("deepseek", [_entry("target", auth_type="api_key", token=TARGET_KEY,
                              provider="deepseek", url=route["override_base_url"])])
        monkeypatch.setattr("tools.delegate_tool._resolve_child_credential_pool", lambda *a, **k: pool)
    elif case.startswith("frozen"):
        parent._auth_authority = AuthAuthority.for_route(
            "deepseek" if case == "frozen-provider" else "zai", ZAI_URL, "api_key", "fixture")
        if case == "frozen-scheme":
            parent.base_url = ZAI_URL.replace("https:", "http:")
            parent._client_kwargs["base_url"] = parent.base_url
        route, expected = {}, None
    elif case == "oauth":
        route = {"override_provider": "openai-codex", "override_base_url": CODEX_URL}
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_oauth_store_runtime",
                            lambda *a, **k: {"api_key": TARGET_KEY, "base_url": CODEX_URL, "source": "hermes-auth-store"})
    elif case in ("external", "cloud"):
        route = {"override_provider": "copilot-acp" if case == "external" else "bedrock"}
        expected = "no-key-required"
    else:
        # Resolve real provider credentials in two isolated homes, then return to the first home.
        cfg = {"profiles": {"target": {"provider": "deepseek", "auth_type": "api_key"}}}
        for home, key in ((tmp_path / "a", TARGET_KEY), (tmp_path / "b", "SECOND-TARGET-FIXTURE"),
                          (tmp_path / "a", TARGET_KEY)):
            home.mkdir(exist_ok=True)
            monkeypatch.setenv("HERMES_HOME", str(home))
            monkeypatch.setenv("DEEPSEEK_API_KEY", key)
            creds, _ = _resolve_profile_execution(cfg, "target", parent)
            child, constructor = _build(parent, override_provider=creds["provider"],
                override_base_url=creds["base_url"], override_api_key=creds["api_key"],
                override_key_origin=creds["key_origin"], override_key_source=creds["key_source"],
                override_profile=creds["profile"], override_auth_type=creds["auth_type"])
            assert constructor.call_args.kwargs["api_key"] == key
            assert child._auth_authority.profile == "target"
            assert PARENT_KEY not in json.dumps(child._session_init_model_config)
        return
    with caplog.at_level(logging.DEBUG), patch("run_agent.AIAgent", side_effect=_fake_child) as constructor:
        if expected is None:
            with pytest.raises(DelegationAuthError) as failure:
                _build_child_agent(0, "g", None, None, None, 3, 1, parent, **route)
            constructor.assert_not_called()
            assert PARENT_KEY not in str(failure.value)
        else:
            child = _build_child_agent(0, "g", None, None, None, 3, 1, parent, **route)
            assert constructor.call_args.kwargs["api_key"] == expected
            assert PARENT_KEY not in json.dumps(child._session_init_model_config)
    assert PARENT_KEY not in caplog.text
