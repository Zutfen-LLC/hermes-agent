"""OAuth refresh semantics for delegated children (ops-supervisor#216), against a real on-disk auth store.

Children bound to a pooled OAuth authority refresh through the credential pool's existing single-use refresh
serialization (auth-store flock + in-lock resync): concurrent siblings spend the refresh token once, all adopt the
rotated credential, and the rotation is written through for the next parent/child. A child never trades a
refresh-benched or DEAD OAuth authority for an API-key entry.
"""

import base64
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli.auth_constants import AuthError
from tools.delegate_tool import _build_child_agent
from tools.delegate_tool_auth import (
    AUTH_UNAVAILABLE, OAUTH_REFRESH_CONFLICT, OAUTH_RELOGIN_REQUIRED, DelegationAuthError,
)
from tools.delegate_tool_child_run import _lease_child_credential

CODEX_URL = "https://chatgpt.com/backend-api/codex"


def _jwt(exp: float, tag: str) -> str:
    def enc(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")
    return f"{enc({'alg': 'none'})}.{enc({'exp': int(exp), 'tag': tag})}.sig"


EXPIRING = _jwt(time.time() - 60, "expiring")
FRESH = _jwt(time.time() + 7 * 86400, "fresh")


def _write_codex_store(access, refresh):
    from hermes_constants import get_hermes_home
    row = {"id": "dc", "label": "device_code", "auth_type": "oauth", "source": "device_code", "priority": 0,
           "access_token": access, "refresh_token": refresh, "base_url": CODEX_URL}
    get_hermes_home().joinpath("auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "openai-codex",
        "providers": {"openai-codex": {"tokens": {"access_token": access, "refresh_token": refresh},
                                       "last_refresh": "2026-09-01T00:00:00Z", "auth_mode": "chatgpt"}},
        "credential_pool": {"openai-codex": [row]}}), encoding="utf-8")


def _read_codex_store():
    from hermes_constants import get_hermes_home
    return json.loads(get_hermes_home().joinpath("auth.json").read_text(encoding="utf-8"))


def _parent(api_key, pool, entry_id):
    return MagicMock(
        provider="openai-codex", base_url=CODEX_URL, api_key=api_key, api_mode="codex_responses", model="m",
        client=None, _client_kwargs={"base_url": CODEX_URL, "api_key": api_key}, _credential_pool=pool,
        _credential_pool_entry_id=entry_id, acp_command=None, acp_args=[], requested_provider="openai-codex",
        _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(), _session_db=None,
        _print_fn=None, tool_progress_callback=None, thinking_callback=None, request_overrides={},
        reasoning_config=None, _fallback_chain=None, capabilities=None, max_tokens=None)


def _build(parent):
    def fake(**kw):
        child = MagicMock(**{k: kw.get(k) for k in ("provider", "base_url", "api_key", "model")})
        child._session_init_model_config = {}
        return child
    with patch("run_agent.AIAgent", side_effect=fake):
        return _build_child_agent(task_index=0, goal="g", context=None, toolsets=None, model=None,
                                  max_iterations=3, task_count=1, parent_agent=parent)


def test_child_binds_the_refreshed_entry_not_the_parents_pre_refresh_token():
    """Refresh before construction: the pool rotated the parent's entry after the parent read its token; the child
    binds the entry by id and gets the rotated credential, not the spent string it inherited."""
    pool = CredentialPool("openai-codex", [PooledCredential(
        provider="openai-codex", id="dc", label="dc", auth_type="oauth", priority=0, source="device_code",
        access_token=FRESH, refresh_token="r1", base_url=CODEX_URL)])
    child = _build(_parent(EXPIRING, pool, "dc"))
    assert child.api_key == FRESH and child._auth_authority.entry_id == "dc"


def test_concurrent_oauth_children_spend_the_refresh_token_once_and_write_through(monkeypatch):
    """Concurrent siblings on one expiring OAuth authority: exactly one refresh POST (the pool's existing
    serialization), every child adopts the rotated token, and the rotation reaches the store for the next user."""
    _write_codex_store(EXPIRING, "r0")
    posts = []
    lock = threading.Lock()

    def fake_refresh(access_token, refresh_token, **_kw):
        with lock:
            posts.append(refresh_token)
        time.sleep(0.2)  # widen the race window
        return {"access_token": FRESH, "refresh_token": "r1", "last_refresh": "2026-09-25T00:00:00Z"}

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", fake_refresh)
    pool = load_pool("openai-codex")
    [dc] = [e for e in pool.entries() if e.auth_type == "oauth"]
    parent = _parent(EXPIRING, pool, dc.id)
    children = [_build(parent) for _ in range(4)]
    assert all(c._auth_authority.entry_id == dc.id for c in children)

    barrier = threading.Barrier(len(children))
    leases = {}

    def lease(child):
        barrier.wait()
        leases[id(child)] = _lease_child_credential(child)[1]

    threads = [threading.Thread(target=lease, args=(c,)) for c in children]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert posts == ["r0"]  # one refresh, never a replay of the spent token
    assert set(leases.values()) == {dc.id}
    assert all(c._swap_credential.call_args[0][0].access_token == FRESH for c in children)
    store = _read_codex_store()
    assert store["providers"]["openai-codex"]["tokens"]["refresh_token"] == "r1"  # write-through
    [reloaded] = [e for e in load_pool("openai-codex").entries() if e.id == dc.id]
    assert reloaded.access_token == FRESH  # the next parent/child sees the rotation


@pytest.mark.parametrize("error, code", [
    (AuthError("UPSTREAM-TEXT-reused", provider="openai-codex", code="refresh_token_reused", relogin_required=True),
     OAUTH_REFRESH_CONFLICT),
    (AuthError("UPSTREAM-TEXT-revoked", provider="openai-codex", code="invalid_grant", relogin_required=True),
     OAUTH_RELOGIN_REQUIRED),
    (AuthError("UPSTREAM-TEXT-absent", provider="openai-codex", code="codex_auth_missing", relogin_required=True),
     AUTH_UNAVAILABLE),
])
def test_login_store_failures_fail_closed_with_distinct_codes(monkeypatch, error, code):
    """Refresh-token reuse, a revoked login and a missing login stay distinguishable; none leaks upstream text."""
    def raise_it(**_kw):
        raise error
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_codex_runtime_credentials", raise_it)
    with pytest.raises(DelegationAuthError) as failure:
        _build(_parent("opaque", None, None))
    assert failure.value.code == code
    assert str(error) not in str(failure.value)


def test_dead_oauth_authority_never_falls_back_to_an_api_key_entry():
    """The parent's OAuth entry is DEAD: the child moves only to a live compatible OAuth entry for its endpoint
    (the set the pool itself rotates across); with none, it fails closed — never onto the API-key entry."""
    def entry(eid, auth_type, url=CODEX_URL, status=None):
        return PooledCredential(provider="openai-codex", id=eid, label=eid, auth_type=auth_type, priority=0,
                                source="manual", access_token=f"tok-{eid}", base_url=url, last_status=status)
    dead, key = entry("dc", "oauth", status="dead"), entry("key", "api_key")
    pool = CredentialPool("openai-codex", [dead, key, entry("elsewhere", "oauth", url="https://proxy.invalid/codex"),
                                           entry("secondary", "oauth")])
    child = _build(_parent("tok-dc", pool, "dc"))
    assert child._auth_authority.entry_id == "secondary" and child.api_key == "tok-secondary"

    with pytest.raises(DelegationAuthError) as failure:
        _build(_parent("tok-dc", CredentialPool("openai-codex", [dead, key]), "dc"))
    assert failure.value.code == AUTH_UNAVAILABLE
