"""Delegated children bind to the parent's authentication authority, not to any pool entry (#216 prereq).

Reproducer: an ``openai-codex`` parent running on its OAuth device_code credential shares a pool that also holds a
stale api_key-typed entry. Least-leased selection handed that entry to the child, which then sent an
API-key-shaped value to the Codex OAuth endpoint (HTTP 401) while the parent kept working.
"""

import logging
from unittest.mock import MagicMock

from agent.credential_pool import CredentialPool, PooledCredential
from tools.delegate_tool_child_run import _lease_child_credential

CODEX_URL = "https://chatgpt.com/backend-api/codex"
ZAI_URL = "https://api.z.ai/api/paas/v4"
OAUTH_TOKEN = "oauth-access-token-fixture"
STALE_KEY = "sk-svcacct-FIXTURE-NOT-A-REAL-KEY"


def _entry(eid, *, auth_type, token, priority, provider="openai-codex", url=CODEX_URL, source="manual"):
    return PooledCredential(provider=provider, id=eid, label=eid, auth_type=auth_type, priority=priority,
                            source=source, access_token=token, base_url=url)


def _child(pool, *, api_key, provider="openai-codex", url=CODEX_URL, entry_id=None):
    return MagicMock(provider=provider, base_url=url, api_key=api_key, model="gpt-test",
                     _credential_pool=pool, _credential_pool_entry_id=entry_id, _pinned_auth_type=None)


def _bound_key(child):
    swap = child._swap_credential
    return swap.call_args[0][0].runtime_api_key if swap.called else child.api_key


def test_oauth_child_never_binds_an_api_key_entry(caplog):
    """Invariant: an OAuth child ends on OAuth credentials — the parent's own entry when the pool holds it (siblings
    included), its inherited token otherwise — and the bind log names the route without any secret."""
    pool = CredentialPool("openai-codex", [  # the api_key entry wins least-leased selection on priority
        _entry("stale-key", auth_type="api_key", token=STALE_KEY, priority=0),
        _entry("oauth", auth_type="oauth", token=OAUTH_TOKEN, priority=1, source="device_code"),
    ])
    siblings = [_child(pool, api_key=OAUTH_TOKEN) for _ in range(3)]
    with caplog.at_level(logging.INFO, logger="tools.delegate_tool"):
        leases = [_lease_child_credential(c)[1] for c in siblings]
    assert leases == ["oauth"] * 3 and [_bound_key(c) for c in siblings] == [OAUTH_TOKEN] * 3
    assert all(c._pinned_auth_type == "oauth" for c in siblings)
    for fact in ("provider=openai-codex", "auth_type=oauth", "auth_source=device_code", "endpoint=chatgpt.com"):
        assert fact in caplog.text
    assert OAUTH_TOKEN not in caplog.text and STALE_KEY not in caplog.text

    # The parent's singleton token is not pooled: nothing is leased, the child keeps its token, pinned to OAuth.
    key_only = CredentialPool("openai-codex", [_entry("stale-key", auth_type="api_key", token=STALE_KEY, priority=0)])
    orphan = _child(key_only, api_key="singleton-token")
    assert _lease_child_credential(orphan)[1] is None
    assert _bound_key(orphan) == "singleton-token" and orphan._pinned_auth_type == "oauth"
    assert key_only._active_leases == {}


def test_api_key_child_spreads_across_api_key_entries_only():
    """Invariant: API-key children keep least-leased rate-limit spreading, but never onto an OAuth entry."""
    pool = CredentialPool("zai", [
        _entry("k1", auth_type="api_key", token="k1", priority=0, provider="zai", url=ZAI_URL),
        _entry("oauth", auth_type="oauth", token="oauth-z", priority=0, provider="zai", url=ZAI_URL),
        _entry("k2", auth_type="api_key", token="k2", priority=1, provider="zai", url=ZAI_URL),
    ])
    pool.acquire_lease("k1")  # the parent holds k1
    child = _child(pool, api_key="k1", provider="zai", url=ZAI_URL)

    _pool, lease_id = _lease_child_credential(child)

    assert lease_id == "k2" and _bound_key(child) == "k2" and child._pinned_auth_type == "api_key"
