"""Delegated children bind to the parent's authentication authority, not to any pool entry (#216 prereq).

Reproducer: an ``openai-codex`` parent running on its OAuth device_code credential shares a pool that also holds a
stale api_key-typed entry. Least-leased selection handed that entry to the child, which then sent an
API-key-shaped value to the Codex OAuth endpoint (HTTP 401) while the parent kept working.
"""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.credential_pool import CredentialPool, PooledCredential

CODEX_URL = "https://chatgpt.com/backend-api/codex"
OAUTH_TOKEN = "oauth-access-token-fixture"
STALE_KEY = "sk-svcacct-FIXTURE-NOT-A-REAL-KEY"


def _entry(eid, *, auth_type, token, priority, provider="openai-codex", url=CODEX_URL, source="manual"):
    return PooledCredential(provider=provider, id=eid, label=eid, auth_type=auth_type, priority=priority,
                            source=source, access_token=token, base_url=url)


def _codex_pool():
    # The api_key entry wins least-leased selection on priority.
    return CredentialPool("openai-codex", [
        _entry("stale-key", auth_type="api_key", token=STALE_KEY, priority=0),
        _entry("oauth", auth_type="oauth", token=OAUTH_TOKEN, priority=1, source="device_code"),
    ])


def _child(pool, *, api_key=OAUTH_TOKEN, provider="openai-codex", entry_id=None):
    return MagicMock(provider=provider, base_url=CODEX_URL, api_key=api_key, model="gpt-test",
                     _credential_pool=pool, _credential_pool_entry_id=entry_id, _pinned_auth_type=None)


class TestChildLeasesParentAuthority(unittest.TestCase):
    def test_oauth_child_does_not_lease_api_key_entry(self):
        from tools.delegate_tool_child_run import _lease_child_credential
        pool = _codex_pool()
        child = _child(pool)

        _pool, lease_id = _lease_child_credential(child)

        self.assertEqual(lease_id, "oauth")
        self.assertEqual(child._swap_credential.call_args[0][0].id, "oauth")
        self.assertEqual(child._pinned_auth_type, "oauth")
        self.assertEqual(pool._active_leases, {"oauth": 1})

    def test_child_reacquires_parent_bound_entry_by_id(self):
        from tools.delegate_tool_child_run import _lease_child_credential
        pool = CredentialPool("openai-codex", [
            _entry("oauth-a", auth_type="oauth", token="tok-a", priority=0),
            _entry("oauth-b", auth_type="oauth", token="tok-b", priority=1),
        ])
        # Parent is bound to oauth-b; the child's inherited key may lag a refresh the pool already applied.
        child = _child(pool, api_key="tok-b-before-refresh", entry_id="oauth-b")

        _pool, lease_id = _lease_child_credential(child)

        self.assertEqual(lease_id, "oauth-b")
        self.assertEqual(child._swap_credential.call_args[0][0].id, "oauth-b")

    def test_unmatched_codex_credential_pins_oauth_and_skips_api_key_entries(self):
        from tools.delegate_tool_child_run import _lease_child_credential
        pool = CredentialPool("openai-codex", [_entry("stale-key", auth_type="api_key", token=STALE_KEY, priority=0)])
        child = _child(pool, api_key="singleton-token-not-in-pool")

        _pool, lease_id = _lease_child_credential(child)

        self.assertIsNone(lease_id)
        child._swap_credential.assert_not_called()
        self.assertEqual(child._pinned_auth_type, "oauth")
        self.assertEqual(pool._active_leases, {})

    def test_api_key_parent_child_still_leases_from_pool(self):
        from tools.delegate_tool_child_run import _lease_child_credential
        url = "https://api.z.ai/api/paas/v4"
        pool = CredentialPool("zai", [
            _entry("k1", auth_type="api_key", token="k1", priority=0, provider="zai", url=url),
            _entry("k2", auth_type="api_key", token="k2", priority=1, provider="zai", url=url),
        ])
        pool.acquire_lease("k1")  # the parent holds k1: rotation-sharing spreads the child onto k2
        child = MagicMock(provider="zai", base_url=url, api_key="k1", model="glm", _credential_pool=pool,
                          _credential_pool_entry_id=None, _pinned_auth_type=None)

        _pool, lease_id = _lease_child_credential(child)

        self.assertEqual(lease_id, "k1")  # same authority is reacquired, not a different key
        self.assertEqual(child._pinned_auth_type, "api_key")

    def test_unmatched_api_key_credential_keeps_least_leased_behaviour(self):
        """No pool entry backs the credential and the provider has no single native auth kind: existing sharing."""
        from tools.delegate_tool_child_run import _lease_child_credential
        url = "https://api.z.ai/api/paas/v4"
        pool = CredentialPool("zai", [_entry("k1", auth_type="api_key", token="k1", priority=0, provider="zai", url=url)])
        child = MagicMock(provider="zai", base_url=url, api_key="env-key", model="glm", _credential_pool=pool,
                          _credential_pool_entry_id=None, _pinned_auth_type=None)

        _pool, lease_id = _lease_child_credential(child)

        self.assertEqual(lease_id, "k1")
        self.assertIsNone(child._pinned_auth_type)

    def test_concurrent_oauth_children_share_one_authority(self):
        """Siblings bind the same OAuth entry, so refresh stays serialized through the pool's single entry."""
        from tools.delegate_tool_child_run import _lease_child_credential
        pool = _codex_pool()
        children = [_child(pool) for _ in range(3)]

        leases = [_lease_child_credential(c)[1] for c in children]

        self.assertEqual(leases, ["oauth"] * 3)
        self.assertEqual(pool._active_leases, {"oauth": 3})

    def test_bind_log_carries_route_facts_and_no_secret(self):
        from tools.delegate_tool_child_run import _lease_child_credential
        child = _child(_codex_pool())

        with self.assertLogs("tools.delegate_tool", level=logging.INFO) as logs:
            _lease_child_credential(child)

        text = "\n".join(logs.output)
        for fact in ("provider=openai-codex", "model=gpt-test", "auth_type=oauth", "auth_source=device_code",
                     "endpoint=chatgpt.com"):
            self.assertIn(fact, text)
        self.assertNotIn(OAUTH_TOKEN, text)
        self.assertNotIn(STALE_KEY, text)


class TestSwapCredentialHonoursPinnedAuthType(unittest.TestCase):
    def _agent(self, pinned):
        from agent.client_lifecycle import ClientLifecycleMixin  # noqa: F401  (import guard for the mixin path)
        return SimpleNamespace(provider="openai-codex", model="gpt-test", base_url=CODEX_URL, api_key=OAUTH_TOKEN,
                               api_mode="codex_responses", _client_kwargs={}, _pinned_auth_type=pinned)

    def _swap(self, agent, entry):
        from agent.client_lifecycle import ClientLifecycleMixin
        agent._reapply_route_client_config = MagicMock()
        agent._replace_primary_openai_client = MagicMock()
        return ClientLifecycleMixin._swap_credential(agent, entry)

    def test_rotation_refuses_entry_of_another_auth_type(self):
        agent = self._agent("oauth")
        stale = _entry("stale-key", auth_type="api_key", token=STALE_KEY, priority=0)

        self.assertFalse(self._swap(agent, stale))
        self.assertEqual(agent.api_key, OAUTH_TOKEN)
        self.assertEqual(agent._client_kwargs, {})

    def test_rotation_accepts_entry_of_pinned_auth_type(self):
        agent = self._agent("oauth")
        fresh = _entry("oauth-2", auth_type="oauth", token="other-oauth-token", priority=0)

        self.assertTrue(self._swap(agent, fresh))
        self.assertEqual(agent.api_key, "other-oauth-token")

    def test_unpinned_agent_keeps_existing_swap_behaviour(self):
        agent = self._agent(None)
        key_entry = _entry("stale-key", auth_type="api_key", token=STALE_KEY, priority=0)

        self.assertTrue(self._swap(agent, key_entry))


if __name__ == "__main__":
    unittest.main()
