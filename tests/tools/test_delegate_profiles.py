"""Logical delegation profiles late-bind to authenticated provider routes (ops-supervisor#216 Phase 4).

``delegation.profiles.<name>`` names a provider (+ model, endpoint, required auth mechanism, reasoning); the route
and its native credentials resolve at spawn time through the same runtime-provider path as ``delegation.provider``.
Unknown, disabled, or auth-mismatched profiles fail closed before any child is constructed.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_dynamic_schema_overrides, delegate_task

GLM_KEY = "glm-profile-key-FIXTURE"
PROFILES = {
    "ingestion_fast": {"provider": "zai", "model": "glm-flash-fixture", "auth_type": "api_key",
                       "reasoning_effort": "low", "description": "extraction"},
    "coding_frontier": {"provider": "zai", "model": "glm-big-fixture", "auth_type": "oauth"},
    "retired": {"provider": "zai", "model": "old", "enabled": False},
}


def _parent():
    return MagicMock(
        provider="openrouter", base_url="https://openrouter.ai/api/v1", api_key="parent-key-FIXTURE",
        api_mode="chat_completions", model="parent-model", client=None, _client_kwargs={}, _credential_pool=None,
        _credential_pool_entry_id=None, acp_command=None, acp_args=[], requested_provider="openrouter",
        _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(), _session_db=None,
        _print_fn=None, tool_progress_callback=None, thinking_callback=None, request_overrides={},
        reasoning_config=None, _fallback_chain=None, capabilities=None, max_tokens=None)


def _child(**kwargs):
    child = MagicMock(**{k: kwargs.get(k) for k in ("provider", "base_url", "api_key", "model")})
    child._session_init_model_config = {}
    child.run_conversation.return_value = {"final_response": "ok", "completed": True, "api_calls": 1, "messages": []}
    return child


@pytest.fixture
def profiles_cfg(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", GLM_KEY)
    cfg = {"max_iterations": 5, "profiles": PROFILES}
    with patch("tools.delegate_tool._load_config", return_value=cfg):
        yield cfg


def test_profile_late_binds_provider_model_auth_and_reasoning(profiles_cfg):
    built = []
    with patch("run_agent.AIAgent", side_effect=lambda **kw: built.append(_child(**kw)) or built[-1]) as agent_cls:
        delegate_task(tasks=[{"goal": "extract", "profile": "ingestion_fast"}], parent_agent=_parent())
    kwargs = agent_cls.call_args.kwargs
    assert (kwargs["provider"], kwargs["model"], kwargs["api_key"]) == ("zai", "glm-flash-fixture", GLM_KEY)
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}
    authority = built[0]._auth_authority
    assert (authority.profile, authority.auth_type, authority.auth_source) == ("ingestion_fast", "api_key",
                                                                               "env:GLM_API_KEY")


@pytest.mark.parametrize("profile, reason", [
    ("retired", "disabled"), ("missing", "not configured"), ("coding_frontier", "delegation_profile_auth_mismatch"),
])
def test_unusable_profiles_fail_closed_before_any_child_exists(profiles_cfg, profile, reason):
    with patch("run_agent.AIAgent", side_effect=_child) as agent_cls:
        result = json.loads(delegate_task(tasks=[{"goal": "g", "profile": profile}], parent_agent=_parent()))
    agent_cls.assert_not_called()
    assert reason in result["error"] and GLM_KEY not in result["error"]


def test_profile_is_a_complete_route(profiles_cfg):
    """Not combinable with per-task model selection, and only enabled profiles are offered to the model."""
    result = json.loads(delegate_task(tasks=[{"goal": "g", "profile": "ingestion_fast", "model": "x"}],
                                      parent_agent=_parent()))
    assert "cannot be combined" in result["error"]
    schema = _build_dynamic_schema_overrides()["parameters"]["properties"]
    assert schema["profile"]["enum"] == ["coding_frontier", "ingestion_fast"]
    assert schema["tasks"]["items"]["properties"]["profile"]["enum"] == ["coding_frontier", "ingestion_fast"]
