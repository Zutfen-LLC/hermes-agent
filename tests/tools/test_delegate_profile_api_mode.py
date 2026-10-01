"""Issue #33: explicit delegation-profile ``api_mode`` must survive provider/runtime
resolution for the official OpenAI endpoint, so ``input_only_v1`` restricted runs on
``openai``/``chat_completions`` become reachable at all.

Defect chain (RED at 0376d425): ``delegation.profiles.<p>`` with ``provider: openai`` +
explicit ``api_mode: chat_completions`` resolves through ``_runtime_provider_credentials`` →
``resolve_runtime_provider(requested="openai")`` → direct-alias expansion →
``_detect_api_mode_for_url("https://api.openai.com/v1")`` mandates ``codex_responses``; the
profile's explicit mode is never consulted on that branch (only the ``delegation.base_url``
branch honors it), so restricted admission ``_validate_restricted_route`` fails for every
``openai`` profile — the trusted-route contract ``(openai, api.openai.com, chat_completions)``
was unsatisfiable.

Precedence rule established here (one rule, both branches): an operator-explicit valid
``api_mode`` beats URL/runtime detection; an invalid explicit mode fails closed; an absent
mode keeps runtime detection; request-time callers cannot override the mode.
"""

import asyncio
import json
import time

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_restricted_runs import _validate_restricted_route
from tools.delegate_tool_config import (
    _EXPLICIT_API_MODES,
    _resolve_delegation_credentials,
    _resolve_profile_execution,
)

AUTH = {"Authorization": "Bearer test-gateway-key"}
RESTRICTED_AUTH = {"Authorization": "Bearer restricted-scope-key-0123456789"}
OPENAI_KEY = "sk-openai-fixture-33"
REQUEST = {
    "delegation_profile_id": "ops_inputonly",
    "work_class": "context_gather",
    "input": "Analyze this supplied snapshot only.",
    "capability_envelope": "input_only_v1",
}

PROFILE_CFG = {"provider": "openai", "model": "gpt-5-nano", "api_mode": "chat_completions"}


def _parent():
    return SimpleNamespace(request_overrides=None)


def _resolve(mode=..., provider="openai", model="gpt-5-nano", base_url=None):
    """Resolve a delegation profile (or plain delegation block) with the fixture key bound."""
    profile = {k: v for k, v in (("provider", provider), ("model", model),
                                  ("base_url", base_url), ("api_mode", None if mode is ... else mode)) if v is not None}
    if provider is not None and "api_mode" in profile and profile["api_mode"] is None:
        del profile["api_mode"]
    cfg = {"profiles": {"p": profile}}
    return _resolve_profile_execution(cfg, "p", _parent())[0]


# ── Unit coverage of the precedence rule ────────────────────────────────────


def test_explicit_profile_api_mode_survives_official_openai_resolution(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    creds = _resolve("chat_completions")
    assert (creds["provider"], creds["base_url"], creds["api_mode"]) == (
        "openai", "https://api.openai.com/v1", "chat_completions")


def test_explicit_profile_api_mode_survives_when_route_passes_through_agent_init(monkeypatch):
    """The cred bundle feeds AIAgent(override_api_mode=...); construction must keep the mode."""
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    creds = _resolve("chat_completions")
    kwargs = {"provider": creds["provider"], "model": creds["model"], "api_key": creds["api_key"],
              "base_url": creds["base_url"], "api_mode": creds["api_mode"]}
    from utils import base_url_hostname
    agent = SimpleNamespace(provider=kwargs["provider"], api_mode=None,
                            _base_url_hostname=base_url_hostname(kwargs["base_url"]),
                            _base_url_lower=kwargs["base_url"].lower())
    # agent_init._resolve_api_mode ladder: explicit mode wins over URL detection.
    from agent.agent_init import _resolve_api_mode
    _resolve_api_mode(agent, kwargs["api_mode"], None, kwargs["base_url"])
    assert agent.api_mode == "chat_completions"


def test_absent_profile_api_mode_keeps_runtime_detection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    creds = _resolve(...)
    assert creds["api_mode"] == "codex_responses"  # URL detection unchanged


@pytest.mark.parametrize("mode", [" responses ", "RESponses", "codex_responses_aliased"])
def test_unknown_explicit_profile_api_mode_fails_closed(monkeypatch, mode):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="api_mode"):
        _resolve(mode)


def test_empty_profile_api_mode_is_absent_not_invalid(monkeypatch):
    """Empty-string api_mode means unset (repo-wide normalization), so runtime detection runs."""
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    creds = _resolve("")
    assert creds["api_mode"] == "codex_responses"


def test_incompatible_explicit_mode_is_resolved_but_refused_by_restricted_admission(monkeypatch):
    """A valid-but-wrong explicit mode resolves (ordinary delegation may use it); restricted
    admission still rejects it — input_only_v1 is NOT widened for official OpenAI."""
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    creds = _resolve("codex_responses")
    assert creds["api_mode"] == "codex_responses"
    with pytest.raises(RuntimeError, match="restricted"):
        _validate_restricted_route({**creds, "api_key": OPENAI_KEY})


def test_explicit_mode_not_honored_for_unrelated_provider(monkeypatch):
    """A provider whose runtime has no direct-alias expansion must not gain an override:
    the explicit mode is only re-applied when the requested provider survives expansion."""
    import hermes_cli.runtime_provider as rp
    seen = {}

    def fake_resolve(*, requested=None, **kw):
        seen["requested"] = requested
        return {"provider": "zai", "model": "glm-fixture", "base_url": "https://api.z.ai/v1",
                "api_key": "zai-fixture", "api_mode": "chat_completions", "source": "env"}

    monkeypatch.setattr(rp, "resolve_runtime_provider", fake_resolve)
    creds = _resolve("anthropic_messages", provider="zai", model="glm-fixture")
    assert creds["api_mode"] == "chat_completions"  # runtime wins; explicit mode not forced


def test_plain_provider_block_without_explicit_mode_is_unchanged(monkeypatch):
    """Ordinary non-restricted resolution (delegation.provider without api_mode) is untouched."""
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    creds = _resolve_delegation_credentials({"provider": "openai"}, _parent())
    assert creds["api_mode"] == "codex_responses"
    assert creds["base_url"] == "https://api.openai.com/v1"


def test_direct_base_url_branch_explicit_mode_precedence_unchanged(monkeypatch):
    """The pre-existing direct-endpoint branch keeps its behavior (explicit wins over URL)."""
    creds = _resolve_delegation_credentials(
        {"provider": None, "base_url": "https://api.openai.com/v1", "api_key": "k",
         "api_mode": "chat_completions"}, _parent())
    assert creds["api_mode"] == "chat_completions"
    creds = _resolve_delegation_credentials(
        {"provider": None, "base_url": "https://api.openai.com/v1", "api_key": "k"}, _parent())
    assert creds["api_mode"] == "codex_responses"


def test_explicit_profile_mode_restricted_route_passes_admission(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    creds = _resolve("chat_completions")
    _validate_restricted_route({**creds, "api_key": OPENAI_KEY})  # must not raise


def test_untrusted_host_still_refused_even_with_explicit_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com.evil.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    creds = _resolve("chat_completions")
    assert creds["base_url"].hostname if hasattr(creds["base_url"], "hostname") else True
    with pytest.raises(RuntimeError, match="restricted"):
        _validate_restricted_route({**creds, "api_key": OPENAI_KEY, "base_url": creds["base_url"]})


# ── End-to-end through the real gateway restricted-run surface ─────────────


class FakeAgent:
    provider = "openai"
    model = "gpt-5-nano"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = kwargs.get("enabled_toolsets", [])
        self.disabled_toolsets = kwargs.get("disabled_toolsets", [])
        self.provider = kwargs.get("provider", "openai")
        self.model = kwargs.get("model", "gpt-5-nano")
        self.api_mode = kwargs.get("api_mode")
        self.base_url = kwargs.get("base_url")
        self.request_overrides = kwargs.get("request_overrides")
        self.output = kwargs.get("test_output", "bounded result")
        self.stop_event = asyncio.Event()
        self.started_event = asyncio.Event()

    def run_conversation(self, **kwargs):
        self.turn_kwargs = kwargs
        self.started_event.set()
        return {"final_response": self.output, "completed": True}


@pytest_asyncio.fixture
async def openai_profile_service(tmp_path, monkeypatch):
    """Real restricted-runs surface, real profile config file, real resolution chain, real
    ``_new_restricted_agent`` route validation — only ``AIAgent`` is faked (no network)."""
    home = tmp_path / "profile-home"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({
        "delegation": {"profiles": {"ops_inputonly": dict(PROFILE_CFG, enabled=True)}}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": "test-gateway-key", "restricted_key": "restricted-scope-key-0123456789"}))
    observed = {}
    built = []

    def fake_agent_cls(**kwargs):
        agent = FakeAgent(**kwargs)
        built.append(agent)
        return agent

    monkeypatch.setattr("run_agent.AIAgent", fake_agent_cls)
    app = web.Application()
    app.router.add_post("/v1/restricted-runs", adapter._handle_restricted_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    async with TestClient(TestServer(app)) as client:
        yield client, {"built": built, "observed": observed}
    adapter._response_store.close()
    close = getattr(adapter._session_db, "close", None)
    if callable(close):
        close()


async def _terminal_status(client, run_id):
    deadline = time.monotonic() + 5
    status = {}
    while time.monotonic() < deadline:
        response = await client.get(f"/v1/runs/{run_id}", headers=AUTH)
        status = await response.json()
        if status.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
            return status
        await asyncio.sleep(0.02)
    return status


@pytest.mark.asyncio
async def test_official_openai_profile_input_only_run_is_admitted_end_to_end(openai_profile_service):
    """The issue's reproducer: POST /v1/restricted-runs (input_only_v1) for an openai profile
    with explicit chat_completions must reach a 202 admission (not 403 profile-unavailable)."""
    client, spy = openai_profile_service
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "issue-33-red"})
    assert response.status == 202, await response.text()
    admitted = await response.json()
    agent = spy["built"][0]
    assert agent.kwargs["api_mode"] == "chat_completions"
    assert agent.kwargs["provider"] == "openai"
    assert agent.kwargs["base_url"] == "https://api.openai.com/v1"
    status = await _terminal_status(client, admitted["run_id"])
    assert status.get("status") == "completed", status
    assert status.get("output") == "bounded result"


@pytest.mark.asyncio
async def test_request_body_cannot_override_profile_route_or_mode(openai_profile_service):
    """Restricted-run requests carry exactly the four contract fields; provider/model/api_mode
    injection is rejected at the schema boundary (400), never reaches route resolution."""
    client, spy = openai_profile_service
    for field, value in (("provider", "attacker"), ("model", "attacker"),
                         ("api_mode", "codex_responses"), ("base_url", "https://attacker.invalid")):
        response = await client.post("/v1/restricted-runs", json={**REQUEST, field: value},
                                     headers={**AUTH, "Idempotency-Key": f"issue-33-{field}"})
        assert response.status == 400, (field, await response.text())
    assert spy["built"] == []


@pytest.mark.asyncio
async def test_profile_without_explicit_mode_still_fails_admission_end_to_end(
        openai_profile_service, tmp_path):
    """Absent explicit mode keeps codex_responses, which restricted admission must keep
    refusing — the fix admits ONLY the operator-explicit compatible mode."""
    client, _ = openai_profile_service
    # Rewrite the served profile config without api_mode (same isolated HERMES_HOME).
    import os
    from pathlib import Path
    home_file = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    home_file.write_text(json.dumps({
        "delegation": {"profiles": {"ops_inputonly": {"provider": "openai", "model": "gpt-5-nano",
                                                      "enabled": True}}}}))
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "issue-33-no-mode"})
    assert response.status == 403
    body = await response.json()
    assert body["error"]["code"] == "delegation_profile_unavailable"
