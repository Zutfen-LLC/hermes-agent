"""Contract regressions for the low-authority delegated-run transport (#208 Slice 3)."""

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import httpx
from openai import OpenAI
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

AUTH = {"Authorization": "Bearer test-gateway-key"}
RESTRICTED_AUTH = {"Authorization": "Bearer restricted-scope-key-0123456789"}
REQUEST = {
    "delegation_profile_id": "logical-helper",
    "work_class": "context_gather",
    "input": "Analyze this supplied snapshot only.",
    "capability_envelope": "input_only_v1",
}


class FakeAgent:
    provider = "provider-test"
    model = "model-test"
    base_url = "https://provider.invalid"
    api_mode = "chat_completions"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = kwargs.get("enabled_toolsets", [])
        self.output = kwargs.get("test_output", "bounded result")
        self.stop_event = threading.Event()
        self.started_event = threading.Event()

    def run_conversation(self, **kwargs):
        self.turn_kwargs = kwargs
        self.started_event.set()
        self.stop_event.wait(0.1)
        return {"final_response": self.output, "completed": True}

    def interrupt(self, message=None):
        self.interrupted = True
        self.stop_event.set()

    def close(self):
        self.closed = True


@pytest_asyncio.fixture
async def restricted_service(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": "test-gateway-key", "restricted_key": "restricted-scope-key-0123456789"}))
    app = web.Application()
    app.router.add_post("/v1/restricted-runs", adapter._handle_restricted_runs)
    app.router.add_post("/v1/restricted-runs/resolve", adapter._handle_resolve_restricted_identity)
    app.router.add_post("/v1/restricted-runs/identity-checked", adapter._handle_identity_checked_restricted_runs)
    app.router.add_get("/v1/restricted-runs/by-key", adapter._handle_restricted_run_by_key)
    app.router.add_post("/v1/restricted-runs/by-key/stop", adapter._handle_stop_restricted_run_by_key)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    async with TestClient(TestServer(app)) as client:
        yield client, adapter
    adapter._response_store.close()
    close = getattr(adapter._session_db, "close", None)
    if callable(close):
        close()


def _route_mocks(monkeypatch, *, secret="native-secret", output="bounded result"):
    observed = {}

    def resolve(cfg, name, parent):
        observed["name"] = name
        observed["profile_cfg"] = cfg
        return ({"provider": "provider-test", "model": "model-test", "api_key": secret,
                 "base_url": "https://provider.invalid", "auth_type": "api_key"}, None)

    def create(_self, creds, reasoning, authority=None, **_kw):
        config = {**creds, "enabled_toolsets": [], "disabled_toolsets": ["all toolsets"],
                  "test_output": output,
                  "skip_context_files": True, "skip_memory": True}
        observed["agent_config"] = config
        observed["agent"] = FakeAgent(**config)
        return observed["agent"]

    monkeypatch.setattr("tools.delegate_tool_config._resolve_profile_execution", resolve)
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._restricted_profile_config",
                        lambda self, name, **kw: {"profiles": {name: {
                            "enabled": True, "restricted_tool_free": True}}})
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._new_restricted_agent", create)
    return observed


async def _terminal_status(client, run_id):
    deadline = time.monotonic() + 2
    status = {}
    while time.monotonic() < deadline:
        response = await client.get(f"/v1/runs/{run_id}", headers=AUTH)
        status = await response.json()
        if status.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
            return status
        await asyncio.sleep(0.02)
    return status


@pytest.mark.asyncio
async def test_restricted_endpoint_rejects_extra_fields_and_missing_idempotency(restricted_service):
    client, _ = restricted_service
    missing = await client.post("/v1/restricted-runs", json=REQUEST, headers=AUTH)
    assert missing.status == 400
    for field, value in (("provider", "attacker"), ("model", "attacker"),
                         ("base_url", "https://attacker.invalid"), ("api_key", "secret"),
                         ("auth", "attacker")):
        response = await client.post("/v1/restricted-runs", json={**REQUEST, field: value},
                                     headers={**AUTH, "Idempotency-Key": "key-1"})
        assert response.status == 400
    injected = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "key-2",
                                          "X-Hermes-Provider-API-Key": "credential"})
    assert injected.status == 400


@pytest.mark.asyncio
async def test_restricted_route_late_binds_profile_and_never_builds_tools(restricted_service, monkeypatch):
    client, _ = restricted_service
    observed = _route_mocks(monkeypatch)
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "late-bind"})
    assert response.status == 202
    admitted = await response.json()
    status = await _terminal_status(client, admitted["run_id"])
    assert observed["name"] == "logical-helper"
    assert observed["agent_config"]["enabled_toolsets"] == []
    assert observed["agent_config"]["skip_context_files"] is True
    assert observed["agent_config"]["skip_memory"] is True

    assert observed["agent"].enabled_toolsets == []
    assert observed["agent"].tools == []
    assert status["status"] == "completed", status
    assert status["output"] == "bounded result"
    assert status["resolved_identity"]["hermes_delegation_profile_id"] == "logical-helper"
    assert "native-secret" not in str(status)


@pytest.mark.asyncio
async def test_tool_free_envelope_is_durable_and_does_not_change_original_run(restricted_service, monkeypatch):
    client, _ = restricted_service
    _route_mocks(monkeypatch)
    body = {**REQUEST, "capability_envelope": "hermes_tool_free_v1"}
    headers = {**RESTRICTED_AUTH, "Idempotency-Key": "user-route"}
    admitted = await client.post("/v1/restricted-runs", json=body, headers=headers)
    assert admitted.status == 202
    run_id = (await admitted.json())["run_id"]
    recovered = await client.get("/v1/restricted-runs/by-key", headers=headers)
    assert recovered.status == 200
    assert (await recovered.json())["capability_envelope"] == "hermes_tool_free_v1"
    old_envelope = await client.post("/v1/restricted-runs", json=REQUEST, headers=headers)
    assert old_envelope.status == 409
    assert (await _terminal_status(client, run_id))["status"] == "completed"


@pytest.mark.asyncio
async def test_tool_free_requires_operator_profile_opt_in_before_reservation(restricted_service, monkeypatch):
    client, adapter = restricted_service
    observed = _route_mocks(monkeypatch)
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._restricted_profile_config",
                        lambda self, name, **kw: {"profiles": {name: {"enabled": True}}})
    response = await client.post("/v1/restricted-runs",
        json={**REQUEST, "capability_envelope": "hermes_tool_free_v1"},
        headers={**RESTRICTED_AUTH, "Idempotency-Key": "no-opt-in"})
    assert response.status == 403
    assert "agent" not in observed
    assert not adapter._run_idempotency_store._conn.execute(
        "SELECT 1 FROM run_idempotency WHERE idempotency_key='no-opt-in'").fetchone()


def test_restricted_agent_factory_pins_no_tools(monkeypatch):
    from types import SimpleNamespace
    from gateway.platforms import api_server_restricted_runs as restricted

    captured = {}

    def agent_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(enabled_toolsets=kwargs["enabled_toolsets"], tools=[], valid_tool_names=set(),
                               provider=kwargs["provider"], model=kwargs["model"],
                               base_url=kwargs["base_url"], api_mode="chat_completions", request_overrides=None)

    monkeypatch.setattr("run_agent.AIAgent", agent_factory)
    agent = restricted._new_restricted_agent(
        None, {"provider": "openai", "model": "gpt-4o", "base_url": "https://api.openai.com/v1"}, None,
        _api_server=SimpleNamespace())
    assert agent.enabled_toolsets == []
    assert agent.tools == []
    assert captured["disabled_toolsets"]
    assert captured["skip_context_files"] is True
    assert captured["skip_memory"] is True
    assert "Do not claim direct repository" in captured["ephemeral_system_prompt"]


@pytest.mark.asyncio
async def test_restricted_result_secret_is_redacted_before_durable_status(restricted_service, monkeypatch):
    client, adapter = restricted_service
    sentinel = "opaque-native-credential-7a98f3"
    _route_mocks(monkeypatch, secret=sentinel, output=f"evidence {sentinel} must not persist")
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "secret-result"})
    run_id = (await response.json())["run_id"]
    status = await _terminal_status(client, run_id)
    assert sentinel not in str(status)
    db_path = adapter._run_idempotency_store._db_path
    if db_path:
        for suffix in ("", "-wal", "-shm"):
            db_file = Path(db_path + suffix)
            if db_file.exists():
                assert sentinel.encode() not in db_file.read_bytes()


@pytest.mark.asyncio
async def test_secret_as_profile_name_is_refused_before_persistence(restricted_service, monkeypatch):
    client, adapter = restricted_service
    sentinel = "opaque-native-credential-7a98f3"
    _route_mocks(monkeypatch, secret=sentinel)
    response = await client.post("/v1/restricted-runs", json={**REQUEST, "delegation_profile_id": sentinel},
                                 headers={**AUTH, "Idempotency-Key": "secret-identity"})
    assert response.status == 403
    assert not adapter._run_idempotency_store._conn.execute("SELECT 1 FROM run_idempotency").fetchone()


@pytest.mark.asyncio
async def test_result_is_redacted_before_truncation(restricted_service, monkeypatch):
    client, _ = restricted_service
    secret = "opaque-provider-credential-CROSS-BOUNDARY"
    _route_mocks(monkeypatch, secret=secret, output="x" * 15990 + secret)
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "redact-then-cut"})
    run_id = (await response.json())["run_id"]
    status = await _terminal_status(client, run_id)
    assert status["status"] == "completed"
    assert len(status["output"]) <= 16000
    assert secret[:16] not in status["output"]


def test_profile_scope_isolation_for_route_config_and_idempotency(restricted_service, monkeypatch):
    _, adapter = restricted_service
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    active = []
    scopes = []

    @contextmanager
    def profile_scope(name):
        active.append(name)
        scopes.append(name)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(adapter, "_profile_scope", profile_scope)
    monkeypatch.setattr("tools.delegate_tool_config._load_config",
                        lambda: {"owner": active[-1]})
    token = api_server._api_request_profile.set("profile-b")
    try:
        cfg_b = restricted._restricted_profile_config(adapter, "helper", _api_server=api_server)
        assert cfg_b["owner"] == "profile-b"
    finally:
        api_server._api_request_profile.reset(token)
    token = api_server._api_request_profile.set("profile-a")
    try:
        cfg_a = restricted._restricted_profile_config(adapter, "helper", _api_server=api_server)
        assert cfg_a["owner"] == "profile-a"
    finally:
        api_server._api_request_profile.reset(token)
    assert scopes == ["profile-b", "profile-a"]

    request = SimpleNamespace(path="/v1/restricted-runs", headers={}, method="POST")
    token = api_server._api_request_profile.set("profile-a")
    try:
        scope_a = adapter._run_idempotency_scope(request)
    finally:
        api_server._api_request_profile.reset(token)
    token = api_server._api_request_profile.set("profile-b")
    try:
        scope_b = adapter._run_idempotency_scope(request)
    finally:
        api_server._api_request_profile.reset(token)
    assert scope_a != scope_b


@pytest.mark.asyncio
async def test_restricted_replay_conflict_and_profile_scope(restricted_service, monkeypatch):
    client, _ = restricted_service
    _route_mocks(monkeypatch)
    h = {**AUTH, "Idempotency-Key": "retry"}
    first = await client.post("/v1/restricted-runs", json=REQUEST, headers=h)
    first_data = await first.json()
    replay = await client.post("/v1/restricted-runs", json=REQUEST, headers=h)
    replay_data = await replay.json()
    assert replay_data["run_id"] == first_data["run_id"]
    assert replay_data["replayed"] is True
    assert replay_data["resolved_identity"]["hermes_delegation_profile_id"] == "logical-helper"
    changed = await client.post("/v1/restricted-runs", json={**REQUEST, "input": "different"}, headers=h)
    assert changed.status == 409


@pytest.mark.asyncio
async def test_parallel_identical_admissions_create_one_run(restricted_service, monkeypatch):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    headers = {**AUTH, "Idempotency-Key": "parallel-same"}
    replies = await asyncio.gather(*(
        client.post("/v1/restricted-runs", json=REQUEST, headers=headers) for _ in range(4)))
    assert all(reply.status == 202 for reply in replies)
    bodies = await asyncio.gather(*(reply.json() for reply in replies))
    assert len({body["run_id"] for body in bodies}) == 1
    rows = adapter._run_idempotency_store._conn.execute("SELECT COUNT(*) FROM run_idempotency").fetchone()
    assert rows[0] == 1


@pytest.mark.asyncio
async def test_restricted_work_class_allowlist(restricted_service):
    client, _ = restricted_service
    for work_class in ("implementation", "correction", "review_support", "other"):
        response = await client.post("/v1/restricted-runs", json={**REQUEST, "work_class": work_class},
                                     headers={**AUTH, "Idempotency-Key": "denied-" + work_class})
        assert response.status == 400


@pytest.mark.asyncio
async def test_restricted_stop_uses_existing_run_control(restricted_service, monkeypatch):
    client, _ = restricted_service
    response = await client.post("/v1/runs/run_nonexistent/stop", headers=AUTH)
    assert response.status == 404
    observed = _route_mocks(monkeypatch)
    admitted = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "stop-running"})
    run_id = (await admitted.json())["run_id"]
    agent = observed["agent"]
    assert await asyncio.to_thread(agent.started_event.wait, 1)
    stopped = await client.post(f"/v1/runs/{run_id}/stop", headers=AUTH)
    assert stopped.status == 200
    terminal = await _terminal_status(client, run_id)
    assert terminal["status"] == "cancelled"
    assert getattr(agent, "interrupted", False)


@pytest.mark.asyncio
async def test_restricted_key_lookup_recovers_lost_acceptance_without_new_run(restricted_service, monkeypatch):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    headers = {**RESTRICTED_AUTH, "Idempotency-Key": "lost-acceptance"}
    admitted = await client.post("/v1/restricted-runs", json=REQUEST, headers=headers)
    run_id = (await admitted.json())["run_id"]
    assert (await _terminal_status(client, run_id))["status"] == "completed"
    recovered = await client.get("/v1/restricted-runs/by-key", headers=headers)
    assert recovered.status == 200
    assert (await recovered.json())["run_id"] == run_id
    assert adapter._run_idempotency_store._conn.execute("SELECT COUNT(*) FROM run_idempotency").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_restricted_key_recovery_rejects_unknown_invalid_and_general_runs(restricted_service, monkeypatch):
    client, adapter = restricted_service
    for headers in (RESTRICTED_AUTH, {**RESTRICTED_AUTH, "Idempotency-Key": "bad key"}):
        response = await client.get("/v1/restricted-runs/by-key", headers=headers)
        assert response.status == 400
    unknown = {**RESTRICTED_AUTH, "Idempotency-Key": "unknown"}
    assert (await client.get("/v1/restricted-runs/by-key", headers=unknown)).status == 404
    assert (await client.post("/v1/restricted-runs/by-key/stop", headers=unknown)).status == 404
    general_scope = adapter._run_idempotency_scope(SimpleNamespace(path="/v1/runs", headers={}, method="POST"))
    adapter._run_idempotency_store.reserve(general_scope, "general-key", "fp", "run_general",
                                            {"object": "hermes.run", "run_id": "run_general", "status": "completed"})
    for endpoint, method in (("/v1/restricted-runs/by-key", client.get),
                             ("/v1/restricted-runs/by-key/stop", client.post)):
        assert (await method(endpoint, headers={**RESTRICTED_AUTH, "Idempotency-Key": "general-key"})).status == 404
    # Even a wrongly stored ordinary run in the restricted scope cannot be controlled.
    from gateway.platforms import api_server_restricted_runs as restricted, api_server
    scope = restricted._restricted_scope(adapter, None, _api_server=api_server)
    adapter._run_idempotency_store.reserve(scope, "ordinary-in-scope", "fp", "run_ordinary",
                                            {"object": "hermes.run", "run_id": "run_ordinary", "status": "completed"})
    assert (await client.get("/v1/restricted-runs/by-key", headers={**RESTRICTED_AUTH,
        "Idempotency-Key": "ordinary-in-scope"})).status == 404


@pytest.mark.asyncio
async def test_restricted_key_stop_is_repeatable_and_keeps_worker_visible(restricted_service, monkeypatch):
    client, _ = restricted_service
    observed = _route_mocks(monkeypatch)
    release = threading.Event()
    def blocked(agent, text):
        agent.started_event.set()
        release.wait(5)
        return {"final_response": "late result", "completed": True}
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._restricted_agent_turn", blocked)
    headers = {**RESTRICTED_AUTH, "Idempotency-Key": "lost-stop"}
    admitted = await client.post("/v1/restricted-runs", json=REQUEST, headers=headers)
    run_id = (await admitted.json())["run_id"]
    assert await asyncio.to_thread(observed["agent"].started_event.wait, 1)
    try:
        for _ in range(2):
            stopped = await client.post("/v1/restricted-runs/by-key/stop", headers=headers)
            assert stopped.status == 200
            assert (await stopped.json())["status"] == "stopping"
            status = await (await client.get("/v1/restricted-runs/by-key", headers=headers)).json()
            assert status["run_id"] == run_id and status["status"] == "stopping"
        assert observed["agent"].interrupted
    finally:
        release.set()
    assert (await _terminal_status(client, run_id))["status"] == "cancelled"
    settled = await client.post("/v1/restricted-runs/by-key/stop", headers=headers)
    assert settled.status == 200
    assert (await settled.json())["status"] == "cancelled"


@pytest.mark.asyncio
async def test_restricted_key_recovery_auth_and_rotation(restricted_service, monkeypatch):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    first = await client.post("/v1/restricted-runs", json=REQUEST,
                              headers={**RESTRICTED_AUTH, "Idempotency-Key": "rotate-by-key"})
    run_id = (await first.json())["run_id"]
    assert (await _terminal_status(client, run_id))["status"] == "completed"
    key_header = {"Idempotency-Key": "rotate-by-key"}
    assert (await client.get("/v1/restricted-runs/by-key", headers=key_header)).status == 401
    assert (await client.post("/v1/restricted-runs/by-key/stop", headers=key_header)).status == 401
    assert (await client.get("/v1/runs", headers=RESTRICTED_AUTH)).status != 200
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "rotated-master-key-0123456789")
    monkeypatch.setattr(adapter, "_expected_restricted_api_key", lambda: "rotated-restricted-key-0123456789")
    assert (await client.get("/v1/restricted-runs/by-key", headers={**key_header, **RESTRICTED_AUTH})).status == 401
    rotated = {"Authorization": "Bearer rotated-restricted-key-0123456789", **key_header}
    recovered = await client.get("/v1/restricted-runs/by-key", headers=rotated)
    assert recovered.status == 200
    assert (await recovered.json())["run_id"] == run_id
    stopped = await client.post("/v1/restricted-runs/by-key/stop", headers=rotated)
    assert stopped.status == 200 and (await stopped.json())["status"] == "completed"


@pytest.mark.asyncio
async def test_restricted_key_recovery_survives_adapter_restart(restricted_service, monkeypatch):
    client, _ = restricted_service
    _route_mocks(monkeypatch)
    headers = {**RESTRICTED_AUTH, "Idempotency-Key": "recover-after-restart"}
    admitted = await client.post("/v1/restricted-runs", json=REQUEST, headers=headers)
    run_id = (await admitted.json())["run_id"]
    assert (await _terminal_status(client, run_id))["status"] == "completed"

    other = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": "test-gateway-key", "restricted_key": "restricted-scope-key-0123456789"}))
    app = web.Application()
    app.router.add_get("/v1/restricted-runs/by-key", other._handle_restricted_run_by_key)
    app.router.add_post("/v1/restricted-runs/by-key/stop", other._handle_stop_restricted_run_by_key)
    try:
        async with TestClient(TestServer(app)) as restarted:
            recovered = await restarted.get("/v1/restricted-runs/by-key", headers=headers)
            assert recovered.status == 200
            assert (await recovered.json())["run_id"] == run_id
            stopped = await restarted.post("/v1/restricted-runs/by-key/stop", headers=headers)
            assert stopped.status == 200 and (await stopped.json())["status"] == "completed"
            assert not other._active_run_agents and not other._active_run_tasks
    finally:
        other._run_idempotency_store.close()
        other._response_store.close()


def test_restricted_key_recovery_routes_are_advertised_and_registered(restricted_service):
    _, adapter = restricted_service
    paths = {(method, path) for method, path, _ in adapter._http_route_table()}
    assert ("GET", "/v1/restricted-runs/by-key") in paths
    assert ("POST", "/v1/restricted-runs/by-key/stop") in paths
    from gateway.platforms import api_server
    assert ("restricted_run_by_key", ("GET", "/v1/restricted-runs/by-key")) in api_server._CAPABILITY_ENDPOINTS
    assert ("restricted_run_stop_by_key", ("POST", "/v1/restricted-runs/by-key/stop")) in api_server._CAPABILITY_ENDPOINTS


@pytest.mark.asyncio
async def test_restricted_key_recovery_isolated_by_served_profile(restricted_service, monkeypatch):
    _, adapter = restricted_service
    from gateway.platforms import api_server
    _route_mocks(monkeypatch)
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "test-gateway-key")
    monkeypatch.setattr(adapter, "_expected_restricted_api_key", lambda: "restricted-scope-key-0123456789")

    @web.middleware
    async def bind_profile(request, handler):
        token = api_server._api_request_profile.set(request.headers["X-Test-Profile"])
        try:
            return await handler(request)
        finally:
            api_server._api_request_profile.reset(token)

    app = web.Application(middlewares=[bind_profile])
    app.router.add_post("/v1/restricted-runs", adapter._handle_restricted_runs)
    app.router.add_get("/v1/restricted-runs/by-key", adapter._handle_restricted_run_by_key)
    app.router.add_post("/v1/restricted-runs/by-key/stop", adapter._handle_stop_restricted_run_by_key)
    headers = {**RESTRICTED_AUTH, "Idempotency-Key": "same-key"}
    async with TestClient(TestServer(app)) as client:
        created = await client.post("/v1/restricted-runs", json=REQUEST,
                                    headers={**headers, "X-Test-Profile": "profile-a"})
        assert created.status == 202
        run_id = (await created.json())["run_id"]
        for method, path in ((client.get, "/v1/restricted-runs/by-key"),
                             (client.post, "/v1/restricted-runs/by-key/stop")):
            assert (await method(path, headers={**headers, "X-Test-Profile": "profile-b"})).status == 404
        recovered = await client.get("/v1/restricted-runs/by-key",
                                     headers={**headers, "X-Test-Profile": "profile-a"})
        assert recovered.status == 200 and (await recovered.json())["run_id"] == run_id


@pytest.mark.asyncio
async def test_stop_does_not_claim_completion_while_worker_is_still_running(restricted_service, monkeypatch):
    client, adapter = restricted_service
    observed = _route_mocks(monkeypatch)

    release = threading.Event()

    def stubborn_turn(agent, text):
        agent.started_event.set()
        release.wait(10)
        return {"final_response": "stopped", "completed": True}

    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._restricted_agent_turn", stubborn_turn)
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "stubborn-stop"})
    run_id = (await response.json())["run_id"]
    agent = observed["agent"]
    assert await asyncio.to_thread(agent.started_event.wait, 1)
    try:
        stopped = await client.post(f"/v1/runs/{run_id}/stop", headers=AUTH)
        assert stopped.status == 200
        assert agent.interrupted
        status = await (await client.get(f"/v1/runs/{run_id}", headers=AUTH)).json()
        assert status["status"] == "stopping"
    finally:
        release.set()
    assert (await _terminal_status(client, run_id))["status"] == "cancelled"


@pytest.mark.asyncio
async def test_repeated_task_cancellation_is_not_a_proven_worker_stop(restricted_service, monkeypatch):
    client, adapter = restricted_service
    observed = _route_mocks(monkeypatch)
    release = threading.Event()
    def blocked(agent, text):
        agent.started_event.set()
        release.wait(10)
        return {"final_response": "late result", "completed": True}
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._restricted_agent_turn", blocked)
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "double-cancel"})
    run_id = (await response.json())["run_id"]
    assert await asyncio.to_thread(observed["agent"].started_event.wait, 1)
    task = adapter._active_run_tasks[run_id]
    try:
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done(), "live worker must retain its control task after repeated cancellation"
        assert adapter._active_run_tasks.get(run_id) is task
        status = await (await client.get(f"/v1/runs/{run_id}", headers=AUTH)).json()
        assert status["status"] in {"running", "stopping"}
    finally:
        release.set()
    await asyncio.wait_for(task, 2)
    assert (await _terminal_status(client, run_id))["status"] == "cancelled"


@pytest.mark.asyncio
async def test_restricted_rejects_unhashable_work_class(restricted_service):
    client, _ = restricted_service
    response = await client.post("/v1/restricted-runs", json={**REQUEST, "work_class": ["context_gather"]},
                                 headers={**AUTH, "Idempotency-Key": "bad-type"})
    assert response.status == 400


def test_real_agent_has_no_model_or_dispatch_tools(restricted_service):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    with adapter._profile_scope(None):
        agent = restricted._new_restricted_agent(
            adapter, {"provider": "openai", "model": "gpt-4o", "api_key": "test-key",
                      "base_url": "https://api.openai.com/v1", "api_mode": "chat_completions"},
            None, _api_server=api_server)
    assert not agent.tools
    assert not agent.valid_tool_names


def test_real_delegation_profile_config_uses_runtime_credentials_not_request_fields(
        restricted_service, monkeypatch):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    from hermes_constants import get_hermes_home
    _, adapter = restricted_service
    (get_hermes_home() / "config.yaml").write_text(
        "delegation:\n  profiles:\n    logical-helper:\n      provider: openrouter\n"
        "      model: operator-model\n      auth_type: api_key\n")
    observed = []
    def runtime(*, requested, target_model):
        observed.append((requested, target_model))
        return {"provider": requested, "model": target_model, "base_url": "https://provider.invalid/v1",
                "api_key": "runtime-only-secret", "auth_type": "api_key", "source": "provider-runtime"}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", runtime)
    creds, _, raw, authority = restricted._resolve_restricted_route(
        adapter, "logical-helper", _api_server=api_server)
    assert observed == [("openrouter", "operator-model")]
    assert creds["api_key"] == "runtime-only-secret"
    assert authority.auth_type == "api_key"
    identity = restricted._identity(adapter, "logical-helper", creds, raw, authority, _api_server=api_server)
    assert identity["resolved_model"] == "operator-model"
    assert "runtime-only-secret" not in str(identity)


def test_restricted_identity_uses_effective_agent_endpoint(restricted_service):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    creds = {"provider": "openai", "model": "gpt-4.1-mini", "base_url": "https://configured.invalid/v1"}
    agent = FakeAgent()
    agent.provider = "openai"
    agent.model = "gpt-4.1-mini"
    agent.base_url = "https://effective.invalid/v1"
    identity = restricted._identity(adapter, "helper", creds, {}, agent=agent,
                                    _api_server=api_server, work_class="context_gather",
                                    envelope="input_only_v1")
    assert identity["endpoint_identity"] == "https://effective.invalid"
    assert identity["endpoint_identity"] != "https://configured.invalid"


def test_generic_identity_route_revision_preserves_legacy_digest(restricted_service):
    import hashlib
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    raw = {"provider": "openai", "model": "gpt-4.1-mini", "base_url": "https://provider.invalid/v1",
           "api_mode": "chat_completions", "enabled": True, "restricted_tool_free": True}
    keys = ("provider", "model", "base_url", "api_mode", "request_overrides", "fallback_providers", "auth_type")
    payload = {key: raw.get(key) for key in keys}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         default=str).encode()).hexdigest()
    identity = restricted._identity(adapter, "helper", raw, raw, _api_server=api_server)
    assert identity["route_revision"] == expected


def test_restricted_route_identity_strips_url_userinfo_path_query_and_fragment(restricted_service):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    secret = "opaque-native-secret-79834"
    creds = {"provider": "custom", "model": "test", "api_key": secret,
             "base_url": f"https://user:{secret}@provider.invalid/private/{secret}?token={secret}#{secret}"}
    identity = restricted._identity(adapter, "logical-helper", creds, {}, _api_server=api_server)
    assert identity["endpoint_identity"] == "https://provider.invalid"
    assert secret not in str(identity)


@pytest.mark.parametrize("creds,authority", [
    ({"command": "agentic-provider"}, SimpleNamespace(auth_type="external_process")),
    ({"request_overrides": {"tools": [{"type": "function"}]}}, None),
    ({"request_overrides": {"extra_body": {"tool_choice": "required"}}}, None),
])
def test_restricted_agent_refuses_external_process_and_tool_override(
        restricted_service, creds, authority):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    with pytest.raises(RuntimeError, match="restricted"):
        restricted._new_restricted_agent(
            adapter, {"provider": "custom", "model": "test", "api_key": "test-key",
                      "base_url": "https://provider.invalid/v1", **creds}, None, authority,
            _api_server=api_server)

@pytest.mark.parametrize("creds", [
    {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "model": "vendor/model"},
    {"provider": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-4o:online"},
    {"provider": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-4o",
     "request_overrides": {"extra_body": {"plugins": [{"id": "web"}]}}},
    {"provider": "custom", "base_url": "https://provider.invalid/v1", "model": "test"},
    {"provider": "custom", "base_url": "https://", "model": "test"},
    {"provider": "openai", "base_url": "https://api.openai.com.evil.invalid/v1", "model": "test"},
])
def test_restricted_refuses_routes_with_uncontrolled_provider_side_capabilities(
        restricted_service, creds):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    with pytest.raises(RuntimeError, match="restricted"):
        restricted._new_restricted_agent(adapter, {"api_key": "key", **creds}, None,
                                         _api_server=api_server)

def test_identity_uses_effective_agent_model_not_requested_alias(restricted_service, monkeypatch):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: SimpleNamespace(
        model="gpt-4.1-mini", provider="openai", base_url=kw["base_url"],
        api_mode="chat_completions", request_overrides=None, enabled_toolsets=[], tools=[],
        valid_tool_names=set()))
    creds = {"provider": "openai", "model": "gpt-4o", "api_key": "key",
             "base_url": "https://api.openai.com/v1", "api_mode": "chat_completions"}
    agent = restricted._new_restricted_agent(adapter, creds, None, _api_server=api_server)
    identity = restricted._identity(adapter, "helper", creds, {}, agent=agent, _api_server=api_server)
    assert identity["resolved_model"] == "gpt-4.1-mini"

@pytest.mark.parametrize("model", [
    "gpt-5-search-api", "gpt-5-search-api-2025-10-14", "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview-2025-03-11", "gpt-4o-realtime-preview",
    "gpt-4o-audio-preview", "gpt-5-chat-latest", "gpt-5-pro", "gpt-5-codex",
    "gpt-5-2025-08-07-search-api", "unknown-model", "GPT-4o", "gpt-4o ",
])
def test_restricted_route_denies_unapproved_openai_models_before_agent_creation(
        restricted_service, monkeypatch, model):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: pytest.fail("model rejected before agent creation"))
    with pytest.raises(RuntimeError, match="restricted"):
        restricted._new_restricted_agent(adapter, {"provider": "openai", "model": model,
            "base_url": "https://api.openai.com/v1", "api_key": "key"}, None, _api_server=api_server)

def test_restricted_real_chat_completions_wire_is_model_only(restricted_service, monkeypatch):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(adapter, {"provider": "openai", "model": "gpt-4o",
        "api_key": "test-key", "base_url": "https://api.openai.com/v1",
        "api_mode": "chat_completions"}, None, _api_server=api_server)
    wire = []
    def respond(request):
        wire.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"id": "chatcmpl-bounded", "object": "chat.completion",
            "model": "gpt-4o", "created": 1, "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "bounded"}, "finish_reason": "stop"}]})
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        with OpenAI(api_key="test-key", base_url="https://api.openai.com/v1",
                    http_client=transport, max_retries=0) as client:
            setattr(agent, "client", client)
            # Hermes creates a fresh per-request SDK client instead of reusing
            # agent.client; intercept that construction too, with no network.
            monkeypatch.setattr(agent, "_create_openai_client", lambda *args, **kw: client)
            setattr(agent, "_cached_system_prompt", "Supplied input only.")
            setattr(agent, "_use_prompt_caching", False)
            setattr(agent, "_disable_streaming", True)
            result = restricted._restricted_agent_turn(agent, "Only this snapshot")
    assert result["final_response"] == "bounded"
    assert wire and all(url == "https://api.openai.com/v1/chat/completions" for url, _ in wire)
    assert all(body["model"] == "gpt-4o" and "Only this snapshot" in str(body["messages"])
               for _, body in wire)
    assert all(not body.get("tools") and not body.get("web_search_options")
               and not body.get("extra_body") for _, body in wire)


def test_user_selected_deepinfra_route_is_tool_free_on_real_wire(restricted_service, monkeypatch):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    creds = {"provider": "deepinfra", "model": "Qwen/Qwen3.8-Flash", "api_key": "test-key",
             "base_url": "https://api.deepinfra.com/v1/openai", "api_mode": "chat_completions"}
    with pytest.raises(RuntimeError, match="input-only"):
        restricted._new_restricted_agent(adapter, creds, None, _api_server=api_server)
    agent = restricted._new_restricted_agent(
        adapter, creds, None, _api_server=api_server, envelope="hermes_tool_free_v1")
    wire = []

    def respond(request):
        wire.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"id": "chatcmpl-restricted", "object": "chat.completion",
            "model": creds["model"], "created": 1, "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "bounded"}, "finish_reason": "stop"}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        with OpenAI(api_key="test-key", base_url=creds["base_url"],
                    http_client=transport, max_retries=0) as client:
            monkeypatch.setattr(agent, "_create_openai_client", lambda *args, **kw: client)
            monkeypatch.setattr("hermes_cli.middleware.apply_llm_request_middleware",
                                lambda *args, **kw: pytest.fail("request middleware ran"))
            monkeypatch.setattr("hermes_cli.middleware.run_llm_execution_middleware",
                                lambda *args, **kw: pytest.fail("execution middleware ran"))
            monkeypatch.setattr("agent.turn_api_request._fire_pre_api_request_hook",
                                lambda *args, **kw: pytest.fail("request hook ran"))
            agent._cached_system_prompt = "Supplied input only."
            result = restricted._restricted_agent_turn(agent, "Only this snapshot")
    assert result["final_response"] == "bounded"
    assert wire and all(url == "https://api.deepinfra.com/v1/openai/chat/completions" for url, _ in wire)
    assert all(body["model"] == creds["model"] and not body.get("tools")
               and not body.get("extra_body") and not body.get("web_search_options")
               for _, body in wire)


def test_tool_free_wire_rejects_hosted_actions_and_route_drift_before_network(restricted_service, monkeypatch):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(adapter, {
        "provider": "deepinfra", "model": "Qwen/Qwen3.8-Flash", "api_key": "test-key",
        "base_url": "https://api.deepinfra.com/v1/openai", "api_mode": "chat_completions",
    }, None, _api_server=api_server, envelope="hermes_tool_free_v1")
    safe = {"model": agent.model, "messages": [{"role": "user", "content": "snapshot"}]}
    restricted._validate_tool_free_wire(agent, safe)
    for field, value in (("tools", [{"type": "function"}]),
                         ("web_search_options", {}), ("extra_body", {"plugins": [{"id": "web"}]}),
                         ("tool_choice", "required")):
        with pytest.raises(RuntimeError, match="outbound request"):
            restricted._validate_tool_free_wire(agent, {**safe, field: value})
    agent.model = "different-model"
    with pytest.raises(RuntimeError, match="outbound request"):
        restricted._validate_tool_free_wire(agent, safe)


def test_tool_free_request_client_cannot_follow_provider_redirects(restricted_service):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(adapter, {
        "provider": "deepinfra", "model": "Qwen/Qwen3.8-Flash", "api_key": "test-key",
        "base_url": "https://api.deepinfra.com/v1/openai", "api_mode": "chat_completions",
    }, None, _api_server=api_server, envelope="hermes_tool_free_v1")
    client = agent._create_request_openai_client(reason="restricted-redirect-check")
    try:
        assert client._client.follow_redirects is False
    finally:
        agent._close_request_openai_client(client, reason="restricted-redirect-check")

@pytest.mark.asyncio
async def test_restricted_credential_cannot_invoke_general_endpoints_or_control_general_runs(
        restricted_service, monkeypatch, caplog):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    admitted = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**RESTRICTED_AUTH, "Idempotency-Key": "scoped-key"})
    assert admitted.status == 202
    assert "rejected invalid API key" not in caplog.text
    run_id = (await admitted.json())["run_id"]
    assert (await client.get(f"/v1/runs/{run_id}", headers=RESTRICTED_AUTH)).status == 200
    assert (await client.post(f"/v1/runs/{run_id}/stop", headers=RESTRICTED_AUTH)).status == 200
    for path in ("/v1/runs", "/v1/chat/completions", f"/v1/runs/{run_id}/steer"):
        response = await client.post(path, json={}, headers=RESTRICTED_AUTH)
        assert response.status == 401, path
    # General endpoint status and stop are allowed only for restricted-owned runs.
    general_id = "run_general_scope"
    adapter._run_owners[general_id] = adapter._run_idempotency_scope(
        SimpleNamespace(headers=AUTH, path="/v1/runs", method="POST"))
    adapter._run_statuses[general_id] = {"run_id": general_id, "status": "running"}
    assert (await client.get(f"/v1/runs/{general_id}", headers=RESTRICTED_AUTH)).status == 404
    assert (await client.post(f"/v1/runs/{general_id}/stop", headers=RESTRICTED_AUTH)).status == 404
    assert (await client.get(f"/v1/runs/{general_id}", headers=AUTH)).status == 200

@pytest.mark.asyncio
async def test_restricted_credential_requires_distinct_configured_key(restricted_service, monkeypatch):
    client, adapter = restricted_service
    monkeypatch.setattr(adapter, "_expected_restricted_api_key", lambda: "test-gateway-key")
    assert (await client.post("/v1/restricted-runs", json=REQUEST,
        headers={**RESTRICTED_AUTH, "Idempotency-Key": "wrong-key"})).status == 401
    assert (await client.post("/v1/restricted-runs", json=REQUEST,
        headers={**AUTH, "Idempotency-Key": "master-still-valid"})).status != 401

@pytest.mark.asyncio
async def test_restricted_credential_rotation_preserves_replay_and_status(restricted_service, monkeypatch):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    first = await client.post("/v1/restricted-runs", json=REQUEST,
                              headers={**RESTRICTED_AUTH, "Idempotency-Key": "scoped-rotation"})
    run_id = (await first.json())["run_id"]
    assert (await _terminal_status(client, run_id))["status"] == "completed"
    monkeypatch.setattr(adapter, "_expected_restricted_api_key", lambda: "new-restricted-credential-0123456789")
    rotated = {"Authorization": "Bearer new-restricted-credential-0123456789"}
    assert (await client.get(f"/v1/runs/{run_id}", headers=RESTRICTED_AUTH)).status == 401
    replay = await client.post("/v1/restricted-runs", json=REQUEST,
                               headers={**rotated, "Idempotency-Key": "scoped-rotation"})
    assert replay.status == 202
    assert (await replay.json())["run_id"] == run_id
    assert (await client.get(f"/v1/runs/{run_id}", headers=rotated)).status == 200
    assert (await client.post(f"/v1/runs/{run_id}/stop", headers=rotated)).status == 200
    assert (await client.post("/v1/runs", json={}, headers=rotated)).status == 401

def test_restricted_route_revalidates_effective_openai_model(restricted_service, monkeypatch):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: SimpleNamespace(
        model="gpt-5-search-api", provider="openai", base_url=kw["base_url"],
        api_mode="chat_completions", request_overrides=None, enabled_toolsets=[], tools=[],
        valid_tool_names=set()))
    with pytest.raises(RuntimeError, match="restricted"):
        restricted._new_restricted_agent(adapter, {"provider": "openai", "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1", "api_key": "key"}, None, _api_server=api_server)

@pytest.mark.parametrize("provider,model,base_url", [
    ("openai", "gpt-4o-2024-08-06", "https://api.openai.com/v1"),
    ("openai", "gpt-5-mini", "https://api.openai.com/v1"),
    ("anthropic", "claude-sonnet-4-5-20250929", "https://api.anthropic.com"),
])
def test_restricted_approved_model_families(provider, model, base_url):
    from gateway.platforms.api_server_restricted_runs import _validate_restricted_route
    _validate_restricted_route({"provider": provider, "model": model, "base_url": base_url})

def test_restricted_refuses_agent_route_drift_to_tool_capable_router(restricted_service, monkeypatch):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: SimpleNamespace(
        model="vendor/model:online", provider="openrouter", base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions", request_overrides=None, enabled_toolsets=[], tools=[],
        valid_tool_names=set()))
    with pytest.raises(RuntimeError, match="restricted"):
        restricted._new_restricted_agent(adapter, {"provider": "openai", "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1", "api_key": "key"}, None, _api_server=api_server)

@pytest.mark.asyncio
async def test_unsafe_profile_is_rejected_before_durable_reservation(restricted_service, monkeypatch):
    client, adapter = restricted_service
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._resolve_restricted_route",
        lambda *args, **kw: ({"provider": "openrouter", "model": "vendor/model:online",
                              "api_key": "key", "base_url": "https://openrouter.ai/api/v1",
                              "request_overrides": {"extra_body": {"plugins": [{"id": "web"}]}},
                              "auth_type": "api_key"}, None, {}, None))
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "unsafe-profile"})
    assert response.status == 403
    assert not adapter._run_idempotency_store._conn.execute(
        "SELECT 1 FROM run_idempotency WHERE idempotency_key='unsafe-profile'").fetchone()

@pytest.mark.asyncio
async def test_restricted_key_rotation_retains_replay_and_control(restricted_service, monkeypatch):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    headers = {**AUTH, "Idempotency-Key": "rotation"}
    first = await client.post("/v1/restricted-runs", json=REQUEST, headers=headers)
    run_id = (await first.json())["run_id"]
    assert (await _terminal_status(client, run_id))["status"] == "completed"
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "rotated-gateway-key")
    rotated = {"Authorization": "Bearer rotated-gateway-key"}
    replay = await client.post("/v1/restricted-runs", json=REQUEST,
                               headers={**rotated, "Idempotency-Key": "rotation"})
    assert replay.status == 202
    assert (await replay.json())["run_id"] == run_id
    assert (await client.get(f"/v1/runs/{run_id}", headers=rotated)).status == 200
    assert (await client.post(f"/v1/runs/{run_id}/stop", headers=rotated)).status == 200

@pytest.mark.asyncio
async def test_restricted_terminal_reservation_never_expires_after_default_ttl(
        restricted_service, monkeypatch):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    headers = {**AUTH, "Idempotency-Key": "old-terminal"}
    first = await client.post("/v1/restricted-runs", json=REQUEST, headers=headers)
    run_id = (await first.json())["run_id"]
    assert (await _terminal_status(client, run_id))["status"] == "completed"
    from gateway.platforms import api_server_run_idempotency as idem
    real_time = time.time
    monkeypatch.setattr(idem.time, "time", lambda: real_time() + idem.RunIdempotencyStore.RETENTION_SECONDS + 10)
    replay = await client.post("/v1/restricted-runs", json=REQUEST, headers=headers)
    assert (await replay.json())["run_id"] == run_id


def test_real_restricted_agent_refuses_hallucinated_mutation_and_recursive_delegation(
        restricted_service, monkeypatch):
    from gateway.platforms import api_server, api_server_restricted_runs as restricted
    _, adapter = restricted_service
    agent = restricted._new_restricted_agent(
        adapter, {"provider": "openai", "model": "gpt-4o", "api_key": "test-key",
                  "base_url": "https://api.openai.com/v1", "api_mode": "chat_completions"},
        None, _api_server=api_server)
    agent.client = MagicMock()
    agent._cached_system_prompt = "Supplied input only."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    for name in ("terminal", "delegate_task"):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=None, reasoning_content=None, reasoning=None,
                    tool_calls=[SimpleNamespace(id="attack", type="function",
                        function=SimpleNamespace(name=name, arguments="{}"))]),
                finish_reason="tool_calls")], model="test", usage=None)
        agent.client.chat.completions.create.side_effect = [response] * 4
        def forbidden_dispatch(*args, **kwargs):
            pytest.fail(f"forbidden tool {name} reached dispatcher")
        monkeypatch.setattr("model_tools.handle_function_call", forbidden_dispatch)
        result = agent.run_conversation("Analyze the provided text only", conversation_history=[])
        assert agent.client.chat.completions.create.call_count >= 1
        assert result.get("completed") is False or result.get("failed") or result.get("partial")


@pytest.mark.asyncio
async def test_restricted_worker_keeps_owning_profile_across_thread_hops(restricted_service, tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from gateway.platforms import api_server_restricted_runs as restricted
    from agent.secret_scope import get_secret

    _, adapter = restricted_service
    homes = {}
    for name in ("alpha", "beta"):
        home = tmp_path / name
        home.mkdir()
        (home / ".env").write_text(f"RESTRICTED_TEST_SECRET={name}-only\n")
        homes[name] = home
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])

    class ScopeAgent(FakeAgent):
        def run_conversation(self, **kwargs):
            return {"final_response": f"{get_hermes_home()}:{get_secret('RESTRICTED_TEST_SECRET')}",
                    "completed": True}

    for name in ("alpha", "beta", "alpha"):
        agent = ScopeAgent(enabled_toolsets=[])
        result = await asyncio.get_running_loop().run_in_executor(
            None, restricted._restricted_agent_turn_scoped, adapter, agent, "input", name)
        assert result["final_response"] == f"{homes[name]}:{name}-only"


@pytest.mark.asyncio
async def test_restricted_failure_is_not_reported_as_success(restricted_service, monkeypatch):
    client, _ = restricted_service
    _route_mocks(monkeypatch)
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._restricted_agent_turn",
                        lambda agent, text: {"final_response": "incomplete", "partial": True, "completed": False})
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "partial"})
    run_id = (await response.json())["run_id"]

    status = await _terminal_status(client, run_id)
    assert status["status"] == "failed"
    assert status.get("completed") is False
    assert not status.get("output")


@pytest.mark.asyncio
async def test_result_persistence_failure_never_claims_completed(restricted_service, monkeypatch):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    store = adapter._run_idempotency_store
    original = store.update_status
    def fail_completion(run_id, status):
        if status.get("status") == "completed":
            raise OSError("fake durable storage outage")
        return original(run_id, status)
    monkeypatch.setattr(store, "update_status", fail_completion)
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "fail-final-persist"})
    run_id = (await response.json())["run_id"]
    status = await _terminal_status(client, run_id)
    assert status["status"] != "completed"
    record = store.status_for_run(adapter._run_owners[run_id], run_id)
    assert record["status"]["status"] != "completed"


@pytest.mark.asyncio
async def test_restricted_steer_is_forbidden(restricted_service, monkeypatch):
    client, _ = restricted_service
    observed = _route_mocks(monkeypatch)
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "steer"})
    run_id = (await response.json())["run_id"]
    agent = observed["agent"]
    agent.steer = lambda text: True
    assert await asyncio.to_thread(agent.started_event.wait, 1)
    reply = await client.post(f"/v1/runs/{run_id}/steer", json={"input": "expand capabilities"}, headers=AUTH)
    assert reply.status == 409


@pytest.mark.asyncio
async def test_restricted_oversubscribed_admission_refused(restricted_service, monkeypatch):
    client, adapter = restricted_service
    adapter._max_concurrent_runs = 1
    observed = _route_mocks(monkeypatch)
    release = threading.Event()
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._restricted_agent_turn",
                        lambda agent, text: (release.wait(5) or {"final_response": "done"}))
    try:
        first = await client.post("/v1/restricted-runs", json=REQUEST,
                                  headers={**AUTH, "Idempotency-Key": "cap-first"})
        assert first.status == 202
        first_id = (await first.json())["run_id"]
        retry = await client.post("/v1/restricted-runs", json=REQUEST,
                                  headers={**AUTH, "Idempotency-Key": "cap-first"})
        assert retry.status == 202
        assert (await retry.json())["run_id"] == first_id
        second = await client.post("/v1/restricted-runs", json=REQUEST,
                                   headers={**AUTH, "Idempotency-Key": "cap-second"})
        assert second.status == 429
    finally:
        release.set()


@pytest.mark.asyncio
async def test_missing_durable_store_rejected_before_route_resolution(restricted_service, monkeypatch):
    client, adapter = restricted_service
    adapter._run_idempotency_store._db_path = None
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._resolve_restricted_route",
                        lambda *args, **kwargs: pytest.fail("no route resolution without durable admission"))
    response = await client.post("/v1/restricted-runs", json=REQUEST,
                                 headers={**AUTH, "Idempotency-Key": "non-durable"})
    assert response.status == 503


@pytest.mark.asyncio
async def test_restricted_replay_and_status_survive_adapter_restart(restricted_service, monkeypatch):
    client, _ = restricted_service
    _route_mocks(monkeypatch)
    headers = {**AUTH, "Idempotency-Key": "restart-replay"}
    first = await client.post("/v1/restricted-runs", json=REQUEST, headers=headers)
    run_id = (await first.json())["run_id"]
    assert (await _terminal_status(client, run_id))["status"] == "completed"

    other = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-gateway-key"}))
    app = web.Application()
    app.router.add_post("/v1/restricted-runs", other._handle_restricted_runs)
    app.router.add_get("/v1/runs/{run_id}", other._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/stop", other._handle_stop_run)
    try:
        monkeypatch.setattr("tools.delegate_tool_config._resolve_profile_execution",
                            lambda *_: pytest.fail("replay must not resolve the profile again"))
        async with TestClient(TestServer(app)) as restarted:
            replay = await restarted.post("/v1/restricted-runs", json=REQUEST, headers=headers)
            assert replay.status == 202
            replay_data = await replay.json()
            assert replay_data["run_id"] == run_id
            assert replay_data["status"] == "completed"
            assert replay.headers["Idempotency-Replayed"] == "true"
            status = await restarted.get(f"/v1/runs/{run_id}", headers=AUTH)
            assert (await status.json())["status"] == "completed"
            stop = await restarted.post(f"/v1/runs/{run_id}/stop", headers=AUTH)
            assert (await stop.json())["status"] == "completed"
    finally:
        other._run_idempotency_store.close()
        other._response_store.close()


@pytest.mark.asyncio
async def test_stale_restricted_run_is_interrupted_not_falsely_stopped(restricted_service, monkeypatch):
    client, adapter = restricted_service
    _route_mocks(monkeypatch)
    release = threading.Event()
    monkeypatch.setattr("gateway.platforms.api_server_restricted_runs._restricted_agent_turn",
                        lambda agent, text: (release.wait(5) or {"final_response": "done"}))
    try:
        response = await client.post("/v1/restricted-runs", json=REQUEST,
                                     headers={**AUTH, "Idempotency-Key": "stale"})
        run_id = (await response.json())["run_id"]
        adapter._run_idempotency_store._conn.execute(
            "UPDATE run_idempotency SET owner_pid=? WHERE run_id=?", (-1, run_id))
        adapter._run_idempotency_store._conn.commit()
        other = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-gateway-key"}))
        app = web.Application()
        app.router.add_get("/v1/runs/{run_id}", other._handle_get_run)
        app.router.add_post("/v1/runs/{run_id}/stop", other._handle_stop_run)
        try:
            async with TestClient(TestServer(app)) as restarted:
                status = await restarted.get(f"/v1/runs/{run_id}", headers=AUTH)
                assert (await status.json())["status"] == "interrupted"
                stop = await restarted.post(f"/v1/runs/{run_id}/stop", headers=AUTH)
                assert (await stop.json())["status"] == "interrupted"
        finally:
            other._run_idempotency_store.close()
            other._response_store.close()
    finally:
        release.set()


def test_checked_restricted_identity_endpoints_are_advertised_and_registered(restricted_service):
    _, adapter = restricted_service
    from gateway.platforms import api_server

    paths = {(method, path) for method, path, _ in adapter._http_route_table()}
    assert ("POST", "/v1/restricted-runs/resolve") in paths
    assert ("POST", "/v1/restricted-runs/identity-checked") in paths
    advertised = {name: {"method": method, "path": path}
                  for name, (method, path) in api_server._CAPABILITY_ENDPOINTS}
    assert advertised["restricted_run_identity_resolve"] == {
        "method": "POST", "path": "/v1/restricted-runs/resolve"}
    assert advertised["restricted_run_identity_checked"] == {
        "method": "POST", "path": "/v1/restricted-runs/identity-checked"}
    assert api_server._RESTRICTED_IDENTITY_CONTRACT["version"] == 1


@pytest.mark.asyncio
async def test_restricted_identity_resolution_is_nonexecuting_and_checked_admission_matches(
        restricted_service, monkeypatch):
    client, adapter = restricted_service
    observed = _route_mocks(monkeypatch)
    resolve_body = {key: REQUEST[key] for key in (
        "delegation_profile_id", "work_class", "capability_envelope")}
    resolved = await client.post("/v1/restricted-runs/resolve", json=resolve_body, headers=RESTRICTED_AUTH)
    assert resolved.status == 200
    identity = (await resolved.json())["identity"]
    assert identity["hermes_delegation_profile_id"] == REQUEST["delegation_profile_id"]
    assert identity["resolved_provider"] == "provider-test"
    assert identity["resolved_model"] == "model-test"
    assert identity["work_class"] == REQUEST["work_class"]
    assert identity["capability_envelope"] == REQUEST["capability_envelope"]
    assert not observed["agent"].started_event.is_set()
    assert observed["agent"].closed is True
    assert not adapter._run_idempotency_store._conn.execute(
        "SELECT 1 FROM run_idempotency").fetchone()

    headers = {**RESTRICTED_AUTH, "Idempotency-Key": "identity-match"}
    admitted = await client.post("/v1/restricted-runs/identity-checked",
                                 json={**REQUEST, "expected_identity": identity}, headers=headers)
    assert admitted.status == 202
    data = await admitted.json()
    assert data["resolved_identity"] == identity
    status = await _terminal_status(client, data["run_id"])
    assert status["status"] == "completed"
    replay = await client.post("/v1/restricted-runs/identity-checked",
                               json={**REQUEST, "expected_identity": identity}, headers=headers)
    assert replay.status == 202
    assert (await replay.json())["run_id"] == data["run_id"]
    changed_identity = {**identity, "resolved_model": "other-model"}
    conflict = await client.post("/v1/restricted-runs/identity-checked",
        json={**REQUEST, "expected_identity": changed_identity}, headers=headers)
    assert conflict.status == 409


@pytest.mark.parametrize(("field", "wrong"), [
    ("hermes_delegation_profile_id", "other-profile"),
    ("resolved_provider", "other-provider"), ("resolved_model", "other-model"),
    ("effective_api_mode", "other-mode"), ("endpoint_identity", "https://other.invalid"),
    ("auth_type", "oauth"), ("auth_source_category", "credential_pool"),
    ("route_revision", "f" * 64), ("work_class", "log_triage"),
    ("capability_envelope", "hermes_tool_free_v1"), ("source", "other-source"),
])
@pytest.mark.asyncio
async def test_checked_restricted_identity_mismatch_creates_no_run(restricted_service, monkeypatch, field, wrong):
    client, adapter = restricted_service
    observed = _route_mocks(monkeypatch)
    resolve_body = {key: REQUEST[key] for key in (
        "delegation_profile_id", "work_class", "capability_envelope")}
    resolved = await client.post("/v1/restricted-runs/resolve", json=resolve_body, headers=AUTH)
    identity = (await resolved.json())["identity"]
    wrong = {**identity, field: wrong}
    response = await client.post("/v1/restricted-runs/identity-checked",
        json={**REQUEST, "expected_identity": wrong},
        headers={**AUTH, "Idempotency-Key": "identity-mismatch"})
    assert response.status == 409
    assert (await response.json())["error"]["code"] == "restricted_identity_mismatch"
    assert not adapter._run_idempotency_store._conn.execute(
        "SELECT 1 FROM run_idempotency").fetchone()
    assert not adapter._active_run_agents
    assert not observed["agent"].started_event.is_set()
