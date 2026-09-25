"""Real HTTP admission keeps exact provider secrets only for active invocations."""

import asyncio
import logging
import threading
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import redact
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

SECRET = "ARBITRARY-CREDENTIAL-7f19-not-a-known-key-pattern"
AUTH = {"Authorization": "Bearer gateway-test-key-0123456789"}
BODY = {"input": "exercise", "provider": "deepinfra", "model": "test-model"}


def _app(adapter):
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _count(secret=SECRET):
    with redact._VAULT_REDACTION_LOCK:
        return redact._PROVIDER_REDACTION_VALUES.get(redact._vault_scope(), {}).get(secret, 0)


async def _until(condition):
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0.02)
    assert condition(), "worker did not reach expected state"


class BlockingAgent:
    def __init__(self, gates):
        self.gates = gates
        self.started = threading.Event()
        self.provider = "deepinfra"
        self.model = "test-model"
        self.session_prompt_tokens = self.session_completion_tokens = self.session_total_tokens = 0

    def run_conversation(self, **kwargs):
        self.started.set()
        self.gates[kwargs["task_id"]].wait(timeout=5)
        return {"final_response": "done", "completed": True}

    def interrupt(self):
        return None


@pytest.mark.asyncio
async def test_two_concurrent_runs_keep_shared_secret_until_both_workers_end():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    gates = defaultdict(threading.Event)
    agents = []

    def create(**kwargs):
        agent = BlockingAgent(gates)
        agents.append(agent)
        return agent

    headers = {**AUTH, "X-Hermes-Provider-API-Key": SECRET}
    with patch.object(adapter, "_create_agent", side_effect=create):
        async with TestClient(TestServer(_app(adapter))) as cli:
            try:
                runs = []
                for n in range(2):
                    response = await cli.post("/v1/runs", json={**BODY, "input": f"run {n}"}, headers=headers)
                    assert response.status == 202, await response.text()
                    runs.append((await response.json())["run_id"])
                    gates[runs[-1]]
                await _until(lambda: len(agents) == 2 and all(a.started.is_set() for a in agents))
                assert _count() == 2
                assert SECRET not in redact.redact_sensitive_text("upstream echoed " + SECRET, force=True)
                gates[runs[0]].set()
                await _until(lambda: _count() == 1)
                assert SECRET not in redact.redact_sensitive_text(SECRET, force=True)
                gates[runs[1]].set()
                await _until(lambda: _count() == 0)
                for run_id in runs:
                    state = await (await cli.get(f"/v1/runs/{run_id}", headers=AUTH)).json()
                    assert state["status"] == "completed"
            finally:
                for gate in gates.values():
                    gate.set()
                await _until(lambda: _count() == 0)


@pytest.mark.asyncio
async def test_cancelled_chat_await_retains_redaction_until_executor_exits():
    from gateway.platforms.api_server_provider_credentials import ProviderCredentialOverride
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    gates = defaultdict(threading.Event)
    agent = BlockingAgent(gates)
    credential = ProviderCredentialOverride(provider="deepinfra", api_key=SECRET)
    with patch.object(adapter, "_create_agent", return_value=agent):
        task = asyncio.create_task(adapter._run_agent(
            user_message="worker", conversation_history=[], session_id="cancel-thread",
            provider_credential=credential))
        try:
            await _until(agent.started.is_set)
            assert _count() == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert _count() == 1
            gates["cancel-thread"].set()
            await _until(lambda: _count() == 0)
        finally:
            gates["cancel-thread"].set()
            await _until(lambda: _count() == 0)


@pytest.mark.asyncio
async def test_cancelled_run_task_keeps_redaction_until_executor_thread_exits():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    gates = defaultdict(threading.Event)
    agents = []
    def create(**kwargs):
        agent = BlockingAgent(gates)
        agents.append(agent)
        return agent
    with patch.object(adapter, "_create_agent", side_effect=create):
        async with TestClient(TestServer(_app(adapter))) as cli:
            response = await cli.post("/v1/runs", json=BODY,
                                      headers={**AUTH, "X-Hermes-Provider-API-Key": SECRET})
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            try:
                await _until(lambda: agents and agents[0].started.is_set())
                assert _count() == 1
                stop = await cli.post(f"/v1/runs/{run_id}/stop", json={}, headers=AUTH)
                assert stop.status == 200
                adapter._active_run_tasks[run_id].cancel()
                await _until(lambda: adapter._active_run_tasks.get(run_id) is None)
                assert _count() == 1, "cancelled asyncio task must not unredact running thread"
                gates[run_id].set()
                await _until(lambda: _count() == 0)
            finally:
                gates[run_id].set()
                await _until(lambda: _count() == 0)


@pytest.mark.asyncio
async def test_steer_error_does_not_log_active_provider_secret(caplog):
    caplog.set_level(logging.DEBUG)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    gates = defaultdict(threading.Event)
    agent = BlockingAgent(gates)
    with patch.object(adapter, "_create_agent", return_value=agent), \
         patch.object(agent, "steer", create=True, side_effect=RuntimeError(SECRET)):
        async with TestClient(TestServer(_app(adapter))) as cli:
            response = await cli.post("/v1/runs", json=BODY,
                headers={**AUTH, "X-Hermes-Provider-API-Key": SECRET})
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            try:
                await _until(lambda: agent.started.is_set() and
                    adapter._run_statuses.get(run_id, {}).get("status") == "running")
                result = await cli.post(f"/v1/runs/{run_id}/steer",
                    json={"input": "continue"}, headers=AUTH)
                assert result.status == 500
                assert SECRET not in await result.text(), "steer response exposed provider key"
                assert SECRET not in caplog.text, "steer traceback exposed provider key"
            finally:
                gates[run_id].set()
                await _until(lambda: _count() == 0)


@pytest.mark.asyncio
async def test_approval_error_does_not_echo_active_provider_secret(caplog):
    caplog.set_level(logging.DEBUG)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    gates = defaultdict(threading.Event)
    agent = BlockingAgent(gates)
    with patch.object(adapter, "_create_agent", return_value=agent), \
         patch("tools.approval.resolve_gateway_approval", side_effect=RuntimeError(SECRET)):
        async with TestClient(TestServer(_app(adapter))) as cli:
            response = await cli.post("/v1/runs", json=BODY,
                headers={**AUTH, "X-Hermes-Provider-API-Key": SECRET})
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            try:
                await _until(lambda: agent.started.is_set() and
                    adapter._run_statuses.get(run_id, {}).get("status") == "running")
                result = await cli.post(f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"}, headers=AUTH)
                assert result.status == 500
                assert SECRET not in await result.text(), "approval response exposed provider key"
                assert SECRET not in caplog.text, "approval traceback exposed provider key"
            finally:
                gates[run_id].set()
                await _until(lambda: _count() == 0)


@pytest.mark.asyncio
async def test_missing_gateway_fingerprint_secret_rejects_before_run_admission():
    from gateway import hosted_room_peer
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    with patch.object(hosted_room_peer, "gateway_room_grant_secret", side_effect=OSError("unavailable")), \
         patch.object(adapter, "_create_agent") as create:
        async with TestClient(TestServer(_app(adapter))) as cli:
            response = await cli.post("/v1/runs", json=BODY,
                headers={**AUTH, "X-Hermes-Provider-API-Key": SECRET, "Idempotency-Key": "fp-missing"})
            data = await response.json()
            assert response.status == 503
            assert data["error"]["code"] == "provider_credential_fingerprint_unavailable"
            assert SECRET not in str(data)
    create.assert_not_called()
    assert not adapter._run_statuses
    assert adapter._run_idempotency_store.lookup(
        adapter._run_idempotency_scope(type("Request", (), {"headers": AUTH})()),
        "fp-missing", "unused")[0] == "missing"


@pytest.mark.parametrize("provider", ["deepinfra", "anthropic"])
@pytest.mark.asyncio
async def test_base_only_http_request_never_substitutes_gateway_key_into_caller_endpoint(
    provider, tmp_path, monkeypatch, caplog
):
    static = "STATIC_GATEWAY_KEY_DO_NOT_SEND"
    home = Path(tmp_path) / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    caplog.set_level(logging.DEBUG)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    delivered = []
    async def fake_endpoint(request):
        delivered.append(dict(request.headers))
        return web.json_response({"error": "unexpected provider invocation"}, status=500)
    remote = web.Application()
    remote.router.add_route("*", "/{path:.*}", fake_endpoint)
    with patch("gateway.run._resolve_runtime_agent_kwargs",
               return_value={"provider": provider, "api_key": static}), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolve, \
         patch.object(adapter, "_create_agent") as create:
        async with TestServer(remote) as endpoint, TestClient(TestServer(_app(adapter))) as cli:
            response = await cli.post("/v1/runs", json={**BODY, "provider": provider,
                "provider_base_url": str(endpoint.make_url("/v1"))}, headers=AUTH)
            data = await response.json()
    assert response.status == 400
    assert data["error"]["code"] == "provider_api_key_required"
    assert static not in str(data)
    assert static not in caplog.text
    for path in home.rglob("*"):
        if path.is_file():
            assert static.encode() not in path.read_bytes(), f"static key persisted in {path.name}"
    assert delivered == []
    resolve.assert_not_called()
    create.assert_not_called()
    assert not adapter._run_statuses


@pytest.mark.asyncio
async def test_invalid_bearer_cannot_reach_credential_extraction():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    with patch("gateway.platforms.api_server_provider_credentials.extract_provider_credential",
               side_effect=AssertionError("credential extraction reached")) as extract:
        async with TestClient(TestServer(_app(adapter))) as cli:
            response = await cli.post("/v1/runs", json=BODY,
                headers={"Authorization": "Bearer invalid", "X-Hermes-Provider-API-Key": SECRET})
            assert response.status == 401
    extract.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_chat_request_releases_credential_after_handler_returns():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    async with TestClient(TestServer(_app(adapter))) as cli:
        response = await cli.post("/v1/chat/completions", json={"provider": "deepinfra", "messages": []},
                                  headers={**AUTH, "X-Hermes-Provider-API-Key": SECRET})
        assert response.status == 400
        assert SECRET not in await response.text()
        await _until(lambda: _count() == 0)
