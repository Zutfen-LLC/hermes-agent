"""Real HTTP sentinel sweep: request provider keys must not escape into durable or public state."""

import asyncio
import logging
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_provider_credentials import PROVIDER_API_KEY_HEADER

SECRET = "ordinary-arbitrary-provider-secret-6bbf21c2-no-pattern"
AUTH = {"Authorization": "Bearer gateway-authentication-key-for-tests"}


def _app(adapter):
    app = web.Application()
    app["api_server_adapter"] = adapter
    routes = (
        ("POST", "/v1/runs", adapter._handle_runs),
        ("GET", "/v1/runs/{run_id}", adapter._handle_get_run),
        ("POST", "/v1/runs/{run_id}/stop", adapter._handle_stop_run),
        ("POST", "/v1/chat/completions", adapter._handle_chat_completions),
        ("POST", "/v1/responses", adapter._handle_responses),
        ("GET", "/v1/responses/{response_id}", adapter._handle_get_response),
    )
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    return app


class Agent:
    provider = "deepinfra"
    model = "test-model"
    session_id = "sentinel-session"
    session_prompt_tokens = session_completion_tokens = session_total_tokens = 0

    def __init__(self, *, failure=None, output="safe reply"):
        self.failure, self.output = failure, output

    def run_conversation(self, **kwargs):
        if self.failure:
            raise RuntimeError(self.failure)
        callback = kwargs.get("streaming_callback")
        if callback:
            callback(self.output)
        return {"final_response": self.output, "completed": True}

    def interrupt(self):
        return None


@pytest_asyncio.fixture
async def service(tmp_path, monkeypatch, caplog):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": AUTH["Authorization"].split()[-1]}))
    adapter._create_agent = lambda **kwargs: Agent()
    caplog.set_level(logging.DEBUG)
    async with TestClient(TestServer(_app(adapter))) as client:
        yield client, adapter, home, caplog
    adapter._response_store.close()
    close = getattr(adapter._session_db, "close", None)
    if callable(close):
        close()


def _headers(secret=SECRET):
    return {**AUTH, PROVIDER_API_KEY_HEADER: secret}


def _assert_no_secret(client, home, caplog, *public_values):
    assert SECRET not in caplog.text
    for value in public_values:
        assert SECRET not in str(value)
    for path in home.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), f"provider secret persisted in {path.name}"


@pytest.mark.asyncio
async def test_runs_success_and_lost_acceptance_replay_are_secret_free(service):
    client, adapter, home, caplog = service
    adapter._create_agent = lambda **kwargs: Agent()
    body = {"input": "hello", "provider": "deepinfra", "model": "test-model"}
    response = await client.post("/v1/runs", json=body, headers={**_headers(), "Idempotency-Key": "stable"})
    assert response.status == 202
    first = await response.json()
    await asyncio.sleep(0.05)
    replay = await client.post("/v1/runs", json=body, headers={**_headers(), "Idempotency-Key": "stable"})
    assert replay.status == 202
    replay_data = await replay.json()
    assert replay_data["run_id"] == first["run_id"]
    _assert_no_secret(client, home, caplog, first, replay_data, adapter._run_statuses, adapter._run_idempotency_store)


@pytest.mark.asyncio
async def test_changed_credential_idempotency_replay_returns_secret_free_409(service):
    client, adapter, home, caplog = service
    body = {"input": "hello", "provider": "deepinfra", "model": "test-model"}
    first = await client.post("/v1/runs", json=body, headers={**_headers(), "Idempotency-Key": "rotate"})
    assert first.status == 202
    await asyncio.sleep(0.05)
    changed = "different-arbitrary-secret-8041"
    response = await client.post("/v1/runs", json=body, headers={**_headers(changed), "Idempotency-Key": "rotate"})
    text = await response.text()
    assert response.status == 409
    assert SECRET not in text and changed not in text
    _assert_no_secret(client, home, caplog, adapter._run_statuses, adapter._run_idempotency_store)


@pytest.mark.asyncio
async def test_changed_provider_base_url_and_dropped_key_replay_are_409(service):
    client, adapter, home, caplog = service
    body = {"input": "hello", "provider": "deepinfra", "provider_base_url": "https://example.test/v1"}
    headers = {**_headers(), "Idempotency-Key": "base-url-conflict"}
    admitted = await client.post("/v1/runs", json=body, headers=headers)
    assert admitted.status == 202
    changed = await client.post("/v1/runs", json={**body, "provider_base_url": "https://example.test/v2"}, headers=headers)
    assert changed.status == 409
    changed_data = await changed.json()
    assert changed_data["error"]["code"] == "idempotency_key_conflict"
    dropped = await client.post("/v1/runs", json={k: v for k, v in body.items() if k != "provider_base_url"},
                                headers={**AUTH, "Idempotency-Key": "base-url-conflict"})
    assert dropped.status == 409
    assert (await dropped.json())["error"]["code"] == "idempotency_key_conflict"
    _assert_no_secret(client, home, caplog, changed_data, adapter._run_statuses,
                      adapter._run_idempotency_store)


@pytest.mark.asyncio
async def test_invalid_request_and_failed_run_have_no_secret_in_response_or_storage(service):
    client, adapter, home, caplog = service
    invalid = await client.post("/v1/chat/completions", json={"provider": "deepinfra", "messages": []}, headers=_headers())
    assert invalid.status == 400
    invalid_text = await invalid.text()
    assert SECRET not in invalid_text
    adapter._create_agent = lambda **kwargs: Agent(failure=SECRET)
    failed = await client.post("/v1/runs", json={"input": "hello", "provider": "deepinfra"}, headers=_headers())
    assert failed.status == 202
    run = await failed.json()
    await asyncio.sleep(0.1)
    status = await (await client.get(f"/v1/runs/{run['run_id']}", headers=AUTH)).text()
    assert SECRET not in status
    _assert_no_secret(client, home, caplog, run, status, adapter._run_statuses)


@pytest.mark.asyncio
async def test_normal_chat_and_session_chat_are_secret_free(service):
    client, adapter, home, caplog = service
    adapter._run_agent = lambda **kwargs: asyncio.sleep(0, result=({"final_response": "safe reply", "session_id": "sentinel-session"}, {}))
    chat = await client.post("/v1/chat/completions", json={"provider": "deepinfra", "messages": [{"role": "user", "content": "hello"}]}, headers=_headers())
    chat_text = await chat.text()
    assert chat.status == 200 and SECRET not in chat_text
    # The authenticated session endpoint parses the same scoped header and invokes the real handler.
    session = await client.post("/api/sessions/sentinel-session/chat", json={"message": "hello", "provider": "deepinfra"}, headers=_headers())
    session_text = await session.text()
    assert SECRET not in session_text
    _assert_no_secret(client, home, caplog, chat_text, session_text)


@pytest.mark.asyncio
async def test_responses_stream_snapshot_and_cancellation_sentinel_sweep(service):
    client, adapter, home, caplog = service
    adapter._run_agent = lambda **kwargs: asyncio.sleep(0, result=({"final_response": "safe response", "session_id": "sentinel-session"}, {}))
    response = await client.post("/v1/responses", json={"input": "hello", "provider": "deepinfra", "stream": True}, headers=_headers())
    wire = await response.text()
    assert response.status == 200 and SECRET not in wire
    for (stored,) in adapter._response_store._conn.execute("SELECT data FROM responses"):
        assert SECRET not in stored
    _assert_no_secret(client, home, caplog, wire)

    # A pre-handler cancellation is covered through the real run stop route after admission.
    gate = asyncio.Event()
    async def blocked(**kwargs):
        await gate.wait()
        return {"final_response": "safe"}, {}
    adapter._run_agent = blocked
    admitted = await client.post("/v1/runs", json={"input": "wait", "provider": "deepinfra"}, headers=_headers())
    run = await admitted.json()
    await asyncio.sleep(0.03)
    stopped = await client.post(f"/v1/runs/{run['run_id']}/stop", json={}, headers=AUTH)
    stop_text = await stopped.text()
    gate.set()
    await asyncio.sleep(0.05)
    _assert_no_secret(client, home, caplog, run, stop_text, adapter._run_statuses)
