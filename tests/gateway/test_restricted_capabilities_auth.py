"""Restricted capability discovery and auth-scope regressions for #39."""

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

MASTER = {"Authorization": "Bearer synthetic-master-key-0123456789"}
RESTRICTED = {"Authorization": "Bearer synthetic-restricted-key-0123456789"}
CONTRACT = {
    "version": 1,
    "resolve": {"method": "POST", "path": "/v1/restricted-runs/resolve"},
    "checked_admission": {"method": "POST", "path": "/v1/restricted-runs/identity-checked"},
}
ENDPOINTS = {
    "restricted_run_identity_resolve": CONTRACT["resolve"],
    "restricted_run_identity_checked": CONTRACT["checked_admission"],
}


@pytest_asyncio.fixture
async def capability_service(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": MASTER["Authorization"][7:], "restricted_key": RESTRICTED["Authorization"][7:]}))
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_get("/api/model/options", adapter._handle_model_options)
    async with TestClient(TestServer(app)) as client:
        yield client, adapter
    adapter._run_idempotency_store.close()
    adapter._response_store.close()


@pytest.mark.asyncio
async def test_restricted_capabilities_read_is_contract_only(capability_service):
    client, adapter = capability_service
    response = await client.get("/v1/capabilities", headers=RESTRICTED)
    print("restricted capability HTTP status:", response.status)
    assert response.status == 200
    document = await response.json()
    assert document == {
        "object": "hermes.api_server.capabilities", "platform": "hermes-agent",
        "features": {"restricted_run_identity": CONTRACT}, "endpoints": ENDPOINTS,
    }
    assert not adapter._active_run_agents
    assert adapter._run_idempotency_store._conn.execute(
        "SELECT COUNT(*) FROM run_idempotency").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_master_capabilities_preserve_full_surface(capability_service):
    client, _ = capability_service
    response = await client.get("/v1/capabilities", headers=MASTER)
    assert response.status == 200
    document = await response.json()
    assert document["features"]["restricted_run_identity"] == CONTRACT
    assert document["features"]["chat_completions"] is True
    assert "model" in document and "runtime" in document
    assert "browser_extension_control" in document["features"]


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer invalid-token"},
                                     {"Authorization": "Bearer invalid-λ-token"}])
async def test_capabilities_reject_anonymous_and_invalid(capability_service, headers):
    client, _ = capability_service
    response = await client.get("/v1/capabilities", headers=headers)
    assert response.status == 401
    assert (await response.json())["error"]["code"] == "gateway_auth_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", [("POST", "/v1/runs"),
    ("POST", "/v1/chat/completions"), ("GET", "/api/model/options")])
async def test_restricted_capability_read_does_not_grant_master_access(capability_service, method, path):
    client, _ = capability_service
    response = await client.request(method, path, headers=RESTRICTED)
    assert response.status == 401
    assert (await response.json())["error"]["code"] == "gateway_auth_failed"
