"""Provider base URLs are part of request-scoped runtime identity."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_provider_credentials as pc
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)

SENTINEL = "sk-prv...0001"
PROVIDER = "deepinfra"
URL_A = "https://endpoint-a.test/v1"
URL_B = "https://endpoint-b.test/v1"
GATEWAY_KEY = "sk-gat...6789"


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": GATEWAY_KEY}))


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {GATEWAY_KEY}"}
    headers.update(extra or {})
    return headers


def _create_app(adapter: APIServerAdapter) -> web.Application:
    middleware = [
        mw for mw in (cors_middleware, security_headers_middleware) if mw is not None
    ]
    app = web.Application(middlewares=middleware)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _credential(base_url: str, *, key: str = SENTINEL):
    request = SimpleNamespace(headers={pc.PROVIDER_API_KEY_HEADER: key})
    body = {"provider": PROVIDER, "provider_base_url": base_url}
    return pc.extract_provider_credential(
        _make_adapter(), request, body, scope_fn=lambda: "test-principal"
    )


def _fake_run_agent(captured):
    async def fake_run_agent(**kwargs):
        credential = kwargs.get("provider_credential")
        base_url = credential.base_url if credential else "no-credential"
        captured.append(base_url)
        return (
            {
                "final_response": f"computed-for:{base_url}",
                "session_id": "idem-session",
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    return fake_run_agent


def _chat_body():
    return {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "provider": PROVIDER,
        "provider_base_url": URL_A,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _responses_body():
    return {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "provider": PROVIDER,
        "provider_base_url": URL_A,
        "input": "hi",
        "store": False,
    }


def _response_text(data):
    return " ".join(
        item.get("content", [{}])[0].get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
    )


def test_fingerprint_differs_when_api_key_differs():
    first = _credential(URL_A)
    second = _credential(URL_A, key="sk-prv...0002")
    assert first.fingerprint != second.fingerprint


def test_fingerprint_differs_when_base_url_differs():
    assert _credential(URL_A).fingerprint != _credential(URL_B).fingerprint


def test_fingerprint_stable_for_identical_runtime_identity():
    assert _credential(URL_A).fingerprint == _credential(URL_A).fingerprint


def test_fingerprint_contains_no_plaintext_key():
    fingerprint = _credential(URL_A).fingerprint
    assert SENTINEL not in fingerprint, "provider key material leaked into fingerprint"
    # A hex digest cannot contain the URL's punctuation either.
    assert "://" not in fingerprint


@pytest.mark.asyncio
async def test_runs_changed_base_url_remains_409():
    adapter = _make_adapter()
    app = _create_app(adapter)
    body = {
        "input": "hello",
        "provider": PROVIDER,
        "model": "test-model",
        "provider_base_url": URL_A,
    }
    with patch.object(
        adapter,
        "_create_agent",
        return_value=MagicMock(
            run_conversation=MagicMock(return_value={"final_response": "ok"}),
            session_prompt_tokens=0,
            session_completion_tokens=0,
            session_total_tokens=0,
        ),
    ):
        async with TestClient(TestServer(app)) as client:
            first = await client.post(
                "/v1/runs",
                json=body,
                headers=_auth_headers({
                    pc.PROVIDER_API_KEY_HEADER: SENTINEL,
                    "Idempotency-Key": "url-runs-r2",
                }),
            )
            assert first.status == 202
            changed = await client.post(
                "/v1/runs",
                json={**body, "provider_base_url": URL_B},
                headers=_auth_headers({
                    pc.PROVIDER_API_KEY_HEADER: SENTINEL,
                    "Idempotency-Key": "url-runs-r2",
                }),
            )
            data = await changed.json()
            assert changed.status == 409
            assert data["error"]["code"] == "idempotency_key_conflict"
            # Custom messages keep a failing leak from echoing the whole body.
            assert SENTINEL not in json.dumps(data), (
                "sentinel leaked into conflict body"
            )
            assert URL_A not in json.dumps(data) and URL_B not in json.dumps(data), (
                "endpoint URL echoed in conflict body"
            )


@pytest.mark.asyncio
async def test_chat_same_url_reuses_cached_computation():
    adapter = _make_adapter()
    captured = []
    adapter._run_agent = _fake_run_agent(captured)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        body = _chat_body()
        headers = {
            **_auth_headers({pc.PROVIDER_API_KEY_HEADER: SENTINEL}),
            "Idempotency-Key": "chat-same-r2",
        }
        first = await client.post("/v1/chat/completions", json=body, headers=headers)
        first_data = await first.json()
        second = await client.post("/v1/chat/completions", json=body, headers=headers)
        second_data = await second.json()
    assert first.status == second.status == 200
    assert len(captured) == 1
    assert (
        second_data["choices"][0]["message"]["content"]
        == first_data["choices"][0]["message"]["content"]
    )


@pytest.mark.asyncio
async def test_chat_changed_url_executes_again_and_sees_url_b():
    adapter = _make_adapter()
    captured = []
    adapter._run_agent = _fake_run_agent(captured)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        headers = {
            **_auth_headers({pc.PROVIDER_API_KEY_HEADER: SENTINEL}),
            "Idempotency-Key": "chat-change-r2",
        }
        first = await client.post(
            "/v1/chat/completions", json=_chat_body(), headers=headers
        )
        first_data = await first.json()
        second = await client.post(
            "/v1/chat/completions",
            json={**_chat_body(), "provider_base_url": URL_B},
            headers=headers,
        )
        second_data = await second.json()
    first_text = first_data["choices"][0]["message"]["content"]
    second_text = second_data["choices"][0]["message"]["content"]
    assert first.status == second.status == 200
    assert captured == [URL_A, URL_B]
    assert URL_A in first_text
    assert URL_B in second_text and URL_A not in second_text


@pytest.mark.asyncio
async def test_responses_same_url_reuses_cached_computation():
    adapter = _make_adapter()
    captured = []
    adapter._run_agent = _fake_run_agent(captured)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        body = _responses_body()
        headers = {
            **_auth_headers({pc.PROVIDER_API_KEY_HEADER: SENTINEL}),
            "Idempotency-Key": "responses-same-r2",
        }
        first = await client.post("/v1/responses", json=body, headers=headers)
        first_data = await first.json()
        second = await client.post("/v1/responses", json=body, headers=headers)
        second_data = await second.json()
    assert first.status == second.status == 200
    assert len(captured) == 1
    assert second_data["output"] == first_data["output"]


@pytest.mark.asyncio
async def test_responses_changed_url_executes_again_and_sees_url_b():
    adapter = _make_adapter()
    captured = []
    adapter._run_agent = _fake_run_agent(captured)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        headers = {
            **_auth_headers({pc.PROVIDER_API_KEY_HEADER: SENTINEL}),
            "Idempotency-Key": "responses-change-r2",
        }
        first = await client.post(
            "/v1/responses", json=_responses_body(), headers=headers
        )
        first_data = await first.json()
        second = await client.post(
            "/v1/responses",
            json={**_responses_body(), "provider_base_url": URL_B},
            headers=headers,
        )
        second_data = await second.json()
    first_text = _response_text(first_data)
    second_text = _response_text(second_data)
    assert first.status == second.status == 200
    assert captured == [URL_A, URL_B]
    assert URL_A in first_text
    assert URL_B in second_text and URL_A not in second_text


@pytest.mark.asyncio
async def test_requests_without_credential_contract_keep_existing_cache_behavior():
    adapter = _make_adapter()
    captured = []
    adapter._run_agent = _fake_run_agent(captured)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        body = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
        }
        headers = {**_auth_headers(), "Idempotency-Key": "plain-chat-r2"}
        first = await client.post("/v1/chat/completions", json=body, headers=headers)
        first_data = await first.json()
        second = await client.post("/v1/chat/completions", json=body, headers=headers)
        second_data = await second.json()
    assert first.status == second.status == 200
    assert len(captured) == 1
    assert (
        second_data["choices"][0]["message"]["content"]
        == first_data["choices"][0]["message"]["content"]
    )
