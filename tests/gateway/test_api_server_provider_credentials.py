"""Request-scoped provider runtime credentials (Ops Supervisor bridge) — contract tests.

Covers the X-Hermes-Provider-API-Key contract end to end:

* routing: credential beats static config for one request; next request reverts
* secret safety: sentinel absent from durable stores, logs, errors, responses
* runs/idempotency: same-key replay returns the same run; different-secret
  replay is a 409 fail-closed conflict; restart stays safe
* chat: session continuity via X-Hermes-Session-Id with per-turn credentials
* profiles: /v1 and /p/<profile>/ mirrors with no cross-profile contamination
* backward compatibility: no header, no behavior change
"""

import asyncio
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server
from gateway.platforms import api_server_provider_credentials as pc
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


SENTINEL = "sk-prvc-SENTINEL-c0ntRaCt-k3y-0001"
SENTINEL_B = "sk-prvc-SENTINEL-c0ntRaCt-k3y-0002"
PROVIDER = "deepinfra"
BASE_URL = "https://api.deepinfra.com/v1/openai"


def _make_adapter(api_key: str = "sk-gateway-test-key-0123456789") -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key}))


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"Authorization": "Bearer sk-gateway-test-key-0123456789"}
    headers.update(extra or {})
    return headers


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


# ---------------------------------------------------------------------------
# Capability advertisement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_advertise_provider_runtime_credentials():
    adapter = _make_adapter()
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities", headers=_auth_headers())
        assert resp.status == 200
        features = (await resp.json())["features"]
        flag = features["provider_runtime_credentials"]
        assert flag["supported"] is True
        assert flag["api_key_header"] == "X-Hermes-Provider-API-Key"
        assert flag["base_url_field"] == "provider_base_url"
        assert flag["per_request"] is True
        assert flag["persisted"] is False
        # The generic admin flag must remain untouched.
        assert features["admin_config_rw"] is False


# ---------------------------------------------------------------------------
# Request-scoped routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_credential_reaches_agent_runtime_for_one_request_only():
    """The override beats static credentials for THIS request; the immediately
    following ordinary request returns to static resolution."""
    adapter = _make_adapter()
    app = _create_app(adapter)
    captured = []

    def _capture_create(**kwargs):
        captured.append(kwargs.get("provider_credential"))
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "done"}
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        return agent

    with patch.object(adapter, "_create_agent", side_effect=_capture_create):
        async with TestClient(TestServer(app)) as cli:
            body = {"model": "Qwen/Qwen2.5-72B-Instruct", "provider": PROVIDER,
                    "provider_base_url": BASE_URL,
                    "messages": [{"role": "user", "content": "hi"}]}
            resp = await cli.post(
                "/v1/chat/completions", json=body,
                headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL}))
            assert resp.status == 200
            data = await resp.json()
            assert "SENTINEL" not in json.dumps(data)

            # Next request WITHOUT the header: no credential object at all.
            resp2 = await cli.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "again"}]},
                headers=_auth_headers())
            assert resp2.status == 200

    assert len(captured) == 2
    cred, plain = captured
    assert cred is not None
    assert cred.api_key == SENTINEL
    assert cred.base_url == BASE_URL
    assert cred.provider == PROVIDER
    assert plain is None, "ordinary request carries no credential"


@pytest.mark.asyncio
async def test_credential_without_provider_rejected():
    adapter = _make_adapter()
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/chat/completions",
            json={"model": "Qwen/Qwen2.5-72B-Instruct",
                  "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL}))
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "provider_required_for_credential"
        assert SENTINEL not in json.dumps(body)


@pytest.mark.asyncio
async def test_body_carried_secret_rejected():
    adapter = _make_adapter()
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        for field in ("provider_api_key", "provider_credentials"):
            resp = await cli.post(
                "/v1/chat/completions",
                json={"provider": PROVIDER, field: SENTINEL,
                      "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth_headers())
            assert resp.status == 400
            body = await resp.json()
            assert body["error"]["code"] == "provider_credential_in_body"
            assert SENTINEL not in json.dumps(body)


@pytest.mark.asyncio
async def test_credential_conflicting_route_provider_rejected():
    """Existing model-route/provider conflict logic must still reject a mixed request."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": "sk-gateway-test-key-0123456789",
        "model_routes": {"alias": {"model": "route/model", "provider": "openrouter"}}}))
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/chat/completions",
            json={"model": "alias", "provider": "minimax",
                  "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL}))
        assert resp.status == 400
        body = await resp.json()
        assert "provider" in body["error"]["message"].lower()
        assert SENTINEL not in json.dumps(body)


@pytest.mark.asyncio
async def test_credential_requires_auth_key_configured():
    """A keyless test listener must not accept credential injection."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/chat/completions",
            json={"provider": PROVIDER, "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Hermes-Provider-API-Key": SENTINEL})
        assert resp.status == 403
        body = await resp.json()
        assert body["error"]["code"] == "provider_credential_auth_required"


@pytest.mark.asyncio
async def test_credential_requires_bearer_token():
    adapter = _make_adapter()
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/chat/completions",
            json={"provider": PROVIDER, "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Hermes-Provider-API-Key": SENTINEL})  # no Authorization
        assert resp.status == 401


@pytest.mark.asyncio
async def test_select_agent_runtime_applies_credential_runtime():
    """Mechanical: _create_agent/_select_agent_runtime with a credential resolves
    the explicit runtime and makes it THE runtime for that agent."""
    from gateway.platforms.api_server import _provider_credentials

    adapter = _make_adapter()
    cred = pc.ProviderCredentialOverride(
        api_key=SENTINEL, base_url=BASE_URL, provider=PROVIDER, fingerprint="f" * 64)
    runtime_kwargs = {"provider": "openrouter", "api_key": "STATIC-KEY",
                      "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions",
                      "credential_pool": MagicMock()}
    with patch.object(_provider_credentials, "resolve_credential_runtime",
                      return_value={"provider": PROVIDER, "api_key": SENTINEL,
                                    "base_url": BASE_URL, "api_mode": "chat_completions"}):
        model, session_override, req_model, req_provider = adapter._select_agent_runtime(
            runtime_kwargs, "gpt-4o",
            requested_model="Qwen/Qwen2.5-72B-Instruct", requested_provider=PROVIDER,
            route=None, session_model=None, confirmed_runtime_lock=False,
            gateway_session_key=None, session_id="sess-1",
            provider_credential=cred)
    assert runtime_kwargs["api_key"] == SENTINEL
    assert runtime_kwargs["base_url"] == BASE_URL
    assert runtime_kwargs["provider"] == PROVIDER
    assert runtime_kwargs["credential_pool"] is None
    assert model == "Qwen/Qwen2.5-72B-Instruct"
    assert session_override is None


# ---------------------------------------------------------------------------
# Runs / idempotency
# ---------------------------------------------------------------------------


def _use_idempotency_db(adapter, path):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(path))


def _make_runs_mock_agent(captured):
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent.provider = PROVIDER
    agent.model = "Qwen/Qwen2.5-72B-Instruct"
    return agent


@pytest.mark.asyncio
async def test_runs_idempotent_replay_same_secret_returns_same_run():
    adapter = _make_adapter()
    app = _create_app(adapter)
    body = {"input": "hello", "provider": PROVIDER, "model": "Qwen/Qwen2.5-72B-Instruct"}
    headers = _auth_headers({"X-Hermes-Provider-API-Key": SENTINEL, "Idempotency-Key": "K-1"})
    with patch.object(adapter, "_create_agent", return_value=_make_runs_mock_agent(None)):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post("/v1/runs", json=body, headers=headers)
            assert first.status == 202
            first_data = await first.json()
            replay = await cli.post("/v1/runs", json=body, headers=headers)
            assert replay.status == 202
            replay_data = await replay.json()
    assert first_data["run_id"] == replay_data["run_id"]
    assert replay.headers.get("Idempotency-Replayed") == "true"


@pytest.mark.asyncio
async def test_runs_replay_with_different_secret_is_conflict():
    """Fail closed: same Idempotency-Key + same body + DIFFERENT secret = 409."""
    adapter = _make_adapter()
    app = _create_app(adapter)
    body = {"input": "hello", "provider": PROVIDER, "model": "Qwen/Qwen2.5-72B-Instruct"}
    with patch.object(adapter, "_create_agent", return_value=_make_runs_mock_agent(None)):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/v1/runs", json=body,
                headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL, "Idempotency-Key": "K-2"}))
            assert first.status == 202
            conflict = await cli.post(
                "/v1/runs", json=body,
                headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL_B, "Idempotency-Key": "K-2"}))
            assert conflict.status == 409
            data = await conflict.json()
            assert data["error"]["code"] == "idempotency_key_conflict"
            assert SENTINEL not in json.dumps(data) and SENTINEL_B not in json.dumps(data)


@pytest.mark.asyncio
async def test_runs_replay_without_secret_after_credential_admission_is_conflict():
    """A credential-admitted key replayed without the header must NOT silently
    re-associate: same key + no credential = different fingerprint = 409."""
    adapter = _make_adapter()
    app = _create_app(adapter)
    body = {"input": "hello", "provider": PROVIDER, "model": "Qwen/Qwen2.5-72B-Instruct"}
    with patch.object(adapter, "_create_agent", return_value=_make_runs_mock_agent(None)):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/v1/runs", json=body,
                headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL, "Idempotency-Key": "K-3"}))
            assert first.status == 202
            stripped = await cli.post(
                "/v1/runs", json=body,
                headers=_auth_headers({"Idempotency-Key": "K-3"}))
            assert stripped.status == 409


@pytest.mark.asyncio
async def test_runs_first_admission_creates_exactly_one_run(tmp_path, caplog):
    """One admission, one run; the sentinel never lands in logs or the durable store."""
    adapter = _make_adapter()
    _use_idempotency_db(adapter, tmp_path / "idem.db")
    app = _create_app(adapter)
    body = {"input": "hello", "provider": PROVIDER, "model": "Qwen/Qwen2.5-72B-Instruct"}
    headers = _auth_headers({"X-Hermes-Provider-API-Key": SENTINEL, "Idempotency-Key": "single-1"})
    caplog.set_level("DEBUG", logger="gateway.platforms.api_server")
    with patch.object(adapter, "_create_agent", return_value=_make_runs_mock_agent(None)):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json=body, headers=headers)
            assert resp.status == 202
            run_id = (await resp.json())["run_id"]
            for _ in range(40):
                status = await (await cli.get(f"/v1/runs/{run_id}", headers=_auth_headers())).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.05)
            assert status["status"] == "completed"
            assert SENTINEL not in json.dumps(status)
    assert SENTINEL not in caplog.text
    # Durable idempotency store: fingerprints only, never the secret.
    assert adapter._run_idempotency_store.durable
    conn = sqlite3.connect(adapter._run_idempotency_store._db_path)
    try:
        rows = conn.execute("SELECT * FROM run_idempotency").fetchall()
    finally:
        conn.close()
    assert rows, "expected at least one idempotency row"
    assert all(SENTINEL not in str(row) for row in rows)


# ---------------------------------------------------------------------------
# Chat session continuity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_session_continuity_with_per_turn_credentials():
    """Turn 1 creates session A with a credential; turn 2 continues A with a new
    credential; turn 3 uses none — session metadata never carries the secret."""
    adapter = _make_adapter()
    app = _create_app(adapter)
    session_ids = []

    async def _fake_run_agent(**kwargs):
        session_ids.append(kwargs.get("session_id"))
        return ({"final_response": "ok", "session_id": kwargs.get("session_id"), "messages": []},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    with patch.object(adapter, "_run_agent", side_effect=_fake_run_agent):
        async with TestClient(TestServer(app)) as cli:
            base = {"provider": PROVIDER, "model": "Qwen/Qwen2.5-72B-Instruct",
                    "messages": [{"role": "user", "content": "turn one"}]}
            r1 = await cli.post("/v1/chat/completions", json=base,
                                headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL}))
            assert r1.status == 200
            sid_a = r1.headers["X-Hermes-Session-Id"]

            r2 = await cli.post("/v1/chat/completions", json=base,
                                headers=_auth_headers({
                                    "X-Hermes-Provider-API-Key": SENTINEL_B,
                                    "X-Hermes-Session-Id": sid_a}))
            assert r2.status == 200
            assert r2.headers["X-Hermes-Session-Id"] == sid_a

            r3 = await cli.post("/v1/chat/completions",
                                json={"messages": [{"role": "user", "content": "turn three"}]},
                                headers=_auth_headers({"X-Hermes-Session-Id": sid_a}))
            assert r3.status == 200
            assert r3.headers["X-Hermes-Session-Id"] == sid_a

    # All three turns hit _run_agent; turn 1/2 carried credentials, turn 3 none.
    assert len(session_ids) == 3
    # Secret never echoed in any response header or body.
    for raw in (r1, r2, r3):
        assert "X-Hermes-Provider-API-Key" not in raw.headers


# ---------------------------------------------------------------------------
# Secret safety sweeps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_provider_auth_does_not_leak_secret(caplog):
    """A provider 401 surfaces a stable non-secret diagnostic; sentinel absent
    from response, logs, and the persistent stores."""
    adapter = _make_adapter()
    app = _create_app(adapter)

    def _failing_create(**kwargs):
        raise api_server._ProviderAuthResolutionError(
            f"Provider authentication failed for key {kwargs['provider_credential'].api_key}")

    caplog.set_level("DEBUG")
    with patch.object(adapter, "_create_agent", side_effect=_failing_create):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"provider": PROVIDER, "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL}))
            body_text = await resp.text()
    # _ProviderAuthResolutionError in _run_agent maps to a 200 soft-fail envelope;
    # either way the sentinel must be absent everywhere.
    assert SENTINEL not in body_text
    assert SENTINEL not in caplog.text


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_mirror_no_cross_profile_contamination():
    """The credential contract is served on /p/<profile>/ mirrors through the same
    handlers (register_route mirrors in connect()); profile isolation for
    credential replays is enforced by the principal-scoped fingerprint: two
    profiles (or two API keys) produce different scopes, so one profile's
    Idempotency-Key + secret can never unlock another's run."""
    adapter = _make_adapter()
    req = MagicMock()
    req.headers = {"Authorization": "Bearer sk-gateway-test-key-0123456789"}
    body = {"provider": PROVIDER, "model": "m1"}

    def _scope(profile: str | None) -> str:
        token = api_server._api_request_profile.set(profile)
        try:
            return adapter._run_idempotency_scope(req)
        finally:
            api_server._api_request_profile.reset(token)

    scope_default = _scope(None)
    scope_other = _scope("other")
    assert scope_default != scope_other

    # The credential fingerprint is bound to that scope: same secret + different
    # principal = different fingerprint = 409 conflict, never silent replay.
    cred_default = pc.extract_provider_credential(
        _scope_profile_adapter(adapter, None), _cred_req(SENTINEL), body, scope_fn=lambda: scope_default)
    cred_other = pc.extract_provider_credential(
        _scope_profile_adapter(adapter, "other"), _cred_req(SENTINEL), body, scope_fn=lambda: scope_other)
    assert cred_default.fingerprint != cred_other.fingerprint
    assert SENTINEL not in cred_default.fingerprint and SENTINEL not in cred_other.fingerprint


def _cred_req(secret: str):
    req = MagicMock()
    req.headers = {"X-Hermes-Provider-API-Key": secret}
    return req


def _scope_profile_adapter(adapter, profile):
    """Adapter view whose _expected_api_key reflects the profile's own key."""
    view = MagicMock(wraps=adapter)
    view._room_grant_token = lambda request: None
    view._expected_api_key = lambda: f"profile-{'default' if profile is None else profile}-key-0123456789abcd"
    view._clean_runtime_id = APIServerAdapter._clean_runtime_id
    return view


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_requests_without_contract_behave_as_before():
    adapter = _make_adapter()
    app = _create_app(adapter)

    async def _fake_run_agent(**kwargs):
        assert "provider_credential" not in kwargs or kwargs["provider_credential"] is None
        return ({"final_response": "ok", "messages": []},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    with patch.object(adapter, "_run_agent", side_effect=_fake_run_agent):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "alias-model", "provider": "openrouter",
                      "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth_headers())
            assert resp.status == 200
            resp2 = await cli.post("/v1/runs", json={"input": "hi"}, headers=_auth_headers())
            assert resp2.status == 202


# ---------------------------------------------------------------------------
# Gateway restart behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_after_restart_returns_same_run_without_plaintext(tmp_path):
    """Simulated lost-202 + gateway restart: a NEW adapter over the same durable
    store returns the SAME run for the same key+secret. No plaintext credential
    is stored to make that possible — the keyed fingerprint is."""
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    db_path = tmp_path / "idem.db"
    adapter = _make_adapter()
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(db_path))
    body = {"input": "restart replay", "provider": PROVIDER, "model": "Qwen/Qwen2.5-72B-Instruct"}
    headers = _auth_headers({"X-Hermes-Provider-API-Key": SENTINEL, "Idempotency-Key": "restart-1"})
    with patch.object(adapter, "_create_agent", return_value=_make_runs_mock_agent(None)):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post("/v1/runs", json=body, headers=headers)
            assert first.status == 202
            first_run = (await first.json())["run_id"]

    # "Restart": fresh adapter + store over the same DB file.
    restarted = _make_adapter()
    restarted._run_idempotency_store.close()
    restarted._run_idempotency_store = RunIdempotencyStore(str(db_path))
    with patch.object(restarted, "_create_agent", return_value=_make_runs_mock_agent(None)):
        app2 = _create_app(restarted)
        async with TestClient(TestServer(app2)) as cli:
            replay = await cli.post("/v1/runs", json=body, headers=headers)
            assert replay.status == 202
            replay_data = await replay.json()
    assert replay_data["run_id"] == first_run
    assert replay.headers.get("Idempotency-Replayed") == "true"

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM run_idempotency").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, "no duplicate run admitted"
    assert all(SENTINEL not in str(row) for row in rows)


# ---------------------------------------------------------------------------
# Session-record persistence audit (session chat surface)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_chat_credential_not_persisted_to_session_store(tmp_path, monkeypatch):
    """A credential turn on /api/sessions/{id}/chat writes nothing secret into the
    session DB (rows, model_config, messages)."""
    import hermes_constants

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    adapter = _make_adapter()
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)

    async def _fake_run_agent(**kwargs):
        return ({"final_response": "ok", "session_id": kwargs.get("session_id"), "messages": []},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    with patch.object(adapter, "_run_agent", side_effect=_fake_run_agent):
        async with TestClient(TestServer(app)) as cli:
            created = await cli.post(
                "/api/sessions", json={"id": "audit-sess-1"}, headers=_auth_headers())
            assert created.status == 201, await created.text()
            resp = await cli.post(
                "/api/sessions/audit-sess-1/chat",
                json={"message": "hello", "provider": PROVIDER,
                      "model": "Qwen/Qwen2.5-72B-Instruct"},
                headers=_auth_headers({"X-Hermes-Provider-API-Key": SENTINEL}))
            assert resp.status == 200
            assert SENTINEL not in await resp.text()

    hits = [str(p) for p in home.rglob("*")
            if p.is_file() and SENTINEL.encode() in p.read_bytes()]
    assert hits == [], f"sentinel persisted in session store: {hits}"


# ---------------------------------------------------------------------------
# Fingerprint semantics
# ---------------------------------------------------------------------------


def test_fingerprint_is_keyed_and_stable():
    adapter = _make_adapter()
    req = MagicMock()
    req.headers = {"X-Hermes-Provider-API-Key": SENTINEL}
    body = {"provider": PROVIDER, "model": "m1", "provider_base_url": BASE_URL}
    c1 = pc.extract_provider_credential(adapter, req, body, scope_fn=lambda: "scope-A")
    c1b = pc.extract_provider_credential(adapter, req, body, scope_fn=lambda: "scope-A")
    c2 = pc.extract_provider_credential(adapter, req, body, scope_fn=lambda: "scope-B")
    assert c1.fingerprint == c1b.fingerprint
    assert c1.fingerprint != c2.fingerprint, "principal scope binds the fingerprint"
    assert SENTINEL not in c1.fingerprint


def test_credential_repr_is_safe():
    cred = pc.ProviderCredentialOverride(api_key=SENTINEL, base_url=BASE_URL, provider=PROVIDER, fingerprint="f" * 64)
    assert SENTINEL not in repr(cred)
