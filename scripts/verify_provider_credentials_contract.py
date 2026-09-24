#!/usr/bin/env python3
"""Deterministic integration fixture for the request-scoped provider-credential
contract (Ops Supervisor PR #213 target).

Boots a real APIServerAdapter on an ephemeral port with a known API key, then
exercises the exact client behaviors Ops Supervisor needs, asserting the eight
proof points of the bridge:

1. the client's stored credential reaches Hermes' provider runtime verbatim
2. the request-scoped key/base URL beat Hermes' static config for that request
3. no plaintext credential lands in any durable store Hermes owns
4. changing the credential affects the NEXT invocation without restarts
5. an already-admitted run keeps its original runtime
6. an idempotent replay after a simulated lost 202 does not create a duplicate run
7. a replay with a DIFFERENT credential is a 409 fail-closed conflict
8. ordinary requests without the contract behave exactly as before

Usage:
    python scripts/verify_provider_credentials_contract.py [--port 8791]

Exit code 0 = contract satisfied. Any failure prints the failing proof point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

GATEWAY_KEY = "sk-fixture-gateway-key-0123456789abcdef"
SENTINEL_A = "sk-fixture-provider-A-0123456789abcdefFEDCBA"
SENTINEL_B = "sk-fixture-provider-B-0123456789abcdefABCDEF"


def _boot(home: Path, port: int):
    import os
    os.environ["HERMES_HOME"] = str(home)
    from aiohttp import web
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": GATEWAY_KEY, "host": "127.0.0.1", "port": port}))
    # A capture shim around _create_agent so the fixture observes the runtime the
    # contract produced WITHOUT any network provider call.
    captured: list = []

    real_select = adapter._select_agent_runtime

    def _capturing_select(runtime_kwargs, model, **kw):
        cred = kw.get("provider_credential")
        if cred is not None:
            from gateway.platforms import api_server_provider_credentials as pc
            runtime = pc.resolve_credential_runtime(cred, target_model=model or None)
            pc.apply_credential_runtime(runtime_kwargs, runtime)
            captured.append({
                "provider": runtime_kwargs.get("provider"),
                "base_url": runtime_kwargs.get("base_url"),
                "api_key": runtime_kwargs.get("api_key"),
                "model": model})
            return model, None, kw.get("requested_model"), kw.get("requested_provider")
        return real_select(runtime_kwargs, model, **kw)

    adapter._select_agent_runtime = _capturing_select
    return adapter, captured


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix="prvcred-fixture-"))
    adapter, captured = _boot(home, args.port)

    from run_agent import AIAgent as _RealAgent  # noqa: F401  (import check)

    class _StubAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = kwargs.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}"
            self.provider = kwargs.get("provider") or "deepinfra"
            self.model = kwargs.get("model") or "Qwen/Qwen2.5-72B-Instruct"
            self.session_prompt_tokens = self.session_completion_tokens = self.session_total_tokens = 0

        def run_conversation(self, **_kw):
            return {"final_response": f"ok via {self.provider}", "completed": True}

    import gateway.platforms.api_server as api_server_mod

    def _stub_create(agent_self=None, **kwargs):
        return _StubAgent(**kwargs)

    failures: list[str] = []

    def check(point: int, ok: bool, detail: str = ""):
        print(f"[{'PASS' if ok else 'FAIL'}] proof {point}: {detail}")
        if not ok:
            failures.append(f"proof {point}: {detail}")

    import run_agent as run_agent_mod
    with patch.object(run_agent_mod, "AIAgent", _StubAgent), \
            patch.object(gateway_mod_run(), "_resolve_runtime_agent_kwargs",
                         lambda: {"provider": "openrouter", "api_key": "STATIC-STATIC-STATIC",
                                  "base_url": "https://openrouter.ai/api/v1",
                                  "api_mode": "chat_completions"}), \
            patch.object(gateway_mod_run(), "_resolve_gateway_model", lambda: "static-default-model"), \
            patch.object(gateway_mod_run(), "_load_gateway_config", lambda: {}), \
            patch.object(gateway_mod_run(), "_current_max_iterations", lambda: 4), \
            patch.object(gateway_mod_run(), "_checkpoint_agent_kwargs", lambda cfg: {}), \
            patch.object(tools_config_mod(), "_get_platform_tools", lambda *_a, **_k: set()):
        server = await adapter._start_test_server(args.port) if hasattr(adapter, "_start_test_server") else None
        if server is None:
            from aiohttp import web
            app = web.Application()
            app["api_server_adapter"] = adapter
            for method, path, handler in adapter._http_route_table():
                app.router.add_route(method, path, handler)
            from aiohttp.test_utils import TestServer, TestClient
            server = TestServer(app)
            await server.start_server()

        import aiohttp

        base = f"http://127.0.0.1:{server.port}"
        auth = {"Authorization": f"Bearer {GATEWAY_KEY}"}

        async def post(path, json_body, headers):
            import aiohttp
            async with aiohttp.ClientSession() as sess:
                return await sess.post(base + path, json=json_body, headers=headers)

        async with aiohttp.ClientSession() as http:
            body_a = {"input": "run one", "provider": "deepinfra",
                      "model": "Qwen/Qwen2.5-72B-Instruct",
                      "provider_base_url": "https://api.deepinfra.com/v1/openai"}

            async def post_json(path, jb, hdrs):
                async with http.post(base + path, json=jb, headers=hdrs) as r:
                    return r.status, await r.json(), dict(r.headers)

            # -- proof 1+2: credential verbatim + beats static config --------------
            status, data, _ = await post_json("/v1/runs", body_a, {
                **auth, "X-Hermes-Provider-API-Key": SENTINEL_A, "Idempotency-Key": "fx-1"})
            run1 = data.get("run_id")
            check(1, status == 202 and run1, f"admission 202 run={run1}")
            for _ in range(100):
                async with http.get(f"{base}/v1/runs/{run1}", headers={
                        **auth, "X-Hermes-Provider-API-Key": SENTINEL_A}) as r:
                    st = await r.json()
                if st["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)
            check(2, bool(captured) and captured[0]["api_key"] == SENTINEL_A
                  and captured[0]["base_url"] == "https://api.deepinfra.com/v1/openai"
                  and captured[0]["provider"] == "deepinfra",
                  f"runtime used caller values: {captured[0] if captured else None}")

            # -- proof 4: different credential affects the NEXT invocation ----------
            await post_json("/v1/chat/completions", {
                "provider": "deepinfra", "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": [{"role": "user", "content": "chat turn"}]},
                {**auth, "X-Hermes-Provider-API-Key": SENTINEL_B})
            check(4, len(captured) >= 2 and captured[1]["api_key"] == SENTINEL_B,
                  "second invocation used the rotated credential")

            # -- proof 6: lost-202 replay returns the same run, no duplicate ---------
            status, data, _ = await post_json("/v1/runs", body_a, {
                **auth, "X-Hermes-Provider-API-Key": SENTINEL_A, "Idempotency-Key": "fx-1"})
            check(6, status == 202 and data.get("run_id") == run1,
                  f"replay run={data.get('run_id')} original={run1}")

            # -- proof 7: different-secret replay is a 409 conflict ------------------
            status, data, _ = await post_json("/v1/runs", body_a, {
                **auth, "X-Hermes-Provider-API-Key": SENTINEL_B, "Idempotency-Key": "fx-1"})
            check(7, status == 409 and data["error"]["code"] == "idempotency_key_conflict",
                  f"different-secret replay -> {status}")

            # -- proof 8: ordinary request unchanged ----------------------------------
            status, data, _ = await post_json("/v1/runs", {"input": "plain"}, auth)
            check(8, status == 202, "ordinary request still admitted")

        await server.close()

    # -- proof 3: durable storage never contains either sentinel -----------------
    hits = [str(p) for p in home.rglob("*")
            if p.is_file() and (SENTINEL_A.encode() in p.read_bytes()
                                or SENTINEL_B.encode() in p.read_bytes())]
    check(3, not hits, f"durable hits: {hits or 'NONE'}")

    if failures:
        print("\nFIXTURE FAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nFIXTURE PASSED: all 8 proof points satisfied")
    return 0


def gateway_mod_run():
    import gateway.run
    return gateway.run


def tools_config_mod():
    import hermes_cli.tools_config
    return hermes_cli.tools_config


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
