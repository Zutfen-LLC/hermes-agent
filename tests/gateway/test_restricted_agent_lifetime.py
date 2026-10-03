"""Constructed-agent ownership regressions for #39; all execution is synthetic."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server
from gateway.platforms import api_server_restricted_runs as restricted

AUTH = {"Authorization": "Bearer synthetic-master-key-0123456789"}
REQUEST = {"delegation_profile_id": "logical-helper", "work_class": "context_gather",
           "input": "Supplied synthetic snapshot", "capability_envelope": "input_only_v1"}


@pytest_asyncio.fixture
async def restricted_service(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": AUTH["Authorization"][7:]}))
    app = web.Application()
    app.router.add_post("/v1/restricted-runs/identity-checked", adapter._handle_identity_checked_restricted_runs)
    async with TestClient(TestServer(app)) as client:
        yield client, adapter
    adapter._run_idempotency_store.close()
    adapter._response_store.close()

CREDS = {"provider": "openai", "model": "gpt-4.1", "api_key": "synthetic-provider-key",
         "base_url": "https://api.openai.com/v1", "api_mode": "chat_completions"}


@pytest.fixture
def counting_agents(monkeypatch):
    counters = {"constructed": 0, "closed": 0, "executions": 0}
    agents = []

    class CountingAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.tools = []
            self.valid_tool_names = set()
            self.close_count = 0
            self.fail_close = False
            counters["constructed"] += 1
            agents.append(self)

        def close(self):
            self.close_count += 1
            counters["closed"] += 1
            if self.fail_close:
                raise RuntimeError("synthetic cleanup failure")

        def run_conversation(self, **kwargs):
            assert self.close_count == 0, "agent closed before use"
            counters["executions"] += 1
            return {"final_response": "synthetic result", "completed": True}

    monkeypatch.setattr("run_agent.AIAgent", CountingAgent)
    return counters, agents


def test_factory_post_construction_validation_closes_once(counting_agents, monkeypatch):
    counters, _ = counting_agents
    monkeypatch.setattr(restricted, "_check_restricted_tool_boundary", lambda agent: False)
    with pytest.raises(RuntimeError, match="restricted tool boundary could not be enforced"):
        restricted._new_restricted_agent(None, CREDS, None, _api_server=api_server)
    print("post-construction validation counters:", counters)
    assert counters == {"constructed": 1, "closed": 1, "executions": 0}


@pytest.mark.parametrize("failure", [ValueError("synthetic validation failure"),
                                      LookupError("synthetic unexpected failure")])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_factory_exception_preserves_primary_and_closes(counting_agents, monkeypatch, failure, cleanup_fails):
    counters, agents = counting_agents

    def fail(agent):
        agent.fail_close = cleanup_fails
        raise failure

    monkeypatch.setattr(restricted, "_check_restricted_tool_boundary", fail)
    with pytest.raises(type(failure)) as caught:
        restricted._new_restricted_agent(None, CREDS, None, _api_server=api_server)
    assert caught.value is failure
    assert counters == {"constructed": 1, "closed": 1, "executions": 0}
    assert agents[0].close_count == 1


def mock_route(monkeypatch):
    raw = {"enabled": True, "provider": "openai", "model": "gpt-4.1"}
    monkeypatch.setattr(restricted, "_resolve_restricted_route",
                        lambda *args, **kwargs: (CREDS, None, raw, None))
    return raw


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_path", ["identity_exception", "identity_mismatch",
    "reservation_exception", "reservation_conflict", "reservation_replay"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_admission_failure_owns_constructed_agent(restricted_service, counting_agents,
                                                       monkeypatch, failure_path, cleanup_fails):
    client, adapter = restricted_service
    counters, agents = counting_agents
    raw = mock_route(monkeypatch)
    identity = restricted._identity(adapter, REQUEST["delegation_profile_id"], CREDS, raw,
        _api_server=api_server, work_class=REQUEST["work_class"], envelope=REQUEST["capability_envelope"])
    boundary = restricted._check_restricted_tool_boundary

    def mark(agent):
        agent.fail_close = cleanup_fails
        return boundary(agent)

    monkeypatch.setattr(restricted, "_check_restricted_tool_boundary", mark)
    expected_status, code = 409, "restricted_identity_mismatch"
    if failure_path == "identity_exception":
        def fail_identity(*args, **kwargs):
            raise LookupError("synthetic identity validation failure")
        monkeypatch.setattr(restricted, "_identity", fail_identity)
        expected_status, code = 403, "delegation_profile_unavailable"
    elif failure_path == "identity_mismatch":
        identity["route_revision"] = "changed-revision"
    elif failure_path == "reservation_exception":
        def fail_reservation(*args, **kwargs):
            raise OSError("synthetic reservation failure")
        monkeypatch.setattr(adapter._run_idempotency_store, "reserve", fail_reservation)
        expected_status, code = 503, "run_storage_unavailable"
    else:
        outcome = "conflict" if failure_path == "reservation_conflict" else "reused"
        monkeypatch.setattr(adapter._run_idempotency_store, "reserve", lambda *args, **kwargs:
            (outcome, {"run_id": "run_existing", "status": {"status": "completed", "resolved_identity": identity}}))
        if outcome == "conflict":
            code = "idempotency_key_conflict"
        else:
            expected_status, code = 202, None
    response = await client.post("/v1/restricted-runs/identity-checked",
        json={**REQUEST, "expected_identity": identity},
        headers={**AUTH, "Idempotency-Key": "failure-key"})
    assert response.status == expected_status, await response.text()
    document = await response.json()
    if code:
        assert document["error"]["code"] == code
    else:
        assert document["replayed"] is True
    print(failure_path, counters)
    assert counters == {"constructed": 1, "closed": 1, "executions": 0}
    assert agents[0].close_count == 1
    assert not adapter._active_run_agents
    assert adapter._run_idempotency_store._conn.execute(
        "SELECT COUNT(*) FROM run_idempotency").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_success_transfers_ownership_without_close_before_use(restricted_service, counting_agents, monkeypatch):
    client, adapter = restricted_service
    counters, agents = counting_agents
    raw = mock_route(monkeypatch)
    identity = restricted._identity(adapter, REQUEST["delegation_profile_id"], CREDS, raw,
        _api_server=api_server, work_class=REQUEST["work_class"], envelope=REQUEST["capability_envelope"])
    started, finish = asyncio.Event(), asyncio.Event()
    execute = restricted._execute_restricted

    async def held_execute(*args, **kwargs):
        started.set()
        await finish.wait()
        await execute(*args, **kwargs)

    monkeypatch.setattr(restricted, "_execute_restricted", held_execute)
    response = await client.post("/v1/restricted-runs/identity-checked",
        json={**REQUEST, "expected_identity": identity}, headers={**AUTH, "Idempotency-Key": "success-key"})
    assert response.status == 202
    run_id = (await response.json())["run_id"]
    await asyncio.wait_for(started.wait(), 2)
    assert counters == {"constructed": 1, "closed": 0, "executions": 0}
    task = adapter._active_run_tasks[run_id]
    finish.set()
    await asyncio.wait_for(task, 2)
    assert counters == {"constructed": 1, "closed": 1, "executions": 1}
    assert agents[0].close_count == 1
