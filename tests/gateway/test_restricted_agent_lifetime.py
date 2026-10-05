"""Constructed-agent ownership regressions for #39; all execution is synthetic."""

import asyncio
import run_agent
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


def test_constructor_failure_does_not_close_partially_initialized_agent(counting_agents, monkeypatch):
    counters, agents = counting_agents
    original = run_agent.AIAgent

    def fail_after_allocation(**kwargs):
        partial = object.__new__(original)
        partial.close = lambda: counters.__setitem__("closed", counters["closed"] + 1)
        agents.append(partial)
        raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr("run_agent.AIAgent", fail_after_allocation)
    with pytest.raises(RuntimeError, match="synthetic constructor failure"):
        restricted._new_restricted_agent(None, CREDS, None, _api_server=api_server)
    assert counters == {"constructed": 0, "closed": 0, "executions": 0}


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


@pytest.mark.asyncio
async def test_task_handoff_failure_closes_and_terminalizes_reservation(
        restricted_service, counting_agents, monkeypatch):
    client, adapter = restricted_service
    counters, _ = counting_agents
    raw = mock_route(monkeypatch)
    identity = restricted._identity(adapter, REQUEST["delegation_profile_id"], CREDS, raw,
        _api_server=api_server, work_class=REQUEST["work_class"], envelope=REQUEST["capability_envelope"])
    original_create_task = asyncio.create_task
    monkeypatch.setattr(restricted.asyncio, "create_task", lambda coro, **kwargs: (
        coro.close(), (_ for _ in ()).throw(RuntimeError("synthetic task handoff failure")))[1])
    try:
        response = await client.post("/v1/restricted-runs/identity-checked",
            json={**REQUEST, "expected_identity": identity},
            headers={**AUTH, "Idempotency-Key": "handoff-failure"})
    finally:
        monkeypatch.setattr(restricted.asyncio, "create_task", original_create_task)
    assert response.status == 500
    assert counters == {"constructed": 1, "closed": 1, "executions": 0}
    assert not adapter._active_run_agents
    row = adapter._run_idempotency_store._conn.execute(
        "SELECT status_json FROM run_idempotency WHERE idempotency_key='handoff-failure'").fetchone()
    assert row is not None
    import json
    assert json.loads(row[0])["status"] == "failed"


@pytest.mark.asyncio
async def test_cancelled_wrapper_keeps_agent_until_worker_finishes(restricted_service, counting_agents, monkeypatch):
    _, adapter = restricted_service
    counters, agents = counting_agents
    agent = restricted._new_restricted_agent(adapter, CREDS, None, _api_server=api_server)
    worker_started, worker_release = asyncio.Event(), asyncio.Event()

    async def worker():
        worker_started.set()
        await worker_release.wait()
        return {"final_response": "synthetic result", "completed": True}

    future = asyncio.create_task(worker())
    monkeypatch.setattr(restricted, "_submit_api_worker", lambda *args, **kwargs: future)
    run_id = "run_synthetic_cancel"
    adapter._run_statuses[run_id] = {"run_id": run_id, "status": "queued"}
    adapter._active_run_agents[run_id] = agent
    task = asyncio.create_task(restricted._execute_restricted(
        adapter, run_id, "synthetic input", agent, None, None, _api_server=api_server))
    await worker_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert counters["closed"] == 0
    worker_release.set()
    await asyncio.gather(task, return_exceptions=True)
    assert counters == {"constructed": 1, "closed": 1, "executions": 0}
