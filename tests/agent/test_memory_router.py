"""Tests for the shared memory routing helper."""

from __future__ import annotations

import sys

from agent import memory_router


class FakeMemoryStore:
    def __init__(self, *, fail_native: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_native = fail_native

    def add(self, target: str, content: str):
        self.calls.append((target, content))
        if self.fail_native:
            return {
                "success": False,
                "error": "Memory at 2,200/2,200 chars. Adding this entry would exceed the limit.",
            }
        return {"success": True}


def test_canonicalization_normalizes_whitespace_and_case() -> None:
    a = memory_router.fingerprint_candidate(
        "native_memory",
        "operational_fact",
        "Hello\r\nWorld  ",
        "sess-1",
    )[1]
    b = memory_router.fingerprint_candidate(
        "native_memory",
        "operational_fact",
        "hello\nworld",
        "sess-1",
    )[1]

    assert a == b


def test_route_memory_candidates_dedupes_across_repeated_invocations(monkeypatch) -> None:
    memory_router.clear_seen("sess-1")

    fake_store = FakeMemoryStore()
    mempalace_calls: list[str] = []
    cca_calls: list[str] = []

    monkeypatch.setattr(
        memory_router,
        "_mempalace_route",
        lambda content, invocation_mode, config=None: mempalace_calls.append(content) or {"success": True},
    )
    monkeypatch.setattr(
        memory_router,
        "_route_to_cca_lite",
        lambda **kwargs: cca_calls.append(kwargs["text"]) or {"success": True},
    )

    messages = [
        {"role": "user", "content": "I prefer compact replies."},
        {"role": "assistant", "content": "Got it."},
    ]

    first = memory_router.route_memory_candidates(
        invocation_mode="manual",
        session_id="sess-1",
        messages=messages,
        memory_store=fake_store,
        source_event="cca-remember",
        config={},
    )
    second = memory_router.route_memory_candidates(
        invocation_mode="manual",
        session_id="sess-1",
        messages=messages,
        memory_store=fake_store,
        source_event="cca-remember",
        config={},
    )

    assert first["ok"] is True
    assert second["suppressed"]
    assert len(fake_store.calls) == 1
    assert not mempalace_calls
    assert not cca_calls
    assert fake_store.calls[0][0] == "user"


def test_mempalace_route_missing_backend_returns_disabled(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "plugins.memory.mempalace", object())

    result = memory_router._mempalace_route(
        "Please remember I prefer compact replies.",
        "manual",
        config={},
    )

    assert result["success"] is True
    assert result["mode"] == "disabled"
    assert "MemPalace unavailable" in result["reason"]


def test_session_route_mempalace_missing_backend_falls_back(monkeypatch) -> None:
    memory_router.clear_seen("sess-missing-mp")

    fake_store = FakeMemoryStore()
    monkeypatch.setitem(sys.modules, "plugins.memory.mempalace", object())

    result = memory_router.route_memory_candidates(
        invocation_mode="manual",
        session_id="sess-missing-mp",
        messages=[
            {"role": "user", "content": "Cross-project operational lesson: disabled semantic sinks should fall back."},
            {"role": "assistant", "content": "Noted."},
        ],
        memory_store=fake_store,
        source_event="cca-remember",
        config={},
    )

    assert result["ok"] is True
    assert not result["failed"]
    assert result["routed"]
    assert result["routed"][0]["canonical_destination"] == "mempalace"
    assert result["routed"][0]["destination"] == "native_memory"
    assert result["routed"][0]["fallback"] is True
    assert len(fake_store.calls) == 1


def test_memory_tool_user_preference_routes_to_native_user(monkeypatch) -> None:
    fake_store = FakeMemoryStore()
    mempalace_calls: list[str] = []
    cca_calls: list[str] = []
    monkeypatch.setattr(
        memory_router,
        "_mempalace_route",
        lambda content, invocation_mode, config=None: mempalace_calls.append(content) or {"success": True},
    )
    monkeypatch.setattr(
        memory_router,
        "_route_to_cca_lite",
        lambda **kwargs: cca_calls.append(kwargs["text"]) or {"success": True},
    )

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="I prefer compact replies.",
        native_add=fake_store.add,
        session_id="tool-user",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "native_user"
    assert result["routing"]["actual_sink"] == "native_user"
    assert fake_store.calls == [("user", "I prefer compact replies.")]
    assert not mempalace_calls
    assert not cca_calls


def test_memory_tool_environment_fact_routes_to_native_memory(monkeypatch) -> None:
    fake_store = FakeMemoryStore()
    monkeypatch.setattr(memory_router, "_mempalace_route", lambda *args, **kwargs: {"success": True})
    monkeypatch.setattr(memory_router, "_route_to_cca_lite", lambda **kwargs: {"success": True})

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="Stable setup fact: this machine has uv installed for Python toolchains.",
        native_add=fake_store.add,
        session_id="tool-env",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "native_memory"
    assert result["routing"]["actual_sink"] == "native_memory"
    assert fake_store.calls == [("memory", "Stable setup fact: this machine has uv installed for Python toolchains.")]


def test_memory_tool_cross_project_lesson_routes_to_mempalace(monkeypatch) -> None:
    fake_store = FakeMemoryStore()
    mempalace_calls: list[str] = []
    cca_calls: list[str] = []
    monkeypatch.setattr(
        memory_router,
        "_mempalace_route",
        lambda content, invocation_mode, config=None: mempalace_calls.append(content) or {"success": True},
    )
    monkeypatch.setattr(
        memory_router,
        "_route_to_cca_lite",
        lambda **kwargs: cca_calls.append(kwargs["text"]) or {"success": True},
    )

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="Cross-project operational lesson: keep retry metadata explicit across sessions.",
        native_add=fake_store.add,
        session_id="tool-mp",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "mempalace"
    assert result["routing"]["actual_sink"] == "mempalace"
    assert mempalace_calls == ["Cross-project operational lesson: keep retry metadata explicit across sessions."]
    assert not fake_store.calls
    assert not cca_calls


def test_memory_tool_repo_local_fact_routes_to_cca_lite(monkeypatch) -> None:
    fake_store = FakeMemoryStore()
    mempalace_calls: list[str] = []
    cca_calls: list[str] = []
    monkeypatch.setattr(
        memory_router,
        "_mempalace_route",
        lambda content, invocation_mode, config=None: mempalace_calls.append(content) or {"success": True},
    )
    monkeypatch.setattr(
        memory_router,
        "_route_to_cca_lite",
        lambda **kwargs: cca_calls.append(kwargs["text"]) or {"success": True, "appended": True},
    )

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="This repo convention: route durable project decisions through cca-lite.",
        native_add=fake_store.add,
        session_id="tool-cca",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "cca_lite"
    assert result["routing"]["actual_sink"] == "cca_lite"
    assert cca_calls == ["This repo convention: route durable project decisions through cca-lite."]
    assert not fake_store.calls
    assert not mempalace_calls


def test_memory_tool_agent_review_can_override_heuristic_destination(monkeypatch) -> None:
    fake_store = FakeMemoryStore()
    mempalace_calls: list[str] = []
    cca_calls: list[str] = []
    monkeypatch.setattr(
        memory_router,
        "_mempalace_route",
        lambda content, invocation_mode, config=None: mempalace_calls.append(content) or {"success": True},
    )
    monkeypatch.setattr(
        memory_router,
        "_route_to_cca_lite",
        lambda **kwargs: cca_calls.append(kwargs["text"]) or {"success": True},
    )

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="This repo convention generalizes across projects: keep durable routing metadata explicit.",
        native_add=fake_store.add,
        session_id="tool-agent-review",
        canonical_destination="mempalace",
        classification_reason="Reusable operational lesson rather than repo-local state.",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "mempalace"
    assert result["routing"]["heuristic_destination"] == "cca_lite"
    assert result["routing"]["classification_source"] == "agent_review"
    assert result["routing"]["agent_review"]["accepted"] is True
    assert mempalace_calls == ["This repo convention generalizes across projects: keep durable routing metadata explicit."]
    assert not fake_store.calls
    assert not cca_calls


def test_memory_tool_invalid_agent_review_is_ignored(monkeypatch) -> None:
    fake_store = FakeMemoryStore()
    monkeypatch.setattr(memory_router, "_mempalace_route", lambda *args, **kwargs: {"success": True})
    monkeypatch.setattr(memory_router, "_route_to_cca_lite", lambda **kwargs: {"success": True})

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="I prefer compact replies.",
        native_add=fake_store.add,
        session_id="tool-agent-review-invalid",
        canonical_destination="somewhere_else",
        classification_reason="Invalid destination should not be accepted.",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "native_user"
    assert result["routing"]["classification_source"] == "heuristic"
    assert result["routing"]["agent_review"]["accepted"] is False
    assert fake_store.calls == [("user", "I prefer compact replies.")]


def test_memory_tool_mempalace_unavailable_falls_back_with_metadata(monkeypatch) -> None:
    fake_store = FakeMemoryStore()
    monkeypatch.setattr(
        memory_router,
        "_mempalace_route",
        lambda content, invocation_mode, config=None: {"success": True, "mode": "disabled", "reason": "missing"},
    )
    monkeypatch.setattr(memory_router, "_route_to_cca_lite", lambda **kwargs: {"success": True})

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="Cross-project operational lesson: disabled sinks should degrade gracefully.",
        native_add=fake_store.add,
        session_id="tool-mp-missing",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "mempalace"
    assert result["routing"]["actual_sink"] == "native_memory"
    assert result["routing"]["fallback"] is True
    assert result["routing"]["rerouted"] is True
    assert fake_store.calls == [("memory", "Cross-project operational lesson: disabled sinks should degrade gracefully.")]


def test_memory_tool_cca_lite_unavailable_falls_back_with_metadata(monkeypatch) -> None:
    fake_store = FakeMemoryStore()
    mempalace_calls: list[str] = []
    monkeypatch.setattr(
        memory_router,
        "_route_to_cca_lite",
        lambda **kwargs: {"success": True, "mode": "disabled", "reason": "cca-lite bridge not configured"},
    )
    monkeypatch.setattr(
        memory_router,
        "_mempalace_route",
        lambda content, invocation_mode, config=None: mempalace_calls.append(content) or {"success": True},
    )

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="This repo invariant: durable project state belongs in docs/cca-lite/hermes-memory.json.",
        native_add=fake_store.add,
        session_id="tool-cca-missing",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "cca_lite"
    assert result["routing"]["actual_sink"] == "mempalace"
    assert result["routing"]["fallback"] is True
    assert result["routing"]["rerouted"] is True
    assert mempalace_calls == ["This repo invariant: durable project state belongs in docs/cca-lite/hermes-memory.json."]
    assert not fake_store.calls


def test_memory_tool_native_overflow_is_backup_path_not_classifier(monkeypatch) -> None:
    fake_store = FakeMemoryStore(fail_native=True)
    mempalace_calls: list[str] = []
    monkeypatch.setattr(
        memory_router,
        "_mempalace_route",
        lambda content, invocation_mode, config=None: mempalace_calls.append(content) or {"success": True},
    )
    monkeypatch.setattr(memory_router, "_route_to_cca_lite", lambda **kwargs: {"success": True})

    result = memory_router.route_memory_tool_write(
        requested_target="memory",
        content="Stable setup fact: this environment uses a shared Hermes venv.",
        native_add=fake_store.add,
        session_id="tool-overflow",
    )

    assert result["success"] is True
    assert result["routing"]["canonical_destination"] == "native_memory"
    assert result["routing"]["actual_sink"] == "mempalace"
    assert result["routing"]["fallback"] is True
    assert mempalace_calls == ["Stable setup fact: this environment uses a shared Hermes venv."]
