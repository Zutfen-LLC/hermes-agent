"""Tests for the shared memory routing helper."""

from __future__ import annotations

from agent import memory_router


class FakeMemoryStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def add(self, target: str, content: str):
        self.calls.append((target, content))
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
    assert len(mempalace_calls) == 1
    assert len(cca_calls) == 1
    assert fake_store.calls[0][0] == "user"
