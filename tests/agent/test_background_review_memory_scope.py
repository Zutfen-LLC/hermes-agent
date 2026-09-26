"""The review fork's memory access must follow the trigger that fired (#105921).

``spawn_background_review_thread`` already received ``review_memory`` /
``review_skills`` but used them only to pick the prompt — the tool whitelist
granted the whole ``memory`` toolset whenever the profile had memory enabled,
so a skill-nudge fork held ``remove``/``replace`` on MEMORY.md it was never
asked to use. These tests pin the scope-aware whitelist and the pass-through
from ``spawn_background_review_thread`` down to it.
"""

from __future__ import annotations

import os
import sys
import pytest
from types import SimpleNamespace
from unittest.mock import patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent.background_review as bg  # noqa: E402


def _review_agent(memory_enabled=True, user_profile_enabled=False) -> SimpleNamespace:
    """The whitelist only reads the profile's memory flags off the fork."""
    return SimpleNamespace(_memory_enabled=memory_enabled, _user_profile_enabled=user_profile_enabled)


class TestReviewToolWhitelistScope:
    def test_skill_only_review_omits_memory_tool(self):
        whitelist, _extra = bg._review_tool_whitelist(_review_agent(), None, review_memory=False)
        assert "memory" not in whitelist
        assert "skill_manage" in whitelist  # the skill review keeps its own surface

    def test_memory_review_keeps_memory_tool(self):
        whitelist, _extra = bg._review_tool_whitelist(_review_agent(), None, review_memory=True)
        assert "memory" in whitelist

    def test_memory_disabled_profile_stays_memory_free(self):
        whitelist, _extra = bg._review_tool_whitelist(
            _review_agent(memory_enabled=False, user_profile_enabled=False), None, review_memory=True)
        assert "memory" not in whitelist

    def test_default_scope_is_memoryless_fail_closed(self):
        # Unknown trigger (default) must not grant memory: the incident fork was a
        # skill review that used memory nobody asked it to touch.
        whitelist, _extra = bg._review_tool_whitelist(_review_agent(), None)
        assert "memory" not in whitelist


class TestSpawnForwardsScope:
    def test_target_passes_review_memory_to_worker(self):
        captured = {}

        def fake_worker(agent, messages_snapshot, prompt, task_cfg=None, review_run=None,
                        review_memory=False, explicit=False):
            captured["review_memory"] = review_memory

        agent = SimpleNamespace()
        with patch.object(bg, "_run_review_in_thread", fake_worker):
            target, _prompt = bg.spawn_background_review_thread(
                agent, [], review_memory=False, review_skills=True)
            target()
            assert captured["review_memory"] is False

            target, _prompt = bg.spawn_background_review_thread(
                agent, [], review_memory=True, review_skills=False)
            target()
            assert captured["review_memory"] is True


class TestExplicitRefineOrigin:
    """``/refine`` (explicit) must not inherit the unattended-review origin: the user asked
    for that review, so its fork keeps the full memory operation set and the delete gate
    does not apply (#105921 review follow-up)."""

    def test_target_passes_explicit_to_worker(self):
        captured = {}

        def fake_worker(agent, messages_snapshot, prompt, task_cfg=None, review_run=None,
                        review_memory=False, explicit=False):
            captured["explicit"] = explicit

        with patch.object(bg, "_run_review_in_thread", fake_worker):
            target, _prompt = bg.spawn_background_review_thread(
                SimpleNamespace(), [], review_memory=True, explicit=True)
            target()
            assert captured["explicit"] is True

    @pytest.mark.parametrize("explicit", [True, False])
    def test_fork_keeps_background_review_origin_and_carries_attendedness(self, explicit):
        """Every curator/skill guard keys on the background_review origin, so an explicit /refine
        must NOT change the origin; attendedness rides on the fork as its own flag."""
        forks = []

        def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
            fork = SimpleNamespace(
                _memory_enabled=True, _user_profile_enabled=False, _memory_write_origin=write_origin,
                run_conversation=lambda **kw: None, _session_messages=[])
            forks.append(fork)
            return fork, {}, False

        noop = lambda *a, **k: None
        with patch.object(bg, "build_cache_parity_fork", fake_build), \
                patch.object(bg, "_track_review_fork", noop), \
                patch.object(bg, "_snapshot_review_usage", lambda a: {}), \
                patch.object(bg, "_record_review_usage_to_parent", noop), \
                patch.object(bg, "finish_background_review_run", noop), \
                patch.object(bg, "_release_fork_clients", noop):
            bg._run_review_fork(SimpleNamespace(), [], "p", None, None, bg._ReviewForkState(), True, explicit)
        (fork,) = forks
        assert fork._memory_write_origin == "background_review"
        assert fork._review_attended is explicit


class TestConsolidationProposalSurfaces:
    """The fork's own review summary is never published back, so a consolidation the delete
    gate staged must surface through ``summarize_background_review_actions`` — otherwise the
    near-limit denial path drops both the requested update and the proposal, silently (#105921)."""

    def _store(self, tmp_path, monkeypatch):
        import json as _json
        from tools.memory_tool_store import MemoryStore

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=500, user_char_limit=300)
        store.load_from_disk()
        return store

    def test_staged_proposal_surfaces_in_summary(self, tmp_path, monkeypatch):
        import json

        from tools.memory_tool import memory_tool
        from tools.skill_provenance import set_current_write_origin, reset_current_write_origin

        store = self._store(tmp_path, monkeypatch)
        assert store.add("memory", "standing rule entry")["success"] is True

        token = set_current_write_origin("background_review")
        try:
            raw = memory_tool(
                action="replace", old_text="standing rule", content="consolidated entry", store=store)
        finally:
            reset_current_write_origin(token)
        result = json.loads(raw)
        assert result["staged"] is True and result["proposal_staged"] is True

        review_messages = [
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "memory", "arguments": json.dumps(
                {"action": "replace", "old_text": "standing rule", "content": "consolidated entry"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": raw},
        ]
        actions = bg.summarize_background_review_actions(review_messages, [])
        assert any("staged for your approval" in a for a in actions)

    def test_near_limit_denial_end_to_end(self, tmp_path, monkeypatch):
        """add rejected by the budget -> fork follows the 'consolidate now' hint with a
        replace -> the delete gate stages it -> the proposal surfaces; the store never
        changed and nothing was silently lost."""
        import json

        from tools.memory_tool import memory_tool
        from tools.skill_provenance import set_current_write_origin, reset_current_write_origin

        store = self._store(tmp_path, monkeypatch)
        assert store.add("memory", "seed entry one")["success"] is True
        # Near-limit: a further add is rejected and the store's hint says to consolidate.
        assert store.add("memory", "x" * 600)["success"] is False

        token = set_current_write_origin("background_review")
        try:
            add_raw = memory_tool(action="add", content="y" * 600, store=store)
            replace_raw = memory_tool(
                action="replace", old_text="seed entry one", content="merged entry", store=store)
        finally:
            reset_current_write_origin(token)
        assert json.loads(add_raw)["success"] is False  # the budget still rejects the add
        replace_result = json.loads(replace_raw)
        assert replace_result["staged"] is True and replace_result["proposal_staged"] is True

        # Fail-closed: nothing was applied or dropped.
        assert "seed entry one" in store._entries_for("memory")
        assert "merged entry" not in store._entries_for("memory")

        review_messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "memory", "arguments": json.dumps(
                    {"action": "add", "content": "y" * 600})}},
                {"id": "c2", "function": {"name": "memory", "arguments": json.dumps(
                    {"action": "replace", "old_text": "seed entry one", "content": "merged entry"})}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": add_raw},
            {"role": "tool", "tool_call_id": "c2", "content": replace_raw},
        ]
        actions = bg.summarize_background_review_actions(review_messages, [])
        assert any("staged for your approval" in a for a in actions)


class TestPendingProposalsContext:
    """``pending_memory_proposals_context`` (#81671): a bounded digest of ACTIVE
    background-review memory proposals so the fork does not re-derive maintenance
    already queued for a human — and its wiring into the memory review prompt."""

    def _home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    def _stage(self, tmp_path, *, origin="background_review", status=None,
               action="replace", matched_entry=None, summary="consolidate duplicates", target="memory"):
        """Stage one pending record via the real write_approval path; return its record dict."""
        from tools import write_approval as wa

        payload = {"action": action, "target": target, "old_text": "seed entry", "content": "merged"}
        if action == "batch":
            payload = {"action": "batch", "target": target, "operations": [
                {"action": "replace", "old_text": "seed entry", "content": "merged",
                 **({"matched_entry": matched_entry} if matched_entry else {})}]}
        elif matched_entry is not None:
            payload["matched_entry"] = matched_entry
        record = wa.stage_write("memory", payload, summary=summary, origin=origin)
        if status is not None:  # stage_write has no status param (added by a sibling change)
            path = wa._pending_path("memory", record["id"])
            record["status"] = status
            import json as _json

            path.write_text(_json.dumps(record), encoding="utf-8")
        return record

    def test_empty_queue_returns_empty_and_prompt_unchanged(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        assert bg.pending_memory_proposals_context() == ""

        agent = SimpleNamespace()
        with patch.object(bg, "_run_review_in_thread", lambda *a, **k: None):
            _, prompt = bg.spawn_background_review_thread(agent, [], review_memory=True)
        assert prompt == bg._MEMORY_REVIEW_PROMPT

    def test_qualifying_record_appears_in_context_and_prompt(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        import hashlib

        record = self._stage(tmp_path, matched_entry="seed entry one", summary="a" * 200)
        ctx = bg.pending_memory_proposals_context()
        assert record["id"] in ctx
        assert "replace" in ctx and "memory" in ctx
        # The brief is the OP text, which tells the reviewer WHICH entry is queued; the
        # staging summary's boilerplate never appears (it used to eat the whole 80 chars).
        assert "replace entry matching 'seed entry'" in ctx
        assert ("a" * 80) not in ctx and "background review consolidation" not in ctx
        digest = hashlib.sha256(b"seed entry one").hexdigest()[:8]
        assert f"pinned: {digest}" in ctx and len(digest) == 8

        agent = SimpleNamespace()
        with patch.object(bg, "_run_review_in_thread", lambda *a, **k: None):
            _, prompt = bg.spawn_background_review_thread(agent, [], review_memory=True)
        marker = "Already-proposed memory maintenance awaiting review"
        assert marker in prompt
        assert prompt.endswith(ctx)

    def test_op_brief_is_capped_at_the_character_budget(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        from tools import write_approval as wa

        long_entry = "E" * 150
        wa.stage_write("memory", {"action": "remove", "target": "memory",
                                  "old_text": long_entry, "matched_entry": long_entry},
                       summary="staging summary that must not be used", origin="background_review")
        ctx = bg.pending_memory_proposals_context()
        # "- remove: " (10 chars) + 70 E's fills the 80-char brief exactly.
        assert ("E" * 70) in ctx and ("E" * 71) not in ctx
        assert "staging summary that must not be used" not in ctx

    def test_non_qualifying_records_skipped(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        # foreground origin
        self._stage(tmp_path, origin="foreground", matched_entry="seed entry one")
        assert bg.pending_memory_proposals_context() == ""
        # archived status
        self._stage(tmp_path, status="stale", matched_entry="seed entry one")
        assert bg.pending_memory_proposals_context() == ""
        # adds only — harmless duplicates the fork can already see
        self._stage(tmp_path, action="add")
        assert bg.pending_memory_proposals_context() == ""

    def test_caps_limit_lines_and_characters(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        for i in range(12):
            self._stage(tmp_path, matched_entry=f"seed entry {i}", summary=f"consolidate dupes {i:02d}")
        ctx = bg.pending_memory_proposals_context()
        lines = [ln for ln in ctx.splitlines() if ln.startswith("- ")]
        assert len(lines) == 8  # default record limit
        assert "(+4 more — /memory pending)" in ctx
        assert len(ctx) < 1500  # char budget + slack for the truncation note

    def test_import_failure_returns_empty_and_spawn_survives(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)

        def boom(subsystem):
            raise RuntimeError("queue unavailable")

        from tools import write_approval as wa

        with patch.object(wa, "list_pending", boom):
            assert bg.pending_memory_proposals_context() == ""
            agent = SimpleNamespace()
            with patch.object(bg, "_run_review_in_thread", lambda *a, **k: None):
                _, prompt = bg.spawn_background_review_thread(agent, [], review_memory=True)
            assert "Already-proposed" not in prompt
            assert prompt == bg._MEMORY_REVIEW_PROMPT