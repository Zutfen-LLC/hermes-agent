"""Safe, opt-in model and reasoning selection for delegate_task."""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_children,
    _build_dynamic_schema_overrides,
    delegate_task,
)
from tools.delegate_tool_config import (
    _SELECTION_UNSET,
    _resolve_child_runtime,
    _resolve_task_execution_overrides,
)


def _creds(model="gpt-6-luna"):
    return {
        "model": model,
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "",
        "api_mode": "codex_responses",
        "request_overrides": {},
        "command": None,
        "args": None,
    }


def _parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.model = "gpt-6-luna"
    parent.provider = "openai-codex"
    parent.api_mode = "codex_responses"
    parent._active_children = []
    return parent


class TestSelectionConfig:
    def test_disabled_by_default(self):
        cfg = DEFAULT_CONFIG["delegation"]
        assert cfg["allow_model_selection"] is False
        assert cfg["allowed_models"] == []
        assert cfg["allowed_reasoning_efforts"] == []

    def test_omitted_selection_preserves_credential_object(self):
        creds = _creds()
        resolved, reasoning = _resolve_task_execution_overrides(
            {}, creds, _SELECTION_UNSET, _SELECTION_UNSET
        )
        assert resolved is creds
        assert reasoning is None

    def test_approved_selection_changes_compute_only(self):
        creds = _creds()
        creds["api_key"] = "operator-secret"
        creds["request_overrides"] = {"service_tier": "priority"}
        selection = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-6-luna", "gpt-6-sol"],
            "allowed_reasoning_efforts": ["high", "max"],
        }

        resolved, reasoning = _resolve_task_execution_overrides(
            selection, creds, "gpt-6-sol", "max"
        )

        assert resolved is not creds
        assert resolved["model"] == "gpt-6-sol"
        assert reasoning == {"enabled": True, "effort": "max"}
        for key in (
            "provider",
            "base_url",
            "api_key",
            "api_mode",
            "request_overrides",
            "command",
            "args",
        ):
            assert resolved[key] == creds[key]
        assert creds["model"] == "gpt-6-luna"

    @pytest.mark.parametrize(
        ("model", "effort", "match"),
        [
            ("gpt-6-sol", _SELECTION_UNSET, "disabled"),
            ("gpt-6-sol ", _SELECTION_UNSET, "not allowed"),
            ("other-model", _SELECTION_UNSET, "not allowed"),
            (_SELECTION_UNSET, "ultra", "not allowed"),
        ],
    )
    def test_selection_fails_closed(self, model, effort, match):
        selection = {
            "allow_model_selection": match != "disabled",
            "allowed_models": ["gpt-6-sol"],
            "allowed_reasoning_efforts": ["high", "max"],
        }
        with pytest.raises(ValueError, match=match):
            _resolve_task_execution_overrides(selection, _creds(), model, effort)

    def test_invalid_effort_is_rejected_even_if_operator_listed_it(self):
        selection = {
            "allow_model_selection": True,
            "allowed_reasoning_efforts": ["definitely-not-real"],
        }
        with pytest.raises(ValueError, match="invalid"):
            _resolve_task_execution_overrides(
                selection, _creds(), _SELECTION_UNSET, "definitely-not-real"
            )


class TestSelectionSchema:
    def test_fields_hidden_when_disabled(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            params = _build_dynamic_schema_overrides()["parameters"]

        assert "model" not in params["properties"]
        assert "reasoning_effort" not in params["properties"]
        task_props = params["properties"]["tasks"]["items"]["properties"]
        assert "model" not in task_props
        assert "reasoning_effort" not in task_props

    def test_enabled_schema_uses_exact_operator_allowlists(self):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-6-luna", "gpt-6-sol"],
            "allowed_reasoning_efforts": ["high", "max"],
        }
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            params = _build_dynamic_schema_overrides()["parameters"]

        assert params["properties"]["model"]["enum"] == cfg["allowed_models"]
        assert params["properties"]["reasoning_effort"]["enum"] == cfg[
            "allowed_reasoning_efforts"
        ]
        task_props = params["properties"]["tasks"]["items"]["properties"]
        assert task_props["model"]["enum"] == cfg["allowed_models"]
        assert task_props["reasoning_effort"]["enum"] == cfg[
            "allowed_reasoning_efforts"
        ]

        static_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "model" not in static_props
        assert "reasoning_effort" not in static_props
        assert "model" not in static_props["tasks"]["items"]["properties"]

    def test_real_config_loader_exposes_selection(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        (tmp_path / "config.yaml").write_text(
            "delegation:\n"
            "  allow_model_selection: true\n"
            "  allowed_models: [gpt-6-luna, gpt-6-sol]\n"
            "  allowed_reasoning_efforts: [high, max]\n",
            encoding="utf-8",
        )

        params = _build_dynamic_schema_overrides()["parameters"]

        assert params["properties"]["model"]["enum"] == ["gpt-6-luna", "gpt-6-sol"]
        assert params["properties"]["reasoning_effort"]["enum"] == ["high", "max"]


class TestSelectionPropagation:
    def test_build_children_uses_task_local_model_and_routing_config(self):
        route = {
            "model": "gpt-6-sol",
            "provider": "openai-codex",
            "reasoning_effort": "max",
        }
        child = MagicMock()

        with patch(
            "tools.delegate_tool._build_child_preserving_parent_tools",
            return_value=child,
        ) as build:
            children, err = _build_children(
                [{"goal": "Implement the bounded delegation routing change"}],
                [None],
                [(_creds("gpt-6-sol"), {"enabled": True, "effort": "max"})],
                top_role="leaf",
                max_iterations=50,
                parent_agent=_parent(),
                routing_cfg=route,
                live_deleg_id=None,
                live_writers=[None],
            )

        assert err is None
        assert children[0][2] is child
        kwargs = build.call_args.kwargs
        assert kwargs["model"] == "gpt-6-sol"
        assert kwargs["routing_cfg"] is route
        assert kwargs["override_reasoning_config"] == {"enabled": True, "effort": "max"}
        assert kwargs["override_provider"] == "openai-codex"
        assert kwargs["override_api_mode"] == "codex_responses"

    def test_runtime_prefers_validated_task_reasoning_over_global_effort(self):
        parent = _parent()
        parent.base_url = "https://chatgpt.com/backend-api/codex"
        parent.api_key = "parent-key"
        parent.request_overrides = {}
        parent.reasoning_config = {"enabled": True, "effort": "low"}
        parent._fallback_chain = None
        override = {"enabled": True, "effort": "max"}

        runtime = _resolve_child_runtime(
            parent,
            {"reasoning_effort": "medium"},
            "parent-key",
            model="gpt-6-sol",
            override_provider=None,
            override_base_url=None,
            override_api_key=None,
            override_api_mode=None,
            override_acp_command=None,
            override_acp_args=None,
            override_reasoning_config=override,
            routing_cfg={},
        )

        assert runtime["reasoning_config"] == override

    def test_call_default_and_per_task_override_precedence(self):
        parent = _parent()
        cfg = {
            "max_iterations": 50,
            "max_concurrent_children": 10,
            "allow_model_selection": True,
            "allowed_models": ["gpt-6-luna", "gpt-6-sol"],
            "allowed_reasoning_efforts": ["high", "max"],
        }
        captured = {}

        def build(task_list, task_schemas, task_execution, **kwargs):
            captured["execution"] = task_execution
            return [], None

        tasks = [
            {"goal": "Inspect the routine compatibility path carefully"},
            {
                "goal": "Resolve the ambiguous concurrency failure carefully",
                "model": "gpt-6-sol",
                "reasoning_effort": "max",
            },
        ]
        with (
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch("tools.delegate_tool._resolve_delegation_credentials", return_value=_creds()),
            patch("tools.delegate_tool._build_children", side_effect=build),
            patch("tools.delegate_tool._oneshot_spawn_budget", return_value=None),
            patch("tools.delegate_tool._announce_batch"),
            patch("tools.delegate_tool._capture_origin", return_value=("", "", None, None, False)),
            patch("tools.delegate_tool._run_batch", return_value='{"ok": true}'),
            patch(
                "tools.delegation_live_log.create_live_transcripts",
                return_value=(None, [None, None], []),
            ),
        ):
            result = delegate_task(
                tasks=tasks,
                model="gpt-6-luna",
                reasoning_effort="high",
                background=False,
                parent_agent=parent,
            )

        assert result == '{"ok": true}'
        execution = captured["execution"]
        assert [creds["model"] for creds, _ in execution] == [
            "gpt-6-luna",
            "gpt-6-sol",
        ]
        assert [reasoning for _, reasoning in execution] == [
            {"enabled": True, "effort": "high"},
            {"enabled": True, "effort": "max"},
        ]

    def test_invalid_later_task_fails_before_side_effects(self):
        parent = _parent()
        cfg = {
            "max_iterations": 50,
            "max_concurrent_children": 10,
            "allow_model_selection": True,
            "allowed_models": ["gpt-6-luna"],
        }
        tasks = [
            {"goal": "Inspect the first routine compatibility path"},
            {
                "goal": "Inspect the second routine compatibility path",
                "model": "not-approved",
            },
        ]

        with (
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value=_creds(),
            ),
            patch("tools.delegate_tool._build_children") as build,
            patch("tools.delegate_tool._oneshot_spawn_budget") as budget,
            patch("tools.delegation_live_log.create_live_transcripts") as transcripts,
        ):
            result = delegate_task(tasks=tasks, background=False, parent_agent=parent)

        assert "not allowed" in result
        budget.assert_not_called()
        transcripts.assert_not_called()
        build.assert_not_called()
