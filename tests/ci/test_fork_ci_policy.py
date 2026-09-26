"""The fork CI policy transform (scripts/ci/fork_ci_policy.py) that the validated
upstream sync re-applies after taking upstream's side of a workflow conflict."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fork_ci_policy", REPO / "scripts/ci/fork_ci_policy.py")
policy = importlib.util.module_from_spec(_spec)
sys.modules["fork_ci_policy"] = policy  # @dataclass resolves its module here
_spec.loader.exec_module(policy)

PRIVATE_LABEL = re.compile(r"\b(ubuntu|windows)-latest-\d+-(arm-)?core\b")

UPSTREAM_STYLE = """\
jobs:
  build:
    # sized for ubuntu-latest-32-core
    runs-on: ${{ inputs.arm && 'windows-latest-32-arm-core' || 'windows-latest-32-core' }}
    strategy:
      matrix:
        include:
          - runner: ubuntu-latest-32-arm-core
          - runner: ubuntu-latest-96-core
"""


def test_no_private_runner_label_survives_and_policy_is_idempotent():
    once = policy.apply_policy("some-new-upstream-workflow.yml", UPSTREAM_STYLE)
    assert not PRIVATE_LABEL.search(once)
    assert policy.apply_policy("some-new-upstream-workflow.yml", once) == once


def test_a_rule_whose_upstream_anchor_changed_fails_by_name():
    with pytest.raises(policy.PolicyError, match="js: check concurrency"):
        policy.apply_policy("js-tests.yml", "run: node .github/scripts/renamed-runner.mjs\n")


def test_committed_upstream_workflows_are_already_policy_fixed_points():
    workflows = sorted((REPO / policy.WORKFLOWS).glob("*.y*ml"))
    assert {p.name for p in workflows} >= set(policy._RULES)
    for path in workflows:
        if path.name in policy.FORK_OWNED:
            continue
        text = path.read_text(encoding="utf-8")
        assert policy.apply_policy(path.name, text) == text, path.name
