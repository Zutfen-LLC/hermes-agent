"""The fork CI policy transform (scripts/ci/fork_ci_policy.py) that the validated
upstream sync re-applies after taking upstream's side of a workflow conflict."""

import importlib.util
import re
import sys
from pathlib import Path
import runpy

import pytest
from ruamel.yaml import YAML

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "fork_ci_policy", REPO / "scripts/ci/fork_ci_policy.py"
)
policy = importlib.util.module_from_spec(_spec)
sys.modules["fork_ci_policy"] = policy  # @dataclass resolves its module here
_spec.loader.exec_module(policy)

PRIVATE_LABEL = re.compile(r"\b(ubuntu|windows)-latest-\d+-(arm-)?core\b")

UPSTREAM_STYLE = """\
jobs:
  build:
    # sized for ubuntu-latest-32-core
    runs-on: ${{ inputs.arm && 'windows-latest-32-arm-core' || 'windows-latest-32-core' }}
  linux:
    runs-on: ${{ matrix.runner }}
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
    jobs = YAML(typ="base").load(once)["jobs"]
    assert jobs["build"]["if"].startswith("false")
    assert "if" not in jobs["linux"]


def test_a_rule_whose_upstream_anchor_changed_fails_by_name():
    with pytest.raises(policy.PolicyError, match="js: check concurrency"):
        policy.apply_policy(
            "js-tests.yml", "run: node .github/scripts/renamed-runner.mjs\n"
        )


def test_committed_upstream_workflows_are_already_policy_fixed_points():
    workflows = sorted((REPO / policy.WORKFLOWS).glob("*.y*ml"))
    assert {p.name for p in workflows} >= set(policy._RULES)
    for path in workflows:
        if path.name in policy.FORK_OWNED:
            continue
        text = path.read_text(encoding="utf-8")
        assert policy.apply_policy(path.name, text) == text, path.name


def test_native_clients_are_disabled_and_linux_validation_remains_strict():
    yaml = YAML(typ="base")
    workflows = {
        path.name: yaml.load(path.read_text(encoding="utf-8"))
        for path in (REPO / policy.WORKFLOWS).glob("*.y*ml")
    }
    for filename, workflow in workflows.items():
        for name, job in workflow["jobs"].items():
            selector = str(job.get("runs-on", "")) + str(job.get("strategy", {}))
            native = any(label in selector for label in ("windows-", "macos-"))
            if native or name in policy._DISABLED_JOBS.get(filename, ()):
                assert job.get("if") == "false", (filename, name)
    ci = workflows["ci.yaml"]["jobs"]
    required = ci["all-checks-pass"]["needs"]
    assert "tests-os" not in required
    assert {"tests", "lint", "js-tests", "bootstrap-installer"} <= set(required)
    assert ci["tests"].get("if") != "false"
    assert workflows["bootstrap-installer.yml"]["jobs"]["posix"].get("if") != "false"
    gate = runpy.run_path(str(REPO / "scripts/ci/required_results.py"))["evaluate_gate"]
    results = {name: {"result": "success"} for name in required}
    assert gate(results, release=True)["ok"]
    results["tests"] = {"result": "skipped"}
    assert gate(results, release=True)["failed"] == ["tests"]
    with pytest.raises(policy.PolicyError, match="mixed Linux/native client matrix"):
        policy.apply_policy(
            "new.yml",
            """jobs:
  build:
    runs-on: ${{ matrix.runner }}
    strategy:
      matrix:
        include:
          - runner: ubuntu-latest
          - runner: windows-latest
""",
        )


@pytest.mark.parametrize("upstream_workers", [16, 32, 64])
def test_test_resources_preserve_commands_and_force_hosted_budgets(upstream_workers):
    jobs = "jobs:\n"
    for name in ("test", "e2e", "e2e-upgrade"):
        jobs += f"  {name}:\n    runs-on: ubuntu-latest-{upstream_workers}-core\n    timeout-minutes: 30\n"
        if name == "test":
            jobs += "    name: Run tests (${{ matrix.slice }}/2)\n    strategy:\n      matrix:\n        slice: [1, 2]\n"
        jobs += "    steps:\n      - run: scripts/run_tests.sh\n        env:\n"
        jobs += f'          HERMES_TEST_WORKERS: "{upstream_workers}"\n'
        if name == "test":
            jobs += "          HERMES_TEST_SLICE: ${{ matrix.slice }}/2\n"
        jobs += '          HERMES_TEST_FILE_TIMEOUT: "3000"\n'
    fixed = policy.TestResources().apply(jobs)
    assert policy.TestResources().apply(fixed) == fixed
    parsed = YAML(typ="base").load(fixed)["jobs"]
    for name, (minutes, workers) in {
        "test": (45, 4),
        "e2e": (60, 2),
        "e2e-upgrade": (90, 2),
    }.items():
        job = parsed[name]
        assert job["runs-on"] == "ubuntu-latest"
        assert int(job["timeout-minutes"]) == minutes
        (step,) = job["steps"]
        assert step["run"] == "scripts/run_tests.sh"
        assert int(step["env"]["HERMES_TEST_WORKERS"]) == workers
        assert step["env"]["HERMES_TEST_FILE_TIMEOUT"] == "3000"
    assert parsed["test"]["strategy"]["matrix"]["slice"] == ["1", "2", "3", "4"]
    assert (
        parsed["test"]["steps"][0]["env"]["HERMES_TEST_SLICE"]
        == "${{ matrix.slice }}/4"
    )
    with pytest.raises(policy.PolicyError, match="expected one job"):
        policy.apply_policy(
            "tests.yml", jobs.replace("  e2e-upgrade:", "  renamed-upgrade:")
        )
