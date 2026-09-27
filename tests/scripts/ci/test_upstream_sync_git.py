"""Exercise candidate preparation and promotion against disposable Git remotes."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[3]


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def sync_repos(tmp_path):
    fork = tmp_path / "fork"
    fork.mkdir()
    git(fork, "init", "--initial-branch=main")
    git(fork, "config", "user.name", "sync-test")
    git(fork, "config", "user.email", "sync-test@example.com")
    shutil.copytree(ROOT / ".github/workflows", fork / ".github/workflows")
    for path in (
        "scripts/ci/fork_ci_policy.py",
        "tools/delegate_tool_config.py",
        "tests/tools/test_delegate_model_selection.py",
        "hermes_cli/config_defaults.py",
    ):
        target = fork / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    workflow = fork / ".github/workflows/tests.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8-sig") + "\n# shared configuration note\n",
        encoding="utf-8",
    )
    (fork / "feature.txt").write_text("base\n", encoding="utf-8")
    git(fork, "add", ".")
    git(fork, "commit", "-m", "test base")
    origin, upstream = tmp_path / "origin.git", tmp_path / "upstream.git"
    git(tmp_path, "clone", "--bare", str(fork), str(origin))
    git(tmp_path, "clone", "--bare", str(fork), str(upstream))
    git(fork, "remote", "add", "origin", str(origin))
    upstream_work = tmp_path / "upstream-work"
    git(tmp_path, "clone", str(upstream), str(upstream_work))
    git(upstream_work, "config", "user.name", "upstream-test")
    git(upstream_work, "config", "user.email", "upstream-test@example.com")
    (upstream_work / "new-feature.txt").write_text(
        "upstream feature\n", encoding="utf-8"
    )
    workflow = YAML(typ="safe").load(
        (ROOT / ".github/workflows/upstream-sync.yml").read_text(encoding="utf-8-sig")
    )
    env = {
        **os.environ,
        "UPSTREAM_URL": str(upstream),
        "SYNC_BRANCH": "automation/upstream-sync",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }
    return fork, origin, upstream_work, workflow, env


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "change", ["clean", "workflow-conflict", "source-conflict", "policy-drift"]
)
def test_prepare_only_pushes_supported_candidates_and_preserves_main(
    sync_repos, change
):
    fork, origin, upstream, workflow, env = sync_repos
    if change in {"workflow-conflict", "source-conflict"}:
        path = (
            ".github/workflows/tests.yml"
            if change == "workflow-conflict"
            else "feature.txt"
        )
        for repo, note in ((fork, "fork note"), (upstream, "upstream note")):
            target = repo / path
            if change == "workflow-conflict":
                target.write_text(
                    target.read_text(encoding="utf-8-sig").replace(
                        "shared configuration note", note
                    ),
                    encoding="utf-8",
                )
            else:
                target.write_text(note + "\n", encoding="utf-8")
        git(fork, "add", ".")
        git(fork, "commit", "-m", "fork change")
        git(fork, "push", "origin", "main")
    elif change == "policy-drift":
        target = upstream / ".github/workflows/js-tests.yml"
        target.write_text(
            target.read_text(encoding="utf-8-sig").replace(
                "run-workspace-checks.mjs", "renamed-checks.mjs"
            ),
            encoding="utf-8",
        )
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "upstream change")
    git(upstream, "push", "origin", "main")
    base, upstream_sha = (
        git(origin, "rev-parse", "main"),
        git(upstream, "rev-parse", "HEAD"),
    )
    prepare = next(
        step
        for step in workflow["jobs"]["prepare"]["steps"]
        if step.get("id") == "prepare"
    )
    result = subprocess.run(
        ["bash", "-c", prepare["run"]],
        cwd=fork,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert git(origin, "rev-parse", "main") == base
    supported = change in {"clean", "workflow-conflict"}
    assert (result.returncode == 0) == supported, result.stdout + result.stderr
    candidate_ref = git(
        origin,
        "for-each-ref",
        "--format=%(objectname)",
        "refs/heads/automation/upstream-sync",
    )
    if supported:
        assert candidate_ref
        git(fork, "merge-base", "--is-ancestor", base, candidate_ref)
        git(fork, "merge-base", "--is-ancestor", upstream_sha, candidate_ref)
        check = subprocess.run(
            ["python3", "scripts/ci/fork_ci_policy.py", "--check"],
            cwd=fork,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        assert check.returncode == 0, check.stdout + check.stderr
    else:
        assert not candidate_ref
        summary = Path(env["GITHUB_STEP_SUMMARY"]).read_text(encoding="utf-8-sig")
        assert (
            "outside CI workflows"
            if change == "source-conflict"
            else "js: check concurrency"
        ) in summary


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("moved", [None, "main", "automation/upstream-sync"])
def test_promote_requires_the_validated_base_and_candidate(sync_repos, moved):
    fork, origin, upstream, workflow, env = sync_repos
    base = git(fork, "rev-parse", "HEAD")
    upstream_sha = git(upstream, "rev-parse", "HEAD")
    (fork / "candidate.txt").write_text("validated candidate\n", encoding="utf-8")
    git(fork, "add", ".")
    git(fork, "commit", "-m", "candidate")
    candidate = git(fork, "rev-parse", "HEAD")
    git(fork, "push", "origin", "HEAD:refs/heads/automation/upstream-sync")
    if moved:
        git(fork, "checkout", "--detach", base if moved == "main" else candidate)
        (fork / "concurrent.txt").write_text("concurrent change\n", encoding="utf-8")
        git(fork, "add", ".")
        git(fork, "commit", "-m", "concurrent change")
        git(fork, "push", "origin", f"HEAD:refs/heads/{moved}")
    before = git(origin, "rev-parse", "main")
    step = next(
        step
        for step in workflow["jobs"]["promote"]["steps"]
        if step.get("id") == "promote"
    )
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=fork,
        env={
            **env,
            "BASE_SHA": base,
            "UPSTREAM_SHA": upstream_sha,
            "CANDIDATE_SHA": candidate,
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    assert (result.returncode == 0) == (moved is None), result.stdout + result.stderr
    assert git(origin, "rev-parse", "main") == (candidate if moved is None else before)
    if moved is None:
        assert "promoted=true" in Path(env["GITHUB_OUTPUT"]).read_text(
            encoding="utf-8-sig"
        )
