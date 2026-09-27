"""Exercise sync dispatch and reporting without publishing to GitHub."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/ci/upstream_sync_report.py"
spec = importlib.util.spec_from_file_location("upstream_sync_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.mark.parametrize(
    "stage",
    ["prepare", "validate", "promote", "cancelled", "skipped", "promoted", "current"],
)
@pytest.mark.parametrize("issues", ["disabled", "empty", "existing"])
@pytest.mark.platforms("posix")
def test_report_cli_covers_each_stage_and_preserves_run_identity(
    tmp_path, stage, issues
):
    needs = {
        name: {"result": "success", "outputs": {}}
        for name in ("prepare", "validate", "promote")
    }
    needs["prepare"]["outputs"] = {
        "needed": "true",
        "base_sha": "base",
        "upstream_sha": "upstream",
        "candidate_sha": "candidate",
    }
    needs["validate"]["outputs"] = {"ci_run_id": "12", "nix_run_id": "13"}
    if stage in ("prepare", "validate", "promote"):
        needs[stage]["result"] = "failure"
        for name in list(needs)[list(needs).index(stage) + 1 :]:
            needs[name]["result"] = "skipped"
    elif stage == "cancelled":
        needs["validate"]["result"] = "cancelled"
        needs["promote"]["result"] = "skipped"
    elif stage == "skipped":
        needs["promote"]["result"] = "skipped"
    elif stage == "current":
        needs["prepare"]["outputs"]["needed"] = "false"
        needs["validate"]["result"] = needs["promote"]["result"] = "skipped"

    calls = tmp_path / "calls.jsonl"
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, pathlib, sys
args = sys.argv[1:]
if '--body-file' in args:
    body = pathlib.Path(args[args.index('--body-file') + 1]).read_text()
else:
    body = None
with open(os.environ['CALLS'], 'a') as log:
    log.write(json.dumps({'args': args, 'body': body}) + '\\n')
if args[0] == 'api':
    if '/issues?' in args[1]:
        print(json.dumps([[]] if os.environ['ISSUES'] == 'empty' else
                         [[{'number': 7, 'title': 'Upstream sync blocked'}]]))
    else:
        print(json.dumps({'has_issues': os.environ['ISSUES'] != 'disabled'}))
"""
    )
    gh.chmod(0o755)
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "SYNC_NEEDS": json.dumps(needs),
            "GITHUB_REPOSITORY": "owner/fork",
            "GITHUB_RUN_ID": "11",
            "GITHUB_STEP_SUMMARY": str(summary),
            "BLOCKER_TITLE": "Upstream sync blocked",
            "CALLS": str(calls),
            "ISSUES": issues,
        },
    )
    assert result.returncode == 0, result.stderr
    body = summary.read_text(encoding="utf-8-sig")
    assert "https://github.com/owner/fork/actions/runs/11" in body
    assert "https://github.com/owner/fork/actions/runs/12" in body
    assert "https://github.com/owner/fork/actions/runs/13" in body
    assert all(f"`{sha}`" in body for sha in ("base", "upstream", "candidate"))
    requests = [
        json.loads(line) for line in calls.read_text(encoding="utf-8-sig").splitlines()
    ]
    mutations = [request for request in requests if request["args"][0] == "issue"]
    if issues == "disabled":
        assert not mutations
        assert "::warning::Issues are disabled" in result.stdout
    elif issues == "existing":
        assert mutations[0]["args"][:3] == ["issue", "comment", "7"]
        assert mutations[0]["body"] == body
        assert any(request["args"][1] == "close" for request in mutations) == (
            stage in {"promoted", "current"}
        )
    elif stage in {"promoted", "current"}:
        assert not mutations
    else:
        assert mutations[0]["args"][:2] == ["issue", "create"]
        assert mutations[0]["body"] == body


@pytest.mark.parametrize("stage", ["failure", "skipped", "success"])
def test_report_distinguishes_a_completed_push_from_missing_validation(stage):
    needs = {
        name: {"result": "success", "outputs": {}}
        for name in ("prepare", "validate", "promote")
    }
    needs["promote"] = {
        "result": stage,
        "outputs": {"promoted": "true"} if stage != "skipped" else {},
    }
    action, body = report.build_report(
        needs, "https://github.com/owner/fork/actions/runs/11"
    )
    assert action == ("resolved" if stage == "success" else "blocked")
    if stage == "failure":
        assert "promoted the candidate" in body
        assert "main was not promoted" not in body
    elif stage == "skipped":
        assert "did not complete" in body
    needs["prepare"]["result"] = "skipped"
    needs["promote"]["result"] = "skipped"
    assert report.build_report(needs, "run") == ("none", "")


@pytest.mark.platforms("posix")
def test_dispatch_requests_strict_checks_and_ignores_older_candidates(tmp_path):
    workflow = YAML(typ="safe").load(
        (ROOT / ".github/workflows/upstream-sync.yml").read_text(encoding="utf-8-sig")
    )
    job = workflow["jobs"]["validate"]
    dispatch = next(step for step in job["steps"] if step.get("id") == "dispatch")
    calls = tmp_path / "calls.jsonl"
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        + """
import datetime, json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
if args[:2] == ['run', 'list']:
    query = args[args.index('--jq') + 1]
    run_id = 12 if args[args.index('--workflow') + 1] == 'ci.yaml' else 13
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    runs = [{'databaseId': 1, 'headSha': 'candidate', 'createdAt': '2000-01-01T00:00:00Z'},
            {'databaseId': 2, 'headSha': 'wrong', 'createdAt': now},
            {'databaseId': run_id, 'headSha': 'candidate', 'createdAt': now}]
    subprocess.run(['jq', '-r', query], input=json.dumps(runs), text=True, check=True)
"""
    )
    gh.chmod(0o755)
    output, summary = tmp_path / "output", tmp_path / "summary"
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "CALLS": str(calls),
        "GITHUB_REPOSITORY": "owner/fork",
        "SYNC_BRANCH": "automation/upstream-sync",
        "CANDIDATE_SHA": "candidate",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    result = subprocess.run(
        ["bash", "-c", dispatch["run"]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    requests = [
        json.loads(line) for line in calls.read_text(encoding="utf-8-sig").splitlines()
    ]
    dispatched = [args for args in requests if args[:2] == ["workflow", "run"]]
    assert len(dispatched) == 2
    assert all("release=true" in args for args in dispatched)
    assert output.read_text(encoding="utf-8-sig").splitlines() == [
        "ci_run_id=12",
        "nix_run_id=13",
    ]
    # Both children must be observed, even when the first watch fails.
    watchers = [step for step in job["steps"] if step["name"].startswith("Require")]
    assert len(watchers) == 2
    assert all("always()" in step["if"] for step in watchers)
