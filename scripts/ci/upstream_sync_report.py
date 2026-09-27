"""Report every sync outcome from the final job's dependency results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile


def build_report(needs: dict, run_url: str) -> tuple[str, str]:
    """Return the issue action and a report with the retained run identity."""
    prepare = needs["prepare"]
    outputs = prepare.get("outputs", {})
    failed = [name for name, job in needs.items() if job["result"] in {"failure", "cancelled"}]
    if failed:
        action = "blocked"
        headline = "Upstream sync blocked. main was not promoted."
        # A reporting failure after promotion must not imply that the push failed.
        if needs["promote"].get("outputs", {}).get("promoted") == "true":
            headline = "Upstream sync promoted the candidate, but a later step failed."
    elif needs["promote"]["result"] == "success":
        action, headline = "resolved", "Validated upstream sync succeeded and was promoted to main."
    elif prepare["result"] == "success" and outputs.get("needed") == "false":
        action, headline = "resolved", "Fork main already contains upstream."
    elif prepare["result"] == "skipped":
        return "none", ""
    else:
        action, headline = "blocked", "Upstream sync did not complete. main was not promoted."

    lines = [headline, "", f"- Run: {run_url}"]
    for field in ("base_sha", "upstream_sha", "candidate_sha"):
        lines.append(f"- {field}: `{outputs.get(field) or 'unavailable'}`")
    for name, job in needs.items():
        lines.append(f"- {name}: {job['result']}")
    for field in ("ci_run_id", "nix_run_id"):
        run_id = needs["validate"].get("outputs", {}).get(field)
        if run_id:
            lines.append(f"- {field}: {run_url.rsplit('/', 1)[0]}/{run_id}")
    return action, "\n".join(lines) + "\n"


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True, timeout=60)


def main() -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_url = f"{server}/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    action, body = build_report(json.loads(os.environ["SYNC_NEEDS"]), run_url)
    if action == "none":
        return
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        summary.write(body)
    print(body)
    if action == "blocked":
        print(f"::error::Upstream sync requires attention. See {run_url}")
    metadata = json.loads(gh("api", f"repos/{repo}"))
    if not metadata["has_issues"]:
        print("::warning::Issues are disabled. The run summary contains the sync report.")
        return
    title = os.environ["BLOCKER_TITLE"]
    pages = json.loads(gh("api", f"repos/{repo}/issues?state=open&per_page=100", "--paginate", "--slurp"))
    blockers = [issue for page in pages for issue in page
                if issue["title"] == title and "pull_request" not in issue]
    with tempfile.TemporaryDirectory(prefix="upstream-sync-report-") as directory:
        body_path = Path(directory) / "body.md"
        body_path.write_text(body, encoding="utf-8")
        if action == "blocked" and not blockers:
            gh("issue", "create", "--repo", repo, "--title", title, "--body-file", str(body_path))
        for issue in blockers:
            number = str(issue["number"])
            gh("issue", "comment", number, "--repo", repo, "--body-file", str(body_path))
            if action == "resolved":
                gh("issue", "close", number, "--repo", repo)


if __name__ == "__main__":
    main()
