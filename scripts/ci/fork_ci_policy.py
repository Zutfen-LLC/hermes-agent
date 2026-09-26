#!/usr/bin/env python3
"""Zutfen fork CI policy, applied to upstream's GitHub workflow files.

The fork runs upstream's workflows on standard GitHub-hosted runners, which
cannot provide upstream's private large-runner labels. Instead of carrying
hand edits in upstream-owned files (which conflict every time upstream
touches a nearby line), the fork's differences live here as a transform:

    fork workflow file == apply(upstream workflow file)

The validated upstream-sync workflow resolves a workflow-file merge conflict
by taking upstream's side and re-running this transform. Each specific rule
names the exact text it rewrites; when upstream changes that text the rule
fails loudly with its name instead of guessing, and the sync reports a
blocker so a human can update the rule.

Stdlib only: the sync job runs it before any dependency install.

    python3 scripts/ci/fork_ci_policy.py           # rewrite in place
    python3 scripts/ci/fork_ci_policy.py --check   # exit 1 if any file would change
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

WORKFLOWS = Path(".github/workflows")
# Workflows only the fork has; the policy describes upstream files.
FORK_OWNED = frozenset({"upstream-sync.yml"})

# Private large-runner labels -> the standard hosted equivalent. Applied to the
# whole file text (comments included) so no private label survives anywhere.
_LABELS = (
    (re.compile(r"\bubuntu-latest-\d+-arm-core\b"), "ubuntu-24.04-arm"),
    (re.compile(r"\bubuntu-latest-\d+-core\b"), "ubuntu-latest"),
    (re.compile(r"\bwindows-latest-\d+-arm-core\b"), "windows-11-arm"),
    (re.compile(r"\bwindows-latest-\d+-core\b"), "windows-latest"),
)


class PolicyError(RuntimeError):
    """A rule's anchor text is gone from the upstream file."""


@dataclass(frozen=True)
class Replace:
    """Replace ``old`` with ``new`` exactly once; a no-op when ``new`` is already there."""

    name: str
    old: str
    new: str

    def apply(self, text: str) -> str:
        hits = text.count(self.old)
        if hits == 1:
            return text.replace(self.old, self.new)
        if hits == 0 and self.new in text:
            return text
        raise PolicyError(f"{self.name}: anchor matched {hits} times, expected 1")


@dataclass(frozen=True)
class DropStep:
    """Remove the step named ``step`` (through the line before the next step or job)."""

    name: str
    step: str

    def apply(self, text: str) -> str:
        pattern = re.compile(
            rf"^      - name: {re.escape(self.step)}\n(?:(?!      - |  \S).*\n|\n)*", re.MULTILINE)
        return pattern.sub("", text, count=1)


# Rules run before the generic label map, so their anchors can name upstream's labels.
_RULES: dict[str, tuple] = {
    "nix.yml": (
        # The sync dispatches Nix on the exact candidate commit.
        Replace("nix: workflow_dispatch trigger",
                "on:\n  pull_request:\n",
                "on:\n  workflow_dispatch:\n  pull_request:\n"),
    ),
    "tests.yml": (
        # One quarter of the suite per 4-vCPU runner instead of one 96-core runner.
        Replace("tests: slice matrix",
                "    name: Run tests\n",
                "    name: Run tests (slice ${{ matrix.slice }}/4)\n"
                "    strategy:\n"
                "      fail-fast: false\n"
                "      matrix:\n"
                "        slice: [1, 2, 3, 4]\n"),
        Replace("tests: unit job bound",
                "    runs-on: ubuntu-latest-96-core\n    timeout-minutes: 30\n",
                "    runs-on: ubuntu-latest\n    timeout-minutes: 45\n"),
        Replace("tests: sliced run",
                "          scripts/run_tests.sh\n        env:\n",
                "          scripts/run_tests.sh --slice ${{ matrix.slice }}/4 -j 4\n        env:\n"),
        Replace("tests: unit workers", "HERMES_TEST_WORKERS: 96\n", "HERMES_TEST_WORKERS: 4\n"),
        # Four slices restoring whichever slice saved last would slice the suite
        # differently per job; without the cache every slice splits by file count.
        DropStep("tests: no duration cache restore", "Restore per-file duration cache"),
        DropStep("tests: no duration cache save", "Save per-file duration cache (main only)"),
        # Fewer concurrent process trees on 4 vCPUs, with longer job bounds.
        Replace("tests: e2e bound",
                "    runs-on: ubuntu-latest-32-core\n    timeout-minutes: 30\n",
                "    runs-on: ubuntu-latest\n    timeout-minutes: 60\n"),
        Replace("tests: e2e workers", 'HERMES_TEST_WORKERS: "3"\n', 'HERMES_TEST_WORKERS: "2"\n'),
        Replace("tests: e2e-upgrade bound",
                "    runs-on: ubuntu-latest-32-core\n    timeout-minutes: 60\n",
                "    runs-on: ubuntu-latest\n    timeout-minutes: 90\n"),
        Replace("tests: e2e-upgrade workers",
                'HERMES_TEST_WORKERS: "4"\n          HERMES_TEST_FILE_TIMEOUT: "3000"\n',
                'HERMES_TEST_WORKERS: "2"\n          HERMES_TEST_FILE_TIMEOUT: "3000"\n'),
    ),
    "tests-os.yml": (
        # psutil reports a CREATE_SUSPENDED child as running on the windows-2025
        # hosted image; the job-object tests need the stopped state.
        Replace("tests-os: windows x64 image",
                "            runner: windows-latest-32-core\n",
                "            runner: windows-2022\n"),
    ),
    "js-tests.yml": (
        # Every check sizes its own worker pool to the core count; four at once
        # on 4 vCPUs starves vitest's per-test timeouts.
        Replace("js: check concurrency",
                "run: node .github/scripts/run-workspace-checks.mjs\n",
                "run: node .github/scripts/run-workspace-checks.mjs --concurrency 2\n"),
        Replace("js: check bound",
                "    runs-on: ubuntu-latest-32-core\n    timeout-minutes: 30\n",
                "    runs-on: ubuntu-latest\n    timeout-minutes: 45\n"),
    ),
    "contributor-check.yml": (
        # Judge only fork-authored commits: a sync or catch-up merge brings in
        # upstream history whose authors upstream itself never mapped.
        Replace("contributors: skip upstream history",
                "NEW_EMAILS=$(git log ${MERGE_BASE}..HEAD --format='%ae' --no-merges | sort -u)\n",
                "git fetch --quiet --no-tags https://github.com/NousResearch/hermes-agent.git main\n"
                "          NEW_EMAILS=$(git log ${MERGE_BASE}..HEAD --not FETCH_HEAD --format='%ae' --no-merges | sort -u)\n"),
    ),
}


def apply_policy(filename: str, text: str) -> str:
    """The fork's version of upstream workflow ``filename`` whose content is ``text``."""
    for rule in _RULES.get(filename, ()):
        text = rule.apply(text)
    for pattern, label in _LABELS:
        text = pattern.sub(label, text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report files that would change; write nothing")
    args = parser.parse_args(argv)

    missing = sorted(set(_RULES) - {p.name for p in WORKFLOWS.glob("*.y*ml")})
    if missing:
        print(f"fork CI policy: rule targets no longer exist: {', '.join(missing)}", file=sys.stderr)
        return 2
    changed = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        if path.name in FORK_OWNED:
            continue
        text = path.read_text(encoding="utf-8")
        try:
            new = apply_policy(path.name, text)
        except PolicyError as exc:
            print(f"fork CI policy: {path}: {exc}", file=sys.stderr)
            return 2
        if new != text:
            changed.append(path)
            if not args.check:
                path.write_text(new, encoding="utf-8")
    for path in changed:
        print(f"{'would rewrite' if args.check else 'rewrote'} {path}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
