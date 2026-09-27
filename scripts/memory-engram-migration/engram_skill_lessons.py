#!/usr/bin/env python3
"""Issue #23 reference implementation: skill-lesson migration + retrieval.

Migrate a skill's attached lesson library into Engram
(``--migrate``) and/or retrieve its lessons deterministically
(``--room``). The authoritative contract lives in
``docs/memory-authority-contract.md``; this script is the executable
form of sections 3, 4, and 7.

Subcommands:

  migrate <library.md> --skill <skill-name>
      Parse lesson blocks out of a local skill library markdown file and
      POST them to Engram as kind=skill_lesson items (idempotent via
      external_source/external_id). Agent writes land 'proposed'; a
      human reviewer promotes them through the Engram review queue.

  retrieve --room <skill-name> [--query Q] [--mode M] [--limit N]
           [--index] [--max-chars C]
      Deterministic full-namespace listing (or --index one-liner), or
      narrowed top-N retrieval. Failure exits 3 with one stderr line and
      empty stdout — Engram is authoritative; there is no local fallback.

Environment: ENGRAM_BASE_URL + ENGRAM_API_KEY (read+write agent key).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

WING = "skills"
KIND = "skill_lesson"
SOURCE_TAG = "hermes-skill-migration"
EXIT_FAIL = 3


def _fail(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"[skill-lessons unavailable: {msg}]", file=sys.stderr)
    raise SystemExit(EXIT_FAIL)


def _creds() -> tuple[str, str]:
    base = os.environ.get("ENGRAM_BASE_URL", "").rstrip("/")
    key = os.environ.get("ENGRAM_API_KEY", "")
    if not base or not key:
        _fail("ENGRAM_BASE_URL / ENGRAM_API_KEY not set in environment")
    return base, key


def _request(
    base: str, key: str, method: str, path: str, payload=None, params: str = ""
):
    req = urllib.request.Request(
        f"{base}{path}{params}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")
    except urllib.error.URLError as exc:
        _fail(f"Engram unreachable at {base}: {exc.reason}")


def _elide_head_tail(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    head, tail = int(cap * 0.7), int(cap * 0.3)
    return (
        text[:head]
        + f"\n\n[... {len(text) - head - tail} chars elided — retrieve with --query to narrow ...]\n\n"
        + text[-tail:]
    )


# --- migrate -----------------------------------------------------------------

_LESSON_START = re.compile(r"^(?:- |\d+[a-z0-9-]*\. )\*\*(.+?)\*\*\s*(.*)$")
_SECTION = re.compile(r"^## (.+)$")


def parse_lessons(text: str) -> list[tuple[str, str, str]]:
    """Yield (group, title, body) per lesson block; numbering-preserving."""
    lessons: list[tuple[str, str, str]] = []
    group: str | None = None
    cur: list | None = None  # [title, chunks]
    for line in text.splitlines():
        sec = _SECTION.match(line)
        if sec:
            if cur:
                lessons.append((group or "", cur[0], "\n".join(cur[1]).strip()))
                cur = None
            group = sec.group(1).strip()
            continue
        m = _LESSON_START.match(line)
        if m:
            if cur:
                lessons.append((group or "", cur[0], "\n".join(cur[1]).strip()))
            cur = [m.group(1).strip(), [m.group(2)]]
        elif cur is not None:
            cur[1].append(line)
    if cur:
        lessons.append((group or "", cur[0], "\n".join(cur[1]).strip()))
    return lessons


def slugify(s: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:maxlen].rstrip("-")


def cmd_migrate(args: argparse.Namespace) -> int:
    base, key = _creds()
    text = open(args.library, encoding="utf-8").read()
    lessons = parse_lessons(text)
    if not lessons:
        print(f"no lesson blocks parsed from {args.library}", file=sys.stderr)
        return 1
    print(f"parsed {len(lessons)} lessons from {args.library}")
    created = deduped = failed = 0
    for i, (group, title, body) in enumerate(lessons, 1):
        ext_id = f"{args.skill}:{i:03d}:{slugify(title)}"
        content = f"[{args.skill} / {group}] {title}\n{body}"
        status, data = _request(
            base,
            key,
            "GET",
            "/v1/items",
            params=f"?kind={KIND}&wing={WING}&room={args.skill}&active_only=false&limit=100",
        )
        if status != 200:
            _fail(f"list failed HTTP {status}: {data}")
        if any(it.get("external_id") == ext_id for it in data.get("items", [])):
            deduped += 1
            continue
        payload = {
            "content": content,
            "kind": KIND,
            "wing": WING,
            "room": args.skill,
            "visibility": "tenant",
            "source_type": "migration",
            "importance": 0.5,
            "subject_type": "skill",
            "subject_id": args.skill,
            "subject_name": group,
            "external_source": SOURCE_TAG,
            "external_id": ext_id,
        }
        status, data = _request(base, key, "POST", "/v1/remember", payload)
        if status in (200, 201):
            created += 1
            print(f"+ {ext_id} -> {data.get('id')} ({data.get('review_status')})")
        else:
            failed += 1
            print(f"!! {ext_id} -> {status}: {str(data)[:200]}")
    print(
        f"DONE: created={created} deduped={deduped} failed={failed} total={len(lessons)}"
    )
    return 1 if failed else 0


# --- retrieve ----------------------------------------------------------------


def cmd_retrieve(args: argparse.Namespace) -> int:
    base, key = _creds()
    params = f"?kind={KIND}&wing={WING}&room={args.room}&active_only=true&limit=100"
    header = ""
    body = ""
    try:
        if args.query:
            status, data = _request(
                base,
                key,
                "POST",
                "/v1/search",
                {
                    "query": args.query,
                    "mode": args.mode,
                    "kind": KIND,
                    "wing": WING,
                    "room": args.room,
                    "limit": max(1, min(args.limit, 20)),
                },
            )
            if status != 200:
                _fail(f"Engram returned HTTP {status} for room {args.room}")
            results = data.get("results", [])
            header = (
                f"## Engram lessons for `{args.room}`"
                f" — top {len(results)} for query {args.query!r} ({args.mode})\n\n"
            )
            body = "\n\n".join(
                f"### lesson [{i}] (score {r.get('score', 0):.3f})\n{r.get('content', '')}"
                for i, r in enumerate(results, 1)
            )
        else:
            status, data = _request(base, key, "GET", "/v1/items", params=params)
            if status != 200:
                _fail(f"Engram returned HTTP {status} for room {args.room}")
            items = data.get("items", [])
            header = (
                f"## Engram lessons for `{args.room}` — {len(items)} active\n"
                f"_(deterministic namespace listing; add a new lesson by writing kind={KIND}, "
                f"wing={WING}, room={args.room} — the skill file itself never grows)_\n\n"
            )
            if args.index:
                body = "\n".join(
                    f"- [{i}] {it.get('content', '').lstrip().splitlines()[0][:160]}"
                    for i, it in enumerate(items, 1)
                )
                header += "_(index view: one line per lesson; narrow with --query for full text)_\n\n"
            else:
                body = "\n\n".join(
                    f"### lesson [{i}]\n{it.get('content', '')}"
                    for i, it in enumerate(items, 1)
                )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — fail-visible boundary
        _fail(f"{type(exc).__name__}: {exc}")
    print(_elide_head_tail(header + body, args.max_chars))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, add_help=True)
    sub = ap.add_subparsers(dest="command", required=True)

    mg = sub.add_parser("migrate", help="import a local lesson library into Engram")
    mg.add_argument("library", help="path to the local library markdown file")
    mg.add_argument("--skill", required=True, help="skill name (room/subject_id)")

    rt = sub.add_parser("retrieve", help="retrieve a skill's lessons from Engram")
    rt.add_argument("--room", required=True, help="skill namespace (room)")
    rt.add_argument("--query", default=None)
    rt.add_argument(
        "--mode", default="keyword", choices=["keyword", "semantic", "hybrid"]
    )
    rt.add_argument("--limit", type=int, default=5)
    rt.add_argument("--max-chars", type=int, default=12000)
    rt.add_argument(
        "--index",
        action="store_true",
        help="one line per lesson instead of full content",
    )

    args = ap.parse_args(argv)
    if args.command == "migrate":
        return cmd_migrate(args)
    return cmd_retrieve(args)


if __name__ == "__main__":
    raise SystemExit(main())
