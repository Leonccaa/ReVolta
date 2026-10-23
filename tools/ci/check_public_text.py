# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that a pull request adds no internal text.

Run by .github/workflows/checks.yml with BASE and HEAD (commit shas) and
PR_TITLE / PR_BODY in the environment. It scans the PR title and body, the
messages and author/committer emails of the PR's commits, and the lines the
PR adds.

- Generic patterns are public: private IPv4 addresses, ".lan" host names,
  absolute home and mount paths.
- Project-specific names come from the PUBLIC_TEXT_DENYLIST secret (one
  regular expression, case-insensitive). PRs from forks do not get secrets,
  so they skip that part. Its matches are reported by location only, so the
  log does not reveal the pattern.

It also warns about hard-wrapped paragraphs in the PR body: GitHub shows a
single newline there as a line break.

Exit status 1 if any error was found.
"""

import os
import subprocess
import sys

import regex as re

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
GENERIC = {
    "private IPv4 address": re.compile(
        rf"(?<![\d.])(?:10(?:\.{_OCTET}){{3}}|192\.168(?:\.{_OCTET}){{2}}"
        rf"|172\.(?:1[6-9]|2\d|3[01])(?:\.{_OCTET}){{2}})(?!\d|\.\d)"
    ),
    "'.lan' host name": re.compile(
        r"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.lan\b", re.I
    ),
    "home directory path": re.compile(r"(?<![\w.])(?:/home/[a-z_][a-z0-9_-]*/|~/ops/)"),
    "mount path": re.compile(r"(?<![\w.])/mnt/[A-Za-z0-9_]"),
}

errors = 0


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, errors="replace"
    ).stdout


def scan(
    location: str,
    text: str,
    denylist: re.Pattern | None,
    at: str = " line ",
    start: int = 1,
) -> None:
    """Report each matching line of ``text`` as ``location + at + line number``."""
    global errors
    for lineno, line in enumerate(text.splitlines(), start):
        where = f"{location}{at}{lineno}"
        for what, pattern in GENERIC.items():
            m = pattern.search(line)
            if m:
                errors += 1
                print(f"::error title=Public text::{where}: {what} {m.group(0)!r}")
        if denylist is not None and denylist.search(line):
            errors += 1
            print(f"::error title=Public text::{where}: matches the private denylist")


def added_lines(base: str, head: str):
    """Yield (path, line number, text) for every line the PR adds."""
    path, lineno = None, 0
    diff = git("diff", "--unified=0", "--no-color", "--no-ext-diff", base, head)
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and path is not None:
            yield path, lineno, line[1:]
            lineno += 1


def hard_wraps(body: str) -> int:
    """Count places where a long plain-text line continues on the next line."""
    count, fence = 0, False
    lines = body.replace("\r\n", "\n").split("\n")
    plain = re.compile(r"^\s*(?:[-*+>|#<]|\d+[.)]\s|```|~~~|$)")
    for cur, nxt in zip(lines, lines[1:]):
        if cur.strip().startswith(("```", "~~~")):
            fence = not fence
        if fence:
            continue
        if (
            len(cur) >= 60
            and not cur.endswith("  ")
            and not plain.match(cur)
            and nxt.strip()
            and not plain.match(nxt)
        ):
            count += 1
    return count


def main() -> int:
    base, head = os.environ["BASE"], os.environ["HEAD"]
    merge_base = git("merge-base", base, head).strip()
    secret = os.environ.get("DENYLIST", "").strip()
    denylist = re.compile(secret, re.I) if secret else None
    if denylist is None:
        print("::notice title=Public text::no private denylist (fork PR): generic only")

    scan("PR title", os.environ.get("PR_TITLE", ""), denylist)
    body = os.environ.get("PR_BODY", "")
    scan("PR body", body, denylist)
    wraps = hard_wraps(body)
    if wraps:
        print(
            f"::warning title=PR body::{wraps} hard-wrapped line(s); GitHub shows a "
            "single newline as a line break, so write each paragraph and list item "
            "on one line"
        )

    log = git("log", "--format=%H%x00%ae%x00%ce%x00%B%x1e", f"{merge_base}..{head}")
    for record in filter(None, (r.strip("\n") for r in log.split("\x1e"))):
        sha, author, committer, message = record.split("\x00", 3)
        scan(f"commit {sha[:10]} message", message, denylist)
        scan(f"commit {sha[:10]} emails", f"{author}\n{committer}", denylist)

    for path, lineno, text in added_lines(merge_base, head):
        scan(path, text, denylist, at=":", start=lineno)
    print(f"public text check: {errors} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
