# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the commits and the added files of a pull request.

Run by .github/workflows/checks.yml with BASE and HEAD (commit shas) in the
environment.

- Every commit except merges carries a ``Signed-off-by:`` trailer with its
  author's email (CONTRIBUTING.md: sign off every commit with
  ``git commit -s``).
- No file the PR adds is larger than 1 MiB: model weights, logs and archives
  do not belong in the repository. Paths listed one per line in
  .github/large-files-allowed.txt are exempt.

Exit status 1 if any check fails.
"""

import os
import subprocess
import sys
from pathlib import Path

import regex as re

MAX_ADDED_BYTES = 1 << 20
ALLOWED_FILE = Path(".github/large-files-allowed.txt")
SIGNOFF = re.compile(r"^Signed-off-by:\s*.+<([^<>\s]+)>\s*$", re.M)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, errors="replace"
    ).stdout


def main() -> int:
    base, head = os.environ["BASE"], os.environ["HEAD"]
    merge_base = git("merge-base", base, head).strip()
    errors = 0

    log = git(
        "log", "--no-merges", "--format=%H%x00%ae%x00%B%x1e", f"{merge_base}..{head}"
    )
    for record in filter(None, (r.strip("\n") for r in log.split("\x1e"))):
        sha, author, message = record.split("\x00", 2)
        signed = {email.lower() for email in SIGNOFF.findall(message)}
        if author.lower() not in signed:
            errors += 1
            print(
                f"::error title=Sign-off::commit {sha[:10]} has no 'Signed-off-by:' "
                f"line for its author <{author}>; amend it with 'git commit -s'"
            )

    allowed = set()
    if ALLOWED_FILE.is_file():
        allowed = {
            line.strip()
            for line in ALLOWED_FILE.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
    added = git("diff", "--diff-filter=A", "-z", "--name-only", merge_base, head)
    for path in filter(None, added.split("\0")):
        size = int(git("cat-file", "-s", f"{head}:{path}").strip())
        if size > MAX_ADDED_BYTES and path not in allowed:
            errors += 1
            print(
                f"::error file={path},title=Large file::{path} adds {size:,} bytes "
                f"(limit {MAX_ADDED_BYTES:,}); keep weights, logs and archives out of "
                f"the repository, or list the path in {ALLOWED_FILE}"
            )

    print(f"commit check: {errors} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
