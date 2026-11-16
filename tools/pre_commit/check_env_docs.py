# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Require ReVolta environment variables to be documented and well named.

Every variable registered in vllm/envs.py under a ReVolta prefix must appear in
one of the reference documents (docs/configuration.md, docs/kv-offload.md).
Newly registered names must follow the naming rule:

  VLLM_SM70_<AREA>_<NAME>        kernel and path switches on SM70
  REVOLTA_RESEARCH_<AREA>_<NAME> research switches (default off)

Variables that predate this check are listed in env_docs_baseline.txt with the
reason they are exempt (``undocumented`` or ``legacy-name``). The baseline may
only shrink: an entry that is now documented, renamed or removed must be
deleted from it, so the file always lists exactly the remaining debt.

Static only: never imports vllm or evaluates getters.
"""

import argparse
import ast
import sys
from pathlib import Path

import regex as re

PREFIXES = ("VLLM_SM70_", "VLLM_DFLASH", "VLLM_QWEN4EXP_", "VLLM_REVOLTA_", "REVOLTA_")
DOCS = ("docs/configuration.md", "docs/kv-offload.md")
BASELINE = Path(__file__).with_name("env_docs_baseline.txt")
NAME_RULES = (
    re.compile(r"^VLLM_SM70_[A-Z0-9]+(?:_[A-Z0-9]+)+$"),
    re.compile(r"^REVOLTA_RESEARCH_[A-Z0-9]+(?:_[A-Z0-9]+)+$"),
)
REASONS = ("undocumented", "legacy-name")


def registered(envs_source: str) -> set[str]:
    tree = ast.parse(envs_source)
    for node in tree.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else None
        )
        if (
            isinstance(target, ast.Name)
            and target.id == "environment_variables"
            and isinstance(node.value, ast.Dict)
        ):
            return {
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    raise RuntimeError("environment_variables not found in vllm/envs.py")


def read_baseline(path: Path) -> dict[tuple[str, str], int]:
    entries: dict[tuple[str, str], int] = {}
    if not path.exists():
        return entries
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, reason = line.partition(" ")
        entries[(name, reason.strip())] = lineno
    return entries


def documented_names(root: Path, names: set[str]) -> set[str]:
    text = "\n".join(
        (root / doc).read_text(encoding="utf-8")
        for doc in DOCS
        if (root / doc).exists()
    )
    found = set(re.findall(r"(?<![A-Za-z0-9_])((?:VLLM|REVOLTA)_[A-Z0-9_]+)", text))
    return names & found


def check(root: Path, baseline_path: Path = BASELINE) -> list[str]:
    names = {
        n
        for n in registered((root / "vllm/envs.py").read_text(encoding="utf-8"))
        if n.startswith(PREFIXES)
    }
    documented = documented_names(root, names)
    baseline = read_baseline(baseline_path)
    errors = []
    for (name, reason), lineno in sorted(baseline.items(), key=lambda kv: kv[1]):
        where = f"{baseline_path.name}:{lineno}"
        if reason not in REASONS:
            errors.append(f"{where}: unknown reason {reason!r} for {name}")
        elif name not in names:
            errors.append(f"{where}: {name} is no longer registered; remove it")
        elif reason == "undocumented" and name in documented:
            errors.append(f"{where}: {name} is documented now; remove it")
        elif reason == "legacy-name" and any(r.match(name) for r in NAME_RULES):
            errors.append(f"{where}: {name} follows the naming rule; remove it")
    for name in sorted(names):
        if name not in documented and (name, "undocumented") not in baseline:
            errors.append(
                f"{name}: registered but not documented in {' or '.join(DOCS)}"
            )
        if (
            not any(r.match(name) for r in NAME_RULES)
            and (
                name,
                "legacy-name",
            )
            not in baseline
        ):
            errors.append(
                f"{name}: name must be VLLM_SM70_<AREA>_<NAME> or "
                "REVOLTA_RESEARCH_<AREA>_<NAME>"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline to the current debt (for the initial import)",
    )
    args = parser.parse_args(argv)
    if args.write_baseline:
        names = {
            n
            for n in registered((args.root / "vllm/envs.py").read_text())
            if n.startswith(PREFIXES)
        }
        documented = documented_names(args.root, names)
        lines = [
            "# ReVolta environment variables exempt from check_env_docs.py.",
            "# Format: NAME REASON (undocumented | legacy-name). Only delete lines.",
        ]
        for name in sorted(names):
            if name not in documented:
                lines.append(f"{name} undocumented")
            if not any(r.match(name) for r in NAME_RULES):
                lines.append(f"{name} legacy-name")
        BASELINE.write_text("\n".join(lines) + "\n")
        return 0
    errors = check(args.root)
    for error in errors:
        print(f"error: {error}")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
