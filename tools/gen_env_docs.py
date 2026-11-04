# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the environment variable reference in docs/configuration.md.

Type, default, category and description come from vllm/envs.py (static
parse; never imports vllm). Release, range, trade-off and metrics come from
tools/env_docs_notes.py. The ReVolta profile value comes from
vllm/revolta_profile.py. The speculative_config.sm70_dflash2 table reads the
Sm70DFlash2Config fields.

  python tools/gen_env_docs.py           rewrite the generated block
  python tools/gen_env_docs.py --check   fail if the block is out of date
"""

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.env_docs_notes import (  # noqa: E402
    DFLASH2_NOTES,
    NOTES,
    PROFILE_NOTE,
    SECTIONS,
)
from tools.pre_commit.check_env_metadata import read_metadata  # noqa: E402

DOC = ROOT / "docs/configuration.md"
BEGIN = "<!-- BEGIN GENERATED: tools/gen_env_docs.py -->"
END = "<!-- END GENERATED: tools/gen_env_docs.py -->"


def annotations(envs_source: str) -> dict[str, str]:
    """NAME -> type from the TYPE_CHECKING declarations."""
    out: dict[str, str] = {}
    for node in ast.walk(ast.parse(envs_source)):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    out[stmt.target.id] = ast.unparse(stmt.annotation)
    return out


def profile_values(source: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    labels = {
        "COMMON_DEFAULTS": "",
        "AWQ_DEFAULTS": "AWQ",
        "AWQ_DFLASH_DEFAULTS": "AWQ+DFlash2",
        "NVFP4_DEFAULTS": "NVFP4",
        "CUSTOM_AR_DEFAULTS": "P2P/NVLink",
    }
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in labels
        ):
            label = labels[node.targets[0].id]
            for name, value in ast.literal_eval(node.value).items():
                values.setdefault(name, []).append(
                    f"`{value}` ({label})" if label else f"`{value}`"
                )
    return values


def dflash2_fields(source: str) -> dict[str, tuple[str, str]]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "Sm70DFlash2Config":
            return {
                stmt.target.id: (
                    ast.unparse(stmt.annotation),
                    ast.unparse(stmt.value) if stmt.value is not None else "",
                )
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    raise RuntimeError("Sm70DFlash2Config not found")


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def render() -> str:
    envs_source = (ROOT / "vllm/envs.py").read_text(encoding="utf-8")
    metadata, errors = read_metadata(envs_source)
    if errors:
        raise SystemExit("envs.py metadata errors:\n" + "\n".join(errors))
    types = annotations(envs_source)
    profile = profile_values((ROOT / "vllm/revolta_profile.py").read_text())
    missing = sorted(set(NOTES) - set(metadata))
    if missing:
        raise SystemExit(f"notes for unregistered variables: {missing}")
    lines = [
        BEGIN,
        "<!-- Edit tools/env_docs_notes.py, then run python tools/gen_env_docs.py -->",
        "",
        "## Environment variable reference",
        "",
        "Every variable below is read once per process (deployment scope). Columns:",
        "type and default as registered in `vllm/envs.py`; *Since* is the first",
        "release with the variable (1Cat = inherited from 1Cat-vLLM); *Category*",
        "is configuration, tuning, experimental, debug or deprecated.",
    ]
    for key, title in SECTIONS.items():
        names = [n for n, note in NOTES.items() if note["section"] == key]
        if not names:
            continue
        lines += ["", f"### {title}", ""]
        columns = ["Name", "Type", "Default", "Range", "Category", "Since"]
        if key == "profile":
            lines += [PROFILE_NOTE, ""]
            columns.insert(3, "Profile")
        columns.append("Effect and trade-off")
        lines += [
            "| " + " | ".join(columns) + " |",
            "|" + "---|" * len(columns),
        ]
        for name in sorted(names):
            meta, note = metadata[name], NOTES[name]
            effect = note["tradeoff"]
            if note.get("metrics"):
                effect += f" Metrics: {note['metrics']}."
            row = [
                f"`{name}`",
                cell(types.get(name, "str")),
                f"`{cell(meta['declared_default'])}`",
            ]
            if key == "profile":
                row.append(cell(", ".join(profile.get(name, ["-"]))))
            row += [
                cell(note["range"]),
                meta["category"],
                note["since"],
                cell(effect),
            ]
            lines.append("| " + " | ".join(row) + " |")
    fields = dflash2_fields((ROOT / "vllm/config/sm70_dflash2.py").read_text())
    lines += [
        "",
        "### speculative_config.sm70_dflash2 keys",
        "",
        "Pass these inside `--speculative-config '{..., \"sm70_dflash2\": {...}}'`",
        "(see [DFlash2](dflash2.md)). `None` selects the model-qualified policy;",
        "operator shape, dtype and native checks still decide each dispatch.",
        "",
        "| Key | Type | Default | Effect and trade-off |",
        "|---|---|---|---|",
    ]
    for key in sorted(DFLASH2_NOTES):
        if key not in fields:
            raise SystemExit(f"sm70_dflash2 note for unknown field {key}")
        annotation, default = fields[key]
        lines.append(
            f"| `{key}` | {cell(annotation)} | `{cell(default)}` | "
            f"{cell(DFLASH2_NOTES[key])} |"
        )
    lines += ["", END]
    return "\n".join(lines)


def updated(text: str, block: str) -> str:
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        return head + block + tail
    anchor = "\n## See also"
    if anchor not in text:
        raise SystemExit("docs/configuration.md: '## See also' anchor not found")
    head, tail = text.split(anchor, 1)
    return head.rstrip("\n") + "\n\n" + block + "\n" + anchor + tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = DOC.read_text(encoding="utf-8")
    new = updated(text, render())
    if args.check:
        if new != text:
            print(
                "docs/configuration.md: generated reference is out of date; "
                "run python tools/gen_env_docs.py"
            )
            return 1
        return 0
    DOC.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
