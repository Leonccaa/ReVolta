# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static environment metadata and ReVolta default-policy inventory.

Never imports vLLM, evaluates a getter, or initializes CUDA. This checkout
retains env metadata checks without the upstream ownership inventory.
"""

import argparse
import ast
import json
from pathlib import Path

from tools.pre_commit.check_env_metadata import read_metadata

ROOT = Path(__file__).resolve().parents[1]
POLICY = "vllm/revolta_profile.py:apply_revolta_profile"


def collect(root: Path) -> tuple[dict, list[str]]:
    metadata, errors = read_metadata((root / "vllm/envs.py").read_text())
    profile = root / "vllm/revolta_profile.py"
    tables = {}
    kernel_defaults = {}
    for node in ast.parse(profile.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in (
                "COMMON_DEFAULTS",
                "AWQ_DEFAULTS",
                "NVFP4_DEFAULTS",
                "CUSTOM_AR_DEFAULTS",
            )
        ):
            tables[node.targets[0].id] = ast.literal_eval(node.value)
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "KERNEL_DEFAULTS"
        ):
            kernel_defaults = ast.literal_eval(node.value)
    for group, defaults in tables.items():
        for name, value in defaults.items():
            row = metadata.get(name)
            if row is None:
                errors.append(f"{name}: profile default is unregistered")
                continue
            conditions = " ".join(row["automatic_conditions"])
            if POLICY not in conditions or repr(value) not in conditions:
                errors.append(f"{name}: missing profile automatic_conditions")
            if group == "AWQ_DEFAULTS" and "Only AWQ checkpoints" not in conditions:
                errors.append(f"{name}: missing AWQ condition")
            if (
                group == "NVFP4_DEFAULTS"
                and "Only ModelOpt NVFP4 checkpoints" not in conditions
            ):
                errors.append(f"{name}: missing NVFP4 condition")
            if group == "CUSTOM_AR_DEFAULTS" and "no_p2p" not in conditions:
                errors.append(f"{name}: missing interconnect condition")
    if len(tables) != 4:
        errors.append("Missing ReVolta default tables")
    kernel_tree = ast.parse((root / "vllm/config/kernel.py").read_text())
    kernel_fields = {
        node.target.id
        for cls in kernel_tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "KernelConfig"
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    if kernel_defaults != {"ple_disk_cascade": False}:
        errors.append("Missing ReVolta cascade default")
    for name in kernel_defaults:
        if name not in kernel_fields or f"{name}_source" not in kernel_fields:
            errors.append(f"{name}: missing KernelConfig field or provenance")
    return {
        "registrations": len(metadata),
        "profile_defaults": tables,
        "kernel_profile_defaults": kernel_defaults,
    }, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    inventory, errors = collect(args.root)
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "Environment metadata and ReVolta defaults are complete."
        if args.check
        else json.dumps(inventory, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
