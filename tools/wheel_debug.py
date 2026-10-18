# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split host debug info from a wheel without changing runtime ELF sections.

Usage: python tools/wheel_debug.py strip dist/revolta-*.whl --output-dir debug
       python tools/wheel_debug.py verify original.so stripped.so
Requires GNU binutils and wheel. Never uses strip --strip-all.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

SECTIONS = (".text", ".nv_fatbin", ".rodata", ".dynsym", ".dynstr")


def section_hashes(path: Path) -> dict[str, str | None]:
    listing = subprocess.check_output(["readelf", "-SW", str(path)], text=True)
    hashes = {}
    with tempfile.TemporaryDirectory() as temp:
        for section in SECTIONS:
            # A host-only library need not have a CUDA fatbin or rodata.
            if section not in listing.split():
                hashes[section] = None
                continue
            target = Path(temp) / section
            subprocess.run(
                ["objcopy", "--dump-section", f"{section}={target}", str(path)],
                check=True,
                capture_output=True,
            )
            hashes[section] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def verify(before: Path, after: Path) -> dict[str, str | None]:
    original = section_hashes(before)
    stripped = section_hashes(after)
    if original != stripped:
        raise ValueError(f"Runtime sections changed: {before} -> {after}")
    if original[".text"] is None or original[".dynsym"] is None:
        raise ValueError(f"Expected a shared ELF library: {before}")
    listing = subprocess.check_output(["readelf", "-SW", str(after)], text=True)
    original_listing = subprocess.check_output(
        ["readelf", "-SW", str(before)], text=True
    )
    if ".symtab" in original_listing.split() and ".symtab" not in listing.split():
        raise ValueError(f"Static symbol table was removed: {after}")
    return original


def strip_wheel(wheel: Path, output_dir: Path) -> dict:
    from wheel.wheelfile import WheelFile

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{wheel.stem}.debug.tar.xz"
    manifest_path = output_dir / f"{wheel.stem}.sections.json"
    if archive.exists() or manifest_path.exists():
        raise FileExistsError(
            "Debug artifacts already exist; use a new output directory"
        )
    report = {
        "wheel": wheel.name,
        "original_bytes": wheel.stat().st_size,
        "libraries": {},
    }
    with tempfile.TemporaryDirectory(prefix="revolta-debug-") as temp:
        root = Path(temp)
        unpacked, symbols = root / "wheel", root / "symbols"
        with WheelFile(wheel) as source:
            source.extractall(unpacked)
            # zipfile extraction does not restore executable bits (vllm-rs).
            for member in source.infolist():
                mode = (member.external_attr >> 16) & 0o777
                if mode:
                    (unpacked / member.filename).chmod(mode)
        for library in sorted(unpacked.rglob("*.so")):
            with library.open("rb") as stream:
                header = stream.read(4)
            if header != b"\x7fELF":
                raise ValueError(f"Not an ELF library: {library}")
            relative = library.relative_to(unpacked)
            debug = symbols / (str(relative) + ".debug")
            debug.parent.mkdir(parents=True, exist_ok=True)
            before = root / "original.so"
            shutil.copy2(library, before)
            subprocess.run(["objcopy", "--only-keep-debug", library, debug], check=True)
            subprocess.run(["strip", "--strip-debug", library], check=True)
            subprocess.run(
                ["objcopy", f"--add-gnu-debuglink={debug}", library], check=True
            )
            report["libraries"][str(relative)] = verify(before, library)
        if not report["libraries"]:
            raise ValueError("Wheel contains no shared libraries")
        with tarfile.open(archive, "w:xz") as target:
            for debug in sorted(symbols.rglob("*.debug")):
                target.add(debug, arcname=str(debug.relative_to(symbols)))
        # WheelFile regenerates RECORD, including sizes and hashes of changed ELFs.
        staged = root / wheel.name
        with WheelFile(staged, "w") as target:
            target.write_files(unpacked)
        # Keep the destination on the same filesystem for atomic replacement.
        replacement = wheel.with_suffix(".whl.tmp")
        try:
            shutil.copy2(staged, replacement)
            os.replace(replacement, wheel)
        finally:
            replacement.unlink(missing_ok=True)
    report["stripped_bytes"] = wheel.stat().st_size
    report["debug_bytes"] = archive.stat().st_size
    report["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify")
    check.add_argument("before", type=Path)
    check.add_argument("after", type=Path)
    strip = commands.add_parser("strip")
    strip.add_argument("wheel", type=Path)
    strip.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.before, args.after), indent=2))
    else:
        print(json.dumps(strip_wheel(args.wheel, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
