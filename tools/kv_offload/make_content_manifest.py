#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write the content manifest of a checkpoint for the revolta_fs KV tier.

Run it once after preparing (or copying) a checkpoint, with the same paths the
engine will load, for example inside the serving container:

    python tools/kv_offload/make_content_manifest.py \
        --role target=/model --role draft=/draft \
        --out /model-manifests/revolta-content-manifest.json

Roles: ``target`` is the --model path and ``draft`` the speculative model
path; any other role (e.g. KV scale overlays) is verified file by file. Then
set ``"content_manifest_path": "<out>"`` in the revolta_fs tier config. If a listed
file is rewritten later, the tier disables itself until a new manifest is
made. Keep the manifest outside the model directories: files inside a role
root are part of that role.
"""

import argparse
import json
import os
import sys

from vllm.v1.kv_offload.tiering.fs.content_manifest import build_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--role",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="role name and the path the engine loads (repeatable)",
    )
    parser.add_argument("--out", required=True, help="manifest file to write")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    roles = {}
    for item in args.role:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--role needs NAME=PATH, got {item!r}")
        if name in roles:
            parser.error(f"duplicate role {name!r}")
        roles[name] = path
    out = os.path.abspath(args.out)
    for name, path in roles.items():
        root = os.path.realpath(path)
        if out.startswith(root + os.sep):
            parser.error(f"--out must not be inside the {name} root {path}")
    manifest = build_manifest(roles, workers=args.workers)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    files = sum(len(r["files"]) for r in manifest["roles"].values())
    print(f"wrote {out}: {files} files, digest {manifest['digest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
