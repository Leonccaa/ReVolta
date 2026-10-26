#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize a revision-checked QSA E4M3 KV-scale checkpoint view.

Target: nvidia/Qwen3.8-Flash-Next-FP4-Local. Its weight shards live under
``tensors/`` and are found only through ``model.safetensors.index.json``. A
vLLM-style loader first globs ``*.safetensors`` in the checkpoint root and
reads the index only when that glob is empty, so a scale file in the root
would hide every weight shard. The scale file is therefore placed in the
``calibration/`` subdirectory of the view and registered in the merged index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any

INDEX_FILENAME = "model.safetensors.index.json"
CONFIG_FILENAME = "config.json"
MANIFEST_FILENAME = "kvscales-manifest.json"
PROVENANCE_FILENAME = "kvscales-provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def inspect_scale_file(path: Path) -> tuple[dict[str, int], dict[str, str], int]:
    """Return {tensor name: FP32 bit pattern}, metadata and payload byte count."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size < 2 or header_size > file_size - 8:
            raise ValueError(f"Invalid safetensors header size: {header_size}")
        header = json.loads(handle.read(header_size))
        if not isinstance(header, dict):
            raise TypeError("Safetensors header is not a JSON object")

        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict):
            raise TypeError("Invalid safetensors metadata")
        data_start = 8 + header_size
        values: dict[str, int] = {}
        payload_bytes = 0
        intervals = []

        for name, details in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(details, dict):
                raise TypeError(f"Invalid tensor header for {name}")
            if details.get("dtype") != "F32" or details.get("shape") not in ([], [1]):
                raise ValueError(f"Expected an FP32 scalar for {name}")
            offsets = details.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(offset, int) for offset in offsets)
                or offsets[1] - offsets[0] != 4
            ):
                raise ValueError(f"Invalid scalar data offsets for {name}")
            if offsets[0] < 0 or offsets[1] > file_size - data_start:
                raise ValueError(f"Out-of-bounds tensor data for {name}")
            intervals.append(tuple(offsets))
            handle.seek(data_start + offsets[0])
            raw_value = handle.read(4)
            if len(raw_value) != 4:
                raise ValueError(f"Truncated tensor data for {name}")
            value = struct.unpack("<f", raw_value)[0]
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Invalid calibrated scale for {name}: {value}")
            values[name] = struct.unpack("<I", raw_value)[0]
            payload_bytes += offsets[1] - offsets[0]

        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("Scale payload has overlapping or unused bytes")
            cursor = end
        if cursor != file_size - data_start:
            raise ValueError("Scale payload has trailing bytes")

    return values, {str(k): str(v) for k, v in metadata.items()}, payload_bytes


DRAFT_SCALE_FILENAME = "model-kvscales-mtp.safetensors"
DRAFT_MANIFEST_FILENAME = "draft-kvscales-manifest.json"
DRAFT_NAMES = {f"mtp.layers.0.self_attn.{kind}_scale" for kind in ("k", "v")}


def validate_draft(pack_dir: Path, manifest: dict[str, Any]):
    """An absent optional file preserves the original 24-scale view exactly."""
    path = pack_dir / DRAFT_SCALE_FILENAME
    if not path.exists():
        return None
    draft = load_json(pack_dir / DRAFT_MANIFEST_FILENAME)
    if draft.get("schema_version") != 1:
        raise ValueError("Unsupported draft manifest version")
    if draft.get("base_checkpoint") != manifest["base_checkpoint"]:
        raise ValueError("Draft scale base revision/index/config binding mismatch")
    if draft.get("scale_contract") != manifest["scale_contract"]:
        raise ValueError("Draft scale converter contract mismatch")
    details = draft["scale_file"]
    if details["filename"] != DRAFT_SCALE_FILENAME:
        raise ValueError("Unexpected draft scale filename")
    if path.stat().st_size != details["size_bytes"]:
        raise ValueError("Draft scale file size mismatch")
    if sha256(path) != details["sha256"]:
        raise ValueError("Draft scale file SHA-256 mismatch")
    values, metadata, payload = inspect_scale_file(path)
    if (
        set(values) != DRAFT_NAMES
        or set(details["tensor_names"]) != DRAFT_NAMES
        or len(details["tensor_names"]) != 2
        or details["tensor_count"] != 2
    ):
        raise ValueError("Draft scale tensor coverage mismatch")
    entries = {entry["name"]: entry for entry in draft["tensors"]}
    if set(entries) != DRAFT_NAMES or len(draft["tensors"]) != 2:
        raise ValueError("Draft manifest tensor coverage mismatch")
    for name, entry in entries.items():
        if values[name] != int(entry["f32_hex"], 16):
            raise ValueError(f"Draft scale value does not match manifest: {name}")
    if metadata.get("negative_control") != "false":
        raise ValueError("Refusing a draft negative-control file")
    if metadata.get("qsa_kv_scale_contract") != manifest["scale_contract"]:
        raise ValueError("Draft scale metadata converter contract mismatch")
    view = draft["derived_view"]
    if view["scale_path"] != "calibration/" + DRAFT_SCALE_FILENAME:
        raise ValueError("Invalid draft scale view path")
    return draft, path, payload


def materialize(args: argparse.Namespace) -> None:
    pack_dir = Path(args.pack_dir).resolve()
    base = Path(args.base_checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    manifest_path = pack_dir / MANIFEST_FILENAME

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if base == output or base in output.parents:
        raise ValueError("Output must not be inside the base checkpoint directory")

    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported scale-pack manifest version")
    base_details = manifest["base_checkpoint"]
    view_details = manifest["derived_view"]

    # 1. The base checkpoint must be the exact revision the scales belong to.
    base_index_path = base / INDEX_FILENAME
    if not base_index_path.is_file():
        raise FileNotFoundError(f"Base checkpoint lacks {INDEX_FILENAME}: {base}")
    actual_base_hash = sha256(base_index_path)
    if actual_base_hash != base_details["index_sha256"]:
        raise ValueError(
            "Base checkpoint revision mismatch: model index SHA-256 is "
            f"{actual_base_hash}, expected {base_details['index_sha256']}"
        )
    base_config_path = base / CONFIG_FILENAME
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base checkpoint lacks {CONFIG_FILENAME}: {base}")
    actual_config_hash = sha256(base_config_path)
    if actual_config_hash != base_details["config_sha256"]:
        raise ValueError(
            "Base checkpoint config mismatch: config.json SHA-256 is "
            f"{actual_config_hash}, expected {base_details['config_sha256']}"
        )

    # 2. The scale file must be exactly the published one.
    scale_details = manifest["scale_file"]
    scale_path = pack_dir / scale_details["filename"]
    if not scale_path.is_file():
        raise FileNotFoundError(f"Scale file is missing: {scale_path}")
    if scale_path.stat().st_size != scale_details["size_bytes"]:
        raise ValueError("Scale file size does not match the manifest")
    actual_scale_hash = sha256(scale_path)
    if actual_scale_hash != scale_details["sha256"]:
        raise ValueError("Scale file SHA-256 does not match the manifest")

    values, scale_metadata, payload_bytes = inspect_scale_file(scale_path)
    expected_names = set(scale_details["tensor_names"])
    if set(values) != expected_names:
        missing = sorted(expected_names - set(values))
        extra = sorted(set(values) - expected_names)
        raise ValueError(
            f"Scale tensor coverage mismatch; missing={missing}, extra={extra}"
        )
    if len(values) != scale_details["tensor_count"]:
        raise ValueError("Scale tensor count does not match the manifest")
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    if set(entries) != expected_names or len(entries) != len(manifest["tensors"]):
        raise ValueError("Manifest tensor entries do not match the tensor names")
    for name, entry in entries.items():
        if values[name] != int(entry["f32_hex"], 16):
            raise ValueError(f"Scale value does not match the manifest: {name}")
    if scale_metadata.get("negative_control") != "false":
        raise ValueError("Refusing a negative-control scale file")
    if scale_metadata.get("qsa_kv_scale_contract") != manifest["scale_contract"]:
        raise ValueError("Scale converter contract does not match the manifest")

    draft_info = validate_draft(pack_dir, manifest)
    all_names = expected_names | (DRAFT_NAMES if draft_info else set())
    selected_view = draft_info[0]["derived_view"] if draft_info else view_details

    # 3. The scales must be new entries in the base index.
    base_index = load_json(base_index_path)
    weight_map = base_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("Base checkpoint index lacks a weight_map object")
    collisions = sorted(all_names & set(weight_map))
    if collisions:
        raise ValueError(
            f"Base checkpoint already contains scale tensors: {collisions}"
        )

    view_scale_path = Path(view_details["scale_path"])
    if view_scale_path.is_absolute() or len(view_scale_path.parts) != 2:
        raise ValueError("The view scale path must be one subdirectory deep")
    view_scale_dir = view_scale_path.parts[0]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")

    excluded = {
        ".cache",
        ".git",
        INDEX_FILENAME,
        MANIFEST_FILENAME,
        PROVENANCE_FILENAME,
        scale_details["filename"],
        view_scale_dir,
    }
    try:
        staging.mkdir()
        for source in base.iterdir():
            if source.name in excluded:
                continue
            relative_target = os.path.relpath(source, start=staging)
            (staging / source.name).symlink_to(
                relative_target, target_is_directory=source.is_dir()
            )

        # A real copy, not a symlink: the runtime may require the scale shard
        # to resolve inside the checkpoint directory.
        (staging / view_scale_dir).mkdir()
        copied_scale = staging / view_scale_path
        shutil.copy2(scale_path, copied_scale)
        if sha256(copied_scale) != actual_scale_hash:
            raise ValueError("Copied scale file failed its integrity check")

        if draft_info:
            draft, draft_path, draft_payload = draft_info
            copied_draft = staging / draft["derived_view"]["scale_path"]
            shutil.copy2(draft_path, copied_draft)
            if sha256(copied_draft) != draft["scale_file"]["sha256"]:
                raise ValueError("Copied draft scale file failed integrity check")
            payload_bytes += draft_payload

        # Keep the base order and append the scale entries in sorted order.
        merged_index = dict(base_index)
        merged_weight_map = dict(weight_map)
        for name in sorted(expected_names):
            merged_weight_map[name] = view_scale_path.as_posix()
        if draft_info:
            for name in sorted(DRAFT_NAMES):
                merged_weight_map[name] = draft_info[0]["derived_view"]["scale_path"]
        merged_index["weight_map"] = merged_weight_map
        metadata = dict(base_index.get("metadata", {}))
        if isinstance(metadata.get("total_size"), int):
            metadata["total_size"] += payload_bytes
        merged_index["metadata"] = metadata
        index_path = staging / INDEX_FILENAME
        index_path.write_text(
            json.dumps(merged_index, indent=2) + "\n", encoding="utf-8"
        )

        # The derived index is deterministic; it must equal the validated one.
        actual_view_hash = sha256(index_path)
        if actual_view_hash != selected_view["index_sha256"]:
            raise ValueError(
                "Derived index SHA-256 is "
                f"{actual_view_hash}, expected {selected_view['index_sha256']}"
            )
        if list(staging.glob("*.safetensors")):
            raise ValueError("A top-level safetensors file would hide the shards")
        indexed_files = sorted(set(merged_weight_map.values()))
        if len(indexed_files) != selected_view["indexed_file_count"]:
            raise ValueError(
                f"Expected {selected_view['indexed_file_count']} indexed files, "
                f"found {len(indexed_files)}"
            )
        broken = [name for name in indexed_files if not (staging / name).is_file()]
        if broken:
            raise FileNotFoundError(f"Indexed files are missing: {broken[:4]}")

        provenance = {
            "schema_version": 1,
            "artifact_id": manifest["artifact_id"],
            "release_tag": manifest["release_tag"],
            "base_model": base_details["repo_id"],
            "base_revision": base_details["revision"],
            "base_index_sha256": actual_base_hash,
            "base_config_sha256": actual_config_hash,
            "derived_index_sha256": actual_view_hash,
            "scale_file": view_scale_path.as_posix(),
            "scale_file_sha256": actual_scale_hash,
            "tensor_count": len(expected_names),
            "loader_contract": manifest["loader_contract"],
        }
        if draft_info:
            provenance.update(
                release_tag=draft_info[0]["release_tag"],
                tensor_count=len(all_names),
                draft_scale_file=draft_info[0]["derived_view"]["scale_path"],
                draft_scale_file_sha256=draft_info[0]["scale_file"]["sha256"],
                draft_manifest_sha256=sha256(pack_dir / DRAFT_MANIFEST_FILENAME),
            )
        (staging / PROVENANCE_FILENAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"Materialized overlay: {output}")
    print(f"Base index SHA-256: {actual_base_hash}")
    print(f"Derived index SHA-256: {actual_view_hash}")
    print(f"Scale SHA-256: {actual_scale_hash}")
    print(f"Indexed files: {len(indexed_files)} (scale file: {view_scale_path})")
    print(f"Loaded scale entries: {len(all_names)}/{len(all_names)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pack-dir",
        default=str(Path(__file__).resolve().parent),
        help="Downloaded scale-pack directory (defaults to this script's directory)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    materialize(parse_args())
