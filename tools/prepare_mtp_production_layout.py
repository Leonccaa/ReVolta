# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a read-only-source random MTP fixture with production shard naming.

Run on CPU: python tools/prepare_mtp_production_layout.py SOURCE NEW_DIRECTORY
The source must remain available: target shards and metadata are symlinked.
Mount both directories at the same absolute paths when testing in a container.
Only MTP tensors are read and rewritten; target tensors are index-filtered
views of the original shards. Missing KV scales are synthesized as unit FP32
scales for random-fixture functionality only, never calibration or quality.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BODY_FILE = "model-00001-of-00001.safetensors"
SCALE_FILE = "model-bf16-kvscales-mtp.safetensors"
INDEX_FILE = "model.safetensors.index.json"


def is_mtp(name: str) -> bool:
    return name.startswith("mtp.") or ".mtp." in name


def build(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("Source must be a checkpoint directory with config.json")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(source):
        raise ValueError("Destination must be outside the read-only source")

    index_path = source / INDEX_FILE
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        weight_map = {}
        for path in sorted(source.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():  # noqa: SIM118
                    if name in weight_map:
                        raise ValueError(f"Duplicate source tensor: {name}")
                    weight_map[name] = path.name
    if not weight_map:
        raise ValueError("No source safetensors weights")
    for filename in set(weight_map.values()):
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Expected a checkpoint-relative shard path: {filename}")

    tensors = {}
    for filename in sorted(set(weight_map.values())):
        names = [n for n, fn in weight_map.items() if fn == filename and is_mtp(n)]
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
        if names:
            with safe_open(source / filename, framework="pt", device="cpu") as f:
                for name in names:
                    tensors[name] = f.get_tensor(name).clone()
    if not tensors:
        raise ValueError("Source has no MTP weights")

    config = json.loads((source / "config.json").read_text())
    text_config = config.get("text_config", config)
    num_experts = text_config["num_experts"]
    # Preserve layer/prefix names and every value; fuse only the routed experts.
    prefixes = {
        name.split(".experts.")[0] + ".experts."
        for name in tensors
        if ".experts." in name
    }
    if not prefixes:
        raise ValueError("Source has no routed MTP experts")
    for prefix in sorted(prefixes):
        fused_names = [prefix + "gate_up_proj", prefix + "down_proj"]
        if any(name in tensors for name in fused_names) and not all(
            name in tensors for name in fused_names
        ):
            raise ValueError(f"Incomplete fused routed experts: {prefix}")
        if not all(name in tensors for name in fused_names):
            projections = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                names = [f"{prefix}{e}.{projection}.weight" for e in range(num_experts)]
                projections[projection] = torch.stack([tensors.pop(n) for n in names])
            tensors[fused_names[0]] = torch.cat(
                (projections["gate_proj"], projections["up_proj"]), dim=1
            )
            tensors[fused_names[1]] = projections["down_proj"]
        gate_up, down = (tensors[name] for name in fused_names)
        if (
            gate_up.dtype != torch.bfloat16
            or down.dtype != torch.bfloat16
            or gate_up.ndim != 3
            or down.ndim != 3
            or gate_up.shape != (num_experts, 2 * down.shape[2], down.shape[1])
        ):
            raise ValueError(f"Expected BF16 [E,2I,H]/[E,H,I] experts: {prefix}")
        if any(name.startswith(prefix) and name not in fused_names for name in tensors):
            raise ValueError(f"Mixed or quantized routed experts: {prefix}")

    scales = {}
    synthesized = []
    for prefix in sorted(prefixes):
        attention = prefix.removesuffix("mlp.experts.") + "self_attn."
        for kind in ("k", "v"):
            # Accept the canonical MTP QSA scale names without guessing aliases.
            name = f"{attention}{kind}_scale"
            if name not in tensors:
                aliases = [
                    n
                    for n in tensors
                    if n.startswith(attention)
                    and (
                        n.endswith(f"{kind}_scale")
                        or n == f"{attention}{kind}_proj.output_scale"
                    )
                ]
                if aliases:
                    raise ValueError(
                        f"Normalize source KV scale aliases first: {aliases}"
                    )
                tensors[name] = torch.tensor(1.0, dtype=torch.float32)
                synthesized.append(name)
            scales[name] = tensors.pop(name)

    destination.mkdir(parents=True)
    output_map = {}
    for filename in sorted(set(weight_map.values())):
        names = [n for n, fn in weight_map.items() if fn == filename and not is_mtp(n)]
        if names:
            relative = Path("source-shards") / filename
            link = destination / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((source / filename).resolve())
            output_map.update(dict.fromkeys(names, str(relative)))
    for path in source.iterdir():
        if path.name == INDEX_FILE or path.suffix == ".safetensors":
            continue
        if path.name in {"source-shards", "layout-manifest.json"}:
            continue
        (destination / path.name).symlink_to(path.resolve())
    for filename, data in ((BODY_FILE, tensors), (SCALE_FILE, scales)):
        save_file(
            {n: t.contiguous() for n, t in data.items()}, str(destination / filename)
        )
        output_map.update(dict.fromkeys(data, filename))
    (destination / INDEX_FILE).write_text(
        json.dumps({"weight_map": output_map}, indent=2) + "\n"
    )
    manifest = {
        "source": str(source),
        "source_config_sha256": hashlib.sha256(
            (source / "config.json").read_bytes()
        ).hexdigest(),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()
        if index_path.exists()
        else None,
        "mtp_body_file": BODY_FILE,
        "mtp_body_tensor_count": len(tensors),
        "scale_file": SCALE_FILE,
        "scale_tensor_names": sorted(scales),
        "synthetic_unit_scales": synthesized,
        "experts": {
            n: {"shape": list(t.shape), "dtype": str(t.dtype)}
            for n, t in tensors.items()
            if ".experts." in n
        },
        "scope": "Random fixture functionality only; no acceptance/quality evidence",
    }
    (destination / "layout-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)
