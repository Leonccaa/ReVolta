# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check ModelOpt NVFP4/FP4-Local components before runtime repacking.

Track successful copies, not just fused parameter names. CSF pairs have already
been decoded by the default loader. PLE and FP8 MTP retain their own checks.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_MISSING = object()


def _kinds(module: nn.Module) -> tuple[str, ...]:
    if not {"weight", "w13_weight"}.intersection(module._parameters):
        return ()
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMxFp8LinearMethod,
        ModelOptNvFp4FusedMoE,
        ModelOptNvFp4LinearMethod,
        ModelOptNvFp4W4A16LinearMethod,
    )
    from vllm.model_executor.layers.quantization.modelopt_vision import (
        ModelOptNvFp4VisionFC2Method,
    )

    method = getattr(module, "quant_method", None)
    if isinstance(method, ModelOptMxFp8LinearMethod):
        return ("weight", "weight_scale")
    if isinstance(
        method,
        (
            ModelOptNvFp4LinearMethod,
            ModelOptNvFp4W4A16LinearMethod,
            ModelOptNvFp4FusedMoE,
            ModelOptNvFp4VisionFC2Method,
        ),
    ):
        kinds: tuple[str, ...] = ("weight", "weight_scale", "weight_scale_2")
        # Native W4A16 has no activation scales. Its runtime placeholder is
        # optional; W4A4 serialized checkpoints carry one per projection.
        config = getattr(method, "quant_config", None)
        if getattr(config, "quant_method", None) == "NVFP4":
            kinds += ("input_scale",)
        return kinds
    return ()


class _Coverage:
    def __init__(self, module: nn.Module, prefix: str, kinds: tuple[str, ...]):
        self.module = module
        self.prefix = prefix or "<root>"
        params = dict(module.named_parameters(recurse=False))
        self.moe = "w13_weight" in params
        names = (
            tuple(f"{p}_{k}" for p in ("w13", "w2") for k in kinds)
            if self.moe
            else kinds
        )
        self.params = {id(params[name]): name for name in names}
        self.allowed = set(params) | {"bias"}
        if self.moe:
            # Invert the loader's substring substitution. layer_name can be
            # fully qualified while the mapping starts at "experts.".
            self.allowed = {
                target.replace(param_name, weight_name)
                for param_name, weight_name, _, _ in module.expert_mapping
                for name in params
                for target in (f"{module.layer_name}.{name}",)
                if param_name in target
            }
        self.expected: set[tuple[str, int | str | None, int | None]] = set()
        self.loaded: set[tuple[str, int | str | None, int | None]] = set()
        self.shards: tuple[int | str | None, ...]
        if isinstance(module, QKVParallelLinear):
            self.shards = ("q", "k", "v")
        elif isinstance(module, MergedColumnParallelLinear):
            self.shards = tuple(range(len(module.output_sizes)))
        else:
            self.shards = (None,)
        for name in names:
            if self.moe:
                shards = ("w1", "w3") if name.startswith("w13_") else ("w2",)
                self.expected.update(
                    (name, shard, expert)
                    for shard in shards
                    for expert in range(params[name].shape[0])
                )
            else:
                self.expected.update((name, shard, None) for shard in self.shards)

    def wrap_copy(self, loader):
        @wraps(loader)
        def tracked(param, loaded_weight, *args, **kwargs):
            result = loader(param, loaded_weight, *args, **kwargs)
            name = self.params.get(id(param))
            if name is None or result is False:
                return result
            if self.moe:
                shard = kwargs.get("shard_id", args[1] if len(args) > 1 else None)
                expert = kwargs.get("expert_id", args[2] if len(args) > 2 else None)
                local = self.module._map_global_expert_id_to_local_expert_id(expert)
                if local >= 0:
                    experts = (
                        range(param.shape[0]) if loaded_weight.ndim == 3 else (local,)
                    )
                    self.loaded.update((name, shard, e) for e in experts)
            else:
                shard = kwargs.get(
                    "loaded_shard_id", kwargs.get("shard_id", args[0] if args else None)
                )
                shards = (
                    self.shards
                    if shard is None
                    else shard
                    if isinstance(shard, tuple)
                    else (shard,)
                )
                self.loaded.update((name, s, None) for s in shards)
            return result

        return tracked

    def checked_weights(self, weights):
        # This boundary is after the model's own name mapping and stage/mtp
        # skips. Do not inspect raw global names using a second set of aliases.
        for name, value in weights:
            key = f"{self.module.layer_name}.{name}" if self.moe else name
            if key not in self.allowed:
                raise ValueError(
                    f"NVFP4 checkpoint unexpected tensor: {self.prefix}.{name}"
                )
            yield name, value

    def wrap_load(self, loader):
        def checked(weights):
            return loader(self.checked_weights(weights))

        return checked

    def validate(self):
        missing = self.expected - self.loaded
        if missing:
            raise ValueError(
                f"NVFP4 checkpoint has incomplete coverage for {self.prefix}: "
                f"missing {len(missing)} tensor components (parameter, shard, expert): "
                + ", ".join(sorted(map(str, missing))[:12])
            )


@contextmanager
def nvfp4_checkpoint_coverage(model: nn.Module) -> Iterator[None]:
    """Run only for NVFP4-bearing checkpoints, never AWQ or FP8-only drafts."""
    from vllm.model_executor.models.utils import AutoWeightsLoader

    coverage = []
    restore = []

    def replace(obj, name, value):
        restore.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, value)

    try:
        for prefix, module in model.named_modules():
            kinds = _kinds(module)
            if not kinds:
                continue
            state = _Coverage(module, prefix, kinds)
            coverage.append(state)
            if state.moe:
                replace(module, "weight_loader", state.wrap_copy(module.weight_loader))
            for param in module.parameters(recurse=False):
                if id(param) in state.params:
                    replace(
                        param,
                        "weight_loader",
                        state.wrap_copy(
                            getattr(param, "weight_loader", default_weight_loader)
                        ),
                    )
            loader = getattr(module, "load_weights", None)
            if loader is None:
                # Same AutoWeightsLoader parameter-copy path, with the legacy
                # unexpected-scale exemption removed at this quantized leaf.
                loader = AutoWeightsLoader(
                    module, ignore_unexpected_suffixes=["bias"]
                ).load_weights
            replace(module, "load_weights", state.wrap_load(loader))
        yield
        for state in coverage:
            state.validate()
    finally:
        for obj, name, previous in reversed(restore):
            if previous is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)
