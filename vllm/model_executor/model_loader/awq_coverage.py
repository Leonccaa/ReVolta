# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Track serialized AWQ components before fused parameters are repacked.

One loaded fused parameter name cannot prove that every expert or logical
projection arrived. Track successful loader calls instead, across all files.
This covers AWQ's qweight/qzeros/scales format, including local expert shards.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)

_KINDS = ("qweight", "qzeros", "scales")
_MOE_NAMES = tuple(f"{proj}_{kind}" for proj in ("w13", "w2") for kind in _KINDS)
_MISSING = object()


class _Coverage:
    def __init__(self, module: nn.Module, prefix: str, moe: bool):
        self.module = module
        self.prefix = prefix
        self.moe = moe
        names = _MOE_NAMES if moe else _KINDS
        self.params = {id(getattr(module, name)): name for name in names}
        self.expected: set[tuple[str, int | str | None, int | None]] = set()
        self.loaded: set[tuple[str, int | str | None, int | None]] = set()
        self.dense_shards: tuple[int | str | None, ...]
        if isinstance(module, QKVParallelLinear):
            self.dense_shards = ("q", "k", "v")
        elif isinstance(module, MergedColumnParallelLinear):
            self.dense_shards = tuple(range(len(module.output_sizes)))
        else:
            self.dense_shards = (None,)
        for name in names:
            if moe:
                shards = ("w1", "w3") if name.startswith("w13_") else ("w2",)
                self.expected.update(
                    (name, shard, expert)
                    for shard in shards
                    for expert in range(getattr(module, name).shape[0])
                )
            else:
                self.expected.update((name, shard, None) for shard in self.dense_shards)

    def wrap(self, loader):
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
                    self.loaded.update((name, shard, expert) for expert in experts)
            else:
                shard = kwargs.get(
                    "loaded_shard_id", kwargs.get("shard_id", args[0] if args else None)
                )
                shards = (
                    self.dense_shards
                    if shard is None
                    else shard
                    if isinstance(shard, tuple)
                    else (shard,)
                )
                self.loaded.update((name, shard, None) for shard in shards)
            return result

        return tracked

    def validate(self):
        missing = self.expected - self.loaded
        if missing:
            examples = sorted(map(str, missing))[:12]
            raise ValueError(
                "AWQ checkpoint has incomplete coverage for "
                f"{self.prefix or '<root>'}: "
                f"missing {len(missing)} tensor components (parameter, shard, expert): "
                + ", ".join(examples)
            )


@contextmanager
def awq_checkpoint_coverage(model: nn.Module) -> Iterator[None]:
    """Validate AWQ components after loading; restore every loader on exit.

    Called only by real AWQ checkpoint loading. Dummy initialization and other
    quantization formats retain their own loading/coverage contracts.
    """
    coverage = []
    restore = []
    try:
        for prefix, module in model.named_modules():
            params = dict(module.named_parameters(recurse=False))
            moe = all(name in params for name in _MOE_NAMES)
            if not moe and not all(name in params for name in _KINDS):
                continue
            state = _Coverage(module, prefix, moe)
            coverage.append(state)
            # MoE loaders can call either the module's method or the per-param
            # callback. Observe both without changing copies or return values.
            if moe:
                restore.append(
                    (
                        module,
                        "weight_loader",
                        module.__dict__.get("weight_loader", _MISSING),
                    )
                )
                module.weight_loader = state.wrap(module.weight_loader)
            for name in _MOE_NAMES if moe else _KINDS:
                param = params[name]
                restore.append((param, "weight_loader", param.weight_loader))
                param.weight_loader = state.wrap(param.weight_loader)
        yield
        for state in coverage:
            state.validate()
    finally:
        for obj, name, previous in reversed(restore):
            if previous is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)
