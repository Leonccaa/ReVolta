# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the shared MRV2 EAGLE/MTP loader and mixed KV pages."""

import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm.config import CacheConfig
from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
from vllm.models.qwen4_exp.nvidia.model import _finalize_qsa_e4m3_scale_load
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.core.kv_cache_utils import (
    _get_csa_linear_tensor_layout,
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.attn_utils import _packed_attention_members, _reshape_kv_cache
from vllm.v1.worker.gpu.spec_decode.eagle import utils
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.skip_global_cleanup


def _qsa(dtype):
    layer = Qwen4ExpQSAAttention.__new__(Qwen4ExpQSAAttention)
    nn.Module.__init__(layer)
    layer.layer_name = "mtp.layers.48.self_attn.attn"
    layer.num_kv_heads = 1
    layer.head_dim = 256
    layer.kv_cache_dtype = dtype
    layer.kv_cache_torch_dtype = torch.uint8 if dtype == "fp8_e4m3" else torch.float16
    layer._qsa_kv_scales_finalized = dtype != "fp8_e4m3"
    layer.register_buffer("_k_scale", torch.tensor(0.0))
    layer.register_buffer("_v_scale", torch.tensor(0.0))
    if dtype == "fp8_e4m3":
        layer.k_scale = nn.Parameter(torch.tensor(0.125), requires_grad=False)
        layer.v_scale = nn.Parameter(torch.tensor(0.25), requires_grad=False)
    return layer


@dataclass
class _LoaderConfig:
    cache_config: CacheConfig
    speculative_config: object
    parallel_config: object
    is_speculative_draft: bool = False


def _load(monkeypatch, dtype, *, calibrated=True):
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(4, 4)
    target.lm_head = nn.Linear(4, 4, bias=False)
    config = _LoaderConfig(
        CacheConfig(cache_dtype="fp8_e4m3", block_size=1600),
        SimpleNamespace(kv_cache_dtype=dtype, draft_model_config=object()),
        SimpleNamespace(decode_context_parallel_size=2),
    )
    loaded = []

    def get_model(**kwargs):
        draft_config = kwargs["vllm_config"]
        loaded.append(draft_config)
        draft = nn.Module()
        draft.model = nn.Module()
        draft.model.self_attn = _qsa(draft_config.cache_config.cache_dtype)
        scales = (
            {"model.self_attn.k_scale", "model.self_attn.v_scale"}
            if calibrated
            else set()
        )
        _finalize_qsa_e4m3_scale_load(
            draft,
            scales,
            draft_config.cache_config.cache_dtype,
            require_calibrated_speculative_draft=True,
        )
        return draft

    monkeypatch.setattr(utils, "get_model", get_model)
    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(
        "vllm.compilation.backends.set_model_tag", lambda _: nullcontext()
    )
    return config, loaded, lambda: utils.load_eagle_model(target, config)


@pytest.mark.parametrize("dtype", [None, "fp8_e4m3"])
def test_inherited_and_explicit_e4m3_keep_the_original_config(monkeypatch, dtype):
    target, loaded, load = _load(monkeypatch, dtype)
    draft = load()
    assert loaded == [target] and loaded[0] is target
    spec = draft.model.self_attn.get_kv_cache_spec(target)
    assert spec.dtype == torch.uint8
    assert spec.block_size == 1600 and not spec.dcp_sharded
    assert spec.page_size_bytes == 819200
    assert draft.model.self_attn._k_scale_float == 0.125
    assert draft.model.self_attn._v_scale_float == 0.25
    assert target.cache_config.cache_dtype == "fp8_e4m3"


@pytest.mark.parametrize("dtype", ["auto", "float16"])
def test_fp16_draft_loader_avoids_the_e4m3_scale_gate(monkeypatch, dtype):
    target, loaded, load = _load(monkeypatch, dtype, calibrated=False)
    draft = load()
    assert loaded[0].is_speculative_draft
    assert loaded[0].cache_config.cache_dtype == dtype
    assert loaded[0].cache_config.compute_hash() != target.cache_config.compute_hash()
    assert target.cache_config.cache_dtype == "fp8_e4m3"
    spec = draft.model.self_attn.get_kv_cache_spec(target)
    assert spec.dtype == torch.float16 and spec.page_size_bytes == 1638400
    assert draft.model.self_attn._qsa_kv_scales_finalized


def test_e4m3_draft_still_requires_calibrated_scales(monkeypatch):
    _, _, load = _load(monkeypatch, "fp8_e4m3", calibrated=False)
    with pytest.raises(ValueError, match="Loaded 0/2 draft K/V scales"):
        load()


def test_production_e4m3_contract_snapshot(monkeypatch):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    options = {
        "method": "mtp",
        "num_speculative_tokens": 4,
        "kv_cache_dtype": "fp8_e4m3",
        "draft_sample_method": "greedy",
    }
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(
        [
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--mm-encoder-tp-mode",
            "weights",
            "--speculative-config",
            json.dumps(options),
        ]
    )
    target, loaded, load = _load(
        monkeypatch, parsed.speculative_config["kv_cache_dtype"]
    )
    draft = load()
    snapshot = {
        "kv_cache_dtype": parsed.kv_cache_dtype,
        "mm_encoder_tp_mode": parsed.mm_encoder_tp_mode,
        "speculative_config": parsed.speculative_config,
        "cache_config": asdict(loaded[0].cache_config),
        "cache_hash": loaded[0].cache_config.compute_hash(),
        "draft_spec": asdict(draft.model.self_attn.get_kv_cache_spec(target)),
        "scales": [
            draft.model.self_attn._k_scale_float,
            draft.model.self_attn._v_scale_float,
        ],
    }
    assert snapshot["speculative_config"] == options
    if output := os.getenv("FP4_CONTRACT_SNAPSHOT"):
        Path(output).write_text(
            json.dumps(snapshot, default=str, sort_keys=True, indent=2) + "\n"
        )


def _config(dtype="float16"):
    model = SimpleNamespace(
        dtype=torch.float16,
        use_mla=False,
        is_hybrid=True,
        hf_text_config=SimpleNamespace(indexer_n_heads=8),
        max_model_len=8192,
        get_num_kv_heads=lambda _: 1,
        get_total_num_hidden_layers=lambda: 48,
    )
    return SimpleNamespace(
        model_config=model,
        is_speculative_draft=False,
        use_v2_model_runner=True,
        speculative_config=SimpleNamespace(
            method="mtp", kv_cache_dtype=dtype, draft_model_config=model
        ),
        cache_config=SimpleNamespace(
            cache_dtype="fp8_e4m3",
            block_size=64,
            num_gpu_blocks_override=3,
            mamba_cache_mode="none",
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=8192,
            max_num_seqs=2,
        ),
        kv_transfer_config=None,
    )


@pytest.mark.parametrize("connector", [None, "OffloadingConnector"])
@pytest.mark.parametrize("method", ["mtp", "eagle", "eagle3"])
def test_mixed_dtype_admission(connector, method):
    validate = pytest.importorskip(
        "vllm.config.speculative_kv"
    ).validate_speculative_kv_cache_dtype
    config = _config()
    config.speculative_config.method = method
    if method != "mtp":
        config.parallel_config.decode_context_parallel_size = 1
    if connector:
        config.kv_transfer_config = SimpleNamespace(
            kv_connector=connector, kv_connector_module_path=None, kv_role="kv_both"
        )
    validate(config)


@pytest.mark.parametrize(
    "owner,field,value,reason",
    [
        ("", "use_v2_model_runner", False, "Model Runner V2"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline parallelism"),
        ("parallel_config", "prefill_context_parallel_size", 2, "prefill CP"),
        ("parallel_config", "decode_context_parallel_size", 4, "DCP2"),
        ("scheduler_config", "disable_hybrid_kv_cache_manager", True, "hybrid KV"),
        ("model_config", "use_mla", True, "MLA"),
        ("speculative_config", "kv_cache_dtype", "fp8_e5m2", "quantized draft"),
        ("speculative_config", "kv_cache_dtype", "bfloat16", "activation dtype"),
        ("cache_config", "cache_dtype", "nvfp4", "target KV layout"),
        ("cache_config", "cache_dtype", "fp8_e5m2", "target KV layout"),
        ("cache_config", "cache_dtype", "int8", "target KV layout"),
        ("speculative_config", "method", "eagle", "native QSA MTP"),
        ("speculative_config", "method", "eagle3", "native QSA MTP"),
    ],
)
def test_mixed_dtype_rejects_incompatible_combinations(owner, field, value, reason):
    validate = pytest.importorskip(
        "vllm.config.speculative_kv"
    ).validate_speculative_kv_cache_dtype
    config = _config()
    setattr(getattr(config, owner) if owner else config, field, value)
    with pytest.raises(ValueError, match=reason):
        validate(config)


@pytest.mark.parametrize(
    "connector,module,role",
    [
        ("NixlConnector", None, "kv_both"),
        ("OffloadingConnector", "custom.connector", "kv_both"),
        ("OffloadingConnector", None, "kv_consumer"),
    ],
)
def test_mixed_dtype_rejects_unqualified_transfer(connector, module, role):
    validate = pytest.importorskip(
        "vllm.config.speculative_kv"
    ).validate_speculative_kv_cache_dtype
    config = _config()
    config.kv_transfer_config = SimpleNamespace(
        kv_connector=connector, kv_connector_module_path=module, kv_role=role
    )
    with pytest.raises(ValueError, match="built-in OffloadingConnector"):
        validate(config)


def test_mixed_dtype_rejects_non_qsa_metadata_schedulers():
    validate = pytest.importorskip(
        "vllm.config.speculative_kv"
    ).validate_speculative_kv_cache_dtype
    config = _config()
    config.model_config.hf_text_config.indexer_n_heads = 0
    with pytest.raises(ValueError, match="only for QSA"):
        validate(config)


def test_mixed_dtype_allows_a_disabled_transfer_config():
    from vllm.config import KVTransferConfig

    validate = pytest.importorskip(
        "vllm.config.speculative_kv"
    ).validate_speculative_kv_cache_dtype
    config = _config()
    config.kv_transfer_config = KVTransferConfig()
    validate(config)


@pytest.mark.parametrize("dtype", ["fp8_e4m3", "int8"])
def test_same_dtype_keeps_existing_unsupported_options_outside_the_new_gate(dtype):
    validate = pytest.importorskip(
        "vllm.config.speculative_kv"
    ).validate_speculative_kv_cache_dtype
    config = _config(dtype)
    config.cache_config.cache_dtype = dtype
    config.use_v2_model_runner = False
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    validate(config)


def test_int8_cli_selection_and_inherited_draft_remain_unchanged():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    validate = pytest.importorskip(
        "vllm.config.speculative_kv"
    ).validate_speculative_kv_cache_dtype
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(["--kv-cache-dtype", "int8"])
    cache = CacheConfig(cache_dtype=parsed.kv_cache_dtype)
    config = _config(None)
    config.cache_config = cache
    validate(config)
    assert config.cache_config is cache
    assert cache.cache_dtype == "int8"


def _mixed_specs(dtype):
    specs = {}
    for index in (3, 7, 48):
        name = f"model.layers.{index}.self_attn"
        specs[name] = FullAttentionSpec(
            block_size=32 if index < 48 else 64,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.uint8 if index < 48 else dtype,
            dcp_sharded=index < 48,
        )
        specs[name + ".compressed"] = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
            dcp_sharded=False,
        )
        specs[name + ".compressor_state"] = CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
            dcp_sharded=False,
        )
    for index in (0, 1, 2, 4, 5, 6):
        specs[f"model.layers.{index}.linear_attn"] = MambaSpec(
            block_size=64,
            shapes=((1, 12000),),
            dtypes=(torch.float16,),
            dcp_sharded=False,
        )
    return specs


@pytest.mark.parametrize("draft_dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("cache_layout", ["NHD", "HND"])
def test_dcp2_mixed_pages_reshape_and_offload_are_byte_preserving(
    monkeypatch, draft_dtype, cache_layout
):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.v1.kv_offload.base import OffloadingSpec

    monkeypatch.setattr(
        "vllm.v1.attention.backends.flash_attn.get_kv_cache_layout",
        lambda: cache_layout,
    )
    config = _config()
    specs = _mixed_specs(draft_dtype)
    groups = get_kv_cache_groups(config, specs)
    cache = get_kv_cache_config_from_groups(config, groups, available_memory=1 << 28)
    layout = _get_csa_linear_tensor_layout(groups)
    assert layout is not None
    assert layout.main_kv_owners == [[0, 1], [2]]
    assert layout.main_kv_page_sizes == [
        16384,
        16384,
        32768 if draft_dtype == torch.uint8 else 65536,
    ]
    scheduler = generate_scheduler_kv_cache_config([cache])
    assert scheduler.num_blocks == cache.num_blocks == 3

    kv_caches = {}
    raw_tensors = []
    generator = torch.Generator().manual_seed(20261018)
    for tensor in cache.kv_cache_tensors:
        raw = torch.randint(
            -128, 128, (tensor.size,), dtype=torch.int8, generator=generator
        )
        raw_tensors.append(raw)
        for gid, group in enumerate(groups):
            per_layer = group.kv_cache_spec
            per_layer = (
                per_layer.kv_cache_specs
                if isinstance(per_layer, UniformTypeKVCacheSpecs)
                else {name: per_layer for name in group.layer_names}
            )
            for name in set(tensor.shared_by) & per_layer.keys():
                spec = per_layer[name]
                if type(spec) is FullAttentionSpec:
                    backend = FlashAttentionBackend
                elif isinstance(spec, MambaSpec):
                    backend = QSAStateBackend
                else:
                    kv_caches[name] = raw
                    continue
                kv_caches.update(
                    _reshape_kv_cache(
                        [AttentionGroup(backend, [name], spec, gid)],
                        {name: raw},
                        "fp8_e4m3",
                        [16] * len(groups),
                        {},
                        _packed_attention_members(cache),
                    )
                )
    assert kv_caches["model.layers.3.self_attn"].dtype == torch.uint8
    assert kv_caches["model.layers.48.self_attn"].dtype == draft_dtype
    # Independent byte offsets for the two packed target members. Checking a
    # random byte stream also catches a wrongly repeated/interleaved page.
    for member, layer_index in enumerate((3, 7)):
        view = kv_caches[f"model.layers.{layer_index}.self_attn"]
        for kernel_page in (0, 1, 2):
            offset = (kernel_page * 2 + member) * 8192
            assert torch.equal(
                view[kernel_page, 0, 0, 0, :].view(torch.int8),
                raw_tensors[0][offset : offset + 256],
            )

    offload = MagicMock(spec=OffloadingSpec)
    offload.kv_cache_config = cache
    offload.vllm_config = MagicMock()
    offload.get_handlers.return_value = iter([])
    worker = OffloadingConnectorWorker(spec=offload)
    worker.worker = MagicMock()
    worker.register_kv_caches(kv_caches)
    canonical = offload.get_handlers.call_args[0][0]
    assert len(canonical.tensors) == len(raw_tensors)
    for registered, raw in zip(canonical.tensors, raw_tensors):
        assert (
            registered.tensor.untyped_storage().data_ptr()
            == raw.untyped_storage().data_ptr()
        )
        assert torch.equal(registered.tensor.flatten(), raw)
        saved = registered.tensor.clone()
        registered.tensor.zero_()
        registered.tensor.copy_(saved)
        assert torch.equal(registered.tensor, saved)
