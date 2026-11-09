# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared builders for the long-context (Active Host KV) CPU tests.

``production_specs`` models the Qwen4Exp DFlash2 geometry under
TP4 + DCP2 (12 QSA layers, 36 GDN layers in six owners, one PLE state and a
five-layer DFlash drafter): it reproduces the 6,389,760-byte pool block and the
per-request state demand derived from the shapes below.
"""

from types import SimpleNamespace

import torch

from vllm.config.kernel import LongContextKVConfig
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

SPAN = 1664
DCP = 2
POOL_BLOCK_BYTES = 6_389_760
MAIN_KV_BLOCK_BYTES = 5_111_808
INDEX_BLOCK_BYTES = 1_277_952
MI = 1 << 20


def host_step_receipts(scheduler, output):
    """Retire a scheduler-only fixture's simulated host worker execution."""
    if output.host_kv is None:
        return None
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )

    mask = scheduler.host_history_pool.host_pages.all_ranks
    return OffloadingWorkerMetadata(
        seal_results={t: (mask, 0, 0) for t in output.host_kv.seals},
        completed_steps={output.host_kv.step: mask},
    )


class ProductionModelConfig:
    hf_text_config = SimpleNamespace(indexer_budget=2048)

    def __init__(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len

    def get_num_kv_heads(self, parallel_config) -> int:
        del parallel_config
        return 1

    def get_num_attention_heads(self, parallel_config) -> int:
        del parallel_config
        return 24

    def get_total_num_hidden_layers(self) -> int:
        return 48


def production_config(
    *,
    max_model_len: int = MI,
    num_speculative_tokens: int = 7,
    max_num_seqs: int = 8,
    long_context: LongContextKVConfig | None = None,
    dflash: bool = True,
):
    return SimpleNamespace(
        model_config=ProductionModelConfig(max_model_len),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=DCP,
            prefill_context_parallel_size=1,
            tensor_parallel_size=4,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=8192,
            max_num_seqs=max_num_seqs,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None, mamba_cache_mode="align"
        ),
        speculative_config=SimpleNamespace(
            use_dflash=lambda: dflash,
            num_speculative_tokens=num_speculative_tokens,
        ),
        max_in_flight_tokens=8192,
        kernel_config=SimpleNamespace(
            long_context_kv=long_context or LongContextKVConfig()
        ),
        kv_transfer_config=(
            SimpleNamespace(kv_connector="OffloadingConnector")
            if long_context is not None
            and long_context.enabled
            and long_context.main_kv_placement == "host"
            else None
        ),
    )


def production_specs(
    *, speculative_blocks: int = 7, drafter: bool = True
) -> dict[str, KVCacheSpec]:
    """Worker specs of the reference geometry; s = 0 is GDN state replay."""
    specs: dict[str, KVCacheSpec] = {}
    for layer in range(3, 48, 4):
        prefix = f"model.layers.{layer}.self_attn"
        specs[prefix] = FullAttentionSpec(
            block_size=SPAN // DCP,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.uint8,
            dcp_sharded=True,
        )
        specs[f"{prefix}.compressed"] = MLAAttentionSpec(
            block_size=SPAN,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
            dcp_sharded=False,
        )
        specs[f"{prefix}.compressor_state"] = CircularBufferSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
            dcp_sharded=False,
        )
    for layer in range(48):
        if layer % 4 != 3:
            specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
                block_size=SPAN,
                shapes=((1, 300_000),),
                dtypes=(torch.float16,),
                mamba_cache_mode="align",
                num_speculative_blocks=speculative_blocks,
            )
    specs["model.layers.0.ple"] = MambaSpec(
        block_size=SPAN,
        shapes=((1, 1_000),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        tp_replicated=True,
    )
    if drafter:
        for layer in range(48, 53):
            specs[f"model.layers.{layer}.self_attn.attn"] = SlidingWindowSpec(
                block_size=SPAN,
                num_kv_heads=2,
                head_size=128,
                dtype=torch.float16,
                sliding_window=2048,
            )
    return specs
