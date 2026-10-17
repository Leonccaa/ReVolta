# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU allocation regressions using real managers and a shared block pool."""

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec


def mamba(pool, group=0, block_size=1632):
    return MambaManager(
        MambaSpec(
            block_size=block_size,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=4,
        ),
        pool,
        enable_caching=True,
        kv_cache_group_id=group,
    )


def test_mamba_shrink_and_regrow():
    pool = BlockPool(64, True, 1632)
    m = mamba(pool)
    m.allocate_new_blocks("r", 1633, 1633)
    before = list(m.req_to_blocks["r"])
    assert len(before) == 6
    assert m.allocate_new_blocks("r", 1631, 1631) == []
    assert m.req_to_blocks["r"] == before
    assert m.allocate_new_blocks("r", 1634, 1634) == []
    m.allocate_new_blocks("r", 3265, 3265)
    assert len(m.req_to_blocks["r"]) == 7
