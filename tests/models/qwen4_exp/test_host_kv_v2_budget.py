# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact storage bills: static static host allocation and the explicit frozen
shared-arena plan."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv.budget import shared_history_memory_bytes
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryStore,
    tensor_storage_bytes,
)

pytestmark = pytest.mark.cpu_test


def test_storage_bill_walks_layer_lists_and_counts_backing_once():
    raw = torch.empty(128, dtype=torch.uint8)
    other = torch.empty(33, dtype=torch.int32)
    bill = tensor_storage_bytes(
        {
            "hot": [raw[16:32], raw[64:80], other[:2]],
            "views": {"same": other[4:]},
            "scalar": 42,
        }
    )
    assert bill == {"hot": 128 + 33 * 4, "views": 0}
    store = object.__new__(HostHistoryStore)
    store.hot = [raw[16:32], other[:2]]
    store.alias = {"again": raw[32:]}
    store.host = raw
    store.host_device = raw
    store.ready = raw
    assert store.fixed_device_memory_bytes() == {"hot": 260, "alias": 0}


@pytest.mark.parametrize(
    "layers,tokens,stride,slot_bytes,requests,write_rows",
    [
        (12, 832, 53248, 2048, 8, 8192),
        (1, 1568, 106496, 8192, 8, 2048),
    ],
)
def test_shared_bill_matches_independent_tensor_storages(
    layers, tokens, stride, slot_bytes, requests, write_rows
):
    # Meta storages provide exact allocation lengths without allocating these
    # future page ownership/shared hot region/value arbitration buffers or consuming
    # device/host capacity.
    n, cap, width = 49, 256, 2051
    slots = n * (stride // slot_bytes)

    def tensor(shape, dtype=torch.int32):
        return torch.empty(shape, dtype=dtype, device="meta")

    tensors = {
        "P_pad": [tensor((n, stride), torch.uint8) for _ in range(layers)],
        "ready": tensor((n, tokens), torch.uint8),
        "generation": tensor((n,)),
        "group_to_slot": [tensor((n, tokens // 4)) for _ in range(layers)],
        "wanted": tensor((n, tokens // 4)),
        "page_of": tensor((n,)),
        "shadow": tensor((n, tokens // 4), torch.uint8),
        "slot_owner": [tensor((slots,)) for _ in range(layers)],
        "age": [tensor((slots,), torch.int64) for _ in range(layers)],
        "wanted_groups": tensor((slots,)),
        "miss_groups": tensor((slots,)),
        "miss_slots": tensor((slots,)),
        "victim_slots": tensor((slots,)),
        "hot_block_table": tensor((requests, slots)),
    }
    fixed = {
        "clock": tensor((layers,), torch.int64),
        "eviction_hand": tensor((layers,)),
        "wanted_count": tensor(()),
        "miss_count": tensor(()),
        "victim_counts": tensor((3, min((slots + 511) // 512, 256))),
        "indices": tensor((cap, width)),
        "staging": tensor((write_rows, slot_bytes // 4), torch.uint8),
        "staging_slots": tensor((write_rows,), torch.int64),
        "v2_counters": tensor((8,), torch.int64),
    }
    per_id, planned_fixed = shared_history_memory_bytes(
        num_blocks=n,
        page_strides=(stride,) * layers,
        slot_bytes_per_layer=slot_bytes,
        local_tokens=tokens,
        max_requests=requests,
        selection_width=width,
        max_write_rows=write_rows,
        read_rows=cap,
    )
    assert tensor_storage_bytes(tensors) == {
        name: size * n for name, size in per_id.items()
    }
    assert tensor_storage_bytes(fixed) == planned_fixed
    # Null is charged as one whole ID, not subtracted before byte accounting.
    assert sum(tensor_storage_bytes(tensors).values()) == n * sum(per_id.values())
    if layers == 12:
        assert sum(per_id.values()) == 655832
        assert per_id["shadow"] == 208
    assert planned_fixed["indices"] == 2_100_224
    assert planned_fixed["v2_counters"] == 64
    # Aliased hot payload adds zero bytes; duplicate layer views add zero too.
    combined = tensor_storage_bytes({**tensors, "hot": tensors["P_pad"]})
    assert combined["hot"] == 0


def test_shadow_encoding_is_an_explicit_budget_input():
    args = dict(
        num_blocks=8079,
        page_strides=(53248,) * 12,
        slot_bytes_per_layer=2048,
        local_tokens=832,
        max_requests=8,
        selection_width=2051,
        max_write_rows=8192,
    )
    compact, fixed = shared_history_memory_bytes(**args)
    wide, same_fixed = shared_history_memory_bytes(**args, shadow_bytes_per_group=9)
    assert fixed == same_fixed
    assert sum(wide.values()) - sum(compact.values()) == 8 * 208
    assert sum(fixed.values()) == 6_363_352
