# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _sm70_flashnext_prefill_ops as prefill_ops


def test_sm70_gdn_prefill_fake_op(monkeypatch):
    lib = torch.library.Library("vllm_sm70_fn_prefill", "DEF")
    lib.define(
        "gdn_chunk_fwd_fused_sm70(Tensor q, Tensor k, Tensor v, "
        "Tensor(a!) a_bar_buf, Tensor(b!) p_buf, Tensor(c!) g_cs_buf, "
        "Tensor g_raw, Tensor b, Tensor(d!) initial_state, Tensor cu_seqlens, "
        "Tensor chunk_indices, Tensor? state_indices, Tensor? has_initial_state, "
        "Tensor(e!) out, Tensor(f!)? final_state, float scale, "
        "bool use_initial_state, bool state_layout_vlk, "
        "bool inplace_final_state, bool store_final_state) -> ()"
    )
    monkeypatch.setattr(prefill_ops.importlib, "import_module", lambda _: None)
    prefill_ops.has_sm70_flashnext_prefill_ops.cache_clear()
    try:
        assert prefill_ops.has_sm70_flashnext_prefill_ops()
        x = torch.empty(1, device="meta")
        assert (
            torch.ops.vllm_sm70_fn_prefill.gdn_chunk_fwd_fused_sm70(
                x,
                x,
                x,
                x,
                x,
                x,
                x,
                x,
                x,
                x,
                x,
                None,
                None,
                x,
                x,
                1.0,
                True,
                True,
                False,
                True,
            )
            is None
        )
    finally:
        prefill_ops.has_sm70_flashnext_prefill_ops.cache_clear()
