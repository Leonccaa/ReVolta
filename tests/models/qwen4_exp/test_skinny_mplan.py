# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_SM70_GDN_SKINNY_MPLAN: per-M column tiling for the GDN input skinny GEMM must
keep the K partition of PLAN (so every row's summation order) and, on SM70,
produce byte-identical outputs."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import sm70_gdn_input_rowmajor as rm


def test_plan_table(monkeypatch):
    monkeypatch.setattr(rm, "_SKINNY_MPLAN", False)
    assert all(rm.plan_for(m) == rm.PLAN for m in range(2, 49))
    monkeypatch.setattr(rm, "_SKINNY_MPLAN", True)
    for m in range(2, 49):
        plan = rm.plan_for(m)
        assert (plan.warps_k, plan.splits) == (rm.PLAN.warps_k, rm.PLAN.splits)
        expect_ng2 = 9 <= m <= 24 or 33 <= m <= 48
        assert plan.ng == (2 if expect_ng2 else rm.PLAN.ng)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not rm.native_available(),
    reason="SM70 skinny GEMM op",
)
def test_outputs_bitwise_equal_to_plan(monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(5)
    qkvz = (torch.randn(*rm.QKVZ_SHAPE, generator=g, device="cuda") * 0.02).half()
    ba = (torch.randn(*rm.BA_SHAPE, generator=g, device="cuda") * 0.02).half()
    for m in range(2, 49):
        x = torch.randn(m, 2560, generator=g, device="cuda").half()
        monkeypatch.setattr(rm, "_SKINNY_MPLAN", False)
        ref = rm.gdn_input_rowmajor(x, qkvz, ba)
        monkeypatch.setattr(rm, "_SKINNY_MPLAN", True)
        got = rm.gdn_input_rowmajor(x, qkvz, ba)
        assert ref is not None and got is not None
        assert all(torch.equal(a, b) for a, b in zip(ref, got)), m
