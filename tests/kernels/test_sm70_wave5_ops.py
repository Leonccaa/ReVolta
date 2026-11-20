# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70",
)


def _gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    rrms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((xf * rrms) * (1.0 + w.float())).half()


def _neox_rope(
    x: torch.Tensor, cos_sin: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    rot = cos_sin.shape[1]
    half_rot = rot // 2
    cs = cos_sin[pos.long()].float()
    cos = cs[:, None, :half_rot]
    sin = cs[:, None, half_rot:]
    xr = x[..., :rot].float()
    a, b = xr[..., :half_rot], xr[..., half_rot:]
    rotated = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).half()
    return torch.cat([rotated, x[..., rot:]], dim=-1)


@pytest.mark.parametrize("k_down", [160, 320, 640])
@pytest.mark.parametrize("rows", [1, 5, 10, 20, 40])
def test_shared_down_gated_matches_reference_and_cudagraph(rows: int, k_down: int):
    torch.manual_seed(1000 + rows * 7 + k_down)
    dev = torch.device("cuda")
    hidden_x = torch.randn(rows, 2560, device=dev, dtype=torch.float16) * 0.5
    act_out = torch.randn(rows, k_down, device=dev, dtype=torch.float16) * 0.5
    down_w = torch.randn(2560, k_down, device=dev, dtype=torch.float16) * 0.05
    gate_w = torch.randn(1, 2560, device=dev, dtype=torch.float16) * 0.05
    out = torch.empty(rows, 2560, device=dev, dtype=torch.float16)

    def ref_fn():
        down = (act_out.float() @ down_w.float().T).half()
        gate_logit = (hidden_x.float() @ gate_w.float().T).half().float()
        gate = torch.sigmoid(gate_logit)
        return (down.float() * gate).half()

    ops.qwen38_shared_down_gated_sm70_out(out, hidden_x, act_out, down_w, gate_w)
    torch.testing.assert_close(out, ref_fn(), rtol=2e-3, atol=2e-3)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ops.qwen38_shared_down_gated_sm70_out(out, hidden_x, act_out, down_w, gate_w)
    hidden_x.mul_(-0.75)
    act_out.mul_(0.5)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref_fn(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows", [1, 4, 5, 10, 20, 40, 64])
def test_router_top10_batch_matches_reference(rows: int):
    torch.manual_seed(2000 + rows)
    dev = torch.device("cuda")
    # Guarantee distinct FP16 logits per row so top-10 ordering is unambiguous.
    perm = torch.stack([torch.randperm(512, device=dev) for _ in range(rows)])
    logits_fp16 = ((perm.float() - 256.0) * 0.015625).half().contiguous()
    weights = torch.empty(rows, 10, device=dev, dtype=torch.float32)
    ids = torch.empty(rows, 10, device=dev, dtype=torch.int32)
    source = torch.empty(rows, 10, device=dev, dtype=torch.int32)
    ops.sm70_router_top10_batch(weights, ids, source, logits_fp16)

    logits = logits_fp16.float()
    top_l, top_i = torch.topk(logits, 10, dim=-1)
    raw_w = torch.exp(top_l - top_l[:, :1])
    ref_w = raw_w / raw_w.sum(dim=-1, keepdim=True)
    ref_src = (
        torch.arange(10, device=dev, dtype=torch.int32)[None, :] * rows
        + torch.arange(rows, device=dev, dtype=torch.int32)[:, None]
    )
    assert torch.equal(ids, top_i.int())
    assert torch.equal(source, ref_src)
    torch.testing.assert_close(weights, ref_w, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("is_bytes", [False, True])
@pytest.mark.parametrize("rows", [1, 5, 15])
def test_ple_local_kv_and_gate_norm_match_eager(rows: int, is_bytes: bool):
    torch.manual_seed(4000 + rows + int(is_bytes))
    dev = torch.device("cuda")
    eps = 1e-6
    if is_bytes:
        emb = torch.randint(0, 120, (rows, 1024), device=dev, dtype=torch.uint8)
        scale = torch.tensor([0.03125], device=dev, dtype=torch.float16)
        emb_fp16 = (emb.view(torch.float8_e4m3fn).float() * scale.float()).half()
    else:
        emb = torch.randn(rows, 1024, device=dev, dtype=torch.float16) * 0.2
        scale = torch.empty(0, device=dev, dtype=torch.float32)
        emb_fp16 = emb
    key_w = torch.randn(10240, 1024, device=dev, dtype=torch.float16) * 0.03
    val_w = torch.randn(2560, 1024, device=dev, dtype=torch.float16) * 0.03
    hidden = torch.randn(rows, 10240, device=dev, dtype=torch.float16) * 0.5
    nkw = torch.randn(10240, device=dev, dtype=torch.float16) * 0.1
    nqw = torch.randn(10240, device=dev, dtype=torch.float16) * 0.1
    ncw = torch.randn(10240, device=dev, dtype=torch.float16) * 0.1

    key_out = torch.empty(rows, 10240, device=dev, dtype=torch.float16)
    val_out = torch.empty(rows, 2560, device=dev, dtype=torch.float16)
    ops.qwen38_ple_kv_proj_local_sm70_out(key_out, val_out, emb, scale, key_w, val_w)
    gv_out = torch.empty(rows, 10240, device=dev, dtype=torch.float16)
    norm_out = torch.empty(rows, 10240, device=dev, dtype=torch.float16)
    ops.qwen38_ple_gate_norm_sm70_out(
        gv_out, norm_out, hidden, key_out, val_out, 0, hidden, nkw, nqw, ncw, eps
    )

    k_ref = (emb_fp16.float() @ key_w.float().T).half()
    v_ref = (emb_fp16.float() @ val_w.float().T).half()
    torch.testing.assert_close(key_out, k_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(val_out, v_ref, rtol=2e-3, atol=2e-3)

    kn = _gemma_rmsnorm(k_ref.view(rows, 4, 2560), nkw.view(4, 2560), eps)
    qn = _gemma_rmsnorm(hidden.view(rows, 4, 2560), nqw.view(4, 2560), eps)
    dot = (kn * qn).sum(dim=-1, keepdim=True) / (2560**0.5)
    min_c = torch.tensor(1e-6, device=dev, dtype=torch.float16)
    gate = torch.sigmoid(
        torch.sign(dot) * torch.sqrt(torch.clamp(dot.abs(), min=min_c))
    )
    gv_ref = (gate * v_ref.unsqueeze(1)).reshape(rows, 10240)
    norm_ref = _gemma_rmsnorm(gv_ref.view(rows, 4, 2560), ncw.view(4, 2560), eps).view(
        rows, 10240
    )
    torch.testing.assert_close(gv_out, gv_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(norm_out, norm_ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("hq", [6, 12, 24])
@pytest.mark.parametrize("rows", [1, 5, 15])
def test_qsa_prep_write_matches_norm_rope_and_history(rows, hq):
    # The native ABI operates directly on the history and hot-cache buffers.
    torch.manual_seed(3000 + rows * 31 + hq)
    dev = "cuda"
    eps = 1e-6
    qkv = torch.randn(rows, hq * 512 + 512, device=dev, dtype=torch.float16)
    pos = torch.randint(0, 200, (rows,), device=dev, dtype=torch.int64)
    slots = torch.arange(16, 16 + rows, device=dev, dtype=torch.int32)
    cos_sin = torch.randn(256, 64, device=dev, dtype=torch.float32)
    qw = torch.randn(256, device=dev, dtype=torch.float16) * 0.1
    kw = torch.randn_like(qw) * 0.1
    q_out = torch.empty(rows, hq, 256, device=dev, dtype=torch.float16)
    k_out = torch.empty(rows, 1, 256, device=dev, dtype=torch.float16)
    history = torch.zeros(1, 2, 64, 1, 256, device=dev, dtype=torch.float16)
    scales = torch.ones(2, device=dev, dtype=torch.float32)
    tags = torch.full((16,), -1, device=dev, dtype=torch.int32)
    page_slots = torch.full_like(tags, -1)
    hot = torch.empty(64, 2, 256, device=dev, dtype=torch.float16)
    epoch = torch.zeros(1, device=dev, dtype=torch.int32)
    ops.sm70_qsa_device_prep_write(
        qkv,
        pos,
        cos_sin,
        qw,
        kw,
        eps,
        q_out,
        k_out,
        slots,
        history,
        scales,
        tags,
        page_slots,
        hot,
        None,
        epoch,
    )
    q = qkv[:, : hq * 512].view(rows, hq, 512)[..., :256]
    k = qkv[:, hq * 512 : hq * 512 + 256].view(rows, 1, 256)
    v = qkv[:, hq * 512 + 256 :].view(rows, 1, 256)
    q_ref = _neox_rope(_gemma_rmsnorm(q, qw, eps), cos_sin, pos)
    k_ref = _neox_rope(_gemma_rmsnorm(k, kw, eps), cos_sin, pos)
    torch.testing.assert_close(q_out, q_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(k_out, k_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(history[0, 0, slots], k_out, rtol=0, atol=0)
    torch.testing.assert_close(history[0, 1, slots], v, rtol=0, atol=0)


@pytest.mark.parametrize("qlen", [5, 8])
def test_ple_spec_updates_state_and_clears_padded_rows(qlen):
    torch.manual_seed(1570 + qlen)
    dev = "cuda"
    rows, hidden = 64, 10240
    x = torch.randn(rows, hidden, device=dev, dtype=torch.float16)
    weight = torch.randn(hidden, 4, device=dev, dtype=torch.float16) * 0.125
    state = torch.randn(9, hidden, 8 + qlen, device=dev, dtype=torch.float16)
    before = state.clone()
    ids = torch.arange(1, 9, device=dev, dtype=torch.int32)
    starts = torch.arange(9, device=dev, dtype=torch.int32) * (qlen - 1)
    accepted = torch.full((8,), 2, device=dev, dtype=torch.int32)
    out = torch.full_like(x, float("nan"))
    ops.qwen38_ple_spec_sm70_v154_out(out, state, x, weight, ids, starts, accepted)
    count = qlen - 1
    for request, source in enumerate(x[: count * 8].split(count)):
        stream = torch.cat([before[request + 1, :, 1:10], source.T], dim=1)
        expected = torch.empty_like(source)
        for row in range(count):
            values = stream[:, row : row + 10 : 3].float()
            total = (values * weight.float()).sum(dim=1).half().float()
            expected[row] = torch.nn.functional.silu(total).half()
        torch.testing.assert_close(
            out[request * count : (request + 1) * count],
            expected,
            rtol=2e-3,
            atol=2e-3,
        )
        assert torch.equal(state[request + 1, :, : 8 + count], stream[:, 1:])
    assert torch.count_nonzero(out[count * 8 :]) == 0
