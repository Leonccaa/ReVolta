# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.models.hy_v4_flash.sm70 import kernels
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cpu",
            marks=pytest.mark.skipif(
                not HAS_TRITON or os.environ.get("TRITON_INTERPRET") != "1",
                reason="Triton CPU interpreter required",
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda() or not torch.accelerator.is_available(),
                reason="CUDA required",
            ),
        ),
    ],
)
@pytest.mark.parametrize("world", [1, 4])
def test_prefill_scores_match_paged(monkeypatch, device, world):
    rng = torch.Generator().manual_seed(41)
    rows, heads, dim = 7, 16, 128
    first, ratio, page, pages_per_block = 2303, 4, 64, 2
    groups = cdiv(first + rows, ratio)
    local_groups = cdiv(groups, world)
    num_blocks = cdiv(local_groups, page * pages_per_block)
    q = torch.randn(rows, heads, dim, generator=rng).half()
    weights = torch.randn(rows, heads, generator=rng) / 32
    positions = torch.arange(first, first + rows)
    global_upper = positions // ratio
    reference = torch.empty(rows, groups)
    old, new = torch.empty_like(reference), torch.empty_like(reference)
    # Exercise the same gather buffer across group tiles on a small fixture.
    monkeypatch.setattr(kernels, "_INDEX_PREFILL_GROUP_TILE", 128)
    for rank in range(world):
        pool = torch.randn(
            num_blocks * pages_per_block, page + 1, dim, generator=rng
        ).half()
        table = torch.randperm(num_blocks, generator=rng, dtype=torch.int32)
        g = torch.arange(local_groups)
        page_ids = (
            table[g // (page * pages_per_block)] * pages_per_block
            + g // page % pages_per_block
        )
        keys = pool[page_ids, g % page].float()
        expected = ((q.float() @ keys.T).relu() * weights[..., None]).sum(1)
        masked = (g * world + rank)[None, :] >= global_upper[:, None]
        expected.masked_fill_(masked, -torch.inf)
        args = (
            q.to(device),
            weights.to(device),
            pool.to(device)[:, :page],
            table.to(device),
            first,
            local_groups,
            ratio,
            pages_per_block,
        )
        baseline = kernels.index_scores(*args, world=world, rank=rank).cpu()
        actual = kernels.index_prefill_scores(*args, world=world, rank=rank).cpu()
        assert torch.equal(actual.isneginf(), masked)
        assert torch.equal(baseline.isneginf(), masked)
        cols = reference[:, rank::world].shape[1]
        reference[:, rank::world] = expected[:, :cols]
        old[:, rank::world] = baseline[:, :cols]
        new[:, rank::world] = actual[:, :cols]

    valid = reference.isfinite()
    old_error = (old[valid] - reference[valid]).abs()
    new_error = (new[valid] - reference[valid]).abs()
    bitwise = torch.equal(old.view(torch.int32), new.view(torch.int32))
    print(
        f"{device=} {world=} {bitwise=} max_abs "
        f"old={old_error.max().item():.8g} new={new_error.max().item():.8g} "
        f"l2 old={old_error.norm().item():.8g} new={new_error.norm().item():.8g}"
    )
    assert new_error.max() <= old_error.max()
    assert new_error.norm() <= old_error.norm()

    old_top = old.topk(511)
    new_top = new.topk(511)
    cutoff = old_top.values[:, -1:]
    ties = (old > cutoff).sum(1) + (old == cutoff).sum(1) > 511
    print(f"top-511 cutoff ties={ties.sum().item()}/{rows}")
    assert torch.equal(
        old_top.indices[~ties].sort().values, new_top.indices[~ties].sort().values
    )
    # A cutoff tie permits any of the tied groups after all strict winners.
    picked = old[ties].gather(1, new_top.indices[ties])
    assert (picked >= cutoff[ties]).all()
    assert torch.equal(
        (picked > cutoff[ties]).sum(1), (old[ties] > cutoff[ties]).sum(1)
    )
