# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.skip_global_cleanup


class Buffer:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.np = np.zeros(shape, dtype=np.int32)
        self.gpu = torch.zeros(shape, dtype=torch.int32)

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            self.gpu.copy_(torch.from_numpy(self.np))
        else:
            self.gpu[:n].copy_(torch.from_numpy(self.np[:n]))
        return self.gpu


def make_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.uses_flash_engram = True
    runner.ngram_context_len = 2
    runner.ngram_eos_token_id = 99
    runner.enable_prompt_embeds = False
    runner._sm70_async_staged_input_prep_active = False
    runner.ngram_context = Buffer((4, 2))
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([0, 1, 3, 5], dtype=np.int32),
        token_ids_cpu=np.array([[7, 8, 9, 10, 11, 12]] * 4, dtype=np.int32),
        is_token_ids=np.ones((4, 6), dtype=bool),
    )
    return runner


def test_history_uses_computed_cursor_and_raw_token_ids() -> None:
    runner = make_runner()
    kwargs: dict[str, torch.Tensor] = {}
    runner._maybe_add_flash_engram_kwargs(
        kwargs,
        num_reqs=4,
        num_reqs_padded=4,
        is_first_rank=True,
        is_encoder_decoder=False,
        use_dummy_context=False,
    )
    # Cursors can represent the first chunk, the next chunk, a cached prefix,
    # or a resumed request; each uses the current request's committed tokens.
    torch.testing.assert_close(
        kwargs["lookback_token_ids"],
        torch.tensor([[99, 99], [99, 7], [8, 9], [10, 11]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    runner.input_batch.num_computed_tokens_cpu[3] = 2
    result = runner._prepare_ngram_context(4, 4)
    torch.testing.assert_close(result[3], torch.tensor([7, 8], dtype=torch.int32))


def test_dummy_and_padding_use_engram_pad() -> None:
    runner = make_runner()
    kwargs: dict[str, torch.Tensor] = {}
    runner._maybe_add_flash_engram_kwargs(
        kwargs,
        num_reqs=2,
        num_reqs_padded=4,
        is_first_rank=True,
        is_encoder_decoder=False,
        use_dummy_context=True,
    )
    torch.testing.assert_close(
        kwargs["lookback_token_ids"], torch.full((4, 2), 99, dtype=torch.int32)
    )
    context = runner._prepare_ngram_context(2, 4)
    assert context.data_ptr() == kwargs["lookback_token_ids"].data_ptr()
    torch.testing.assert_close(context[2:], torch.full((2, 2), 99, dtype=torch.int32))
