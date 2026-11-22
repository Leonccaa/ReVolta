# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampler staged writes are flushed only on steps that admit requests
(vllm-project/vllm#62748). Sampler.add_request is the only writer."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _runner(events, sampler_present=True, last_pp=True):
    r = SimpleNamespace()
    r.is_last_pp_rank = last_pp
    r.encoder_cache = None
    r._remove_request = lambda req_id: None
    r.req_states = MagicMock()
    r.req_states.req_id_to_index = {}
    r.req_states.add_request.side_effect = lambda req_id, **kw: (
        r.req_states.req_id_to_index.__setitem__(
            req_id, len(r.req_states.req_id_to_index)
        )
    )
    r.req_states.apply_staged_writes.side_effect = lambda: events.append("req")
    r.model_state = MagicMock()
    r.model_state.apply_staged_writes.side_effect = lambda: events.append("model")
    r.block_tables = MagicMock()
    r.lora_state = MagicMock()
    r.prompt_logprobs_worker = MagicMock()
    if sampler_present:
        r.sampler = MagicMock()
        r.sampler.add_request.side_effect = lambda *a: events.append("s_add")
        r.sampler.apply_staged_writes.side_effect = lambda: events.append("s_flush")
    else:
        r.sampler = None
    return r


def _req(req_id, sp):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        prefill_token_ids=[1, 2, 3],
        num_computed_tokens=0,
        mm_features=[],
        block_ids=([0],),
        lora_request=None,
        sampling_params=sp,
    )


def _sched(reqs):
    return SimpleNamespace(scheduled_new_reqs=reqs)


def test_no_new_requests_skips_sampler_flush():
    ev: list[str] = []
    r = _runner(ev)
    GPUModelRunner.add_requests(r, _sched([]))
    assert ev == []
    r.sampler.apply_staged_writes.assert_not_called()


# Admission, preemption resume (preempted ids are removed and re-admitted as
# new reqs) and slot reuse all arrive as scheduled_new_reqs.
@pytest.mark.parametrize("n", [1, 2])
def test_admission_flushes_after_add_request(n):
    ev: list[str] = []
    r = _runner(ev)
    sp = SamplingParams(
        temperature=0.7,
        repetition_penalty=1.1,
        logit_bias={5: 1.0},
        bad_words=["x"],
        logprobs=3,
    )
    GPUModelRunner.add_requests(r, _sched([_req(f"a{i}", sp) for i in range(n)]))
    assert ev == ["s_add"] * n + ["req", "model", "s_flush"]


def test_resume_and_slot_reuse_flush_each_time():
    ev: list[str] = []
    r = _runner(ev)
    sp = SamplingParams(temperature=0.5)
    for _ in range(3):  # same req_id re-admitted / slot reused
        GPUModelRunner.add_requests(r, _sched([_req("a", sp)]))
        GPUModelRunner.add_requests(r, _sched([]))
    assert ev == ["s_add", "req", "model", "s_flush"] * 3


def test_non_last_pp_rank_no_sampler():
    ev: list[str] = []
    r = _runner(ev, sampler_present=False, last_pp=False)
    GPUModelRunner.add_requests(r, _sched([_req("a", None)]))
    assert ev == ["req", "model"]
