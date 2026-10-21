# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request isolation for matcher failures; see vllm-project/vllm#48200, #58612."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import torch
from transformers import OPTConfig

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar
from vllm.v1.structured_output.request import StructuredOutputRequest


def _grammar():
    grammar = Mock(spec=StructuredOutputGrammar)
    grammar.is_terminated.return_value = False
    grammar.accept_tokens.return_value = True
    grammar.validate_tokens.side_effect = lambda tokens: tokens
    grammar.fill_bitmask.side_effect = lambda mask, row: mask[row].fill_(7)
    return grammar


def _batch(tmp_path):
    OPTConfig().save_pretrained(tmp_path)
    scheduler = create_scheduler(
        model=str(tmp_path), skip_tokenizer_init=True, num_speculative_tokens=4
    )
    requests = create_requests(3, num_tokens=2, req_ids=["bad", "good", "plain"])
    for request in requests[:2]:
        request.structured_output_request = StructuredOutputRequest(
            params=StructuredOutputsParams(json_object=True), _grammar=_grammar()
        )
    for request in requests:
        scheduler.add_request(request)
    manager = scheduler.structured_output_manager
    manager._grammar_bitmask = torch.full((15, 1), -99, dtype=torch.int32)
    return scheduler, requests


def _model_output(scheduled):
    ids = list(scheduled.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=[[123] for _ in ids],
    )


def _check_isolated(scheduler, requests, scheduled):
    bad, good, plain = requests
    result = scheduler.update_from_output(scheduled, _model_output(scheduled))
    by_id = {o.request_id: o for batch in result.values() for o in batch.outputs}
    assert by_id["bad"].finish_reason == FinishReason.ERROR
    assert bad.status == RequestStatus.FINISHED_ERROR
    assert "bad" not in scheduler.requests
    assert bad not in scheduler.running
    assert bad.num_in_flight_tokens == 0
    assert by_id["good"].new_token_ids == by_id["plain"].new_token_ids == [123]
    assert by_id["good"].finish_reason is None
    assert by_id["plain"].finish_reason is None
    assert list(good.output_token_ids) == list(plain.output_token_ids) == [123]
    # Blocks were released and the next scheduling/grammar round still works.
    assert (
        "bad"
        not in scheduler.kv_cache_manager.coordinator.single_type_managers[
            0
        ].req_to_blocks
    )
    next_step = scheduler.schedule()
    assert "bad" not in next_step.num_scheduled_tokens
    scheduler.get_grammar_bitmask(next_step)
    scheduler.update_from_output(next_step, _model_output(next_step))
    assert list(good.output_token_ids) == list(plain.output_token_ids) == [123, 123]


@pytest.mark.parametrize(
    "failure", ["reject", "accept", "fill", "rollback", "terminated"]
)
def test_runtime_mask_failure_is_request_scoped(failure, tmp_path):
    scheduler, requests = _batch(tmp_path)
    bad, good, _ = requests
    grammar = bad.structured_output_request.grammar
    if failure == "reject":
        grammar.accept_tokens.return_value = False
    else:
        method = {
            "accept": "accept_tokens",
            "fill": "fill_bitmask",
            "rollback": "rollback",
            "terminated": "is_terminated",
        }[failure]
        getattr(grammar, method).side_effect = RuntimeError(f"injected {failure}")
    scheduled = scheduler.schedule()
    # Exercise several temporary advances and row offsets, as in MTP4.
    scheduled.scheduled_spec_decode_tokens = {"bad": [3, 4], "good": [5]}
    masks = scheduler.get_grammar_bitmask(scheduled)
    assert masks.grammar_bitmask.shape == (5, 1)
    assert (masks.grammar_bitmask[3:] == 7).all()
    good.structured_output_request.grammar.rollback.assert_called_once_with(1)
    _check_isolated(scheduler, requests, scheduled)


def test_parallel_fill_failure_is_request_scoped(tmp_path):
    scheduler, requests = _batch(tmp_path)
    manager = scheduler.structured_output_manager
    manager.vllm_config.speculative_config = None
    manager.fill_bitmask_parallel_threshold = 0
    manager.fill_bitmask_parallel_batch_size = 1
    requests[
        0
    ].structured_output_request.grammar.fill_bitmask.side_effect = RuntimeError(
        "injected parallel fill"
    )
    with ThreadPoolExecutor(2) as pool:
        manager.executor_for_fillmask = pool
        scheduled = scheduler.schedule()
        masks = scheduler.get_grammar_bitmask(scheduled)
        assert (masks.grammar_bitmask[1] == 7).all()
        _check_isolated(scheduler, requests, scheduled)


@pytest.mark.parametrize("in_output", [False, True])
def test_draft_validation_failure_is_request_scoped(in_output, tmp_path):
    scheduler, requests = _batch(tmp_path)
    scheduled = scheduler.schedule()
    requests[
        0
    ].structured_output_request.grammar.validate_tokens.side_effect = RuntimeError(
        "injected validation"
    )
    drafts = DraftTokenIds(req_ids=["bad", "good"], draft_token_ids=[[3, 4], [5]])
    if in_output:
        scheduled.scheduled_spec_decode_tokens = {"bad": [0, 0], "good": [0]}
        scheduler.update_draft_token_ids_in_output(drafts, scheduled)
    else:
        scheduler.update_draft_token_ids(drafts)
    scheduler.get_grammar_bitmask(scheduled)
    _check_isolated(scheduler, requests, scheduled)


def test_sampled_token_accept_exception_is_request_scoped(tmp_path):
    scheduler, requests = _batch(tmp_path)
    scheduled = scheduler.schedule()
    scheduler.get_grammar_bitmask(scheduled)
    requests[
        0
    ].structured_output_request.grammar.accept_tokens.side_effect = RuntimeError(
        "injected sampled accept"
    )
    _check_isolated(scheduler, requests, scheduled)
