# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from transformers import OPTConfig
from xgrammar import (
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    allocate_token_bitmask,
)

from tests.v1.core.utils import create_scheduler
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


@pytest.mark.parametrize("draft_width", [4, 7])
@pytest.mark.parametrize("async_scheduling", [True, False], ids=["async", "sync"])
def test_non_deferred_drafts_constrain_grammar(draft_width, async_scheduling, tmp_path):
    OPTConfig(architectures=["OPTForCausalLM"]).save_pretrained(tmp_path)
    scheduler = create_scheduler(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        async_scheduling=async_scheduling,
        num_speculative_tokens=None if async_scheduling else draft_width,
        max_num_seqs=2,
        max_num_batched_tokens=64,
    )
    scheduler.use_v2_model_runner = True
    if async_scheduling:
        # The scheduler helper configures CPU n-gram drafts, which cannot run
        # asynchronously. Supply the slot count for the mocked worker instead.
        scheduler.num_spec_tokens = draft_width
        scheduler._spec_token_placeholders = (-1,) * draft_width
    vocab = ["r", "[reasoning end]", "x", "{", "}", " ", "[eos]"]
    reason, marker, invalid, open_brace, close_brace, space, eos = range(len(vocab))
    info = TokenizerInfo(vocab, stop_token_ids=[eos])
    compiled = GrammarCompiler(info).compile_builtin_json_grammar()
    reasoner = SimpleNamespace(
        is_reasoning_end=lambda tokens: marker in tokens,
        is_reasoning_end_streaming=lambda _tokens, delta: marker in delta,
    )
    manager = scheduler.structured_output_manager
    manager.reasoner_cls = lambda _: reasoner
    manager._grammar_bitmask = allocate_token_bitmask(2 * (draft_width + 1), len(vocab))
    req_ids = ["crosses_marker", "after_marker"]
    structured_requests = []
    for req_id in req_ids:
        params = SamplingParams(
            temperature=0,
            max_tokens=24,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )
        params.update_from_generation_config({}, eos)
        request = Request(req_id, [reason] * 8, params, None)
        structured = request.structured_output_request
        assert structured is not None
        structured.grammar = XgrammarGrammar(
            vocab_size=len(vocab),
            matcher=GrammarMatcher(compiled, max_rollback_tokens=draft_width),
            ctx=compiled,
        )
        structured.reasoner = reasoner
        scheduler.add_request(request)
        structured_requests.append(structured)

    prefill: Future[ModelRunnerOutput] = Future()
    prefill.set_result(
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[reason], [marker]],
        )
    )
    # The worker can return drafts in an order different from the scheduler.
    drafts = DraftTokenIds(
        req_ids[::-1],
        [
            [open_brace] + [space] * (draft_width - 1),
            [reason, marker, open_brace] + [invalid] * (draft_width - 3),
        ],
    )
    executor = SimpleNamespace(
        execute_model=Mock(side_effect=lambda *args, **kwargs: Future()),
        sample_tokens=Mock(side_effect=[prefill, Future()]),
        take_draft_token_ids=Mock(return_value=drafts),
    )
    engine = SimpleNamespace(
        batch_queue=deque(maxlen=2),
        batch_queue_size=2,
        _sm70_async_cpu_trace_step=0,
        is_ec_consumer=True,
        is_pooling_model=False,
        scheduler=scheduler,
        async_scheduling=async_scheduling,
        use_spec_decode=True,
        model_executor=executor,
        log_error_detail=lambda _: nullcontext(),
        log_iteration_details=lambda _: nullcontext(),
        _process_aborts_queue=lambda: None,
    )
    _, model_executed = EngineCore.step_with_batch_queue(engine)
    EngineCore.post_step(engine, model_executed)
    assert not engine.batch_queue
    assert all(req.num_output_placeholders == 0 for req in scheduler.requests.values())
    assert [request.reasoning_ended for request in structured_requests] == [False, True]
    executor.take_draft_token_ids.reset_mock()

    EngineCore.step_with_batch_queue(engine)
    _, scheduled, _ = engine.batch_queue[0]
    assert not scheduled.pending_structured_output_tokens
    mask = executor.sample_tokens.call_args.args[0]
    assert mask.structured_output_request_ids == req_ids
    assert mask.grammar_bitmask.shape == (2 * (draft_width + 1), 1)

    # Reference rows at JSON start and after the opening brace. They differ,
    # so this also proves the grammar advances through the actual draft IDs.
    matcher = GrammarMatcher(compiled)
    reference = allocate_token_bitmask(2, len(vocab))
    matcher.fill_next_token_bitmask(reference, 0)
    assert matcher.accept_token(open_brace)
    matcher.fill_next_token_bitmask(reference, 1)
    start, inside_object = reference[:2, 0].tolist()
    assert start & (1 << open_brace) and not start & (1 << invalid)
    assert inside_object & (1 << close_brace) and not inside_object & (1 << invalid)
    np.testing.assert_array_equal(
        mask.grammar_bitmask[:, 0],
        [-1, -1, start]
        + [inside_object] * (draft_width - 2)
        + [start]
        + [inside_object] * draft_width,
    )
    assert executor.take_draft_token_ids.call_count == int(async_scheduling)
