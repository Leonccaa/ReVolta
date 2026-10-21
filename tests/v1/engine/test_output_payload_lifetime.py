# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Payload ownership regressions for vllm-project/vllm#50053 (Nick Hill)."""

from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import zmq

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


@pytest.mark.parametrize("request_id_size", [16, zmq.COPY_THRESHOLD - 1, 2**18])
@pytest.mark.parametrize("tensor_rows", [0, 24, 8192])
def test_output_thread_preserves_payload_and_logprobs(request_id_size, tensor_rows):
    """Queue messages before the reader drains, exercising the actual send loop."""
    proc = object.__new__(EngineCoreProc)
    proc.output_queue = Queue()
    expected = []
    for i in range(8):
        tensors = None
        if tensor_rows:
            tensors = LogprobsTensors(
                torch.full((tensor_rows, 2), i, dtype=torch.int64),
                torch.full((tensor_rows, 2), -float(i), dtype=torch.float32),
                torch.ones(tensor_rows, dtype=torch.int32),
            )
        message = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id=f"{i}:" + "r" * request_id_size,
                    new_token_ids=[i],
                    new_prompt_logprobs_tensors=tensors,
                )
            ]
        )
        expected.append(message)
        proc.output_queue.put((0, message))
    proc.output_queue.put(EngineCoreProc.ENGINE_CORE_DEAD)
    if request_id_size >= zmq.COPY_THRESHOLD - 1:
        assert len(MsgpackEncoder().encode(expected[0])[0]) >= zmq.COPY_THRESHOLD

    # The production context creates PUSH sockets. Inproc keeps delivery queued
    # until recv and makes premature bytearray reuse deterministic on CPU.
    with zmq.Context() as ctx:
        pull = ctx.socket(zmq.PULL)
        pull.bind("inproc://payload-lifetime")
        facade = SimpleNamespace(**dict(vars(zmq), Context=lambda: nullcontext(ctx)))
        with patch("vllm.v1.engine.core.zmq", facade):
            proc.process_output_sockets(["inproc://payload-lifetime"], None, 0)
        decoder = MsgpackDecoder(EngineCoreOutputs)
        for sent in expected:
            got = decoder.decode(pull.recv_multipart(copy=False))
            actual, original = got.outputs[0], sent.outputs[0]
            assert actual.request_id == original.request_id
            assert actual.new_token_ids == original.new_token_ids
            if tensor_rows:
                for left, right in zip(
                    actual.new_prompt_logprobs_tensors,
                    original.new_prompt_logprobs_tensors,
                ):
                    if left is None:
                        assert right is None
                    else:
                        assert torch.equal(left, right)
        assert pull.recv() == EngineCoreProc.ENGINE_CORE_DEAD
        pull.close(linger=0)
