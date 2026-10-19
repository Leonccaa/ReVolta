# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A request resumed through the prefix cache must not be served KV that
another request computed for a different image.

The cache layout is the production Qwen3.8 Flash-Next shape used by
``test_scheduler_kv_capacity``: QSA main KV sharded over DCP2 (1632-token
global blocks), a QSA key ring of 8 tokens (MTP4), and Mamba groups in align
mode. Block hashes are therefore computed every 8 tokens. Two requests share
every token and differ only in the first of two one-token images that sit in
the last partial 8-token block of the prompt; output tokens complete that
block, so it is hashed incrementally.
"""

import time
from dataclasses import replace

import pytest

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .test_scheduler_kv_capacity import POOL_BLOCK_BYTES, SPAN, make_config, make_specs
from .utils import EOS_TOKEN_ID

pytestmark = pytest.mark.cpu_test

# The prompt ends two tokens before the first 1632-token boundary, and both
# images are in its last partial 8-token block [1624, 1632).
PROMPT_LEN = SPAN - 2
IMAGE_OFFSETS = (SPAN - 8, SPAN - 5)
OUTPUT_TOKEN = 11
NUM_BLOCKS = 400


def make_scheduler() -> tuple[Scheduler, int]:
    # Production scheduler knobs (TP4, DCP2, MTP4, Mamba align, prefix
    # caching) on the cached text-only OPT config, as in the capacity tests.
    config = make_config(max_model_len=4096)
    config.cache_config.block_size = SPAN

    # With MTP4 the QSA key ring holds 4 + 4 rows, rounded to whole groups.
    specs = {
        name: replace(spec, block_size=8) if type(spec) is CircularBufferSpec else spec
        for name, spec in make_specs().items()
    }
    (worker_cache,) = get_kv_cache_configs(config, [specs], [64 * POOL_BLOCK_BYTES])
    kv_cache_config = generate_scheduler_kv_cache_config([worker_cache])
    block_size, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, config)
    assert (block_size, hash_block_size) == (SPAN, 8)
    kv_cache_config.num_blocks = NUM_BLOCKS
    config.cache_config.num_gpu_blocks = NUM_BLOCKS
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_block_size,
    )
    assert scheduler.use_eagle
    # The OPT config has no encoder budget; give the scheduler one so it can
    # schedule the two one-token images (the cache path under test does not
    # depend on the model's encoder).
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=64)
    return scheduler, hash_block_size


def make_request(request_id: str, first_image: str, hash_block_size: int):
    sampling_params = SamplingParams(max_tokens=1700, ignore_eos=True)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="image",
            identifier=identifier,
            mm_position=PlaceholderRange(offset=offset, length=1),
        )
        for identifier, offset in zip((first_image, "shared"), IMAGE_OFFSETS)
    ]
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(PROMPT_LEN)),
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=features,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def step(scheduler: Scheduler) -> None:
    output = scheduler.schedule()
    ids = list(output.num_scheduled_tokens)
    if not ids:
        return
    # Every request "generates" the same token, so the two requests keep
    # sharing every token while they decode.
    sampled = [
        [OUTPUT_TOKEN]
        if scheduler.requests[r].num_computed_tokens >= scheduler.requests[r].num_tokens
        else []
        for r in ids
    ]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def run_until(scheduler: Scheduler, done, limit: int = 5000) -> None:
    for _ in range(limit):
        if done():
            return
        step(scheduler)
    raise AssertionError("scheduler made no progress")


def test_resume_does_not_reuse_kv_of_a_different_earlier_image():
    init_none_hash(sha256)
    scheduler, hash_block_size = make_scheduler()
    req_a = make_request("A", "red", hash_block_size)
    req_b = make_request("B", "green", hash_block_size)
    assert req_a.block_hashes == req_b.block_hashes  # no image in full blocks

    # A runs alone until it has cached two 1632-token blocks, then finishes;
    # its blocks stay in the prefix cache.
    a_blocks: list[int] = []
    scheduler.add_request(req_a)

    def a_done():
        if req_a.num_computed_tokens >= 2 * SPAN and not a_blocks:
            a_blocks.extend(scheduler.kv_cache_manager.get_block_ids("A")[0])
        return req_a.is_finished()

    run_until(scheduler, a_done)
    assert a_blocks

    # B decodes the same tokens past the second boundary (MTP drops one
    # matched block, so a 1632-token hit needs two matched blocks).
    scheduler.add_request(req_b)
    run_until(scheduler, lambda: req_b.num_tokens >= 2 * SPAN + 2)
    assert req_b.mm_features[0].identifier != req_a.mm_features[0].identifier

    def preempt_b():
        # Preempt B the way the scheduler does when it runs out of KV blocks.
        # A request preempted inside schedule() is not part of that step, so
        # it is not in the previous step's scheduled set when it resumes.
        scheduler.running.remove(req_b)
        scheduler._preempt_request(req_b, time.monotonic())
        scheduler.prev_step_scheduled_req_ids.discard("B")
        assert req_b.status == RequestStatus.PREEMPTED

    lookups = []
    get_computed_blocks = scheduler.kv_cache_manager.get_computed_blocks

    def traced(request):
        blocks, num_hit = get_computed_blocks(request)
        lookups.append((request.request_id, num_hit, blocks.get_block_ids()[0]))
        return blocks, num_hit

    scheduler.kv_cache_manager.get_computed_blocks = traced

    # First resume: no Mamba state was kept at the 1632 boundary while B
    # decoded across it, so nothing hits yet. B recomputes its first
    # 1632-token chunk, which keeps that state.
    preempt_b()
    step(scheduler)
    assert lookups == [("B", 0, [])]
    assert req_b.num_computed_tokens == SPAN

    # B is preempted again (e.g. KV pressure persists) and resumes through
    # the prefix cache.
    preempt_b()
    step(scheduler)
    (request_id, num_hit, hit_blocks) = lookups[1]
    assert request_id == "B"
    # The resume is served from the prefix cache (the test is not vacuous).
    assert num_hit == SPAN
    # The block holding B's first image must be B's own KV, never the block
    # A computed for its different image.
    assert not set(hit_blocks) & set(a_blocks)
