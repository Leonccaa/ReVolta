# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery from P0/P1 multi-modal processor cache drift (cf. upstream #46747).

P0 (frontend) keeps a metadata-only shadow of the P1 (engine) cache and sends
no data for items it believes P1 has. A request that P0 rendered but that
never reached P1 (e.g. rejected as too long once its images are expanded)
leaves the shadow claiming items P1 never received; every later request with
those items then failed in P1 until P0 happened to evict them.
"""

import asyncio
import queue
from functools import partial
from types import SimpleNamespace

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal import cache as mm_cache
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc

pytestmark = pytest.mark.cpu_test


class _StubModelConfig:
    def __init__(self, mm_processor_cache_gb: float = 1) -> None:
        self._mm_config = MultiModalConfig(mm_processor_cache_gb=mm_processor_cache_gb)

    def get_multimodal_config(self) -> MultiModalConfig:
        return self._mm_config


def _feature(data: MultiModalKwargsItem | None, mm_hash: str) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=data,
        modality="image",
        identifier=mm_hash,
        mm_position=PlaceholderRange(offset=0, length=4),
        mm_hash=mm_hash,
    )


# ------------------------------ P0 / P1 caches ------------------------------ #
def test_rejected_request_drift_raises_then_recovers_after_invalidate():
    config = _StubModelConfig()
    p0 = mm_cache.MultiModalProcessorSenderCache(config)  # type: ignore[arg-type]
    p1 = mm_cache.MultiModalReceiverCache(config)  # type: ignore[arg-type]
    item = MultiModalKwargsItem.dummy(64)
    mm_hash = "image-A"

    # The first request is rendered (P0 records the item and sends its data)
    # but is rejected before it reaches P1.
    sent, _ = p0.get_and_update_item((item, []), mm_hash)
    assert sent is item

    # The retry is a P0 shadow hit, so it carries no data, and P1 never got
    # the item: P1 reports the drift instead of asserting.
    retry, _ = p0.get_and_update_item((item, []), mm_hash)
    assert retry is None
    with pytest.raises(mm_cache.MultiModalCacheMissError) as excinfo:
        p1.get_and_update_features([_feature(retry, mm_hash)])
    assert excinfo.value.mm_hashes == [mm_hash]

    # The frontend drops the stale shadow entry; the next retry sends data.
    p0.invalidate(mm_hash)
    resent, _ = p0.get_and_update_item((item, []), mm_hash)
    assert resent is item
    (feature,) = p1.get_and_update_features([_feature(resent, mm_hash)])
    assert feature.data is item

    # From then on the caches agree again: no data is sent, P1 supplies it.
    hit, _ = p0.get_and_update_item((item, []), mm_hash)
    assert hit is None
    (feature,) = p1.get_and_update_features([_feature(hit, mm_hash)])
    assert feature.data is item


def test_cache_miss_reports_every_drifted_item_of_a_request():
    config = _StubModelConfig()
    p1 = mm_cache.MultiModalReceiverCache(config)  # type: ignore[arg-type]
    present = MultiModalKwargsItem.dummy(16)
    p1.get_and_update_features([_feature(present, "B")])

    features = [_feature(None, "A"), _feature(None, "B"), _feature(None, "C")]
    with pytest.raises(mm_cache.MultiModalCacheMissError) as excinfo:
        p1.get_and_update_features(features)
    assert excinfo.value.mm_hashes == ["A", "C"]
    assert features[1].data is present


def test_invalidate_is_a_noop_for_caches_without_a_p1_shadow():
    config = _StubModelConfig()
    cache = mm_cache.MultiModalProcessorOnlyCache(config)  # type: ignore[arg-type]
    cache.invalidate("unknown")  # must not raise


# -------------------------------- EngineCore -------------------------------- #
def test_engine_core_fails_the_request_on_a_cache_miss():
    engine = SimpleNamespace(output_queue=queue.Queue())
    engine._send_finish_outputs_to_client = partial(
        EngineCoreProc._send_finish_outputs_to_client, engine
    )
    engine._send_error_outputs_to_client = partial(
        EngineCoreProc._send_error_outputs_to_client, engine
    )
    request = SimpleNamespace(request_id="req-1", client_index=3)
    EngineCoreProc._handle_mm_cache_miss(
        engine,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        mm_cache.MultiModalCacheMissError(["A"]),
    )
    client_index, outputs = engine.output_queue.get_nowait()
    assert client_index == 3
    (output,) = outputs.outputs
    assert output.request_id == "req-1"
    assert output.new_token_ids == []
    assert output.finish_reason == FinishReason.ERROR


# --------------------------------- AsyncLLM --------------------------------- #
class _FakeEngineCore:
    def __init__(self) -> None:
        self.pending: list[EngineCoreOutputs] = []
        self.added: list[str] = []

    async def add_request_async(self, request) -> None:
        self.added.append(request.request_id)

    async def abort_requests_async(self, request_ids) -> None:
        pass

    async def get_output_async(self) -> EngineCoreOutputs:
        while not self.pending:
            await asyncio.sleep(0)
        return self.pending.pop(0)

    def shutdown(self, timeout=None) -> None:
        pass


class _FakeOutputProcessor:
    def __init__(self, events: list) -> None:
        self.events = events

    def add_request(self, request, prompt, parent_req, index, queue) -> None:
        pass

    def abort_requests(self, request_ids, internal) -> list[str]:
        return list(request_ids)

    def process_outputs(self, outputs, timestamp, iteration_stats):
        self.events.append(("process", [o.request_id for o in outputs]))
        return SimpleNamespace(request_outputs=[], reqs_to_abort=[])

    def update_scheduler_stats(self, scheduler_stats) -> None:
        pass

    def propagate_error(self, error) -> None:
        self.events.append(("error", [repr(error)]))


class _FakeRenderer:
    mm_processor_cache = object()

    def __init__(self, events: list) -> None:
        self.events = events

    def invalidate_mm_cache_items_async(self, mm_hashes) -> asyncio.Future:
        # Like the real renderer, the work is submitted when this is called.
        self.events.append(("invalidate", list(mm_hashes)))
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future

    def stat_mm_cache(self):
        return None

    def shutdown(self) -> None:
        pass


def _make_async_llm(events: list) -> AsyncLLM:
    llm = object.__new__(AsyncLLM)
    llm.output_handler = None
    llm.engine_core = _FakeEngineCore()
    llm.output_processor = _FakeOutputProcessor(events)
    llm.renderer = _FakeRenderer(events)
    llm.log_stats = False
    llm.log_requests = False
    llm.logger_manager = None
    llm._mm_shadow_hits = {}
    return llm


def _mm_request(request_id: str) -> SimpleNamespace:
    # "A" was a sender-cache hit (no data sent); "B" was sent with data.
    return SimpleNamespace(
        request_id=request_id,
        mm_features=[
            _feature(None, "A"),
            _feature(MultiModalKwargsItem.dummy(4), "B"),
        ],
    )


def _finish(request_id: str, reason: FinishReason) -> EngineCoreOutputs:
    return EngineCoreOutputs(
        finished_requests={request_id},
        outputs=[EngineCoreOutput(request_id, [], finish_reason=reason)],
    )


async def _drain(llm: AsyncLLM, events: list, num_processed: int) -> None:
    llm._run_output_handler()
    for _ in range(1000):
        if sum(kind == "process" for kind, _ in events) >= num_processed:
            break
        await asyncio.sleep(0)
    llm.output_handler.cancel()


def test_async_llm_drops_sent_without_data_items_of_a_failed_request():
    """When the engine fails a request that sent items without data, the
    frontend forgets them before the failure reaches the client, so the
    client's retry is rendered with the data."""
    events: list = []
    llm = _make_async_llm(events)

    async def run() -> None:
        await llm._add_request(_mm_request("req-1"), None, None, 0, None)
        assert llm._mm_shadow_hits == {"req-1": ["A"]}
        llm.engine_core.pending.append(_finish("req-1", FinishReason.ERROR))
        await _drain(llm, events, 1)

    asyncio.run(run())
    assert events == [("invalidate", ["A"]), ("process", ["req-1"])]
    assert llm._mm_shadow_hits == {}


def test_async_llm_keeps_the_cache_when_requests_succeed_or_abort():
    events: list = []
    llm = _make_async_llm(events)

    async def run() -> None:
        await llm._add_request(_mm_request("ok"), None, None, 0, None)
        await llm._add_request(_mm_request("aborted"), None, None, 0, None)
        assert set(llm._mm_shadow_hits) == {"ok", "aborted"}
        await llm.abort("aborted", internal=True)
        assert set(llm._mm_shadow_hits) == {"ok"}
        llm.engine_core.pending.append(_finish("ok", FinishReason.STOP))
        await _drain(llm, events, 1)

    asyncio.run(run())
    assert events == [("process", ["ok"])]
    assert llm._mm_shadow_hits == {}
