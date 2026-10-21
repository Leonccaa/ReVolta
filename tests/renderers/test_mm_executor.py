# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

from vllm.renderers.base import BaseRenderer


def test_mm_preprocessing_does_not_block_tokenization_and_keeps_cache_order():
    events = []
    started = threading.Event()
    release = threading.Event()
    cache = SimpleNamespace(
        invalidate=lambda key: events.append(("invalidate", key)),
        clear_cache=lambda: events.append(("clear", None)),
        close=lambda: None,
    )

    class Renderer(BaseRenderer):
        def _process_multimodal(self, *args, **kwargs):
            events.append(("process", None))
            started.set()
            assert release.wait(5)

    # No model or tokenizer download; initialize the real executor wiring.
    with (
        patch.object(Renderer, "__abstractmethods__", set()),
        patch(
            "vllm.renderers.base.mm_registry.supports_multimodal_inputs",
            return_value=False,
        ),
    ):
        renderer = Renderer(
            SimpleNamespace(
                model_config=SimpleNamespace(renderer_num_workers=1),
                parallel_config=SimpleNamespace(_api_process_rank=0),
            ),
            None,
        )
    renderer.mm_processor = SimpleNamespace(cache=cache)

    async def run():
        processing = renderer._process_multimodal_async()
        try:
            assert await asyncio.to_thread(started.wait, 2)
            invalidate = renderer.invalidate_mm_cache_items_async(["A"])
            clear = asyncio.create_task(renderer.clear_mm_cache_async())
            # A blocked MM task must not occupy the tokenizer's worker.
            probe = asyncio.get_running_loop().run_in_executor(
                renderer._executor, lambda: "tokenized"
            )
            assert await asyncio.wait_for(probe, 0.5) == "tokenized"
            assert events == [("process", None)]
        finally:
            release.set()
            await processing
            await invalidate
            await clear
        assert events == [("process", None), ("invalidate", "A"), ("clear", None)]

    try:
        asyncio.run(run())
    finally:
        release.set()
        renderer.shutdown()
