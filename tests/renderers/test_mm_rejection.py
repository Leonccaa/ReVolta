# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.utils import get_max_tokens
from vllm.multimodal.cache import MultiModalProcessorSenderCache
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.renderers.base import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


@pytest.fixture
def renderer():
    config = SimpleNamespace(get_multimodal_config=lambda: MultiModalConfig())
    cache = MultiModalProcessorSenderCache(config)
    item = MultiModalKwargsItem.dummy(16)

    class Renderer(BaseRenderer):
        def _process_multimodal(self, prompt, data, **kwargs):
            sent, _ = cache.get_and_update_item((item, []), data["image"])
            return {
                "type": "multimodal",
                "prompt_token_ids": prompt,
                "mm_kwargs": {"image": [sent]},
                "mm_hashes": {"image": [data["image"]]},
                "mm_placeholders": {},
            }

    with (
        patch.object(Renderer, "__abstractmethods__", set()),
        patch(
            "vllm.renderers.base.mm_registry.supports_multimodal_inputs",
            return_value=False,
        ),
    ):
        r = Renderer(
            SimpleNamespace(
                model_config=SimpleNamespace(renderer_num_workers=1),
                parallel_config=SimpleNamespace(_api_process_rank=0),
            ),
            None,
        )
    r.mm_processor = SimpleNamespace(cache=cache)
    yield r
    r.shutdown()


async def render(renderer, image="A"):
    return await renderer.process_for_engine_async(
        {"prompt_token_ids": [1, 2, 3], "multi_modal_data": {"image": image}}, 0
    )


@pytest.mark.parametrize("mode", ["length", "error_response", "cancelled"])
def test_serving_rejection_resends_image_on_first_retry(renderer, mode):
    serving = object.__new__(OpenAIServing)
    serving.renderer = renderer
    serving.has_kv_connector = False
    request = SimpleNamespace(kv_transfer_params=None)

    async def reject():
        assert (await render(renderer))["mm_kwargs"]["image"][0] is not None
        if mode == "error_response":
            return OpenAIServing.create_error_response("too long")
        if mode == "cancelled":
            raise asyncio.CancelledError
        get_max_tokens(2, 1, 3, {})

    async def run():
        wrapped = serving._with_kv_transfer_rejection_cleanup(reject(), request, None)
        if mode == "error_response":
            assert isinstance(await wrapped, ErrorResponse)
        else:
            error = asyncio.CancelledError if mode == "cancelled" else ValueError
            with pytest.raises(error):
                await wrapped
        assert (await render(renderer))["mm_kwargs"]["image"][0] is not None
        assert (await render(renderer))["mm_kwargs"]["image"][0] is None

    asyncio.run(run())


def test_input_validation_rejection_resends_image_on_first_retry(renderer):
    llm = object.__new__(AsyncLLM)
    llm.output_handler = None
    llm.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False), shutdown=lambda **kw: None
    )
    llm.renderer = renderer
    llm.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )

    def reject(*args, **kwargs):
        raise ValueError("invalid sampling parameters")

    llm.input_processor = SimpleNamespace(process_inputs=reject)

    async def supported():
        return ("generate",)

    llm.get_supported_tasks = supported

    async def run():
        prompt = await render(renderer)
        with pytest.raises(ValueError, match="invalid sampling"):
            await llm.add_request("rejected", prompt, SamplingParams(max_tokens=1))
        assert (await render(renderer))["mm_kwargs"]["image"][0] is not None

    asyncio.run(run())


def test_successful_render_keeps_sender_hit(renderer):
    serving = object.__new__(OpenAIServing)
    serving.renderer = renderer
    serving.has_kv_connector = False

    async def run():
        async def accept():
            await render(renderer)
            return "accepted"

        assert (
            await serving._with_kv_transfer_rejection_cleanup(
                accept(), SimpleNamespace(kv_transfer_params=None), None
            )
            == "accepted"
        )
        assert (await render(renderer))["mm_kwargs"]["image"][0] is None

    asyncio.run(run())


def test_concurrent_rejection_only_invalidates_its_own_items(renderer):
    serving = object.__new__(OpenAIServing)
    serving.renderer = renderer
    serving.has_kv_connector = False
    request = SimpleNamespace(kv_transfer_params=None)

    async def run():
        ready = asyncio.Event()
        accepted = asyncio.Event()

        async def reject_a():
            await render(renderer, "A")
            ready.set()
            await accepted.wait()
            raise ValueError("rejected A")

        async def accept_b():
            await ready.wait()
            await render(renderer, "B")
            accepted.set()
            return "accepted B"

        a: object
        b: object
        a, b = await asyncio.gather(
            serving._with_kv_transfer_rejection_cleanup(reject_a(), request, None),
            serving._with_kv_transfer_rejection_cleanup(accept_b(), request, None),
            return_exceptions=True,
        )
        assert isinstance(a, ValueError) and b == "accepted B"
        assert (await render(renderer, "B"))["mm_kwargs"]["image"][0] is None
        assert (await render(renderer, "A"))["mm_kwargs"]["image"][0] is not None

    asyncio.run(run())


def test_text_rejection_keeps_existing_media_cache(renderer):
    serving = object.__new__(OpenAIServing)
    serving.renderer = renderer
    serving.has_kv_connector = False

    async def run():
        await render(renderer)

        async def reject_text():
            await renderer.process_for_engine_async({"prompt_token_ids": [1]}, 0)
            raise ValueError("rejected text")

        with pytest.raises(ValueError):
            await serving._with_kv_transfer_rejection_cleanup(
                reject_text(), SimpleNamespace(kv_transfer_params=None), None
            )
        assert (await render(renderer))["mm_kwargs"]["image"][0] is None

    asyncio.run(run())


def test_rejection_preserves_kv_connector_notification(renderer):
    serving = object.__new__(OpenAIServing)
    serving.renderer = renderer
    serving.has_kv_connector = True
    notifications = []

    async def notify(request_id, params, **kwargs):
        notifications.append(request_id)

    serving.engine_client = SimpleNamespace(notify_kv_transfer_request_rejected=notify)

    async def run():
        async def reject():
            await render(renderer)
            raise ValueError("rejected")

        request = SimpleNamespace(
            request_id="rejected", kv_transfer_params={"do_remote_prefill": True}
        )
        with pytest.raises(ValueError):
            await serving._with_kv_transfer_rejection_cleanup(reject(), request, None)
        assert notifications == ["rejected"]
        assert (await render(renderer))["mm_kwargs"]["image"][0] is not None

    asyncio.run(run())
