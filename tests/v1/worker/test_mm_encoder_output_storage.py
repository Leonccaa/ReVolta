# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for per-item encoder-cache storage and source lifetimes."""

import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from transformers import BatchFeature

from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

pytestmark = pytest.mark.skip_global_cleanup


def mm_inputs(modality, sizes):
    items = MultiModalKwargsItems.from_hf_inputs(
        BatchFeature({"sizes": torch.tensor(sizes)}),
        {"sizes": MultiModalFieldConfig.batched(modality)},
    )
    return [(modality, item) for item in items[modality]]


def values(tokens, dtype):
    tensor = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 4)
    # Include signed zero, infinity and NaN; compare their bits below.
    tensor[0] = torch.tensor([0.0, -0.0, float("inf"), float("nan")])
    return tensor.to(dtype)


def assert_same_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def storage_bytes(cache):
    storages = {
        out.untyped_storage().data_ptr(): out.untyped_storage() for out in cache
    }
    return sum(storage.nbytes() for storage in storages.values())


class Outputs:
    def __init__(self, kind="split", dtype=torch.float16, check_lifetime=False):
        self.kind = kind
        self.dtype = dtype
        self.check_lifetime = check_lifetime
        self.sources = []
        self.expected = []
        self.originals = []
        self.calls = 0

    def __call__(self, sizes):
        if self.check_lifetime:
            assert all(source.expired() for source in self.sources), (
                "Previous batch source is still alive during the next encoder call"
            )
        self.calls += 1
        sizes = sizes.tolist()
        source = values(sum(sizes) * 4, self.dtype)
        if self.kind == "tensor":
            assert len(set(sizes)) == 1
            outputs = source[: sum(sizes)].reshape(len(sizes), sizes[0], 4)
        elif self.kind == "strided":
            outputs = source[::4].split(sizes)
        elif self.kind == "padded":
            outputs = source[: sum(sizes)].split(sizes)
        elif self.kind == "split":
            source = values(sum(sizes), self.dtype)
            outputs = source.split(sizes)
        elif self.kind in ("owned", "owned_transposed"):
            outputs = [values(size, self.dtype) for size in sizes]
            if self.kind == "owned_transposed":
                outputs = [out.t().contiguous().t() for out in outputs]
        elif self.kind == "mixed":
            outputs = [*source[: sum(sizes[:-1])].split(sizes[:-1])]
            outputs.append(values(sizes[-1], self.dtype))
        else:
            raise AssertionError(self.kind)

        self.expected.extend(out.clone() for out in outputs)
        self.originals.extend(weakref.ref(out) for out in outputs)
        self.sources.append(StorageWeakRef(source.untyped_storage()))
        return outputs


def make_runner(kind, outputs, *, sequential=False, graph=False):
    cache = EncoderCache()
    if kind == "mrv2":
        runner = EncoderRunner(
            SimpleNamespace(embed_multimodal=outputs),
            max_num_tokens=1,
            hidden_size=4,
            encoder_cache=cache,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )

        def run(inputs, hashes):
            result = runner.execute_mm_encoder(inputs)
            cache.encoder_outputs.update(zip(hashes, result))
            return result

    else:
        graph_manager = (
            SimpleNamespace(
                supports_modality=lambda _: True, execute=lambda kw: outputs(**kw)
            )
            if graph
            else None
        )
        runner = SimpleNamespace(
            device=torch.device("cpu"),
            pin_memory=False,
            model=SimpleNamespace(embed_multimodal=outputs),
            encoder_cache=cache.encoder_outputs,
            observability_config=None,
            lora_config=None,
            is_multimodal_pruning_enabled=False,
            requires_sequential_video_encoding=sequential,
            encoder_cudagraph_manager=graph_manager,
            timed_encoder_operation=lambda *args: nullcontext(),
            maybe_save_ec_to_connector=lambda *args: None,
        )

        def run(inputs, hashes):
            runner._batch_mm_inputs_from_scheduler = lambda _: (hashes, inputs, [])
            return gpu_model_runner.GPUModelRunner._execute_mm_encoder(
                runner, SimpleNamespace(scheduled_encoder_inputs={})
            )

    return run, cache


@pytest.fixture(autouse=True)
def disable_peak_logging(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_LOG_MEMORY_PEAKS", "0")


@pytest.mark.parametrize("runner_kind", ["mrv1", "mrv2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "output_kind", ["split", "tensor", "padded", "strided", "mixed"]
)
def test_cached_items_own_storage_and_keep_output_bits(runner_kind, dtype, output_kind):
    outputs = Outputs(output_kind, dtype)
    run, cache = make_runner(runner_kind, outputs)
    sizes = {"tensor": [2, 2], "padded": [2], "mixed": [4, 2, 3]}.get(
        output_kind, [6, 2]
    )
    hashes = [f"item-{i}" for i in range(len(sizes))]
    result = run(mm_inputs("image", sizes), hashes)

    for out, expected in zip(result, outputs.expected):
        assert_same_bits(out, expected)
        assert out.untyped_storage().nbytes() == out.nbytes
    assert len({out.untyped_storage().data_ptr() for out in result}) == len(result)
    assert storage_bytes(cache.encoder_outputs.values()) == sum(
        out.nbytes for out in result
    )
    if output_kind == "mixed":
        assert result[-1] is outputs.originals[-1]()

    # The return value must not keep evicted entries alive in this check.
    del result
    cache.free_encoder_cache(hashes[0])
    remaining = list(cache.encoder_outputs.values())
    assert storage_bytes(remaining) == sum(out.nbytes for out in remaining)


@pytest.mark.parametrize("runner_kind", ["mrv1", "mrv2"])
@pytest.mark.parametrize("layout", ["owned", "owned_transposed"])
def test_owned_outputs_are_reused_without_copy(runner_kind, layout):
    outputs = Outputs(layout)
    run, _ = make_runner(runner_kind, outputs)
    result = run(mm_inputs("image", [3, 2]), ["a", "b"])
    assert all(out is original() for out, original in zip(result, outputs.originals))
    for out, expected in zip(result, outputs.expected):
        assert_same_bits(out, expected)
    if layout == "owned_transposed":
        assert all(not out.is_contiguous() for out in result)


@pytest.mark.parametrize("runner_kind", ["mrv1", "mrv2"])
@pytest.mark.parametrize("output_kind", ["split", "tensor", "padded"])
def test_source_released_before_next_modality_batch(runner_kind, output_kind):
    outputs = Outputs(output_kind, check_lifetime=True)
    run, _ = make_runner(runner_kind, outputs)
    inputs = [
        item
        for modality in ["image", "video", "image"]
        for item in mm_inputs(modality, [2, 2])
    ]
    result = run(inputs, [f"item-{i}" for i in range(len(inputs))])
    assert outputs.calls == 3
    assert all(source.expired() for source in outputs.sources)
    for out, expected in zip(result, outputs.expected):
        assert_same_bits(out, expected)


@pytest.mark.parametrize("runner_kind", ["mrv1", "mrv2"])
@pytest.mark.parametrize("output_kind", ["split", "padded"])
def test_partial_eviction_does_not_accumulate_batch_storage(runner_kind, output_kind):
    outputs = Outputs(output_kind)
    run, cache = make_runner(runner_kind, outputs)
    for step in range(30):
        big, small = f"big-{step}", f"small-{step}"
        run(mm_inputs("image", [6, 2]), [big, small])
        cache.free_encoder_cache(big)
        retained = list(cache.encoder_outputs.values())
        logical = sum(out.nbytes for out in retained)
        assert logical == (step + 1) * 2 * 4 * 2
        assert storage_bytes(retained) == logical


@pytest.mark.parametrize("sequential,graph", [(True, False), (False, True)])
def test_mrv1_releases_sequential_and_graph_output_sources(sequential, graph):
    outputs = Outputs("padded", check_lifetime=True)
    run, _ = make_runner("mrv1", outputs, sequential=sequential, graph=graph)
    inputs = mm_inputs("video", [2, 2, 2]) + mm_inputs("image", [2])
    result = run(inputs, [f"item-{i}" for i in range(len(inputs))])
    assert outputs.calls == (4 if sequential else 2)
    assert all(source.expired() for source in outputs.sources)
    for out, expected in zip(result, outputs.expected):
        assert_same_bits(out, expected)


@pytest.mark.parametrize("runner_kind", ["mrv1", "mrv2"])
def test_no_multimodal_inputs_skip_encoder(runner_kind):
    outputs = Outputs()
    run, cache = make_runner(runner_kind, outputs)
    assert run([], []) == []
    assert outputs.calls == 0
    assert cache.encoder_outputs == {}
