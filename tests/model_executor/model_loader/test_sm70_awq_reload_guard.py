# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject reload before any checkpoint views or runtime pointers change."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
from vllm.model_executor.layers.quantization.awq_sm70_moe import AWQSM70MoEMethod
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

pytestmark = pytest.mark.skip_global_cleanup


def _model(kind):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    if kind == "dense":
        model[1]._awq_sm70_prepared = True
    elif kind == "moe":
        model[1].quant_method = object.__new__(AWQSM70MoEMethod)
    elif kind == "wrapped-moe":
        model[1].quant_method = SimpleNamespace()
        model[1].base_quant_method = object.__new__(AWQSM70MoEMethod)
    elif kind == "other-awq":
        model[1].quant_method = object.__new__(AWQLinearMethod)
        model[1]._awq_sm70_prepared = False
    record_metadata_for_reloading(model)
    return model


def _snapshot(model):
    return [
        (name, param, param.data_ptr(), param.clone())
        for name, param in model.named_parameters()
    ]


def _assert_unchanged(model, snapshot):
    for name, original, pointer, value in snapshot:
        param = model.get_parameter(name)
        assert param is original
        assert param.device.type == "cpu"
        assert param.data_ptr() == pointer
        assert torch.equal(param, value)
    assert not hasattr(model, "_original_do_torchao_reload")


@pytest.mark.parametrize("kind", ["dense", "moe", "wrapped-moe"])
def test_layerwise_reload_rejected_before_restoring_any_layer(kind):
    model = _model(kind)
    before = _snapshot(model)
    with pytest.raises(NotImplementedError, match="SM70 AWQ.*cold"):
        initialize_layerwise_reload(model)
    _assert_unchanged(model, before)


@pytest.mark.parametrize("kind", ["dense", "moe", "wrapped-moe"])
@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("source", ["iterator", "disk"])
def test_runner_reload_rejected_before_loading_or_changing_path(
    kind, checkpoint, source
):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    model = _model(kind)
    model.load_weights = MagicMock(side_effect=AssertionError("must not load"))
    before = _snapshot(model)
    config = SimpleNamespace(model="original")
    runner = SimpleNamespace(get_model=lambda: model, model_config=config)
    with pytest.raises(NotImplementedError, match="SM70 AWQ.*cold"):
        GPUModelRunner.reload_weights(
            runner,
            is_checkpoint_format=checkpoint,
            weights_iterator=iter([]) if source == "iterator" else None,
            weights_path="replacement" if source == "disk" else None,
        )
    model.load_weights.assert_not_called()
    assert config.model == "original"
    _assert_unchanged(model, before)


@pytest.mark.parametrize("kind", ["dense", "moe", "wrapped-moe"])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_weight_transfer_start_rejected_before_mutation(kind, checkpoint):
    from vllm.v1.worker.gpu_worker import Worker

    model = _model(kind)
    before = _snapshot(model)
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        _weight_update_active=False,
        _is_checkpoint_format=None,
        model_runner=SimpleNamespace(model=model),
        device=torch.device("cpu"),
    )
    with pytest.raises(NotImplementedError, match="SM70 AWQ.*cold"):
        Worker.start_weight_update(worker, is_checkpoint_format=checkpoint)
    assert not worker._weight_update_active
    assert worker._is_checkpoint_format is None
    _assert_unchanged(model, before)


@pytest.mark.parametrize("kind", ["ordinary", "other-awq"])
def test_non_sm70_awq_and_unquantized_reload_remain_available(kind):
    model = _model(kind)
    initialize_layerwise_reload(model)
    assert all(param.device.type == "meta" for param in model.parameters())


@pytest.mark.parametrize("kind", ["dense", "moe", "wrapped-moe"])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_weight_transfer_receive_rejected_without_receiving_weights(kind, checkpoint):
    from vllm.v1.worker.gpu_worker import Worker

    model = _model(kind)
    before = _snapshot(model)
    transfer = MagicMock()
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        _weight_update_active=True,
        _is_checkpoint_format=checkpoint,
        weight_transfer_engine=transfer,
        model_runner=SimpleNamespace(model=model),
        device=torch.device("cpu"),
    )
    with pytest.raises(NotImplementedError, match="SM70 AWQ.*cold"):
        Worker.update_weights(worker, {})
    transfer.parse_update_info.assert_not_called()
    transfer.receive_weights.assert_not_called()
    _assert_unchanged(model, before)
