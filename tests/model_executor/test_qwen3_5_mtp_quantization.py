# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of MTP quantization exclusions without loading model kernels.

Run with --confcutdir=tests/model_executor to avoid model-loading parent fixtures.
"""

import ast
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest


@pytest.fixture(scope="module")
def mtp_quant_disabled() -> Callable[..., bool]:
    # Execute the actual standalone predicate, without importing GPU dependencies.
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm/model_executor/models/qwen3_5_mtp.py"
    )
    tree = ast.parse(path.read_text())
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MTP"
    )
    method = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_mtp_quant_disabled_in_hf_config"
    )
    method.decorator_list = []
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return cast(Callable[..., bool], namespace[method.name])


@pytest.mark.parametrize("as_object", [False, True], ids=["dict", "object"])
@pytest.mark.parametrize(
    ("quant_cfg", "expected"),
    [
        pytest.param(
            {"quant_method": "awq", "modules_to_not_convert": [pattern]},
            True,
            id=f"awq-{name}",
        )
        for name, pattern in [
            ("literal", "mtp"),
            ("model-literal", "model.mtp"),
            ("prefix-wildcard", "mtp*"),
            ("qualified-literal", "*.mtp"),
            ("qualified-wildcard", "*.mtp*"),
            ("qualified-prefix", "model.language_model.mtp*"),
            ("normalized-literal", "  MTP  "),
            ("normalized-model", "  MODEL.MTP  "),
            ("normalized-wildcard", "  *.MtP*  "),
        ]
    ]
    + [
        pytest.param(
            {"quant_method": "compressed-tensors", "ignore": [pattern]},
            True,
            id=f"compressed-tensors-{name}",
        )
        for name, pattern in [
            ("literal", "mtp"),
            ("wildcard", "*.mtp*"),
            ("projection", "mtp.fc"),
            ("submodule", "mtp.layers.0.self_attn.q_proj"),
            ("qualified-submodule", "model.language_model.mtp.layers.0*"),
            ("normalized-submodule", "  MODEL.MTP.LAYERS.0.MLP.GATE  "),
        ]
    ]
    + [
        pytest.param(
            {"quant_method": "fp8", "modules_to_not_convert": [pattern]},
            False,
            id=f"fp8-selective-{name}",
        )
        for name, pattern in [
            ("layernorm", "mtp.layers.0.input_layernorm"),
            ("gate", "mtp.layers.0.mlp.gate"),
            ("q-norm", "mtp.layers.0.self_attn.q_norm"),
            ("qualified-gate", "model.mtp.layers.0.mlp.gate"),
            ("projection", "mtp.fc"),
            ("norm", "mtp.norm"),
        ]
    ]
    + [
        pytest.param(
            {"quant_method": "fp8", "modules_to_not_convert": ["model.mtp"]},
            True,
            id="fp8-whole-branch",
        ),
        pytest.param(
            {"quant_method": "fp8", "ignore": ["model.mtp.layers.0.mlp.gate"]},
            False,
            id="fp8-selective-ignore",
        ),
        pytest.param(
            {"quant_method": "awq", "ignore": ["mtp*"]},
            True,
            id="awq-whole-branch-ignore",
        ),
        pytest.param(
            {
                "quant_method": "COMPRESSED-TENSORS",
                "ignore": ["mtp.layers.0.self_attn.q_proj"],
            },
            True,
            id="compressed-tensors-normalized-method",
        ),
        pytest.param(
            {
                "quant_method": "compressed-tensors",
                "modules_to_not_convert": ["mtp.layers.0.mlp.gate"],
            },
            False,
            id="compressed-tensors-submodule-outside-ignore",
        ),
        pytest.param(
            {"ignore": ["mtp.layers.0.mlp.gate"]},
            False,
            id="submodule-without-quant-method",
        ),
        pytest.param(
            {
                "quant_method": "compressed-tensors",
                "ignore": ["lm_head", "model.layers.0.self_attn.q_proj"],
            },
            False,
            id="unrelated-ignore",
        ),
        pytest.param(
            {"quant_method": "awq", "modules_to_not_convert": ["attempt", "mtp2"]},
            False,
            id="unrelated-modules",
        ),
        pytest.param(
            {"modules_to_not_convert": None, "ignore": None},
            False,
            id="null-exclusions",
        ),
        pytest.param({}, False, id="empty-config"),
    ],
)
def test_mtp_quantization_exclusions(
    mtp_quant_disabled: Callable[..., bool],
    as_object: bool,
    quant_cfg: dict[str, object],
    expected: bool,
) -> None:
    config = SimpleNamespace(
        quantization_config=SimpleNamespace(**quant_cfg) if as_object else quant_cfg
    )
    assert mtp_quant_disabled(config) is expected


def test_mtp_without_quantization_config(
    mtp_quant_disabled: Callable[..., bool],
) -> None:
    assert not mtp_quant_disabled()
    assert not mtp_quant_disabled(None, SimpleNamespace())
    assert not mtp_quant_disabled(SimpleNamespace(quantization_config=None))


def test_mtp_checks_both_text_and_top_level_configs(
    mtp_quant_disabled: Callable[..., bool],
) -> None:
    selective = SimpleNamespace(
        quantization_config={
            "quant_method": "fp8",
            "modules_to_not_convert": ["mtp.layers.0.mlp.gate"],
        }
    )
    full_precision = SimpleNamespace(
        quantization_config=SimpleNamespace(
            quant_method="awq", modules_to_not_convert=["model.mtp"]
        )
    )
    assert mtp_quant_disabled(None, selective, full_precision)
    assert mtp_quant_disabled(full_precision, selective)
