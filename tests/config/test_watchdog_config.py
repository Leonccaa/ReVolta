# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU CLI/config/hash checks for optional stack diagnostics."""

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.skip_global_cleanup


def test_defaults():
    from vllm.config import WatchdogConfig

    config = WatchdogConfig()
    assert (config.timeout, config.check_interval, config.dump_dir) == (300, 10, "")


@pytest.mark.parametrize("field", ["timeout", "check_interval"])
@pytest.mark.parametrize("value", [0, -1, 0.5, float("inf"), float("nan")])
def test_invalid_intervals(field, value):
    from vllm.config import WatchdogConfig

    with pytest.raises(ValidationError):
        WatchdogConfig(**{field: value})


@pytest.mark.parametrize("cli", [False, True])
def test_cli_and_engine_config(tmp_path, cli):
    from transformers import Qwen3Config

    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=128,
        max_position_embeddings=512,
    ).save_pretrained(tmp_path / "model")
    options = dict(
        model=str(tmp_path / "model"),
        skip_tokenizer_init=True,
        enforce_eager=True,
        dtype="float16",
        max_model_len=512,
    )
    watchdog = dict(dump_dir=str(tmp_path / "dumps"), timeout=30, check_interval=1)
    if cli:
        import json

        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        parsed = parser.parse_args(
            ["--model", options["model"], "--watchdog-config", json.dumps(watchdog)]
        )
        args = EngineArgs.from_cli_args(parsed)
        for key, value in options.items():
            setattr(args, key, value)
    else:
        args = EngineArgs(**options, watchdog_config=watchdog)
    config = args.create_engine_config()
    assert config.watchdog_config.dump_dir == watchdog["dump_dir"]
    assert config.watchdog_config.timeout == 30
    assert config.watchdog_config.check_interval == 1


def test_watchdog_does_not_change_graph_hash():
    from vllm.config import VllmConfig, WatchdogConfig

    config = VllmConfig()
    baseline = config.compute_hash()
    for options in (
        dict(timeout=1, check_interval=1),
        dict(dump_dir="/tmp/watchdog", timeout=30, check_interval=2),
        dict(dump_dir="/tmp/other", timeout=900, check_interval=20),
    ):
        config.watchdog_config = WatchdogConfig(**options)
        assert config.compute_hash() == baseline
