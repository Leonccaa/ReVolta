# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""--shutdown-timeout defaults to 0, as upstream."""

import dataclasses

import pytest

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _parse(argv):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    return EngineArgs.from_cli_args(parser.parse_args(argv))


def test_cli_default_is_0():
    assert _parse([]).shutdown_timeout == 0


@pytest.mark.parametrize("value", [0, 1, 30, 120])
def test_cli_explicit_value_is_kept(value):
    assert _parse(["--shutdown-timeout", str(value)]).shutdown_timeout == value


def test_dataclass_and_config_defaults_are_0():
    assert EngineArgs().shutdown_timeout == 0
    field = {f.name: f for f in dataclasses.fields(VllmConfig)}["shutdown_timeout"]
    default = getattr(field.default, "default", field.default)  # pydantic FieldInfo
    assert default == 0


def _engine_like(cleanup_s, ready):
    import signal
    import time

    def on_term(*_):
        time.sleep(cleanup_s)  # Drain, flush and resource cleanup.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    ready.send(True)
    while True:
        time.sleep(0.05)


@pytest.mark.parametrize(
    ("argv", "cleanup_s", "exitcode"),
    [([], 30.0, -9), (["--shutdown-timeout", "5"], 0.3, 0)],
)
def test_real_process_stop_follows_shutdown_timeout(argv, cleanup_s, exitcode):
    """The engine manager's stop: 0 kills at once, a positive value waits."""
    import multiprocessing
    import time

    from vllm.utils.process_reaping import process_reap_timeout
    from vllm.v1.utils import shutdown

    args = _parse(argv)
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    proc = context.Process(target=_engine_like, args=(cleanup_s, writer))
    proc.start()
    try:
        assert reader.poll(30) and reader.recv()
        started = time.monotonic()
        shutdown(
            [proc],
            timeout=args.shutdown_timeout,
            retirements={},
            reap_timeout=process_reap_timeout(args),
        )
        proc.join(10)
        elapsed = time.monotonic() - started
        assert proc.exitcode == exitcode
        assert elapsed < (2.0 if exitcode else 5.0)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)
