# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def _reset_global_watch_dog():
    from vllm.utils import watch_dog as watch_dog_module

    original = watch_dog_module._watch_dog
    yield
    watch_dog_module._watch_dog.stop()
    watch_dog_module._watch_dog = original


def _make_watch_dog(dump_dir, **kwargs):
    from vllm.config import WatchdogConfig
    from vllm.utils.watch_dog import ThreadedWatchDog

    config = WatchdogConfig(dump_dir=str(dump_dir), **kwargs)
    return ThreadedWatchDog("engine_0", config, MagicMock())


def _run_check_loop(wd, ticks: int) -> int:
    """Run the check loop for ``ticks`` checks; return the dump count."""
    with (
        patch.object(wd.stop_event, "wait", side_effect=[False] * ticks + [True]),
        patch.object(wd, "handle_failure") as handle_failure,
    ):
        wd._check_loop()
    return handle_failure.call_count


def test_check_loop_dumps_once_per_stall(tmp_path):
    wd = _make_watch_dog(tmp_path)
    wd.timeout = 0
    assert _run_check_loop(wd, ticks=3) == 1
    # A new feed followed by another stall dumps again.
    wd.feed()
    assert _run_check_loop(wd, ticks=3) == 1


def test_check_loop_stays_quiet_while_fed(tmp_path):
    wd = _make_watch_dog(tmp_path)
    wd.feed()
    assert _run_check_loop(wd, ticks=3) == 0


def test_disable_suspends_detection_and_restarts_window(tmp_path):
    wd = _make_watch_dog(tmp_path, timeout=60)
    wd.last_feed_time = time.monotonic() - 3600
    with wd.disable():
        assert _run_check_loop(wd, ticks=3) == 0
    assert _run_check_loop(wd, ticks=3) == 0


def test_handle_failure_appends_numbered_dumps(tmp_path):
    wd = _make_watch_dog(tmp_path / "dumps")
    wd.handle_failure("feed timeout")
    wd.handle_failure("SIGTERM")

    with open(wd.dump_file) as f:
        content = f.read()
    assert "dump #1" in content and "due to feed timeout" in content
    assert "dump #2" in content and "due to SIGTERM" in content
    assert test_handle_failure_appends_numbered_dumps.__name__ in content


def test_handle_failure_is_reentrant(tmp_path):
    # A second SIGTERM can re-enter handle_failure() while a dump holds the lock.
    wd = _make_watch_dog(tmp_path)
    with wd.lock:
        wd.handle_failure("SIGTERM")
    with open(wd.dump_file) as f:
        assert "due to SIGTERM" in f.read()


def test_handle_failure_does_not_raise_on_error(tmp_path):
    from vllm.config import WatchdogConfig
    from vllm.utils.watch_dog import (
        ThreadedWatchDog,
    )

    logger = MagicMock()
    config = WatchdogConfig(dump_dir=str(tmp_path))
    wd = ThreadedWatchDog("engine_0", config, logger)
    with patch("builtins.open", side_effect=OSError("boom")):
        wd.handle_failure("feed timeout")
    logger.warning.assert_called_once()


def test_start_watch_dog_without_dump_dir_stays_noop():
    from vllm.config import WatchdogConfig
    from vllm.utils.watch_dog import (
        NoopWatchDog,
        get_watch_dog,
        start_watch_dog,
    )

    logger = MagicMock()
    wd = start_watch_dog("engine_0", WatchdogConfig(), logger)
    assert isinstance(wd, NoopWatchDog)
    assert wd is get_watch_dog()
    logger.info.assert_not_called()


def test_start_watch_dog_with_dump_dir(tmp_path):
    from vllm.config import WatchdogConfig
    from vllm.utils.watch_dog import ThreadedWatchDog, get_watch_dog, start_watch_dog

    config = WatchdogConfig(dump_dir=str(tmp_path))
    with patch("faulthandler.enable") as enable:
        wd = start_watch_dog("worker_1", config, MagicMock())
    assert isinstance(wd, ThreadedWatchDog)
    try:
        assert wd is get_watch_dog()
        enable.assert_called_once_with(all_threads=True)
        assert wd.thread is not None
        assert wd.thread.is_alive() and wd.thread.daemon
        wd.handle_failure("shutdown")
        assert os.path.exists(
            tmp_path / f"VLLM_STACK_DUMP_for_worker_1_{os.getpid()}.log"
        )
    finally:
        wd.stop()


def test_disable_restarts_window_without_a_checker_tick(tmp_path):
    wd = _make_watch_dog(tmp_path)
    with patch("time.monotonic", return_value=10), wd.disable():
        pass
    assert wd.last_feed_time == 10


def test_thread_stop_restart_and_duplicate_start(tmp_path):
    wd = _make_watch_dog(tmp_path)
    wd.check_interval = 0.01
    wd.start()
    first = wd.thread
    try:
        wd.start()
        assert wd.thread is first
        wd.stop()
        assert not first.is_alive()
        assert wd.thread is None
        wd.start()
        assert wd.thread is not first
        thread = wd.thread
        assert thread is not None and thread.is_alive()
    finally:
        wd.stop()


def test_default_does_not_start_thread_or_touch_faulthandler():
    from vllm.config import WatchdogConfig
    from vllm.utils.watch_dog import NoopWatchDog, start_watch_dog

    with patch("threading.Thread") as thread, patch("faulthandler.enable") as enable:
        wd = start_watch_dog("engine_0", WatchdogConfig(), MagicMock())
        wd.feed()
        with wd.disable():
            wd.handle_failure("SIGTERM")
        wd.stop()
    assert isinstance(wd, NoopWatchDog)
    thread.assert_not_called()
    enable.assert_not_called()


def test_singleton_can_be_disabled_after_use(tmp_path):
    from vllm.config import WatchdogConfig
    from vllm.utils.watch_dog import NoopWatchDog, start_watch_dog

    wd = start_watch_dog(
        "engine_0", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock()
    )
    thread = wd.thread
    disabled = start_watch_dog("engine_0", WatchdogConfig(), MagicMock())
    assert isinstance(disabled, NoopWatchDog)
    assert not thread.is_alive()


def test_permission_failure_is_logged_and_monitor_survives(tmp_path):
    wd = _make_watch_dog(tmp_path)
    # PermissionError is deterministic even in root-run CPU containers.
    with patch("builtins.open", side_effect=PermissionError("read-only directory")):
        wd.handle_failure("feed timeout")
    wd.logger.warning.assert_called_once()
    wd.handle_failure("SIGTERM")
    assert "SIGTERM" in Path(wd.dump_file).read_text()


def test_signal_during_dump_does_not_recurse(tmp_path):
    import subprocess
    import sys

    script = """
import logging, os, signal, sys
from vllm.config import WatchdogConfig
from vllm.utils.watch_dog import ThreadedWatchDog
config = WatchdogConfig(dump_dir=sys.argv[1])
wd = ThreadedWatchDog("signal", config, logging.getLogger())
original = wd.dump_stack
def dump(reason):
    os.kill(os.getpid(), signal.SIGTERM)
    original(reason)
wd.dump_stack = dump
signal.signal(signal.SIGTERM, lambda *_: wd.handle_failure("SIGTERM"))
os.kill(os.getpid(), signal.SIGTERM)
assert wd.dump_seq == 2
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        timeout=30,
        cwd=tmp_path,
    )
    content = next(tmp_path.glob("VLLM_STACK_DUMP_for_signal_*.log")).read_text()
    assert content.count("Call stack dump") == 1
    assert "SIGTERM" in content


def test_signal_does_not_wait_for_checker_dump(tmp_path):
    import threading

    wd = _make_watch_dog(tmp_path)
    locked = threading.Event()
    release = threading.Event()

    def hold_lock():
        with wd.lock:
            locked.set()
            release.wait(5)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    try:
        assert locked.wait(2)
        wd.handle_failure("SIGTERM")
        assert wd.dump_seq == 1
    finally:
        release.set()
        thread.join(2)
