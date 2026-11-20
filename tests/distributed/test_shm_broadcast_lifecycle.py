# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

import pytest

# Isolate each case: a stuck ZMQ context must not hang the pytest process.
_CHILD = r"""
import gc
import sys
import weakref

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

role, cycle, transport = sys.argv[1:]
for iteration in range(5):
    local = transport == "local"
    writer = MessageQueue(
        n_reader=1, n_local_reader=int(local), connect_ip="127.0.0.1",
        max_chunk_bytes=1024, max_chunks=1,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), 0)
    target = writer if role == "writer" else reader
    socket = target.local_socket if local else target.remote_socket
    refs = [weakref.ref(target), weakref.ref(socket.context), weakref.ref(socket)]
    if target._spin_condition is not None:
        refs.extend(weakref.ref(s) for s in (
            target._spin_condition.local_notify_socket,
            target._spin_condition.read_cancel_socket,
            target._spin_condition.write_cancel_socket,
        ) if s is not None)
    if cycle == "cycle":
        target.cycle = target
    del socket, target
    if role == "writer":
        del writer
    else:
        del reader
    print(f"collect {role} {cycle} {transport} {iteration}", flush=True)
    gc.collect()
    assert all(ref() is None for ref in refs), "queue or ZMQ resources retained"
    if role == "writer":
        del reader
    else:
        del writer
    gc.collect()
print("finished", flush=True)
"""


@pytest.mark.parametrize("role", ["writer", "reader"])
@pytest.mark.parametrize("cycle", ["plain", "cycle"])
@pytest.mark.parametrize("transport", ["local", "remote"])
def test_message_queue_resources_collected_in_subprocess(role, cycle, transport):
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, role, cycle, transport],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "finished" in completed.stdout
