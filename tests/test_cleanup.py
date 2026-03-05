from __future__ import annotations

import os

from reslock.cleanup import is_pid_alive, remove_dead_processes
from reslock.models import Lease, QueueEntry, State


def test_current_pid_is_alive() -> None:
    assert is_pid_alive(os.getpid())


def test_dead_pid() -> None:
    assert not is_pid_alive(999999999)


def test_remove_dead_processes() -> None:
    state = State(
        resources={"vram_mb": 8000},
        leases=[
            Lease(pid=os.getpid(), resources={"vram_mb": 4000}),
            Lease(pid=999999999, resources={"vram_mb": 2000}),
        ],
        queue=[
            QueueEntry(pid=os.getpid(), resources={"vram_mb": 1000}),
            QueueEntry(pid=999999999, resources={"vram_mb": 1000}),
        ],
    )
    remove_dead_processes(state)
    assert len(state.leases) == 1
    assert state.leases[0].pid == os.getpid()
    assert len(state.queue) == 1
    assert state.queue[0].pid == os.getpid()
