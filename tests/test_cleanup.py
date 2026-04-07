from __future__ import annotations

import os

from reslock.cleanup import has_dead_processes, is_pid_alive, remove_dead_processes
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


def test_lease_with_live_child_survives_dead_owner() -> None:
    """A lease whose owner is dead but whose tracked child PID is still alive
    must be preserved — the child is still holding the actual resource.

    This is the scenario that caused SCRIBA-283/284: aiserver's Python parent
    died but its spawned llama-server children kept holding GPU VRAM.
    """
    state = State(
        resources={"vram_mb": 8000},
        leases=[
            Lease(
                pid=999999999,  # dead owner
                resources={"vram_mb": 4000},
                pids=[os.getpid()],  # live child
            ),
        ],
    )
    assert not has_dead_processes(state)
    remove_dead_processes(state)
    assert len(state.leases) == 1


def test_lease_with_all_dead_pids_is_removed() -> None:
    """A lease is reclaimed only when its owner AND every tracked child PID
    are all dead."""
    state = State(
        resources={"vram_mb": 8000},
        leases=[
            Lease(
                pid=999999999,
                resources={"vram_mb": 4000},
                pids=[999999998, 999999997],
            ),
        ],
    )
    assert has_dead_processes(state)
    remove_dead_processes(state)
    assert state.leases == []


def test_lease_with_live_owner_and_dead_child_is_kept() -> None:
    """Dead child PIDs do not invalidate an otherwise-live lease. The owner
    may spawn new children later."""
    state = State(
        resources={"vram_mb": 8000},
        leases=[
            Lease(
                pid=os.getpid(),
                resources={"vram_mb": 4000},
                pids=[999999999],  # dead
            ),
        ],
    )
    assert not has_dead_processes(state)
    remove_dead_processes(state)
    assert len(state.leases) == 1
