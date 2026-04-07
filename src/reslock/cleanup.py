from __future__ import annotations

import os

from reslock.models import Lease, QueueEntry, State


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _check_pid(pid: int, host_pid: int | None) -> bool:
    """Check if a process is alive, preferring host_pid for cross-namespace checks."""
    check = host_pid if host_pid is not None else pid
    return is_pid_alive(check)


def _lease_alive(lease: Lease) -> bool:
    """A lease is alive if its owner OR any tracked child PID is still running.

    Callers can register subprocess PIDs via ``handle.update(pids=[...])`` to
    protect the lease from premature reclaim when the owner dies but its
    children (e.g. an llama-server spawned from a Python parent) are still
    holding the actual resource. Child PIDs are always checked as-is — they
    are assumed to be in the same PID namespace as the caller.
    """
    if _check_pid(lease.pid, lease.host_pid):
        return True
    return any(is_pid_alive(child_pid) for child_pid in lease.pids)


def _queue_entry_alive(entry: QueueEntry) -> bool:
    return _check_pid(entry.pid, entry.host_pid)


def has_dead_processes(state: State) -> bool:
    """Check if any leases or queue entries reference dead processes."""
    return any(not _lease_alive(ls) for ls in state.leases) or any(
        not _queue_entry_alive(e) for e in state.queue
    )


def remove_dead_processes(state: State) -> None:
    state.leases = [ls for ls in state.leases if _lease_alive(ls)]
    state.queue = [e for e in state.queue if _queue_entry_alive(e)]
