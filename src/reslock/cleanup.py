from __future__ import annotations

import os

from reslock.models import State


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


def remove_dead_processes(state: State) -> None:
    state.leases = [ls for ls in state.leases if _check_pid(ls.pid, ls.host_pid)]
    state.queue = [e for e in state.queue if _check_pid(e.pid, e.host_pid)]
