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


def remove_dead_processes(state: State) -> None:
    state.leases = [lease for lease in state.leases if is_pid_alive(lease.pid)]
    state.queue = [e for e in state.queue if is_pid_alive(e.pid)]
