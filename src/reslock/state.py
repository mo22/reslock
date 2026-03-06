from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import portalocker

from reslock.cleanup import remove_dead_processes
from reslock.models import State

T = TypeVar("T")

DEFAULT_STATE_PATH = Path.home() / ".reslock" / "state.json"


def ensure_state_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Make directory world-writable (sticky bit) so multiple users/containers can share it
    try:
        path.parent.chmod(0o1777)
    except OSError:
        pass
    if not path.exists():
        path.write_text(State().model_dump_json(indent=2))
        try:
            path.chmod(0o666)
        except OSError:
            pass


def read_state(path: Path) -> State:
    with portalocker.Lock(str(path), "r", timeout=5) as fh:
        data = fh.read()
    return State.model_validate_json(data)


def write_state(path: Path, state: State) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(state.model_dump_json(indent=2))
        os.chmod(tmp, 0o666)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def transact(path: Path, fn: Callable[[State], T]) -> T:
    """Atomically read, modify, and write the state file under an exclusive lock.

    The callable `fn` receives the current state (with dead processes cleaned up)
    and may mutate it. The modified state is written back. The return value of `fn`
    is returned to the caller.
    """
    with portalocker.Lock(str(path), "r+", timeout=5) as fh:
        data = fh.read()
        state = State.model_validate_json(data)
        remove_dead_processes(state)
        result = fn(state)
        new_data = state.model_dump_json(indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(new_data)
    return result
