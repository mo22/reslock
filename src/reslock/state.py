from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import portalocker

from reslock.cleanup import has_dead_processes, remove_dead_processes
from reslock.models import SCHEMA_VERSION, State

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _load_state(data: str) -> State:
    """Parse state JSON and migrate across schema versions.

    Schema version 2 changed GPU VRAM keys from index-based (``gpu0_vram_mb``)
    to UUID-based (``gpu_GPU-<uuid>_vram_mb``). Older snapshots are dropped —
    ``resources``, ``leases``, and ``queue`` are reset — so the next
    ``set_resources()`` / ``acquire()`` calls repopulate with UUID keys.
    Dead PID cleanup handles stale process entries separately.
    """
    state = State.model_validate_json(data)
    if state.version != SCHEMA_VERSION:
        logger.warning(
            "reslock state schema v%d detected (expected v%d) — resetting "
            "resources, leases, and queue. Consumers must re-register resources.",
            state.version,
            SCHEMA_VERSION,
        )
        return State()
    return state


def _default_state_path() -> Path:
    """System-wide default, overridable via RESLOCK_DIR env var.

    Reslock coordinates shared resources (GPUs) across all processes on a
    machine, regardless of user.  /var/lib/reslock is the canonical location;
    falls back to ~/.reslock if /var/lib is not writable (e.g. unprivileged
    container without the volume mount).

    This function only *picks* the path — directory/file creation is handled
    by ensure_state_file().
    """
    env = os.environ.get("RESLOCK_DIR")
    if env:
        return Path(env) / "state.json"
    system_dir = Path("/var/lib/reslock")
    if system_dir.exists():
        if os.access(system_dir, os.W_OK):
            return system_dir / "state.json"
    elif os.access(system_dir.parent, os.W_OK):
        return system_dir / "state.json"
    user_dir = Path.home() / ".reslock"
    try:
        if os.access(user_dir, os.W_OK) or os.access(user_dir.parent, os.W_OK):
            logger.warning(
                "System state dir /var/lib/reslock is not writable, "
                "falling back to %s — resource coordination will be per-user only",
                user_dir,
            )
            return user_dir / "state.json"
    except (OSError, RuntimeError):
        pass
    logger.warning(
        "No writable state directory (tried /var/lib/reslock and %s) — "
        "reslock will not be able to coordinate resources",
        user_dir,
    )
    return user_dir / "state.json"


DEFAULT_STATE_PATH = _default_state_path()


def ensure_state_file(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Cannot create state directory %s", path.parent)
        return
    # Make directory world-writable (sticky bit) so multiple users/containers can share it
    with contextlib.suppress(OSError):
        path.parent.chmod(0o1777)
    if not path.exists():
        try:
            path.write_text(State().model_dump_json(indent=2))
        except OSError:
            logger.warning("Cannot create state file %s", path)
            return
        with contextlib.suppress(OSError):
            path.chmod(0o666)


def read_state(path: Path) -> State:
    with portalocker.Lock(str(path), "r", timeout=5) as fh:  # pyright: ignore[reportUnknownVariableType]
        data: str = fh.read()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return _load_state(data)  # pyright: ignore[reportUnknownArgumentType]


def read_state_clean(path: Path) -> State:
    """Read state, cleaning up dead processes if any are found.

    Unlike read_state(), this checks for dead PIDs and persists cleanup
    back to the state file when needed. Avoids writes when no dead processes exist.
    """
    state = read_state(path)
    if not has_dead_processes(state):
        return state

    def _identity(st: State) -> State:
        return st

    return transact(path, _identity)


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
    with portalocker.Lock(str(path), "r+", timeout=5) as fh:  # pyright: ignore[reportUnknownVariableType]
        data: str = fh.read()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        state = _load_state(data)  # pyright: ignore[reportUnknownArgumentType]
        remove_dead_processes(state)
        result = fn(state)
        new_data = state.model_dump_json(indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(new_data)  # pyright: ignore[reportUnknownMemberType]
    return result
