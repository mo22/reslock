from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import portalocker

from reslock.cleanup import has_dead_processes, remove_dead_processes
from reslock.models import SCHEMA_VERSION, State

T = TypeVar("T")
logger = logging.getLogger(__name__)


class SchemaVersionMismatch(RuntimeError):
    """The state file was written under a different schema version.

    Raised in both directions — an older *and* a newer file — because the only
    safe reaction to a state file we do not speak is to not touch it.

    Up to v0.11.1 a mismatch silently returned a fresh empty ``State()``.
    ``read_state()`` merely passed that through, but ``transact()`` wrote it
    back: a single ``acquire()`` from an installation with a stale schema
    erased the lease table of every other consumer sharing the file, leaving a
    WARNING as the only trace. The consequence is VRAM handed out twice, and it
    escalates — after the reset the laggard writes *its* schema version, so the
    up-to-date consumers see a mismatch in turn and reset back.

    Failing instead puts the error where it belongs: the consumer that cannot
    speak the file's schema refuses to work, and everyone else keeps running.

    Recovery from a genuinely stale file (i.e. an intentional upgrade) is
    ``reslock reset --force`` / :func:`force_reset_state`, which rewrites the
    file at the current schema without reading it.
    """

    def __init__(self, found: object, expected: int, path: Path | None = None) -> None:
        self.found = found
        self.expected = expected
        self.path = path
        where = f" at {path}" if path is not None else ""
        super().__init__(
            f"reslock state file{where} has schema version {found!r}, "
            f"this reslock speaks v{expected}. Refusing to read or write it — "
            f"writing would erase the leases of consumers running the other "
            f"version. Upgrade all consumers sharing this file to matching "
            f"reslock versions, then 'reslock reset --force' (or delete the "
            f"file) once they are all stopped."
        )


def _load_state(data: str, path: Path | None = None) -> State:
    """Parse state JSON, refusing any schema version other than the current one.

    Raises :class:`SchemaVersionMismatch` when ``version`` differs. Dead PID
    cleanup handles stale process entries separately.

    The version is peeked from the raw JSON before strict Pydantic validation,
    so the mismatch is reported as itself rather than as whatever field-level
    ``extra="forbid"`` violation the foreign schema happens to trip over first
    (a v0.7.x file carries ``Lease.estimated_seconds``, a v4 file carries
    ``QueueEntry.vram_mb``, and so on). Same reason the peek existed when this
    path still reset the file.
    """
    try:
        parsed: object = json.loads(data)
    except json.JSONDecodeError:
        # Corrupt or empty JSON — fall through to model_validate which raises
        # with detail. This is intentionally fail-closed: a truncated state
        # file (e.g. a crash mid-``transact()`` rewrite) must NOT be silently
        # treated as "fresh empty state", which would let new acquires proceed
        # while in-flight leases were lost.
        return State.model_validate_json(data)

    if not isinstance(parsed, dict):
        # Top-level JSON wasn't an object — let pydantic raise the
        # canonical validation error.
        return State.model_validate_json(data)
    found_version = cast("dict[str, object]", parsed).get("version")
    if found_version != SCHEMA_VERSION:
        raise SchemaVersionMismatch(found_version, SCHEMA_VERSION, path)
    return State.model_validate_json(data)


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
    return _load_state(data, path)  # pyright: ignore[reportUnknownArgumentType]


def peek_state_version(path: Path) -> int | None:
    """Read only the ``version`` field, without validating the rest.

    For tools that need to *report on* a state file they may not speak —
    monitoring, ``reslock status``, upgrade checks. Everything else must go
    through :func:`read_state` / :func:`transact`, which refuse a foreign
    schema outright.

    Returns ``None`` when the file is missing, unreadable, not JSON, or has no
    integer ``version``. All four mean "cannot tell", which is deliberately
    not distinguished from each other here.
    """
    try:
        with portalocker.Lock(str(path), "r", timeout=5) as fh:  # pyright: ignore[reportUnknownVariableType]
            data: str = fh.read()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        parsed: object = json.loads(data)  # pyright: ignore[reportUnknownArgumentType]
    except (OSError, json.JSONDecodeError, portalocker.LockException):
        return None
    if not isinstance(parsed, dict):
        return None
    found = cast("dict[str, object]", parsed).get("version")
    # bool is an int subclass; a JSON `true` is not a version.
    return found if isinstance(found, int) and not isinstance(found, bool) else None


def force_reset_state(path: Path) -> None:
    """Overwrite the state file with a fresh, empty state at the current schema.

    The supported recovery from :class:`SchemaVersionMismatch`, and the way a
    coordinated schema upgrade moves the file forward now that reads no longer
    reset it. Does not read the existing content — that is the point, since by
    definition we may not be able to parse it.

    Destructive: every lease and queue entry in the file is dropped, and
    registered capacities go with them (consumers re-register via
    ``set_resources()``). Only safe with all consumers stopped — a running
    holder would keep its resources without the file recording them.
    """
    ensure_state_file(path)
    with portalocker.Lock(str(path), "r+", timeout=5) as fh:  # pyright: ignore[reportUnknownVariableType]
        fh.seek(0)
        fh.truncate()
        fh.write(State().model_dump_json(indent=2))  # pyright: ignore[reportUnknownMemberType]


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

    Raises :class:`SchemaVersionMismatch` before `fn` runs and before anything
    is written when the file was authored under a different schema version —
    the file is left byte-for-byte untouched.
    """
    with portalocker.Lock(str(path), "r+", timeout=5) as fh:  # pyright: ignore[reportUnknownVariableType]
        data: str = fh.read()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        state = _load_state(data, path)  # pyright: ignore[reportUnknownArgumentType]
        remove_dead_processes(state)
        result = fn(state)
        new_data = state.model_dump_json(indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(new_data)  # pyright: ignore[reportUnknownMemberType]
    return result
