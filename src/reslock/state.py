from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, TypeVar, cast

import portalocker
import pydantic

from reslock.cleanup import has_dead_processes, remove_dead_processes
from reslock.models import SCHEMA_VERSION, State

T = TypeVar("T")
logger = logging.getLogger(__name__)

# A reader that finds unparseable JSON re-reads this many times, sleeping
# ``TORN_READ_DELAY_SEC`` between attempts, before raising. Bounded, so a file
# that is genuinely corrupt still fails — after ~100 ms instead of at once.
# See :func:`_load_state_retrying` for what "torn" means and where it comes from.
TORN_READ_RETRIES = 5
TORN_READ_DELAY_SEC = 0.02


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


def _is_torn_read(exc: pydantic.ValidationError) -> bool:
    """True when every error in ``exc`` is a JSON syntax error.

    A state file is written as one JSON object whose last byte is the closing
    brace, so every strict prefix of it is syntactically invalid JSON — which
    is exactly what pydantic reports as ``json_invalid``. Field-level errors
    mean a complete but wrong document; those are never torn reads.
    """
    errors = exc.errors()
    return bool(errors) and all(e["type"] == "json_invalid" for e in errors)


def _load_state_retrying(path: Path, read: Callable[[], str]) -> State:
    """Parse the state read by ``read()``, re-reading a bounded number of times
    while the content is syntactically invalid JSON.

    Why a torn read is possible at all: reslock < 0.12.2 wrote the state file
    in place through a buffered handle and portalocker's ``Lock.release()``
    drops the flock *before* ``close()`` flushes that buffer. For a file below
    the 8 KiB buffer size the truncate had landed and the new content had not,
    so a reader taking the shared lock in that window saw 0 bytes (measured
    on macOS and in the kirk container, 2026-09-03; aiserver's reclaim loop
    died on it after 30 h of dev traffic). Since 0.12.2 every writer flushes
    before releasing the lock, so a torn read can only come from a consumer
    still on an older reslock sharing the file, or from a process crash
    between truncate and write. The retry covers the former during a rolling
    upgrade; the latter stays a hard error after the retries are exhausted.

    Only JSON syntax errors are retried. :class:`SchemaVersionMismatch` and
    field-level validation errors describe a complete document and are raised
    at once. An empty file is a torn read, never "fresh empty state" — the
    0.12.0 rule that a read never resets the file stands.
    """
    for attempt in range(TORN_READ_RETRIES + 1):
        data = read()
        try:
            return _load_state(data, path)
        except pydantic.ValidationError as exc:
            if not _is_torn_read(exc) or attempt == TORN_READ_RETRIES:
                raise
            logger.debug(
                "Torn read of %s (%d bytes), retry %d/%d",
                path,
                len(data),
                attempt + 1,
                TORN_READ_RETRIES,
            )
            time.sleep(TORN_READ_DELAY_SEC)
    raise AssertionError("unreachable")  # pragma: no cover


def _read_shared(path: Path) -> str:
    with portalocker.Lock(str(path), "r", timeout=5) as fh:  # pyright: ignore[reportUnknownVariableType]
        return fh.read()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _flush_locked(fh: IO[str]) -> None:
    """Push a locked handle's buffered content to disk *before* the lock goes.

    portalocker's ``Lock.release()`` is ``unlock(fh); fh.close()`` — the
    flush that ``close()`` implies happens after the flock is gone. Calling
    this as the last statement inside the ``with`` block moves the content
    under the lock. ``fsync`` is cheap here (one small file per lease change)
    and makes the write durable, not merely visible.
    """
    fh.flush()
    os.fsync(fh.fileno())


def read_state(path: Path) -> State:
    """Read and validate the state file under a shared lock.

    Re-reads a few times if the content is not valid JSON (see
    :func:`_load_state_retrying`); the shared lock is released between
    attempts so a writer can finish.
    """
    return _load_state_retrying(path, lambda: _read_shared(path))


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
    parsed: object = None
    for attempt in range(TORN_READ_RETRIES + 1):
        try:
            parsed = json.loads(_read_shared(path))
        except json.JSONDecodeError:
            # Same torn-read window as read_state(); bounded retry, then
            # "cannot tell" rather than an exception — this is a peek.
            if attempt == TORN_READ_RETRIES:
                return None
            time.sleep(TORN_READ_DELAY_SEC)
            continue
        except (OSError, portalocker.LockException):
            return None
        break
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
        _flush_locked(fh)  # pyright: ignore[reportUnknownArgumentType]


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

    The write is in place on the locked inode and flushed + fsynced before
    the lock is released (:func:`_flush_locked`), so a reader holding the
    shared lock always sees the complete document. It is deliberately *not*
    a temp file + ``os.replace``: the flock lives on the state file's inode,
    so after a replace a writer that was blocked on the old inode would get
    its lock on an unlinked file, read stale content and write into the void
    — a lost update, worse than a torn read. Atomic replace would need a
    separate lock file, and that is a locking-protocol change every consumer
    sharing the file has to make at once; the flush is additive.
    """
    with portalocker.Lock(str(path), "r+", timeout=5) as fh:  # pyright: ignore[reportUnknownVariableType]

        def _read_locked() -> str:
            fh.seek(0)
            return fh.read()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        state = _load_state_retrying(path, _read_locked)
        remove_dead_processes(state)
        result = fn(state)
        new_data = state.model_dump_json(indent=2)
        fh.seek(0)
        fh.truncate()
        fh.write(new_data)  # pyright: ignore[reportUnknownMemberType]
        _flush_locked(fh)  # pyright: ignore[reportUnknownArgumentType]
    return result
