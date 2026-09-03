"""Regression tests for the torn-read window in the state file writer (v0.12.2).

Mechanism, measured 2026-09-03 (``tasks/torn-read-of-state-json.md``): ``transact()``
wrote in place through a buffered handle, and portalocker's ``Lock.release()`` is
``unlock(fh); fh.close()`` — the flock is dropped *before* the buffer is flushed. For a
state file below the 8 KiB buffer, the truncate had landed and nothing else had, so a
reader that won the lock in that window read 0 bytes. Above 8 KiB the single write goes
straight through, which is why a first repro with a 9 KB file never tore.

Old code: the three small sizes below report 0 B at unlock, the concurrent test counts
hundreds of torn snapshots, and the retrying-reader tests raise. New code: all green.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from typing import Any, cast

import portalocker
import portalocker.portalocker as portalocker_impl
import pydantic
import pytest

from reslock.models import SCHEMA_VERSION, State
from reslock.state import (
    SchemaVersionMismatch,
    ensure_state_file,
    force_reset_state,
    peek_state_version,
    read_state,
    transact,
)

# resources count -> ~final size: 50 -> 1.2 KB, 150 -> 3.5 KB, 300 -> 7 KB (all inside
# the 8 KiB buffer, 0 B at unlock on 0.12.1); 400 -> 9.3 KB is the positive control that
# was already complete before the fix.
SIZES = [50, 150, 300, 400]


def _fill(n: int):
    def fn(state: State) -> None:
        state.resources.clear()
        for i in range(n):
            state.resources[f"gpu:{i:04d}"] = 24135

    return fn


def _hook_unlock(monkeypatch: pytest.MonkeyPatch, before: Any) -> None:
    """Run ``before(fh)`` at the instant portalocker releases a lock.

    ``portalocker.utils.Lock.release`` calls ``portalocker.unlock`` via the
    ``portalocker.portalocker`` submodule, so that attribute is the hook point.
    """
    real_unlock = portalocker_impl.unlock

    def probe(fh: Any) -> None:
        before(fh)
        real_unlock(fh)

    monkeypatch.setattr(portalocker_impl, "unlock", probe)


@pytest.mark.parametrize("n", SIZES)
def test_transact_content_is_on_disk_when_lock_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    path = tmp_path / "state.json"
    ensure_state_file(path)
    at_unlock: dict[str, Any] = {}

    def snapshot(_fh: Any) -> None:
        at_unlock["size"] = os.stat(path).st_size
        at_unlock["bytes"] = path.read_bytes()

    _hook_unlock(monkeypatch, snapshot)
    transact(path, _fill(n))

    final = path.read_bytes()
    assert n <= 300 and len(final) < 8192 or n > 300 and len(final) > 8192, len(final)
    assert at_unlock["size"] == len(final)
    assert at_unlock["bytes"] == final
    assert State.model_validate_json(at_unlock["bytes"]).resources == {
        f"gpu:{i:04d}": 24135 for i in range(n)
    }


def test_force_reset_content_is_on_disk_when_lock_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    ensure_state_file(path)
    transact(path, _fill(150))
    at_unlock: dict[str, Any] = {}

    def snapshot(_fh: Any) -> None:
        at_unlock.setdefault("bytes", path.read_bytes())

    _hook_unlock(monkeypatch, snapshot)

    force_reset_state(path)

    assert at_unlock["bytes"] == path.read_bytes()
    assert State.model_validate_json(at_unlock["bytes"]) == State()


# --- concurrent reader/writer -------------------------------------------------------------


def _widened_unlock_writer(path_str: str, transactions: int, widen_sec: float) -> None:
    """Child process: loop ``transact`` with the post-unlock gap widened.

    The sleep sits AFTER the real unlock and BEFORE ``close()`` — the same order as
    shipped, only wider, so a reader lands in the window reliably instead of once in
    30 hours. With the flush moved under the lock the gap is empty and the width is
    irrelevant.
    """
    real_unlock = portalocker_impl.unlock

    def slow_unlock(fh: Any) -> None:
        real_unlock(fh)
        time.sleep(widen_sec)

    portalocker_impl.unlock = slow_unlock
    path = Path(path_str)
    for i in range(transactions):
        transact(path, _fill(100 + i % 50))


def test_concurrent_reader_never_sees_torn_state(tmp_path: Path) -> None:
    """Raw locked reads (no retry) while a writer with a 2 ms widened unlock gap
    runs a few hundred transactions: torn snapshots must be 0.

    Reads bypass ``read_state()`` on purpose: its retry would mask a torn writer.
    """
    path = tmp_path / "state.json"
    ensure_state_file(path)
    transactions = 300
    ctx = multiprocessing.get_context("spawn")
    writer = ctx.Process(
        target=_widened_unlock_writer, args=(str(path), transactions, 0.002), daemon=True
    )
    writer.start()
    ok = torn = 0
    sizes: set[int] = set()
    try:
        while writer.is_alive() or ok + torn == 0:
            with portalocker.Lock(str(path), "r", timeout=5) as fh:  # pyright: ignore[reportUnknownVariableType]
                data = cast("str", fh.read())  # pyright: ignore[reportUnknownMemberType]
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                torn += 1
                sizes.add(len(data))
            else:
                assert parsed["version"] == SCHEMA_VERSION
                ok += 1
            if not writer.is_alive():
                break
    finally:
        writer.join(timeout=60)
    assert writer.exitcode == 0
    assert ok >= 50, f"reader barely ran: ok={ok} torn={torn}"
    final = State.model_validate_json(path.read_bytes())
    assert len(final.resources) == 100 + (transactions - 1) % 50
    assert torn == 0, f"{torn} torn reads of {ok + torn} (sizes seen: {sorted(sizes)})"


# --- reader retry ------------------------------------------------------------------------


def _write_after(path: Path, delay: float, content: str) -> threading.Thread:
    def later() -> None:
        time.sleep(delay)
        path.write_text(content)

    t = threading.Thread(target=later, daemon=True)
    t.start()
    return t


def test_read_state_retries_a_torn_file(tmp_path: Path) -> None:
    """A file that is empty / truncated now and whole 30 ms later is read, not raised."""
    path = tmp_path / "state.json"
    state = State()
    state.resources["vram_mb"] = 4096
    whole = state.model_dump_json(indent=2)

    path.write_text("")  # the measured 0-byte case
    t = _write_after(path, 0.03, whole)
    assert read_state(path).resources == {"vram_mb": 4096}
    t.join()

    path.write_text(whole[:20])  # mid-string, like the kirk traceback
    t = _write_after(path, 0.03, whole)
    assert read_state(path).resources == {"vram_mb": 4096}
    t.join()


def test_read_state_gives_up_after_bounded_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that stays invalid raises after TORN_READ_RETRIES re-reads — never a reset."""
    from reslock import state as state_mod

    path = tmp_path / "state.json"
    path.write_text("")
    sleeps: list[float] = []
    monkeypatch.setattr(state_mod.time, "sleep", sleeps.append)
    with pytest.raises(pydantic.ValidationError):
        read_state(path)
    assert len(sleeps) == state_mod.TORN_READ_RETRIES == 5
    assert sleeps == [state_mod.TORN_READ_DELAY_SEC] * 5
    assert path.read_text() == ""  # not reset


def test_schema_mismatch_and_field_errors_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reslock import state as state_mod

    path = tmp_path / "state.json"
    sleeps: list[float] = []
    monkeypatch.setattr(state_mod.time, "sleep", sleeps.append)

    path.write_text('{"version": 1, "resources": {}, "leases": [], "queue": []}')
    with pytest.raises(SchemaVersionMismatch):
        read_state(path)

    path.write_text(f'{{"version": {SCHEMA_VERSION}, "resources": "oops"}}')
    with pytest.raises(pydantic.ValidationError):
        read_state(path)

    assert sleeps == []


def test_transact_retries_a_torn_file_under_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("")
    t = _write_after(path, 0.03, State().model_dump_json(indent=2))
    transact(path, _fill(3))
    t.join()
    assert len(read_state(path).resources) == 3


def test_peek_state_version_retries_a_torn_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("")
    t = _write_after(path, 0.03, State().model_dump_json(indent=2))
    assert peek_state_version(path) == SCHEMA_VERSION
    t.join()
    path.write_text("")
    assert peek_state_version(path) is None
