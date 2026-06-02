"""Tests for the v3 QueueEntry / EntryHandle work-tracking model.

* ``acquire(estimated_seconds=N)`` auto-creates an attached entry exposed via
  ``lease.entry``.
* ``acquire()`` without ``estimated_seconds`` leaves no entry behind.
* ``lease.start_work()`` creates a new entry per call (e.g. one per request
  on a persistent reclaimable lease).
* ``EntryHandle.complete()`` drops the entry.
* ``LeaseHandle.release()`` drops every still-attached entry.
* Reclaim skips leases that have any active QueueEntry attached.
* ``LeaseHandle.update()`` rejects ``estimated_seconds`` / ``progress`` kwargs
  (moved to EntryHandle).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from reslock import ResourcePool
from reslock.detect import gpu_vram_key
from reslock.models import State
from reslock.state import transact

UUID_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"


def _make_pool(tmp_path: Path, **resources: int) -> ResourcePool:
    pool = ResourcePool(tmp_path / "state.json")
    pool.set_resources(resources)
    return pool


# --- auto-tracked entry from acquire(estimated_seconds=...) ---


def test_acquire_with_estimated_seconds_creates_entry(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=4000)
    with pool.acquire(ram_mb=1000, estimated_seconds=30, label="foo") as lease:
        assert lease.entry is not None
        assert lease.entry.estimated_seconds == 30
        st = pool.status()
        assert len(st.queue) == 1
        assert st.queue[0].lease_id == lease.id
        assert st.queue[0].estimated_seconds == 30
        assert st.queue[0].is_active
        assert st.queue[0].started_at is not None
    # After release: lease and entry both gone.
    st = pool.status()
    assert st.leases == []
    assert st.queue == []


def test_acquire_without_estimated_seconds_drops_entry(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=4000)
    with pool.acquire(ram_mb=1000, label="foo") as lease:
        assert lease.entry is None
        st = pool.status()
        assert len(st.leases) == 1
        assert st.queue == []


def test_try_acquire_with_estimated_seconds_creates_entry(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=4000)
    lease = pool.try_acquire(ram_mb=1000, estimated_seconds=20)
    assert lease is not None
    assert lease.entry is not None
    assert lease.entry.estimated_seconds == 20
    assert lease.wait_sec == 0.0
    lease.release()
    assert pool.status().queue == []


# --- start_work creates a new entry per call ---


def test_start_work_creates_attached_entry(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, **{gpu_vram_key(UUID_A): 24000})
    lease = pool.try_acquire(vram_mb_each=8000, num_gpus=1, reclaimable=True)
    assert lease is not None
    assert lease.entry is None  # no auto-track
    entry = lease.start_work(estimated_seconds=30, label="chat")
    assert entry.lease_id == lease.id
    assert entry.estimated_seconds == 30
    assert entry.label == "chat"
    st = pool.status()
    assert len(st.queue) == 1
    assert st.queue[0].lease_id == lease.id
    entry.complete()
    assert pool.status().queue == []
    lease.release()


def test_start_work_multiple_concurrent(tmp_path: Path) -> None:
    """A persistent lease can have several attached entries simultaneously."""
    pool = _make_pool(tmp_path, ram_mb=4000)
    lease = pool.try_acquire(ram_mb=1000, reclaimable=True)
    assert lease is not None
    e1 = lease.start_work(estimated_seconds=10, label="a")
    e2 = lease.start_work(estimated_seconds=20, label="b")
    assert pool.status().queue and len(pool.status().queue) == 2
    e1.complete()
    assert len(pool.status().queue) == 1
    e2.complete()
    assert pool.status().queue == []
    lease.release()


def test_entry_update_modifies_fields(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=4000)
    with pool.acquire(ram_mb=1000, estimated_seconds=30) as lease:
        assert lease.entry is not None
        lease.entry.update(estimated_seconds=15, progress=0.5)
        st = pool.status()
        assert st.queue[0].estimated_seconds == 15
        assert st.queue[0].progress == 0.5


def test_entry_context_manager_completes(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=4000)
    lease = pool.try_acquire(ram_mb=1000, reclaimable=True)
    assert lease is not None
    with lease.start_work(estimated_seconds=30):
        assert len(pool.status().queue) == 1
    assert pool.status().queue == []
    lease.release()


# --- lease release drops attached entries ---


def test_release_drops_all_attached_entries(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=4000)
    lease = pool.try_acquire(ram_mb=1000, estimated_seconds=10, reclaimable=True)
    assert lease is not None
    lease.start_work(estimated_seconds=20)
    lease.start_work(estimated_seconds=30)
    assert len(pool.status().queue) == 3  # auto-track + 2 start_work
    lease.release()
    assert pool.status().queue == []


# --- reclaim skips leases with active entries ---


def test_reclaim_skips_lease_with_active_entry(tmp_path: Path) -> None:
    """A reclaimable lease with an active entry must not be picked for reclaim."""
    pool = _make_pool(tmp_path, ram_mb=4000)
    holder = pool.try_acquire(ram_mb=4000, reclaimable=True)
    assert holder is not None
    entry = holder.start_work(estimated_seconds=30, label="busy")

    # Try to provoke a reclaim by submitting a higher-priority request that
    # can't fit. With the holder's entry alive, the reclaim path must skip it.
    blocked: dict[str, bool] = {}

    def _waiter() -> None:
        try:
            with pool.acquire(priority=10, ram_mb=4000, poll_interval=0.05):
                blocked["got_it"] = True
        except BaseException:
            return

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    time.sleep(0.3)

    # Holder must NOT have been flagged for reclaim while the entry is active.
    state = pool.status()
    held = next(ls for ls in state.leases if ls.id == holder.id)
    assert held.reclaim_requested is False, "reclaim should be blocked by active entry"

    # Complete the entry; reclaim should now flip on the next poll.
    entry.complete()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if holder.reclaim_requested:
            break
        time.sleep(0.02)
    assert holder.reclaim_requested, "reclaim should be requested once entry completes"

    holder.release()
    t.join(timeout=2.0)


# --- LeaseHandle.update rejects moved fields ---


def test_lease_update_rejects_estimated_seconds(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=4000)
    lease = pool.try_acquire(ram_mb=1000)
    assert lease is not None
    with pytest.raises(TypeError, match="estimated_seconds"):
        lease.update(estimated_seconds=10)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError, match="progress"):
        lease.update(progress=0.5)  # pyright: ignore[reportCallIssue]
    lease.release()


def test_lease_handle_entry_property_after_release_is_completed(tmp_path: Path) -> None:
    """After lease.release(), the auto-track entry handle's complete() is a no-op."""
    pool = _make_pool(tmp_path, ram_mb=4000)
    lease = pool.try_acquire(ram_mb=1000, estimated_seconds=30)
    assert lease is not None
    assert lease.entry is not None
    handle = lease.entry
    lease.release()
    # No exception; queue is already empty.
    handle.complete()
    assert pool.status().queue == []


# --- queue_entry persists across active state ---


def test_queue_entry_persists_with_lease_id_after_promotion(tmp_path: Path) -> None:
    """The queueing entry stays in state.queue with lease_id+started_at set
    when the caller asked for tracking."""
    pool = _make_pool(tmp_path, ram_mb=4000)
    blocker = pool.try_acquire(ram_mb=4000)
    assert blocker is not None

    granted: dict[str, str] = {}

    def _waiter() -> None:
        with pool.acquire(ram_mb=2000, estimated_seconds=10, poll_interval=0.05) as lease:
            granted["lease_id"] = lease.id
            assert lease.entry is not None
            assert lease.entry.lease_id == lease.id
            time.sleep(0.1)  # hold so the assertion below sees state.queue

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    time.sleep(0.2)
    blocker.release()

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if "lease_id" in granted:
            break
        time.sleep(0.02)
    assert "lease_id" in granted
    t.join(timeout=2.0)


# --- direct State helpers used by the scheduler ---


def test_active_lease_ids_returns_attached_lease_ids(tmp_path: Path) -> None:
    """active_lease_ids() returns the set of lease_ids referenced by entries.

    Validates the helper used by reclaimable_for_shortfall to skip busy leases.
    """
    path = tmp_path / "state.json"
    pool = ResourcePool(path)
    pool.set_resources({"ram_mb": 4000})
    lease = pool.try_acquire(ram_mb=1000, estimated_seconds=10)
    assert lease is not None

    def _check(state: State) -> None:
        assert state.active_lease_ids() == {lease.id}

    transact(path, _check)
    lease.release()


# --- complete() failure semantics ---


def test_complete_retries_after_transient_transact_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ``complete()`` must surface the error AND leave the handle
    re-callable so a retry can actually clear the entry.

    Regression for the SCRIBA-384 / aiserver-kirk wedge incident
    (2026-05-30): the prior implementation set ``_completed=True`` before
    the state write, so any swallowed exception from ``transact()`` would
    permanently leak the entry — every subsequent ``complete()`` returned
    early without touching the state file.
    """
    from collections.abc import Callable
    from typing import Any

    import reslock.pool as pool_mod

    pool = _make_pool(tmp_path, ram_mb=4000)
    lease = pool.try_acquire(ram_mb=1000, reclaimable=True)
    assert lease is not None
    entry = lease.start_work(estimated_seconds=30, label="probe")
    assert len(pool.status().queue) == 1

    real_transact = pool_mod.transact
    fail_once = {"armed": True}

    def flaky_transact(path: Path, fn: Callable[[State], Any]) -> Any:
        if fail_once["armed"]:
            fail_once["armed"] = False
            raise OSError("simulated transient state-file failure")
        return real_transact(path, fn)

    monkeypatch.setattr(pool_mod, "transact", flaky_transact)

    # 1) First call surfaces the error — callers cannot swallow it silently
    #    without explicitly catching.
    with pytest.raises(OSError, match="simulated transient"):
        entry.complete()

    # 2) Entry is still in the queue; failed complete() must NOT silently
    #    mark the lease as released.
    assert len(pool.status().queue) == 1

    # 3) Retry succeeds — the handle did not latch ``_completed=True`` on the
    #    failed attempt.
    entry.complete()
    assert pool.status().queue == []

    # 4) Subsequent complete() calls are idempotent no-ops.
    entry.complete()
    lease.release()
