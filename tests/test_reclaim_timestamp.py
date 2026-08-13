"""``Lease.reclaim_requested_at`` — since when a reclaim has been pending.

Schema v5 (v0.12.0). Motivating incident (2026-08-01..10): a llama-server
lease held two GPUs with ``reclaim_requested: true`` for nine days. The state
file recorded *that* a reclaim was outstanding but not *since when*, and that
age is the only thing that distinguishes a consumer honouring the request
(seconds) from one that has stopped listening.

Uses RAM rather than VRAM throughout — the reclaim machinery is shared, and
RAM keeps NVML out of the test.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from reslock import ResourcePool
from reslock.models import State
from reslock.state import read_state, transact


def _make_pool(tmp_path: Path, **resources: int) -> ResourcePool:
    state_path = tmp_path / "state.json"
    pool = ResourcePool(state_path)

    def _set(st: State) -> None:
        st.resources.update(resources)

    transact(state_path, _set)
    return pool


def _lease_in_file(pool: ResourcePool, lease_id: str):
    for lease in read_state(pool._path).leases:  # pyright: ignore[reportPrivateUsage]
        if lease.id == lease_id:
            return lease
    return None


def _wait_for_reclaim(pool: ResourcePool, lease_id: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lease = _lease_in_file(pool, lease_id)
        if lease is not None and lease.reclaim_requested:
            return lease
        time.sleep(0.02)
    raise AssertionError("reclaim was never requested")


def test_reclaim_request_is_timestamped(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=700_000)
    holder = pool.try_acquire(ram_mb=650_000, reclaimable=True, priority=0, label="batch")
    assert holder is not None

    before = datetime.now(timezone.utc)

    def _high_pri() -> None:
        with pool.acquire(ram_mb=650_000, priority=10, poll_interval=0.05):
            pass

    t = threading.Thread(target=_high_pri, daemon=True)
    t.start()
    try:
        lease = _wait_for_reclaim(pool, holder.id)
        assert lease.reclaim_requested_at is not None
        assert before <= lease.reclaim_requested_at <= datetime.now(timezone.utc)
    finally:
        holder.release()
        t.join(timeout=2.0)


def test_timestamp_is_not_refreshed_by_later_scheduler_ticks(tmp_path: Path) -> None:
    """The whole point: it must record the *first* request, not the last tick.

    ``_request_reclaim_to_resolve`` runs on every poll of every waiter. If it
    re-stamped, a nine-day-old pending reclaim would read as milliseconds old —
    the field would look healthy exactly when it matters most.
    """
    pool = _make_pool(tmp_path, ram_mb=700_000)
    holder = pool.try_acquire(ram_mb=650_000, reclaimable=True, priority=0, label="batch")
    assert holder is not None

    def _high_pri() -> None:
        with pool.acquire(ram_mb=650_000, priority=10, poll_interval=0.05):
            pass

    t = threading.Thread(target=_high_pri, daemon=True)
    t.start()
    try:
        first = _wait_for_reclaim(pool, holder.id).reclaim_requested_at
        assert first is not None

        # Let the waiter poll several more times without the holder yielding.
        time.sleep(0.4)

        again = _lease_in_file(pool, holder.id)
        assert again is not None
        assert again.reclaim_requested_at == first
    finally:
        holder.release()
        t.join(timeout=2.0)


def test_age_survives_an_unrelated_transact(tmp_path: Path) -> None:
    """A pending reclaim must not be reset by other consumers' writes.

    Every ``acquire`` / ``release`` / ``set_resources`` rewrites the whole file,
    so a field that only lived in one writer's memory would be lost on the next
    foreign transaction — which is what made the isidore-side workaround
    (measuring first-observation per lease id) necessary in the first place.
    """
    pool = _make_pool(tmp_path, ram_mb=700_000)
    holder = pool.try_acquire(ram_mb=10_000, reclaimable=True, label="batch")
    assert holder is not None

    stamped = datetime.now(timezone.utc) - timedelta(days=9)

    def _stamp(st: State) -> None:
        for lease in st.leases:
            if lease.id == holder.id:
                lease.reclaim_requested = True
                lease.reclaim_requested_at = stamped

    transact(pool._path, _stamp)  # pyright: ignore[reportPrivateUsage]

    with pool.acquire(ram_mb=5_000, label="somebody-else"):
        pass
    pool.set_resources({"ram_mb": 700_000})

    lease = _lease_in_file(pool, holder.id)
    assert lease is not None
    assert lease.reclaim_requested_at == stamped

    holder.release()
