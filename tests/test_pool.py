from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from reslock import ResourcePool
from reslock.models import Lease, State
from reslock.state import read_state, transact


def _make_pool(tmp_path: Path, **resources: int) -> ResourcePool:
    state_path = tmp_path / "state.json"
    pool = ResourcePool(state_path)

    def _set(st: State) -> None:
        st.resources.update(resources)

    transact(state_path, _set)
    return pool


def test_acquire_release(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, vram_mb=8000)
    with pool.acquire(vram_mb=4000):
        st = pool.status()
        assert st.available["vram_mb"] == 4000
        assert len(st.leases) == 1
    st = pool.status()
    assert st.available["vram_mb"] == 8000
    assert len(st.leases) == 0


def test_try_acquire_success(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, vram_mb=8000)
    lease = pool.try_acquire(vram_mb=4000)
    assert lease is not None
    assert pool.status().available["vram_mb"] == 4000
    lease.release()
    assert pool.status().available["vram_mb"] == 8000


def test_try_acquire_insufficient(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, vram_mb=4000)
    lease = pool.try_acquire(vram_mb=8000)
    assert lease is None


def test_multiple_leases(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, vram_mb=8000)
    l1 = pool.try_acquire(vram_mb=3000)
    l2 = pool.try_acquire(vram_mb=3000)
    l3 = pool.try_acquire(vram_mb=3000)
    assert l1 is not None
    assert l2 is not None
    assert l3 is None  # only 2000 left
    l1.release()
    l3 = pool.try_acquire(vram_mb=3000)
    assert l3 is not None
    l2.release()
    l3.release()


def test_status(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, vram_mb=8000, ram_mb=16000)
    st = pool.status()
    assert st.resources == {"vram_mb": 8000, "ram_mb": 16000}
    assert st.available == {"vram_mb": 8000, "ram_mb": 16000}


def test_set_resources(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    pool = ResourcePool(state_path)

    # Register initial resources
    pool.set_resources({"gpu0_vram_mb": 24000, "cpu_cores": 16})
    st = pool.status()
    assert st.resources == {"gpu0_vram_mb": 24000, "cpu_cores": 16}

    # Second call overwrites existing keys, leaves others
    pool.set_resources({"gpu0_vram_mb": 48000, "ram_mb": 65536})
    st = pool.status()
    assert st.resources == {"gpu0_vram_mb": 48000, "cpu_cores": 16, "ram_mb": 65536}


def test_set_resources_then_acquire(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    pool = ResourcePool(state_path)
    pool.set_resources({"vram_mb": 8000})

    with pool.acquire(vram_mb=3000):
        assert pool.status().available["vram_mb"] == 5000
    assert pool.status().available["vram_mb"] == 8000


def test_status_cleans_dead_leases(tmp_path: Path) -> None:
    """status() should remove leases from dead PIDs and persist the cleanup."""
    pool = _make_pool(tmp_path, vram_mb=8000)
    state_path = tmp_path / "state.json"

    # Inject a lease with a dead PID directly into the state file
    def _inject(st: State) -> None:
        st.leases.append(Lease(pid=999999999, resources={"vram_mb": 4000}))

    transact(state_path, _inject)

    # Verify the dead lease is in the file
    raw = read_state(state_path)
    assert any(ls.pid == 999999999 for ls in raw.leases)

    # status() should clean it up
    st = pool.status()
    assert len(st.leases) == 0
    assert st.available["vram_mb"] == 8000

    # Verify the cleanup was persisted to disk
    raw = read_state(state_path)
    assert len(raw.leases) == 0


def test_shrink_reduces_available(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=12000)
    lease = pool.try_acquire(disk_mb=12000)
    assert lease is not None
    assert pool.status().available["disk_mb"] == 0
    lease.shrink(disk_mb=4000)
    assert pool.status().available["disk_mb"] == 4000
    lease.shrink(disk_mb=3000)
    assert pool.status().available["disk_mb"] == 7000
    lease.release()
    assert pool.status().available["disk_mb"] == 12000


def test_shrink_to_zero_releases_lease(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=5000)
    lease = pool.try_acquire(disk_mb=5000)
    assert lease is not None
    lease.shrink(disk_mb=5000)
    st = pool.status()
    assert len(st.leases) == 0
    assert st.available["disk_mb"] == 5000
    # Further shrinks are no-ops (already released).
    lease.shrink(disk_mb=1)


def test_shrink_partial_to_zero_keeps_lease(tmp_path: Path) -> None:
    """A multi-resource lease stays alive as long as any key has capacity."""
    pool = _make_pool(tmp_path, disk_mb=8000, ram_mb=4000)
    lease = pool.try_acquire(disk_mb=8000, ram_mb=4000)
    assert lease is not None
    lease.shrink(disk_mb=8000)
    st = pool.status()
    assert len(st.leases) == 1
    assert "disk_mb" not in st.leases[0].resources
    assert st.leases[0].resources == {"ram_mb": 4000}
    assert st.available == {"disk_mb": 8000, "ram_mb": 0}


def test_shrink_negative_raises(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=5000)
    lease = pool.try_acquire(disk_mb=5000)
    assert lease is not None
    with pytest.raises(ValueError, match="non-negative"):
        lease.shrink(disk_mb=-1)
    # State must be unchanged on rejection.
    assert pool.status().available["disk_mb"] == 0
    lease.release()


def test_shrink_below_zero_raises(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=5000)
    lease = pool.try_acquire(disk_mb=1000)
    assert lease is not None
    with pytest.raises(ValueError, match="below zero"):
        lease.shrink(disk_mb=2000)
    assert pool.status().available["disk_mb"] == 4000
    lease.release()


def test_shrink_unknown_key_raises(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=5000)
    lease = pool.try_acquire(disk_mb=1000)
    assert lease is not None
    with pytest.raises(ValueError, match="does not hold"):
        lease.shrink(vram_mb=100)
    lease.release()


def test_shrink_after_release_is_noop(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=5000)
    lease = pool.try_acquire(disk_mb=5000)
    assert lease is not None
    lease.release()
    # No exception, no state change.
    lease.shrink(disk_mb=1000)
    assert pool.status().available["disk_mb"] == 5000


def test_shrink_promotes_queued_waiter(tmp_path: Path) -> None:
    """A waiter blocked on capacity should unblock once shrink() frees enough."""
    pool = _make_pool(tmp_path, disk_mb=10000)
    holder = pool.try_acquire(disk_mb=10000)
    assert holder is not None

    acquired = threading.Event()

    def _waiter() -> None:
        with pool.acquire(disk_mb=4000, poll_interval=0.05):
            acquired.set()

    t = threading.Thread(target=_waiter)
    t.start()
    try:
        # Waiter should still be blocked.
        time.sleep(0.2)
        assert not acquired.is_set()

        holder.shrink(disk_mb=4000)

        assert acquired.wait(timeout=2.0), "waiter did not unblock after shrink"
    finally:
        holder.release()
        t.join(timeout=2.0)
