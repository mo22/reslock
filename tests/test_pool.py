from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from reslock import ResourcePool
from reslock.models import Lease, State
from reslock.pool import LeaseHandle
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


def test_try_acquire_wait_sec_zero(tmp_path: Path) -> None:
    """try_acquire skips the queue path, so wait_sec is exactly 0.0."""
    pool = _make_pool(tmp_path, vram_mb=8000)
    lease = pool.try_acquire(vram_mb=4000)
    assert lease is not None
    assert lease.wait_sec == 0.0
    lease.release()


def test_acquire_wait_sec_reflects_queue_time(tmp_path: Path) -> None:
    """Waiters promoted from the queue see wait_sec ≥ time spent blocked."""
    pool = _make_pool(tmp_path, vram_mb=8000)
    holder = pool.try_acquire(vram_mb=8000)
    assert holder is not None

    poll_interval = 0.05
    block_seconds = 0.3
    waiter_handle: list[LeaseHandle] = []

    def _waiter() -> None:
        h = pool._acquire_blocking(  # pyright: ignore[reportPrivateUsage]
            vram_mb_each=None,
            num_gpus=0,
            non_gpu={"vram_mb": 4000},
            priority=0,
            reclaimable=False,
            estimated_seconds=None,
            label=None,
            poll_interval=poll_interval,
        )
        waiter_handle.append(h)

    t = threading.Thread(target=_waiter)
    t.start()
    try:
        time.sleep(block_seconds)
        holder.release()
        t.join(timeout=2.0)
        assert waiter_handle, "waiter never acquired"
        h = waiter_handle[0]
        assert h.wait_sec is not None
        assert h.wait_sec >= block_seconds * 0.8  # some slack for timing
    finally:
        for h in waiter_handle:
            h.release()


def test_lease_handle_gpu_uuids(tmp_path: Path) -> None:
    """gpu_uuids parses GPU UUIDs out of `gpu_{uuid}_vram_mb` resource keys."""
    from reslock.pool import LeaseHandle as _LeaseHandle

    uuid_a = "GPU-1a2b3c4d-5e6f-7890-abcd-ef1234567890"
    uuid_b = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    pool = ResourcePool(tmp_path / "state.json")
    lease = Lease(
        pid=1,
        resources={
            f"gpu_{uuid_a}_vram_mb": 8000,
            f"gpu_{uuid_b}_vram_mb": 8000,
            "ram_mb": 1000,
        },
    )
    handle = _LeaseHandle(lease, pool)
    assert sorted(handle.gpu_uuids) == sorted([uuid_a, uuid_b])


def test_lease_handle_gpu_uuids_empty_without_gpu_keys(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, vram_mb=8000)
    lease = pool.try_acquire(vram_mb=4000)
    assert lease is not None
    assert lease.gpu_uuids == []
    lease.release()


def test_update_resources_shrinks_reservation(tmp_path: Path) -> None:
    """update(resources=) sets a smaller reservation; state + available reflect it."""
    pool = _make_pool(tmp_path, disk_mb=8000, ram_mb=4000)
    lease = pool.try_acquire(disk_mb=8000, ram_mb=4000)
    assert lease is not None
    assert pool.status().available == {"disk_mb": 0, "ram_mb": 0}

    lease.update(resources={"disk_mb": 4000, "ram_mb": 2000})

    st = pool.status()
    assert st.leases[0].resources == {"disk_mb": 4000, "ram_mb": 2000}
    assert st.available == {"disk_mb": 4000, "ram_mb": 2000}
    # Cached handle reflects the shrunk set.
    assert lease.resources == {"disk_mb": 4000, "ram_mb": 2000}
    lease.release()


def test_update_resources_frees_capacity_for_next_acquire(tmp_path: Path) -> None:
    """A request that didn't fit before the shrink fits after it."""
    pool = _make_pool(tmp_path, vram_mb=8000)
    holder = pool.try_acquire(vram_mb=6000)
    assert holder is not None
    # Only 2000 free — a 4000 request can't fit yet.
    assert pool.try_acquire(vram_mb=4000) is None

    holder.update(resources={"vram_mb": 3000})

    # Now 5000 free — the 4000 request fits.
    other = pool.try_acquire(vram_mb=4000)
    assert other is not None
    other.release()
    holder.release()


def _inject_gpu_lease(pool: ResourcePool, resources: dict[str, int]) -> LeaseHandle:
    """Inject a live (current-pid) lease holding the given resources and return a handle.

    GPU-keyed leases can't be created via the v3 acquire() path (it only takes
    vram_mb_each + num_gpus and binds UUIDs at promotion). Tests that need a
    specific UUID-keyed reservation inject it directly, mirroring how the
    scheduler would have resolved it.
    """
    import os

    lease = Lease(pid=os.getpid(), resources=dict(resources))

    def _inject(st: State) -> None:
        st.leases.append(lease)

    transact(pool._path, _inject)  # pyright: ignore[reportPrivateUsage]
    return LeaseHandle(lease, pool)


def test_update_resources_keeps_per_gpu_ratio(tmp_path: Path) -> None:
    """A proportional multi-GPU shrink preserves the per-GPU split."""
    uuid_a = "GPU-1a2b3c4d-5e6f-7890-abcd-ef1234567890"
    uuid_b = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    key_a = f"gpu_{uuid_a}_vram_mb"
    key_b = f"gpu_{uuid_b}_vram_mb"
    pool = _make_pool(tmp_path, **{key_a: 24000, key_b: 24000})
    lease = _inject_gpu_lease(pool, {key_a: 20000, key_b: 19000})

    reserved_total = sum(lease.resources.values())  # 39000
    target_total = 22000
    scale = target_total / reserved_total
    new = {k: int(v * scale) for k, v in lease.resources.items()}
    lease.update(resources=new)

    st = pool.status()
    assert st.leases[0].resources == new
    assert sum(new.values()) <= reserved_total
    # Ratio preserved within rounding.
    assert abs(new[key_a] / new[key_b] - 20000 / 19000) < 0.01
    lease.release()


def test_update_resources_rejects_grow(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=8000)
    lease = pool.try_acquire(disk_mb=4000)
    assert lease is not None
    with pytest.raises(ValueError, match="shrink-only"):
        lease.update(resources={"disk_mb": 5000})
    # State unchanged on rejection.
    assert pool.status().leases[0].resources == {"disk_mb": 4000}
    lease.release()


def test_update_resources_rejects_unknown_key(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=8000)
    lease = pool.try_acquire(disk_mb=4000)
    assert lease is not None
    with pytest.raises(ValueError, match="not held by the lease"):
        lease.update(resources={"vram_mb": 100})
    assert pool.status().leases[0].resources == {"disk_mb": 4000}
    lease.release()


def test_update_resources_rejects_nonpositive(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, disk_mb=8000)
    lease = pool.try_acquire(disk_mb=4000)
    assert lease is not None
    with pytest.raises(ValueError, match="must be > 0"):
        lease.update(resources={"disk_mb": 0})
    assert pool.status().leases[0].resources == {"disk_mb": 4000}
    lease.release()


def test_update_resources_reflected_in_gpu_uuids(tmp_path: Path) -> None:
    """Dropping a GPU key via update(resources=) removes it from gpu_uuids."""
    uuid_a = "GPU-1a2b3c4d-5e6f-7890-abcd-ef1234567890"
    uuid_b = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    key_a = f"gpu_{uuid_a}_vram_mb"
    key_b = f"gpu_{uuid_b}_vram_mb"
    pool = _make_pool(tmp_path, **{key_a: 24000, key_b: 24000})
    lease = _inject_gpu_lease(pool, {key_a: 20000, key_b: 20000})
    assert sorted(lease.gpu_uuids) == sorted([uuid_a, uuid_b])

    # Shrink to a single GPU.
    lease.update(resources={key_a: 18000})
    assert lease.gpu_uuids == [uuid_a]
    assert lease.resources == {key_a: 18000}
    lease.release()


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
