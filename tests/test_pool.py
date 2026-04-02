from __future__ import annotations

from pathlib import Path

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
