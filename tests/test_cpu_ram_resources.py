"""First-class CPU / RAM resource fields (kirk CPU-LLM serving).

Covers the v1 contract from tasks/first-class-cpu-mem-resources.md:
standardized ``cpu_cores`` / ``ram_mb`` keys with explicit acquire kwargs,
capacity detection, and the same queue / priority / reclaim semantics the
VRAM path has — all over the existing counter machinery, no schema bump.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from reslock import (
    CPU_CORES_KEY,
    RAM_MB_KEY,
    ResourcePool,
    detect_cpu_cores,
    detect_ram_mb,
)
from reslock.models import State
from reslock.state import transact
from tests.conftest import FAKE_GPU_UUIDS


def _make_pool(tmp_path: Path, **resources: int) -> ResourcePool:
    state_path = tmp_path / "state.json"
    pool = ResourcePool(state_path)

    def _set(st: State) -> None:
        st.resources.update(resources)

    transact(state_path, _set)
    return pool


# --- detection ---


def test_detect_ram_mb_returns_standard_key() -> None:
    result = detect_ram_mb()
    assert set(result) == {RAM_MB_KEY}
    assert result[RAM_MB_KEY] > 0


def test_detect_ram_mb_subtracts_reserve() -> None:
    base = detect_ram_mb()[RAM_MB_KEY]
    assert detect_ram_mb(reserve_mb=1024)[RAM_MB_KEY] == base - 1024


def test_detect_ram_mb_rejects_negative_reserve() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        detect_ram_mb(reserve_mb=-1)


def test_detect_ram_mb_rejects_reserve_exceeding_total() -> None:
    with pytest.raises(ValueError, match="leaves no capacity"):
        detect_ram_mb(reserve_mb=2**40)


def test_detect_cpu_cores_returns_standard_key() -> None:
    result = detect_cpu_cores()
    assert set(result) == {CPU_CORES_KEY}
    assert result[CPU_CORES_KEY] >= 1


# --- first-class acquire kwargs ---


def test_acquire_with_first_class_kwargs(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, cpu_cores=56, ram_mb=700_000)
    with pool.acquire(cpu_cores=48, ram_mb=650_000, label="kimi") as lease:
        assert lease.resources == {CPU_CORES_KEY: 48, RAM_MB_KEY: 650_000}
        st = pool.status()
        assert st.available[CPU_CORES_KEY] == 8
        assert st.available[RAM_MB_KEY] == 50_000
    st = pool.status()
    assert st.available[CPU_CORES_KEY] == 56
    assert st.available[RAM_MB_KEY] == 700_000


def test_try_acquire_with_first_class_kwargs(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, cpu_cores=16, ram_mb=64_000)
    lease = pool.try_acquire(cpu_cores=8, ram_mb=32_000)
    assert lease is not None
    assert lease.resources == {CPU_CORES_KEY: 8, RAM_MB_KEY: 32_000}
    lease.release()


def test_first_class_kwargs_reject_nonpositive(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, cpu_cores=16, ram_mb=64_000)
    with pytest.raises(ValueError, match="cpu_cores"):
        pool.try_acquire(cpu_cores=0)
    with pytest.raises(ValueError, match="ram_mb"):
        pool.try_acquire(ram_mb=-5)


# --- queue semantics: second CPU-LLM start queues behind the first ---


def test_second_cpu_llm_queues_until_first_releases(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, cpu_cores=56, ram_mb=700_000)
    first = pool.try_acquire(cpu_cores=48, ram_mb=650_000, label="kimi")
    assert first is not None

    # Non-blocking path refuses immediately.
    assert pool.try_acquire(cpu_cores=48, ram_mb=500_000, label="glm") is None

    got_lease = threading.Event()

    def _second() -> None:
        with pool.acquire(cpu_cores=48, ram_mb=500_000, label="glm", poll_interval=0.05):
            got_lease.set()

    t = threading.Thread(target=_second, daemon=True)
    t.start()
    time.sleep(0.3)
    assert not got_lease.is_set()
    assert any(not e.is_active for e in pool.status().queue)

    first.release()
    assert got_lease.wait(timeout=2.0)
    t.join(timeout=2.0)


# --- reclaim semantics over ram_mb, same as VRAM ---


def test_high_priority_reclaims_ram_lease(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path, ram_mb=700_000)
    holder = pool.try_acquire(ram_mb=650_000, reclaimable=True, priority=0, label="batch")
    assert holder is not None

    got_lease = threading.Event()

    def _high_pri() -> None:
        with pool.acquire(ram_mb=650_000, priority=10, poll_interval=0.05):
            got_lease.set()

    t = threading.Thread(target=_high_pri, daemon=True)
    t.start()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not holder.reclaim_requested:
        time.sleep(0.02)
    assert holder.reclaim_requested, "high-priority RAM waiter should trigger reclaim"

    holder.release()
    assert got_lease.wait(timeout=2.0)
    t.join(timeout=2.0)


# --- isolation: CPU/RAM leases leave VRAM accounting and NVML untouched ---


def test_cpu_ram_lease_does_not_touch_gpu_capacity(tmp_path: Path) -> None:
    gpu_key = f"gpu_{FAKE_GPU_UUIDS[0]}_vram_mb"
    pool = _make_pool(tmp_path, cpu_cores=56, ram_mb=700_000, **{gpu_key: 24_000})
    with pool.acquire(cpu_cores=48, ram_mb=650_000):
        st = pool.status()
        assert st.available[gpu_key] == 24_000


def test_cpu_ram_acquire_skips_nvml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-GPU acquires must never touch pynvml (the cuda extra is optional)."""
    import reslock.pool as pool_mod

    def _boom() -> dict[str, int]:
        raise AssertionError("nvml_free_vram_mb must not be called for non-GPU acquires")

    monkeypatch.setattr(pool_mod, "nvml_free_vram_mb", _boom)
    pool = _make_pool(tmp_path, cpu_cores=16, ram_mb=64_000)
    lease = pool.try_acquire(cpu_cores=8, ram_mb=32_000)
    assert lease is not None
    lease.release()
