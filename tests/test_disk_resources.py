"""Free-disk-space leases (``disk_mb@<mount>``).

Contract: reserve N MB of free space, use it up (write the bytes), release.
Admission is against live statvfs ground truth, not a registered capacity:
grant only while ``request + sum(active disk leases on the mount) <= actual
free``. No registration needed — written bytes show up in statvfs, so
persistent storage is accounted for automatically once the lease returns.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import reslock.pool as pool_mod
from reslock import ResourcePool, disk_mb_key, parse_disk_mb_key
from reslock.models import State
from reslock.state import transact

DATA = disk_mb_key("/data")


def _make_pool(tmp_path: Path, **resources: int) -> ResourcePool:
    state_path = tmp_path / "state.json"
    pool = ResourcePool(state_path)
    if resources:

        def _set(st: State) -> None:
            st.resources.update(resources)

        transact(state_path, _set)
    return pool


def _patch_free(monkeypatch: pytest.MonkeyPatch, free: dict[str, int]) -> None:
    """Fake statvfs: *free* maps mount path -> free MB (mutable for dynamics)."""
    monkeypatch.setattr(pool_mod, "get_disk_free_mb", lambda p: free.get(p))


# --- key convention ---


def test_disk_key_roundtrip() -> None:
    assert disk_mb_key("/data") == "disk_mb@/data"
    assert parse_disk_mb_key("disk_mb@/data") == "/data"
    assert parse_disk_mb_key("ram_mb") is None
    assert parse_disk_mb_key("gpu_GPU-x_vram_mb") is None


def test_disk_key_normalizes_trailing_slash() -> None:
    assert disk_mb_key("/data/") == disk_mb_key("/data")
    assert disk_mb_key("/") == "disk_mb@/"


# --- admission against live free space ---


def test_acquire_grants_within_actual_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_free(monkeypatch, {"/data": 100_000})
    pool = _make_pool(tmp_path)
    with pool.acquire(disk_mb=16_000, disk_path="/data", label="download") as lease:
        assert lease.resources == {DATA: 16_000}
    assert pool.status().leases == []


def test_refuse_when_leased_sum_would_exceed_actual_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_free(monkeypatch, {"/data": 100_000})
    pool = _make_pool(tmp_path)
    first = pool.try_acquire(disk_mb=60_000, disk_path="/data")
    assert first is not None
    # 60k leased + 60k requested > 100k actual free -> do not download.
    assert pool.try_acquire(disk_mb=60_000, disk_path="/data") is None
    # 40k still fits exactly.
    second = pool.try_acquire(disk_mb=40_000, disk_path="/data")
    assert second is not None
    first.release()
    second.release()


def test_external_writes_shrink_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When actual free drops (bytes being written), new requests see it."""
    free = {"/data": 100_000}
    _patch_free(monkeypatch, free)
    pool = _make_pool(tmp_path)
    holder = pool.try_acquire(disk_mb=60_000, disk_path="/data")
    assert holder is not None
    # An external process (or the holder itself) wrote 55 GB.
    free["/data"] = 45_000
    # 10k + 60k leased > 45k actual free -> refused.
    assert pool.try_acquire(disk_mb=10_000, disk_path="/data") is None
    # Holder finished writing and returns its reservation.
    holder.release()
    lease = pool.try_acquire(disk_mb=10_000, disk_path="/data")
    assert lease is not None
    lease.release()


def test_shrink_returns_headroom_as_bytes_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    free = {"/data": 100_000}
    _patch_free(monkeypatch, free)
    pool = _make_pool(tmp_path)
    holder = pool.try_acquire(disk_mb=60_000, disk_path="/data")
    assert holder is not None
    # 50 GB written: statvfs drops, consumer shrinks its remaining reservation.
    free["/data"] = 50_000
    holder.shrink(**{DATA: 50_000})
    assert holder.resources == {DATA: 10_000}
    # 10k leased + 30k requested <= 50k actual free -> grants.
    lease = pool.try_acquire(disk_mb=30_000, disk_path="/data")
    assert lease is not None
    lease.release()
    holder.release()


def test_mounts_are_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_free(monkeypatch, {"/data": 10_000, "/scratch": 100_000})
    pool = _make_pool(tmp_path)
    holder = pool.try_acquire(disk_mb=10_000, disk_path="/data")
    assert holder is not None
    assert pool.try_acquire(disk_mb=1, disk_path="/data") is None
    other = pool.try_acquire(disk_mb=50_000, disk_path="/scratch")
    assert other is not None
    other.release()
    holder.release()


# --- blocking queue path ---


def test_waiter_promoted_when_holder_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_free(monkeypatch, {"/data": 100_000})
    pool = _make_pool(tmp_path)
    holder = pool.try_acquire(disk_mb=90_000, disk_path="/data")
    assert holder is not None

    got_lease = threading.Event()

    def _waiter() -> None:
        with pool.acquire(disk_mb=90_000, disk_path="/data", poll_interval=0.05):
            got_lease.set()

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    time.sleep(0.3)
    assert not got_lease.is_set()
    holder.release()
    assert got_lease.wait(timeout=2.0)
    t.join(timeout=2.0)


# --- validation ---


def test_disk_kwargs_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_free(monkeypatch, {"/data": 100_000})
    pool = _make_pool(tmp_path)
    with pytest.raises(ValueError, match="disk_mb"):
        pool.try_acquire(disk_mb=0, disk_path="/data")
    with pytest.raises(ValueError, match="disk_path"):
        pool.try_acquire(disk_path="/data")


def test_unstattable_path_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_free(monkeypatch, {})  # no mounts known -> get_disk_free_mb returns None
    pool = _make_pool(tmp_path)
    with pytest.raises(ValueError, match="cannot stat"):
        pool.try_acquire(disk_mb=1000, disk_path="/nonexistent")


def test_default_disk_path_is_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_free(monkeypatch, {"/": 100_000})
    pool = _make_pool(tmp_path)
    lease = pool.try_acquire(disk_mb=1000)
    assert lease is not None
    assert lease.resources == {disk_mb_key("/"): 1000}
    lease.release()


# --- end-to-end against the real filesystem ---


def test_real_statvfs_roundtrip(tmp_path: Path) -> None:
    """1 MB on the tmp filesystem exercises the unpatched statvfs path."""
    pool = _make_pool(tmp_path)
    with pool.acquire(disk_mb=1, disk_path=str(tmp_path)) as lease:
        (key,) = lease.resources
        assert parse_disk_mb_key(key) == str(tmp_path)


# --- interplay with counter resources ---


def test_disk_combines_with_cpu_ram(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_free(monkeypatch, {"/data": 100_000})
    pool = _make_pool(tmp_path, cpu_cores=16, ram_mb=64_000)
    with pool.acquire(cpu_cores=4, ram_mb=8_000, disk_mb=16_000, disk_path="/data") as lease:
        assert lease.resources == {"cpu_cores": 4, "ram_mb": 8_000, DATA: 16_000}
        st = pool.status()
        assert st.available["cpu_cores"] == 12
