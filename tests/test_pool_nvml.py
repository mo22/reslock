"""Pool-level tests for the pynvml pre-flight integration.

These exercise the SCRIBA-325 scenario end-to-end: an external (unaccounted)
process holding VRAM, internal lease accounting saying the GPU has free
headroom, NVML correctly reporting the gap, and reslock either refusing the
lease (``try_acquire``) or signalling reclaim on opportunistic leases that
together cover the shortfall (blocking ``acquire``).
"""

from __future__ import annotations

import builtins
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from reslock import nvml
from reslock.detect import gpu_vram_key
from reslock.pool import ResourcePool

UUID_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
UUID_B = "GPU-bbbbbbbb-1111-2222-3333-444444444444"


def _fake_pynvml(devices: list[tuple[str, int, int]]) -> Any:
    module = types.ModuleType("pynvml")

    def _init() -> None:
        return None

    def _count() -> int:
        return len(devices)

    def _by_index(i: int) -> int:
        return i

    def _uuid(handle: int) -> str:
        return devices[handle][0]

    class _Mem:
        def __init__(self, total_mb: int, free_mb: int) -> None:
            self.total = total_mb * 1024 * 1024
            self.free = free_mb * 1024 * 1024
            self.used = self.total - self.free

    def _mem(handle: int) -> _Mem:
        _, total_mb, free_mb = devices[handle]
        return _Mem(total_mb, free_mb)

    module.nvmlInit = _init  # type: ignore[attr-defined]
    module.nvmlDeviceGetCount = _count  # type: ignore[attr-defined]
    module.nvmlDeviceGetHandleByIndex = _by_index  # type: ignore[attr-defined]
    module.nvmlDeviceGetUUID = _uuid  # type: ignore[attr-defined]
    module.nvmlDeviceGetMemoryInfo = _mem  # type: ignore[attr-defined]
    return module


@pytest.fixture(autouse=True)
def reset_nvml_state() -> Iterator[None]:
    nvml.reset_for_test()
    yield
    nvml.reset_for_test()


# --- non-GPU requests must not touch pynvml ---


def test_non_gpu_acquire_does_not_require_pynvml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    real_import = builtins.__import__

    def _block(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pynvml":
            raise ImportError("no pynvml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)

    pool = ResourcePool(tmp_path / "state.json")
    pool.set_resources({"ram_mb": 8000})
    h = pool.try_acquire(ram_mb=1000)
    assert h is not None
    h.release()


# --- try_acquire pre-flight ---


def test_try_acquire_grants_when_nvml_has_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([(UUID_A, 24000, 20000)]))
    pool = ResourcePool(tmp_path / "state.json")
    pool.set_resources({gpu_vram_key(UUID_A): 24000})

    h = pool.try_acquire(**{gpu_vram_key(UUID_A): 14000})  # pyright: ignore[reportArgumentType]
    assert h is not None
    h.release()


def test_try_acquire_refuses_when_nvml_short_despite_internal_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Internal accounting: 24 GB total, no leases → 24 GB free.
    # NVML: only 4 GB free (some external process holds 20 GB).
    # Request: 14 GB. Internal says fit, NVML says short → refuse.
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([(UUID_A, 24000, 4000)]))
    pool = ResourcePool(tmp_path / "state.json")
    pool.set_resources({gpu_vram_key(UUID_A): 24000})

    h = pool.try_acquire(**{gpu_vram_key(UUID_A): 14000})  # pyright: ignore[reportArgumentType]
    assert h is None


def test_try_acquire_raises_when_pynvml_missing_for_gpu_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    real_import = builtins.__import__

    def _block(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pynvml":
            raise ImportError("no pynvml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)

    pool = ResourcePool(tmp_path / "state.json")
    pool.set_resources({gpu_vram_key(UUID_A): 24000})

    with pytest.raises(nvml.NvmlUnavailableError):
        pool.try_acquire(**{gpu_vram_key(UUID_A): 1000})  # pyright: ignore[reportArgumentType]


# --- blocking acquire: reclaim signalling on NVML drift ---


def _make_dynamic_pynvml(free_box: dict[str, int], total_mb: int = 24000) -> Any:
    """Build a stub pynvml module that reports ``free_box['free']`` MB free.

    Tests mutate ``free_box['free']`` between phases to simulate an external
    process arriving / leaving / a holder dropping its model.
    """
    module = types.ModuleType("pynvml")

    def _init() -> None:
        return None

    def _count() -> int:
        return 1

    def _by_index(_i: int) -> int:
        return 0

    def _uuid(_h: int) -> str:
        return UUID_A

    class _Mem:
        def __init__(self, total_mb_: int, free_mb_: int) -> None:
            self.total = total_mb_ * 1024 * 1024
            self.free = free_mb_ * 1024 * 1024
            self.used = self.total - self.free

    def _mem(_h: int) -> _Mem:
        return _Mem(total_mb, free_box["free"])

    module.nvmlInit = _init  # type: ignore[attr-defined]
    module.nvmlDeviceGetCount = _count  # type: ignore[attr-defined]
    module.nvmlDeviceGetHandleByIndex = _by_index  # type: ignore[attr-defined]
    module.nvmlDeviceGetUUID = _uuid  # type: ignore[attr-defined]
    module.nvmlDeviceGetMemoryInfo = _mem  # type: ignore[attr-defined]
    return module


def test_blocking_acquire_requests_reclaim_when_nvml_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Realistic SCRIBA-325 shape — a holder acquired its lease when VRAM was
    available, then an external (unaccounted) process appeared and consumed
    VRAM. Internal accounting still says the GPU has headroom for a new
    lease, but NVML disagrees. Reslock should mark the reclaimable holder for
    reclaim so the holder can drop its cache and free the gap.
    """
    free_box = {"free": 19000}  # holder hasn't started yet, 24-5=19 free
    monkeypatch.setitem(sys.modules, "pynvml", _make_dynamic_pynvml(free_box))

    pool = ResourcePool(tmp_path / "state.json")
    key = gpu_vram_key(UUID_A)
    pool.set_resources({key: 24000})

    # Phase 1: holder grabs a 5 GB reclaimable lease while NVML has headroom.
    holder = pool.try_acquire(reclaimable=True, label="opportunistic", **{key: 5000})  # pyright: ignore[reportArgumentType]
    assert holder is not None

    # Phase 2: an external process consumes 15 GB (invisible to reslock).
    # NVML free drops to 4 GB. Reslock-internal: 24-5=19 free still.
    free_box["free"] = 4000
    nvml.reset_for_test()  # bust 1s cache so the next read sees the new value

    # Waiter tries to acquire 14 GB. Internal says fit (19 free); NVML says
    # short (only 4 free). Reslock should mark the 5 GB reclaimable holder.
    def _waiter() -> None:
        try:
            with pool.acquire(label="needs-vram", poll_interval=0.05, **{key: 14000}):  # pyright: ignore[reportArgumentType]
                pass
        except BaseException:
            return

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()

    deadline = time.monotonic() + 2.0
    flipped = False
    while time.monotonic() < deadline:
        if holder.reclaim_requested:
            flipped = True
            break
        time.sleep(0.02)

    holder.release()
    t.join(timeout=1.0)

    assert flipped, "reclaim_requested should have been set on the opportunistic holder"


def test_blocking_acquire_grants_after_holder_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cooperative reclaim path: holder drops the lease after seeing
    reclaim_requested, NVML free goes up, waiter is granted.
    """
    free_box = {"free": 19000}
    monkeypatch.setitem(sys.modules, "pynvml", _make_dynamic_pynvml(free_box))

    pool = ResourcePool(tmp_path / "state.json")
    key = gpu_vram_key(UUID_A)
    pool.set_resources({key: 24000})

    holder = pool.try_acquire(reclaimable=True, label="opportunistic", **{key: 5000})  # pyright: ignore[reportArgumentType]
    assert holder is not None

    # External process arrives. NVML drops; internal still thinks fit.
    free_box["free"] = 4000
    nvml.reset_for_test()

    granted: dict[str, Any] = {}

    def _waiter() -> None:
        with pool.acquire(label="needs-vram", poll_interval=0.05, **{key: 14000}) as h:  # pyright: ignore[reportArgumentType]
            granted["id"] = h.id

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if holder.reclaim_requested:
            break
        time.sleep(0.02)
    assert holder.reclaim_requested

    # Holder cooperatively releases; the external process also exits at the
    # same time so NVML now reports the full 24 GB free.
    holder.release()
    free_box["free"] = 24000
    nvml.reset_for_test()

    t.join(timeout=3.0)
    assert "id" in granted, "waiter should have been granted a lease"


# --- priority gate must respect NVML pre-flight ---


def test_priority_gate_does_not_block_on_nvml_short_higher_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex review repro: a high-priority GPU waiter that internally fits but
    is NVML-short must NOT block a lower-priority non-GPU waiter from being
    promoted. The old gate checked ``state.can_fit()`` only and would let the
    GPU entry block the RAM waiter forever.
    """
    # NVML reports only 1 GB free on the GPU — internal accounting still
    # thinks the full 24 GB is available because no GPU lease is held.
    free_box = {"free": 1000}
    monkeypatch.setitem(sys.modules, "pynvml", _make_dynamic_pynvml(free_box))

    pool = ResourcePool(tmp_path / "state.json")
    gpu_key = gpu_vram_key(UUID_A)
    pool.set_resources({gpu_key: 24000, "ram_mb": 8000})

    ram_granted: dict[str, Any] = {}

    def _gpu_waiter() -> None:
        # High priority. Internal-fit yes, NVML-short yes → stays queued.
        try:
            with pool.acquire(priority=10, poll_interval=0.05, **{gpu_key: 14000}):  # pyright: ignore[reportArgumentType]
                pass
        except BaseException:
            return

    def _ram_waiter() -> None:
        with pool.acquire(priority=0, poll_interval=0.05, ram_mb=1000) as h:
            ram_granted["id"] = h.id

    gpu_thread = threading.Thread(target=_gpu_waiter, daemon=True)
    gpu_thread.start()

    # Give the GPU waiter a moment to enqueue so the RAM waiter sees it ahead.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if any(e.priority == 10 for e in pool.status().queue):
            break
        time.sleep(0.02)
    assert any(e.priority == 10 for e in pool.status().queue), (
        "GPU waiter should be enqueued before the RAM waiter starts"
    )

    ram_thread = threading.Thread(target=_ram_waiter, daemon=True)
    ram_thread.start()
    ram_thread.join(timeout=2.0)

    assert "id" in ram_granted, (
        "lower-priority RAM waiter must be promoted past an NVML-short higher-pri GPU waiter"
    )

    # Drain the GPU waiter so the test exits cleanly.
    free_box["free"] = 24000
    nvml.reset_for_test()
    gpu_thread.join(timeout=2.0)
