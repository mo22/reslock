"""Unit tests for the pynvml pre-flight wrapper.

These tests never call into a real NVIDIA driver. They exercise the pure
helpers (``request_uses_gpu_vram``, ``compute_nvml_shortfall``) and stub the
pynvml module via ``sys.modules`` for the import-and-init paths.
"""

from __future__ import annotations

import builtins
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from reslock import nvml


def _fake_pynvml(
    devices: list[tuple[str, int, int]],
    *,
    init_raises: bool = False,
    uuid_returns_bytes: bool = False,
) -> Any:
    """Build a stub `pynvml` module exposing the surface reslock uses.

    Each entry in *devices* is ``(uuid, total_mb, free_mb)``.
    """

    module = types.ModuleType("pynvml")

    def _init() -> None:
        if init_raises:
            raise RuntimeError("driver missing")

    handles = list(range(len(devices)))

    def _count() -> int:
        return len(devices)

    def _by_index(i: int) -> int:
        return handles[i]

    def _uuid(handle: int) -> Any:
        u = devices[handle][0]
        return u.encode() if uuid_returns_bytes else u

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


# --- pure helpers ---


def test_request_uses_gpu_vram_detects_gpu_keys() -> None:
    assert nvml.request_uses_gpu_vram({"gpu_GPU-abc_vram_mb": 1000}) is True


def test_request_uses_gpu_vram_ignores_other_keys() -> None:
    assert nvml.request_uses_gpu_vram({"ram_mb": 1000, "cpu_cores": 4}) is False
    # Old index-based key (pre v0.5) is no longer recognised — pre-flight
    # would skip, which is fine because index-based keys are also no longer
    # produced by detect_gpu_vram_mb().
    assert nvml.request_uses_gpu_vram({"gpu0_vram_mb": 1000}) is False


def test_compute_nvml_shortfall_when_free_sufficient() -> None:
    free = {"GPU-aaa": 20000, "GPU-bbb": 18000}
    req = {"gpu_GPU-aaa_vram_mb": 5000, "gpu_GPU-bbb_vram_mb": 18000}
    assert nvml.compute_nvml_shortfall(req, free) == {}


def test_compute_nvml_shortfall_partial_deficit() -> None:
    free = {"GPU-aaa": 5000, "GPU-bbb": 20000}
    req = {"gpu_GPU-aaa_vram_mb": 12000, "gpu_GPU-bbb_vram_mb": 6000}
    short = nvml.compute_nvml_shortfall(req, free)
    assert short == {"gpu_GPU-aaa_vram_mb": 7000}


def test_compute_nvml_shortfall_missing_uuid_treated_as_zero_free() -> None:
    free: dict[str, int] = {}
    req = {"gpu_GPU-aaa_vram_mb": 100}
    assert nvml.compute_nvml_shortfall(req, free) == {"gpu_GPU-aaa_vram_mb": 100}


def test_compute_nvml_shortfall_ignores_non_gpu_keys() -> None:
    free = {"GPU-aaa": 1000}
    req = {"ram_mb": 9999, "gpu_GPU-aaa_vram_mb": 500}
    assert nvml.compute_nvml_shortfall(req, free) == {}


# --- import + init paths ---


def test_nvml_free_vram_mb_reads_via_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_pynvml([("GPU-aaa", 24000, 14600), ("GPU-bbb", 24000, 20100)])
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    out = nvml.nvml_free_vram_mb(cache_seconds=0)
    assert out == {"GPU-aaa": 14600, "GPU-bbb": 20100}


def test_nvml_free_vram_mb_handles_bytes_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_pynvml([("GPU-aaa", 24000, 5000)], uuid_returns_bytes=True)
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    out = nvml.nvml_free_vram_mb(cache_seconds=0)
    assert out == {"GPU-aaa": 5000}


def test_nvml_free_vram_mb_caches_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_pynvml([("GPU-aaa", 24000, 12000)])
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    calls: list[int] = []

    orig = fake.nvmlDeviceGetMemoryInfo

    def _spy(h: int) -> Any:
        calls.append(h)
        return orig(h)

    fake.nvmlDeviceGetMemoryInfo = _spy  # type: ignore[attr-defined]

    nvml.nvml_free_vram_mb(cache_seconds=10)
    nvml.nvml_free_vram_mb(cache_seconds=10)
    nvml.nvml_free_vram_mb(cache_seconds=10)
    # Only one batch of device queries — subsequent calls served from cache.
    assert len(calls) == 1


def test_nvml_free_vram_mb_raises_when_pynvml_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    real_import = builtins.__import__

    def _block(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pynvml":
            raise ImportError("no pynvml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)

    with pytest.raises(nvml.NvmlUnavailableError, match="pynvml"):
        nvml.nvml_free_vram_mb(cache_seconds=0)


def test_nvml_free_vram_mb_raises_when_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_pynvml([("GPU-aaa", 24000, 1000)], init_raises=True)
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    with pytest.raises(nvml.NvmlUnavailableError, match="nvmlInit"):
        nvml.nvml_free_vram_mb(cache_seconds=0)
