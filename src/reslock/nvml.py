"""NVIDIA driver pre-flight check via pynvml.

Lets ``ResourcePool.acquire`` cross-check its internal lease accounting against
the NVIDIA driver's actual free VRAM before granting a GPU lease. The driver is
the only ground truth across all consumers (containerised scriba, host
aiserver, host one-off scripts, kernel allocator overhead). If reslock thinks a
GPU has free VRAM but the driver disagrees — typically because some process
holds VRAM without a registered lease — pre-flight refuses the grant and the
existing reclaim loop proceeds to evict reclaimable leases instead of letting
``cudaMalloc`` fail downstream.

This module is intentionally minimal: it never inits CUDA contexts and never
imports torch. ``nvmlInit()`` is the same call ``nvidia-smi`` makes; cost is in
the milliseconds.

Hard-fail policy: if a lease request includes any ``gpu_<uuid>_vram_mb`` key
and pynvml is missing or ``nvmlInit()`` fails, we raise. CUDA-mode reslock
without NVML means the driver isn't installed correctly, and silently falling
back to ``state.json``-only accounting hides the gap that pre-flight is meant
to expose.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from reslock.detect import parse_gpu_vram_key

_lock = threading.Lock()
_pynvml: Any = None
_initialized = False
_free_cache: dict[str, int] = {}
_free_cache_at: float = 0.0


class NvmlUnavailableError(RuntimeError):
    """pynvml is not importable, or nvmlInit failed."""


def _import_pynvml() -> Any:
    global _pynvml
    if _pynvml is not None:
        return _pynvml
    try:
        import pynvml  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
    except ImportError as ex:
        raise NvmlUnavailableError(
            "reslock: GPU VRAM lease requested but pynvml (nvidia-ml-py) is not installed. "
            "Install with `pip install nvidia-ml-py` or `pip install reslock[cuda]`."
        ) from ex
    _pynvml = pynvml
    return pynvml


def _ensure_initialized() -> Any:
    global _initialized
    if _initialized:
        return _pynvml
    with _lock:
        if _initialized:
            return _pynvml
        m = _import_pynvml()
        try:
            m.nvmlInit()
        except Exception as ex:
            raise NvmlUnavailableError(
                f"reslock: pynvml is installed but nvmlInit() failed: {ex}. "
                "Check the NVIDIA driver is installed and accessible."
            ) from ex
        _initialized = True
    return _pynvml


def request_uses_gpu_vram(resources: dict[str, int]) -> bool:
    """True if any resource key is a per-GPU VRAM key (``gpu_<uuid>_vram_mb``)."""
    return any(parse_gpu_vram_key(k) is not None for k in resources)


def nvml_free_vram_mb(cache_seconds: float = 1.0) -> dict[str, int]:
    """Read free VRAM per GPU UUID from the NVIDIA driver.

    Returns ``{gpu_uuid: free_mb}`` for every visible GPU. Results are cached
    for ``cache_seconds`` so burst-acquires don't hammer the driver.

    Raises ``NvmlUnavailableError`` if pynvml is missing or ``nvmlInit()``
    fails. Callers must guard with :func:`request_uses_gpu_vram` to skip the
    pre-flight when the request has no GPU keys.
    """
    global _free_cache, _free_cache_at
    now = time.monotonic()
    if _free_cache and (now - _free_cache_at) < cache_seconds:
        return dict(_free_cache)
    m = _ensure_initialized()
    out: dict[str, int] = {}
    count = int(m.nvmlDeviceGetCount())
    for i in range(count):
        handle = m.nvmlDeviceGetHandleByIndex(i)
        uuid_raw = m.nvmlDeviceGetUUID(handle)
        uuid_str = uuid_raw.decode() if isinstance(uuid_raw, bytes) else str(uuid_raw)
        mem = m.nvmlDeviceGetMemoryInfo(handle)
        free_bytes = int(mem.free)
        out[uuid_str] = free_bytes // (1024 * 1024)
    _free_cache = out
    _free_cache_at = now
    return dict(out)


def compute_nvml_shortfall(resources: dict[str, int], nvml_free: dict[str, int]) -> dict[str, int]:
    """Per-key shortfall: how much more VRAM the request needs beyond NVML free.

    For each ``gpu_<uuid>_vram_mb`` key in *resources*, compares the requested
    amount against ``nvml_free[uuid]``. Missing UUIDs are treated as 0 free
    (a UUID we expected but didn't see counts as fully held by an external
    consumer). Non-GPU keys are ignored.

    Returns the keys whose deficit is positive; empty dict means the driver
    has enough headroom for the request.
    """
    short: dict[str, int] = {}
    for key, requested in resources.items():
        uuid_str = parse_gpu_vram_key(key)
        if uuid_str is None:
            continue
        free = nvml_free.get(uuid_str, 0)
        if requested > free:
            short[key] = requested - free
    return short


def reset_for_test() -> None:
    """Reset module-level cache and init flags. Tests only."""
    global _initialized, _free_cache, _free_cache_at, _pynvml
    with _lock:
        _initialized = False
        _free_cache = {}
        _free_cache_at = 0.0
        _pynvml = None
