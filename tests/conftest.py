"""Shared pytest config: stub pynvml so non-NVML tests don't trip the pre-flight.

Reslock's NVML pre-flight hard-fails when a GPU is requested but pynvml /
``nvmlInit()`` is unavailable — that's correct production behavior (see
``reslock.nvml`` for the rationale). On a Mac dev box pynvml is installed for
typing but ``nvmlInit()`` raises ``NVMLError_LibraryNotFound``, so any test
that exercises the GPU code paths needs a fake.

Tests in ``test_pool_nvml.py`` install their own pynvml fakes per-test to
exercise driver-vs-accounting drift; this conftest steps out of the way for
that file.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

# UUIDs used across the test suite. Keep in sync with the per-file constants.
FAKE_GPU_UUIDS: list[str] = [
    "GPU-aaaaaaaa-1111-2222-3333-444444444444",
    "GPU-bbbbbbbb-1111-2222-3333-444444444444",
    "GPU-cccccccc-1111-2222-3333-444444444444",
]


def _build_fake_pynvml(uuids: list[str], free_mb: int = 9_999_999) -> Any:
    module = types.ModuleType("pynvml")

    def _init() -> None:
        return None

    def _count() -> int:
        return len(uuids)

    def _by_index(i: int) -> int:
        return i

    def _uuid(handle: int) -> str:
        return uuids[handle]

    class _Mem:
        def __init__(self, mb: int) -> None:
            self.total = mb * 1024 * 1024
            self.free = mb * 1024 * 1024
            self.used = 0

    def _mem(_h: int) -> _Mem:
        return _Mem(free_mb)

    module.nvmlInit = _init  # type: ignore[attr-defined]
    module.nvmlDeviceGetCount = _count  # type: ignore[attr-defined]
    module.nvmlDeviceGetHandleByIndex = _by_index  # type: ignore[attr-defined]
    module.nvmlDeviceGetUUID = _uuid  # type: ignore[attr-defined]
    module.nvmlDeviceGetMemoryInfo = _mem  # type: ignore[attr-defined]
    return module


@pytest.fixture(autouse=True)
def _autouse_fake_pynvml(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Install a permissive pynvml fake — every known GPU has plenty of free VRAM.

    Tests that explicitly want to exercise NVML drift install their own fake
    via ``monkeypatch.setitem(sys.modules, "pynvml", ...)`` later in the test
    body, which overrides this one.
    """
    basename = os.path.basename(str(request.node.fspath))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    if basename == "test_pool_nvml.py":
        # Per-test fakes manage NVML state explicitly; don't shadow them.
        return
    fake = _build_fake_pynvml(FAKE_GPU_UUIDS)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    from reslock import nvml

    nvml.reset_for_test()
