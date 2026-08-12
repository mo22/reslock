"""Tests for the set_resources GPU-capacity-vs-NVML-total tripwire.

Guards against the 2026-07-20 kirk incident: a consumer persisted free-based
VRAM snapshots as *capacity*, which went stale and blocked promotion on a
physically idle host. The guard warns (never clamps or refuses) when a
registered GPU VRAM capacity is below the NVML-reported physical total.

The autouse conftest fake pynvml reports every GPU in ``FAKE_GPU_UUIDS`` with
a total of 9_999_999 MB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from reslock.detect import gpu_vram_key
from reslock.nvml import NvmlUnavailableError
from reslock.pool import GPU_CAPACITY_MIN_RATIO, ResourcePool
from tests.conftest import FAKE_GPU_UUIDS

FAKE_TOTAL_MB = 9_999_999
GPU_KEY = gpu_vram_key(FAKE_GPU_UUIDS[0])

# Real 2026-08-12 measurements, scaled onto the fake card so the ratios — not
# the absolute numbers — are what the tests pin down.
PHYSICAL_MB = 24_576  # nvidia-smi memory.total, RTX 3090


def _scaled(measured_mb: int) -> int:
    """The fake card's equivalent of a value measured against a real 24576 MB card."""
    return int(FAKE_TOTAL_MB * measured_mb / PHYSICAL_MB)


LINUX_DRIVER_MB = _scaled(24_135)  # cuDeviceTotalMem on kirk — 1.8 % below physical
WINDOWS_DRIVER_MB = _scaled(24_575)  # cuDeviceTotalMem on spock/WDDM — 0.004 % below
FREE_SNAPSHOT_MB = _scaled(18_000)  # the 2026-07-20 incident — 27 % below


def test_free_snapshot_still_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The case the tripwire exists for: a free-based snapshot as capacity.

    This is the negative test that keeps the tolerance honest — without it,
    widening the band would look indistinguishable from fixing the noise.
    """
    pool = ResourcePool(tmp_path / "state.json")
    with caplog.at_level(logging.DEBUG, logger="reslock.pool"):
        pool.set_resources({GPU_KEY: FREE_SNAPSHOT_MB})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert GPU_KEY in msg
    assert str(FREE_SNAPSHOT_MB) in msg
    assert str(FAKE_TOTAL_MB) in msg
    assert "pid" in msg
    # Warn only — the submitted value is registered regardless.
    assert pool.status().resources[GPU_KEY] == FREE_SNAPSHOT_MB


def test_cuda_driver_reserve_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A normally configured CUDA host must be silent.

    Both driver models measured on 2026-08-12 register below the NVML
    physical total; neither is a misconfiguration.
    """
    pool = ResourcePool(tmp_path / "state.json")
    with caplog.at_level(logging.DEBUG, logger="reslock.pool"):
        pool.set_resources({GPU_KEY: LINUX_DRIVER_MB})
        pool.set_resources({GPU_KEY: WINDOWS_DRIVER_MB})

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    # Silent, but not invisible: the offset is still on the record at DEBUG.
    assert [r for r in caplog.records if r.levelno == logging.DEBUG] != []
    assert pool.status().resources[GPU_KEY] == WINDOWS_DRIVER_MB


def test_tolerance_boundary(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The band's edge: exactly at the ratio is silent, one MB under warns."""
    edge = int(FAKE_TOTAL_MB * GPU_CAPACITY_MIN_RATIO) + 1
    pool = ResourcePool(tmp_path / "state.json")
    with caplog.at_level(logging.WARNING, logger="reslock.pool"):
        pool.set_resources({GPU_KEY: edge})
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="reslock.pool"):
        pool.set_resources({GPU_KEY: edge - 2})
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_equal_to_total_no_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pool = ResourcePool(tmp_path / "state.json")
    with caplog.at_level(logging.WARNING, logger="reslock.pool"):
        pool.set_resources({GPU_KEY: FAKE_TOTAL_MB, "cpu_cores": 16})

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    assert pool.status().resources[GPU_KEY] == FAKE_TOTAL_MB


def test_nvml_unavailable_fails_soft(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise() -> dict[str, int]:
        raise NvmlUnavailableError("no driver")

    monkeypatch.setattr("reslock.pool.nvml_total_vram_mb", _raise)
    pool = ResourcePool(tmp_path / "state.json")
    with caplog.at_level(logging.WARNING, logger="reslock.pool"):
        pool.set_resources({GPU_KEY: 18_000})

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    assert pool.status().resources[GPU_KEY] == 18_000


def test_above_total_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pool = ResourcePool(tmp_path / "state.json")
    with caplog.at_level(logging.WARNING, logger="reslock.pool"):
        pool.set_resources({GPU_KEY: FAKE_TOTAL_MB + 1})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "above" in warnings[0].getMessage()
    assert pool.status().resources[GPU_KEY] == FAKE_TOTAL_MB + 1


def test_non_gpu_keys_skip_nvml(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> dict[str, int]:  # pragma: no cover - must not be called
        raise AssertionError("nvml_total_vram_mb called for a non-GPU registration")

    monkeypatch.setattr("reslock.pool.nvml_total_vram_mb", _boom)
    pool = ResourcePool(tmp_path / "state.json")
    with caplog.at_level(logging.WARNING, logger="reslock.pool"):
        pool.set_resources({"cpu_cores": 16, "ram_mb": 64_000})

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
