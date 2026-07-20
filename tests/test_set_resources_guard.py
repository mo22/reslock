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
from reslock.pool import ResourcePool
from tests.conftest import FAKE_GPU_UUIDS

FAKE_TOTAL_MB = 9_999_999
GPU_KEY = gpu_vram_key(FAKE_GPU_UUIDS[0])


def test_below_total_warns_but_registers(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pool = ResourcePool(tmp_path / "state.json")
    with caplog.at_level(logging.WARNING, logger="reslock.pool"):
        pool.set_resources({GPU_KEY: 18_000})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert GPU_KEY in msg
    assert "18000" in msg
    assert str(FAKE_TOTAL_MB) in msg
    assert "pid" in msg
    # Warn only — the submitted value is registered regardless.
    assert pool.status().resources[GPU_KEY] == 18_000


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
