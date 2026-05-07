"""Tests for reslock.audit — GPU orphan detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from reslock import OrphanReport, ResourcePool
from reslock.audit import gpu_orphans
from reslock.detect import gpu_vram_key
from reslock.models import Lease, State

UUID_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"


def _no_nvidia_smi(_name: str) -> str | None:
    return None


def _has_nvidia_smi(_name: str) -> str | None:
    return "/usr/bin/nvidia-smi"


def test_gpu_orphans_empty_when_nvidia_smi_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """No nvidia-smi → no diagnosis available → empty list."""
    monkeypatch.setattr("reslock.audit.shutil.which", _no_nvidia_smi)
    state = State()
    assert gpu_orphans(state) == []


def test_gpu_orphans_identifies_unregistered_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PID with VRAM that's not in any lease's PID set is an orphan."""

    def _fake_run(*_a: Any, **_kw: Any) -> Any:
        class _R:
            returncode = 0
            stdout = f"99999, {UUID_A}, 8192\n12345, {UUID_A}, 4096\n"

        return _R()

    monkeypatch.setattr("reslock.audit.shutil.which", _has_nvidia_smi)
    monkeypatch.setattr("reslock.audit.subprocess.run", _fake_run)
    # _read_cmdline returns None when /proc/<pid>/cmdline is unavailable; that's fine on macOS.

    state = State(leases=[Lease(pid=12345, resources={gpu_vram_key(UUID_A): 4096})])

    orphans = gpu_orphans(state)
    assert len(orphans) == 1
    assert orphans[0].pid == 99999
    assert orphans[0].vram_mb == 8192
    assert orphans[0].gpus == {UUID_A: 8192}


def test_gpu_orphans_includes_host_pid_in_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Container scenario: lease.pid is the in-container PID, lease.host_pid is
    what nvidia-smi sees. Cross-namespace registration must be honored."""

    def _fake_run(*_a: Any, **_kw: Any) -> Any:
        class _R:
            returncode = 0
            stdout = f"99999, {UUID_A}, 8192\n"

        return _R()

    monkeypatch.setattr("reslock.audit.shutil.which", _has_nvidia_smi)
    monkeypatch.setattr("reslock.audit.subprocess.run", _fake_run)

    state = State(leases=[Lease(pid=42, host_pid=99999, resources={gpu_vram_key(UUID_A): 8192})])
    assert gpu_orphans(state) == []


def test_gpu_orphans_includes_tracked_child_pids(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lease with `pids=[child]` must register the child as covered."""

    def _fake_run(*_a: Any, **_kw: Any) -> Any:
        class _R:
            returncode = 0
            stdout = f"55555, {UUID_A}, 4096\n"

        return _R()

    monkeypatch.setattr("reslock.audit.shutil.which", _has_nvidia_smi)
    monkeypatch.setattr("reslock.audit.subprocess.run", _fake_run)

    state = State(
        leases=[
            Lease(pid=42, pids=[55555], resources={gpu_vram_key(UUID_A): 4096}),
        ]
    )
    assert gpu_orphans(state) == []


def test_gpu_orphans_sorted_by_vram_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worst offenders first."""

    def _fake_run(*_a: Any, **_kw: Any) -> Any:
        class _R:
            returncode = 0
            stdout = f"100, {UUID_A}, 1000\n200, {UUID_A}, 5000\n300, {UUID_A}, 2000\n"

        return _R()

    monkeypatch.setattr("reslock.audit.shutil.which", _has_nvidia_smi)
    monkeypatch.setattr("reslock.audit.subprocess.run", _fake_run)

    state = State()
    orphans = gpu_orphans(state)
    assert [o.pid for o in orphans] == [200, 300, 100]


def test_gpu_orphans_aggregates_per_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same PID on multiple GPUs → single OrphanReport with summed vram_mb."""
    UUID_B = "GPU-bbbbbbbb-1111-2222-3333-444444444444"

    def _fake_run(*_a: Any, **_kw: Any) -> Any:
        class _R:
            returncode = 0
            stdout = f"42, {UUID_A}, 2000\n42, {UUID_B}, 3000\n"

        return _R()

    monkeypatch.setattr("reslock.audit.shutil.which", _has_nvidia_smi)
    monkeypatch.setattr("reslock.audit.subprocess.run", _fake_run)

    state = State()
    orphans = gpu_orphans(state)
    assert len(orphans) == 1
    assert orphans[0].pid == 42
    assert orphans[0].vram_mb == 5000
    assert orphans[0].gpus == {UUID_A: 2000, UUID_B: 3000}


def test_pool_gpu_orphans_dispatches_to_audit_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ResourcePool.gpu_orphans() exposes the audit primitive on the pool."""
    monkeypatch.setattr("reslock.audit.shutil.which", _no_nvidia_smi)
    pool = ResourcePool(tmp_path / "state.json")
    out = pool.gpu_orphans()
    assert out == []


def test_orphan_report_is_pydantic_model() -> None:
    """OrphanReport is a Pydantic model (serializable via state inspection / RPC)."""
    rep = OrphanReport(pid=os.getpid(), vram_mb=1024, gpus={UUID_A: 1024}, cmdline="/usr/bin/foo")
    dumped = rep.model_dump()
    assert dumped["pid"] == os.getpid()
    assert dumped["vram_mb"] == 1024
    assert dumped["gpus"] == {UUID_A: 1024}
