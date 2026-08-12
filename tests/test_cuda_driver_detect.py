"""Tests for GPU VRAM detection via the CUDA driver API.

The point of this path is that it reports the CUDA-visible total *without*
torch, without the ``nvidia-smi`` binary, and — this is the part worth
protecting — without destroying the caller's ability to fork CUDA workers.
``torch.cuda.is_available()`` opens the ``/dev/nvidia*`` device files, and a
process that has done so can no longer fork a CUDA-capable child.

CI has no GPU, so the actual ``libcuda`` read is not exercised here. What is
exercised: which branch gets taken, the fd-based "already initialized" probe,
and the full fork/pipe/parse plumbing with a stubbed reader.
"""

# The branch logic and the fork plumbing are the whole point of this module and
# have no public entry point of their own — testing them means reaching for the
# underscored helpers.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from reslock import resources

needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="platform has no fork()")


# --- the "is CUDA already open in this process?" probe ---------------------


def _fd_dir(tmp_path: Path, targets: list[str]) -> str:
    """Build a stand-in for /proc/self/fd with the given symlink targets."""
    d = tmp_path / "fd"
    d.mkdir()
    for i, target in enumerate(targets):
        os.symlink(target, d / str(i))  # dangling links are fine, we only readlink
    return str(d)


def test_initialized_detects_nvidia_device_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fds = _fd_dir(tmp_path, ["/dev/null", "/dev/nvidiactl", "/dev/nvidia0"])
    monkeypatch.setattr(resources, "_PROC_SELF_FD", fds)
    assert resources._cuda_driver_initialized() is True


def test_not_initialized_without_nvidia_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bare `import torch` or a dlopen of libcuda looks like this — libcuda is
    # mapped, but no device file is open, and fork still works.
    fds = _fd_dir(tmp_path, ["/dev/null", "/dev/urandom", "/tmp/libcuda.so.1"])
    monkeypatch.setattr(resources, "_PROC_SELF_FD", fds)
    assert resources._cuda_driver_initialized() is False


def test_not_initialized_when_procfs_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resources, "_PROC_SELF_FD", str(tmp_path / "nope"))
    assert resources._cuda_driver_initialized() is False


# --- branch selection ------------------------------------------------------


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record which reader ran instead of touching a real driver."""
    calls: list[str] = []

    def _direct() -> dict[str, int]:
        calls.append("direct")
        return {"gpu_GPU-direct_vram_mb": 1}

    def _forked(_timeout: float = 0.0) -> dict[str, int]:
        calls.append("forked")
        return {"gpu_GPU-forked_vram_mb": 2}

    monkeypatch.setattr(resources, "_read_cuda_driver_totals", _direct)
    monkeypatch.setattr(resources, "_read_cuda_driver_totals_forked", _forked)
    return calls


def test_linux_without_cuda_forks_to_stay_fork_safe(
    spy: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(resources, "_cuda_driver_initialized", lambda: False)
    assert resources.detect_gpu_vram_mb_cuda_driver() == {"gpu_GPU-forked_vram_mb": 2}
    assert spy == ["forked"]


def test_linux_with_cuda_already_open_reads_directly(
    spy: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing left to protect: this process can't fork a CUDA child either way,
    # and cuInit is a no-op, so the fork would be pure cost (and would fail —
    # the child inherits the initialized state).
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(resources, "_cuda_driver_initialized", lambda: True)
    assert resources.detect_gpu_vram_mb_cuda_driver() == {"gpu_GPU-direct_vram_mb": 1}
    assert spy == ["direct"]


def test_windows_reads_directly(spy: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    # No fork on Windows at all — multiprocessing spawns fresh processes, which
    # inherit no CUDA state, so there is nothing the fork dance would buy.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        resources,
        "_cuda_driver_initialized",
        lambda: pytest.fail("the fd probe is Linux-only and must not run here"),
    )
    assert resources.detect_gpu_vram_mb_cuda_driver() == {"gpu_GPU-direct_vram_mb": 1}
    assert spy == ["direct"]


def test_driver_library_matches_platform() -> None:
    expected = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    assert expected == resources._CUDA_DRIVER_LIB


# --- the direct read, without a driver present -----------------------------


def test_direct_read_without_driver_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_lib(_name: str) -> object:
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(resources.ctypes, "CDLL", _no_lib)
    assert resources._read_cuda_driver_totals() == {}


# --- fork plumbing ---------------------------------------------------------


@needs_fork
def test_forked_read_round_trips_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"gpu_GPU-aaaa_vram_mb": 24135, "gpu_GPU-bbbb_vram_mb": 81038}
    monkeypatch.setattr(resources, "_read_cuda_driver_totals", lambda: dict(payload))
    assert resources._read_cuda_driver_totals_forked() == payload


@needs_fork
def test_forked_read_survives_an_empty_child_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "_read_cuda_driver_totals", dict)
    assert resources._read_cuda_driver_totals_forked() == {}


@needs_fork
def test_forked_read_survives_a_crashing_child(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> dict[str, int]:
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(resources, "_read_cuda_driver_totals", _boom)
    assert resources._read_cuda_driver_totals_forked() == {}


@needs_fork
def test_forked_read_times_out_instead_of_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged child must degrade to {}, not block the consumer's startup."""

    def _hang() -> dict[str, int]:
        time.sleep(5)
        return {"gpu_GPU-late_vram_mb": 1}

    monkeypatch.setattr(resources, "_read_cuda_driver_totals", _hang)
    started = time.monotonic()
    assert resources._read_cuda_driver_totals_forked(timeout=0.3) == {}
    assert time.monotonic() - started < 10.0


@needs_fork
def test_forked_read_leaves_no_zombie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "_read_cuda_driver_totals", lambda: {"gpu_GPU-x_vram_mb": 7})
    resources._read_cuda_driver_totals_forked()
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)  # no unreaped children left behind


def test_payload_parser_ignores_garbage() -> None:
    parsed = resources._parse_probe_payload(b"a=1\nnot-a-pair\n=5\nb=\nc=xyz\nd=42")
    assert parsed == {"a": 1, "d": 42}


# --- the chain -------------------------------------------------------------


def test_chain_prefers_driver_over_torch_and_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resources, "detect_gpu_vram_mb_cuda_driver", lambda: {"gpu_GPU-a_vram_mb": 24135}
    )
    monkeypatch.setattr(
        resources,
        "detect_gpu_vram_mb_torch",
        lambda: pytest.fail("torch must not run once the driver answered — it breaks fork"),
    )
    monkeypatch.setattr(
        resources, "detect_gpu_vram_mb_nvidia_smi", lambda: pytest.fail("nvidia-smi must not run")
    )
    assert resources.detect_gpu_vram_mb() == {"gpu_GPU-a_vram_mb": 24135}


def test_chain_falls_through_to_torch_then_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "detect_gpu_vram_mb_cuda_driver", dict)
    monkeypatch.setattr(resources, "detect_gpu_vram_mb_torch", dict)
    monkeypatch.setattr(
        resources, "detect_gpu_vram_mb_nvidia_smi", lambda: {"gpu_GPU-c_vram_mb": 24576}
    )
    assert resources.detect_gpu_vram_mb() == {"gpu_GPU-c_vram_mb": 24576}
