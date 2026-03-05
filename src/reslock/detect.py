from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys


def detect_gpu_vram_mb() -> dict[str, int]:
    """Detect total GPU VRAM via nvidia-smi. Returns empty dict if not available."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        total = 0
        for line in result.stdout.strip().splitlines():
            total += int(line.strip())
        if total > 0:
            return {"vram_mb": total}
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return {}


# --- Per-process resource measurement (portable) ---


def get_host_pid() -> int:
    """Get the host-visible PID of this process.

    In a container with a separate PID namespace, the internal PID differs from
    the host PID. This reads /proc/self/status NSpid to find the host PID.
    Falls back to os.getpid() when not in a container or not on Linux.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("NSpid:"):
                    pids = line.split()[1:]
                    return int(pids[0])  # first PID is in the host namespace
    except (FileNotFoundError, IndexError, ValueError):
        pass
    return os.getpid()


def get_self_rss_mb() -> int | None:
    """Get RSS of the current process in MB using the resource module."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ru_maxrss = usage.ru_maxrss
        # macOS reports bytes, Linux reports KB
        if sys.platform == "darwin":
            return ru_maxrss // (1024 * 1024)
        return ru_maxrss // 1024
    except (OSError, ValueError):
        return None


def get_self_cpu_seconds() -> float | None:
    """Get CPU time (user + system) of the current process in seconds."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime
    except (OSError, ValueError):
        return None


def get_pid_rss_mb(pid: int) -> int | None:
    """Get RSS of a process by PID in MB. Uses `ps` (works on macOS + Linux)."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "rss="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        rss_kb = int(result.stdout.strip())
        return rss_kb // 1024
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


def get_pid_cpu_seconds(pid: int) -> float | None:
    """Get CPU time of a process by PID in seconds. Uses `ps` (works on macOS + Linux)."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "time="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        # Format: [DD-]HH:MM:SS or MM:SS
        text = result.stdout.strip()
        parts = text.replace("-", ":").split(":")
        parts_int = [int(p) for p in parts]
        if len(parts_int) == 2:
            return parts_int[0] * 60.0 + parts_int[1]
        if len(parts_int) == 3:
            return parts_int[0] * 3600.0 + parts_int[1] * 60.0 + parts_int[2]
        if len(parts_int) == 4:
            return (
                parts_int[0] * 86400.0 + parts_int[1] * 3600.0 + parts_int[2] * 60.0 + parts_int[3]
            )
        return None
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


def get_pid_vram_mb(pid: int) -> int | None:
    """Get GPU memory usage of a process by PID via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        total = 0
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2 and int(parts[0].strip()) == pid:
                total += int(parts[1].strip())
        return total if total > 0 else None
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


def get_all_pid_vram_mb() -> dict[int, int]:
    """Get GPU memory usage for all processes via a single nvidia-smi call."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        usage: dict[int, int] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                pid = int(parts[0].strip())
                mb = int(parts[1].strip())
                usage[pid] = usage.get(pid, 0) + mb
        return usage
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return {}


def get_torch_cuda_mb() -> int | None:
    """Get CUDA memory allocated by torch in the current process. Returns None if torch not loaded."""
    if "torch" not in sys.modules:
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total = 0
        for i in range(torch.cuda.device_count()):
            total += torch.cuda.memory_allocated(i)
        return total // (1024 * 1024)
    except Exception:
        return None


def get_self_actual_resources() -> dict[str, int]:
    """Detect actual resource usage of the current process. Returns what it can measure."""
    result: dict[str, int] = {}
    rss = get_self_rss_mb()
    if rss is not None and rss > 0:
        result["ram_mb"] = rss
    # Prefer torch CUDA measurement (more accurate, includes tensors)
    vram = get_torch_cuda_mb()
    if vram is not None and vram > 0:
        result["vram_mb"] = vram
    else:
        vram = get_pid_vram_mb(os.getpid())
        if vram is not None and vram > 0:
            result["vram_mb"] = vram
    return result
