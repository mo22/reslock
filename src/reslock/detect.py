from __future__ import annotations

import os
import shutil
import subprocess
import sys

try:
    import resource
except ImportError:  # pragma: no cover — Windows has no POSIX resource module
    resource = None  # type: ignore[assignment]


def detect_gpu_vram_mb() -> dict[str, int]:
    """Detect per-GPU VRAM via nvidia-smi.

    Returns resources like ``{"gpu0_vram_mb": 24000, "gpu1_vram_mb": 24000}``.
    Returns empty dict if nvidia-smi is not available or no GPUs found.
    """
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        resources: dict[str, int] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                idx = int(parts[0].strip())
                mb = int(parts[1].strip())
                resources[f"gpu{idx}_vram_mb"] = mb
        return resources
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
    if resource is None:
        return None
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
    if resource is None:
        return None
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


def _nvidia_smi_gpu_uuid_to_index() -> dict[str, int]:
    """Build a mapping from GPU UUID to device index via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        mapping: dict[str, int] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                mapping[parts[1].strip()] = int(parts[0].strip())
        return mapping
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return {}


def get_pid_vram_mb(pid: int) -> int | None:
    """Get GPU memory usage of a process by PID via nvidia-smi (total across GPUs)."""
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
    """Get GPU memory usage for all processes via a single nvidia-smi call (total per PID)."""
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


def get_all_pid_vram_per_gpu_mb() -> dict[int, dict[int, int]]:
    """Get per-GPU memory usage for all processes.

    Returns ``{pid: {gpu_index: mb}}``. Requires two nvidia-smi calls
    (one for UUID→index mapping, one for per-process usage).
    """
    if not shutil.which("nvidia-smi"):
        return {}
    uuid_map = _nvidia_smi_gpu_uuid_to_index()
    if not uuid_map:
        return {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        usage: dict[int, dict[int, int]] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 3:
                pid = int(parts[0].strip())
                gpu_uuid = parts[1].strip()
                mb = int(parts[2].strip())
                gpu_idx = uuid_map.get(gpu_uuid)
                if gpu_idx is not None:
                    usage.setdefault(pid, {})[gpu_idx] = usage.get(pid, {}).get(gpu_idx, 0) + mb
        return usage
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return {}


def get_torch_cuda_mb() -> int | None:
    """Get total CUDA memory allocated by torch across all GPUs. Returns None if torch not loaded."""
    if "torch" not in sys.modules:
        return None
    try:
        import torch  # pyright: ignore[reportMissingImports]

        if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            return None
        total = 0
        for i in range(torch.cuda.device_count()):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            total += torch.cuda.memory_allocated(i)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return total // (1024 * 1024)  # pyright: ignore[reportUnknownVariableType]
    except Exception:
        return None


def get_torch_cuda_per_gpu_mb() -> dict[int, int]:
    """Get per-GPU CUDA memory allocated by torch. Returns ``{gpu_index: mb}``."""
    if "torch" not in sys.modules:
        return {}
    try:
        import torch  # pyright: ignore[reportMissingImports]

        if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            return {}
        result: dict[int, int] = {}
        for i in range(torch.cuda.device_count()):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            mb = int(torch.cuda.memory_allocated(i)) // (1024 * 1024)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            if mb > 0:
                result[i] = mb
        return result
    except Exception:
        return {}


def get_self_actual_resources() -> dict[str, int]:
    """Detect actual resource usage of the current process.

    Reports per-GPU VRAM as ``gpu0_vram_mb``, ``gpu1_vram_mb``, etc.
    """
    result: dict[str, int] = {}
    rss = get_self_rss_mb()
    if rss is not None and rss > 0:
        result["ram_mb"] = rss
    # Prefer torch CUDA measurement (more accurate, includes tensors)
    per_gpu = get_torch_cuda_per_gpu_mb()
    if per_gpu:
        for idx, mb in per_gpu.items():
            result[f"gpu{idx}_vram_mb"] = mb
    else:
        vram = get_pid_vram_mb(os.getpid())
        if vram is not None and vram > 0:
            result["vram_mb"] = vram
    return result
