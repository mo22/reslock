from __future__ import annotations

import os
import shutil
import subprocess
import sys

try:
    import resource
except ImportError:  # pragma: no cover — Windows has no POSIX resource module
    resource = None  # type: ignore[assignment]


def gpu_vram_key(gpu_uuid: str) -> str:
    """Build the reslock resource key for a GPU UUID's VRAM."""
    return f"gpu_{gpu_uuid}_vram_mb"


def parse_gpu_vram_key(key: str) -> str | None:
    """Extract the GPU UUID from a ``gpu_{uuid}_vram_mb`` resource key.

    Returns None if the key doesn't match the expected format.
    """
    if key.startswith("gpu_") and key.endswith("_vram_mb"):
        return key[len("gpu_") : -len("_vram_mb")]
    return None


def detect_gpu_vram_mb() -> dict[str, int]:
    """Detect per-GPU VRAM via nvidia-smi, keyed by host-stable GPU UUID.

    Returns resources like ``{"gpu_GPU-<uuid>_vram_mb": 24000, ...}``.
    Returns empty dict if nvidia-smi is not available or no GPUs found.

    Keying by UUID (instead of nvidia-smi index) keeps coordination correct
    across containers that get partial GPU mappings from the NVIDIA container
    runtime — each container sees only its mapped cards renumbered from 0,
    but UUIDs are stable across the host.
    """
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"],
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
                uuid_str = parts[0].strip()
                mb = int(parts[1].strip())
                if uuid_str:
                    resources[gpu_vram_key(uuid_str)] = mb
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


_TORCH_INDEX_TO_UUID_CACHE: dict[int, str] = {}


def _torch_device_uuid(torch_index: int) -> str | None:
    """Read a GPU UUID from torch's device properties. Requires torch 2.0+."""
    if "torch" not in sys.modules:
        try:
            import torch  # pyright: ignore[reportMissingImports]  # noqa: F401
        except ImportError:
            return None
    try:
        import torch  # pyright: ignore[reportMissingImports]

        if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            return None
        if torch_index < 0 or torch_index >= torch.cuda.device_count():  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            return None
        props = torch.cuda.get_device_properties(torch_index)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        uuid_obj = getattr(props, "uuid", None)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        if uuid_obj is None:
            return None
        return f"GPU-{uuid_obj!s}"
    except Exception:
        return None


def _nvidia_smi_index_to_uuid(torch_index: int) -> str | None:
    """Fall back to nvidia-smi to find the UUID of a device index."""
    uuid_to_index = _nvidia_smi_gpu_uuid_to_index()
    for uuid_str, idx in uuid_to_index.items():
        if idx == torch_index:
            return uuid_str
    return None


def gpu_uuid_for_torch_index(torch_index: int) -> str | None:
    """Return the host-stable GPU UUID for a local torch device index.

    Inside a container with partial GPU mapping, ``torch_index`` is the
    container-local index (0..N-1 where N is the number of visible cards).
    The returned UUID is the host's GPU UUID — stable across containers and
    invariant under NVIDIA container runtime renumbering.

    Returns None if the UUID cannot be determined (no CUDA, index out of
    range, or neither torch nor nvidia-smi available).

    Results are cached per-process; GPUs don't hot-swap.
    """
    if torch_index in _TORCH_INDEX_TO_UUID_CACHE:
        return _TORCH_INDEX_TO_UUID_CACHE[torch_index]
    uuid_str = _torch_device_uuid(torch_index)
    if uuid_str is None:
        uuid_str = _nvidia_smi_index_to_uuid(torch_index)
    if uuid_str is not None:
        _TORCH_INDEX_TO_UUID_CACHE[torch_index] = uuid_str
    return uuid_str


def gpu_resource_key(torch_index: int) -> str | None:
    """Return the reslock resource key ``gpu_{uuid}_vram_mb`` for a local torch
    device index, or None if the UUID cannot be determined.

    Inside a container with partial GPU mapping, ``torch_index`` is the
    container-local index, but the returned key uses the host-stable UUID, so
    leases coordinate correctly across containers sharing reslock state.
    """
    uuid_str = gpu_uuid_for_torch_index(torch_index)
    if uuid_str is None:
        return None
    return gpu_vram_key(uuid_str)


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


def get_all_pid_vram_per_gpu_mb() -> dict[int, dict[str, int]]:
    """Get per-GPU memory usage for all processes, keyed by GPU UUID.

    Returns ``{pid: {gpu_uuid: mb}}``. GPU UUIDs match the keys emitted by
    ``detect_gpu_vram_mb()`` (without the ``gpu_``/``_vram_mb`` wrapping).
    """
    if not shutil.which("nvidia-smi"):
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
        usage: dict[int, dict[str, int]] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 3:
                pid = int(parts[0].strip())
                gpu_uuid = parts[1].strip()
                mb = int(parts[2].strip())
                if gpu_uuid:
                    per_pid = usage.setdefault(pid, {})
                    per_pid[gpu_uuid] = per_pid.get(gpu_uuid, 0) + mb
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


def get_torch_cuda_per_gpu_mb() -> dict[str, int]:
    """Get per-GPU CUDA memory allocated by torch, keyed by GPU UUID.

    Returns ``{gpu_uuid: mb}``. Only includes devices whose UUID is resolvable
    (via torch properties or nvidia-smi) and whose current allocation is > 0.
    """
    if "torch" not in sys.modules:
        return {}
    try:
        import torch  # pyright: ignore[reportMissingImports]

        if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            return {}
        result: dict[str, int] = {}
        for i in range(torch.cuda.device_count()):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            mb = int(torch.cuda.memory_allocated(i)) // (1024 * 1024)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            if mb <= 0:
                continue
            uuid_str = gpu_uuid_for_torch_index(i)
            if uuid_str is None:
                continue
            result[uuid_str] = mb
        return result
    except Exception:
        return {}


def get_self_actual_resources() -> dict[str, int]:
    """Detect actual resource usage of the current process.

    Reports per-GPU VRAM as ``gpu_{uuid}_vram_mb`` keys, matching the format
    emitted by ``detect_gpu_vram_mb()``.
    """
    result: dict[str, int] = {}
    rss = get_self_rss_mb()
    if rss is not None and rss > 0:
        result["ram_mb"] = rss
    # Prefer torch CUDA measurement (more accurate, includes tensors)
    per_gpu = get_torch_cuda_per_gpu_mb()
    if per_gpu:
        for uuid_str, mb in per_gpu.items():
            result[gpu_vram_key(uuid_str)] = mb
    else:
        vram = get_pid_vram_mb(os.getpid())
        if vram is not None and vram > 0:
            result["vram_mb"] = vram
    return result
