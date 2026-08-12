"""Detection functions for common system resources.

Each function returns a dict suitable for ``pool.set_resources()``.
Heavy dependencies (torch, subprocess calls) are imported or invoked
lazily inside each function — safe to import this module even when
torch, nvidia-smi, etc. are not available.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import time
import warnings

from reslock.detect import CPU_CORES_KEY, RAM_MB_KEY, gpu_vram_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU VRAM
# ---------------------------------------------------------------------------


def detect_gpu_vram_mb() -> dict[str, int]:
    """Detect per-GPU total VRAM, keyed by host-stable GPU UUID.

    Tries the CUDA driver API first (no torch, no ``nvidia-smi`` binary, and
    it leaves the caller's ability to fork CUDA workers intact), then torch,
    then the nvidia-smi CLI.

    Returns ``{"gpu_GPU-<uuid>_vram_mb": 24000, ...}``. UUIDs are the
    host-stable identifiers reported by nvidia-smi / CUDA, so two containers
    with partial GPU mappings that share a reslock state file coordinate
    correctly on the same physical card.

    Returns an empty dict if no GPUs are detected.

    The driver-API step is first on purpose. The torch step calls
    ``torch.cuda.is_available()``, which opens the ``/dev/nvidia*`` device
    files and thereby makes CUDA unusable in any process the caller forks
    afterwards (measured on kirk: 12 fds opened, forked child then fails with
    "CUDA driver initialization failed"). It also reports nothing on torch
    builds without ``get_device_properties().uuid``, so the registered
    capacity would silently depend on the consumer's torch version.
    """
    result = detect_gpu_vram_mb_cuda_driver()
    if result:
        return result
    result = detect_gpu_vram_mb_torch()
    if result:
        return result
    return detect_gpu_vram_mb_nvidia_smi()


# ---------------------------------------------------------------------------
# GPU VRAM via the CUDA driver API (ctypes)
# ---------------------------------------------------------------------------

_CUDA_DRIVER_LIB = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"

_PROC_SELF_FD = "/proc/self/fd"
"""Where :func:`_cuda_driver_initialized` looks for open GPU device files.

Module-level so tests can point it at a fixture directory.
"""

_FORK_READ_TIMEOUT_S = 10.0
"""Cap on how long the parent waits for the reader child.

``cuInit`` over ten cards takes ~0.3 s (measured on kirk); the cap only
exists so a child that deadlocks — ``fork`` from a multi-threaded process
can, in principle, inherit a held loader lock — degrades into an empty
result and the next detection step, instead of hanging the caller's startup.
"""


def detect_gpu_vram_mb_cuda_driver() -> dict[str, int]:
    """Detect per-GPU total VRAM via ``libcuda``/``nvcuda`` directly, keyed by UUID.

    Needs neither torch nor the ``nvidia-smi`` binary, and reports the same
    number torch does (``cuDeviceTotalMem``, which is what torch's
    ``total_memory`` passes through) without depending on the consumer's
    torch version.

    ``cuInit`` opens the GPU device files, and a process that has done so
    can no longer fork a CUDA-capable child. Where that matters — Linux,
    CUDA not yet initialized here — the read therefore happens in a
    short-lived child process and the caller stays fork-safe. Everywhere
    else the call is direct, because there is nothing left to protect:

    * **Windows** has no ``fork`` at all (``multiprocessing`` spawns fresh
      processes, which inherit no CUDA state).
    * **CUDA already initialized in this process** — the caller is already
      committed; ``cuInit`` is then a no-op and the read costs ~0.2 ms.
    * **macOS / no driver** — loading the library fails and the result is
      an empty dict, so :func:`detect_gpu_vram_mb` moves on.

    Returns an empty dict if the CUDA driver is unavailable or reports no
    devices. Never raises.
    """
    if sys.platform == "linux" and not _cuda_driver_initialized():
        return _read_cuda_driver_totals_forked()
    return _read_cuda_driver_totals()


def _cuda_driver_initialized() -> bool:
    """True if this process already has the CUDA driver open.

    Detected by open file descriptors on ``/dev/nvidia*``. Measured to be an
    exact discriminator for "forking would no longer yield a CUDA-capable
    child": a bare ``import torch`` or a ``dlopen`` of libcuda shows no such
    fds and still forks fine, while ``torch.cuda.is_available()``,
    ``cuInit(0)`` and a real allocation each show them and each break fork.
    Checking for libcuda in ``/proc/self/maps`` would *not* discriminate —
    it is already true after a plain ``import torch``.

    Linux-only; returns False anywhere ``/proc/self/fd`` is unreadable.
    """
    try:
        fds = os.listdir(_PROC_SELF_FD)
    except OSError:
        return False
    for fd in fds:
        try:
            target = os.readlink(os.path.join(_PROC_SELF_FD, fd))
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            return True
    return False


def _read_cuda_driver_totals() -> dict[str, int]:
    """Read per-GPU totals via the CUDA driver API in *this* process.

    Initializes the CUDA driver as a side effect — see
    :func:`detect_gpu_vram_mb_cuda_driver` for when that is acceptable.
    """
    try:
        lib = ctypes.CDLL(_CUDA_DRIVER_LIB)
    except OSError:
        return {}
    try:
        if lib.cuInit(0) != 0:
            return {}
        count = ctypes.c_int()
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return {}
        resources: dict[str, int] = {}
        for index in range(count.value):
            device = ctypes.c_int()
            if lib.cuDeviceGet(ctypes.byref(device), index) != 0:
                continue
            total_bytes = ctypes.c_size_t()
            if lib.cuDeviceTotalMem_v2(ctypes.byref(total_bytes), device) != 0:
                continue
            raw = (ctypes.c_ubyte * 16)()
            if lib.cuDeviceGetUuid(ctypes.byref(raw), device) != 0:
                continue
            d = bytes(raw).hex()
            uuid_str = f"GPU-{d[:8]}-{d[8:12]}-{d[12:16]}-{d[16:20]}-{d[20:]}"
            resources[gpu_vram_key(uuid_str)] = total_bytes.value // (1024 * 1024)
        return resources
    except (AttributeError, OSError, ValueError):
        # Missing symbol (very old driver) or a failing ioctl — the caller
        # falls through to the next detection method.
        return {}


def _read_cuda_driver_totals_forked(timeout: float = _FORK_READ_TIMEOUT_S) -> dict[str, int]:
    """Read the totals in a short-lived child so this process stays fork-safe.

    The child writes ``key=value`` lines to a pipe and ``_exit``s without
    running interpreter shutdown. On any failure — no fork, a child that
    writes nothing, a child that hangs past *timeout* — the result is an
    empty dict.
    """
    try:
        read_fd, write_fd = os.pipe()
    except OSError:
        return {}
    with warnings.catch_warnings():
        # Python 3.12+ warns about fork() in a multi-threaded process. The
        # child does no locking beyond dlopen and exits via os._exit; a stuck
        # one is handled by the timeout below. Warning on every consumer
        # startup would be its own kind of log noise.
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            pid = os.fork()
        except OSError:
            os.close(read_fd)
            os.close(write_fd)
            return {}
    if pid == 0:  # pragma: no cover — child process, never returns
        try:
            os.close(read_fd)
            payload = "\n".join(f"{k}={v}" for k, v in _read_cuda_driver_totals().items())
            os.write(write_fd, payload.encode())
        except BaseException:
            pass
        finally:
            os._exit(0)

    os.close(write_fd)
    buffer = b""
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.debug("reslock: CUDA driver probe child timed out after %.1fs", timeout)
                break
            if not select.select([read_fd], [], [], remaining)[0]:
                logger.debug("reslock: CUDA driver probe child timed out after %.1fs", timeout)
                break
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            buffer += chunk
    except OSError:
        buffer = b""
    finally:
        os.close(read_fd)
        _reap(pid, deadline)
    return _parse_probe_payload(buffer)


def _reap(pid: int, deadline: float) -> None:
    """Wait for the probe child, killing it if it outstayed the deadline."""
    if time.monotonic() >= deadline:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        os.waitpid(pid, 0)


def _parse_probe_payload(buffer: bytes) -> dict[str, int]:
    resources: dict[str, int] = {}
    for line in buffer.decode(errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key and value.isdigit():
            resources[key] = int(value)
    return resources


def detect_gpu_vram_mb_torch() -> dict[str, int]:
    """Detect per-GPU total VRAM using torch, keyed by GPU UUID.

    Requires torch >= 2.0 for the ``get_device_properties(i).uuid`` attribute.
    Returns an empty dict if torch or CUDA is unavailable, or if the UUID
    attribute is missing (older torch).
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        return {}
    if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
        return {}
    resources: dict[str, int] = {}
    for i in range(torch.cuda.device_count()):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        props = torch.cuda.get_device_properties(i)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        total = getattr(props, "total_memory", None) or props.total_mem  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
        uuid_obj = getattr(props, "uuid", None)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        if uuid_obj is None:
            return {}
        resources[gpu_vram_key(f"GPU-{uuid_obj!s}")] = total // (1024 * 1024)  # pyright: ignore[reportUnknownArgumentType]
    return resources


def detect_gpu_vram_mb_nvidia_smi() -> dict[str, int]:
    """Detect per-GPU total VRAM by shelling out to nvidia-smi, keyed by UUID.

    Useful when torch is not installed (e.g. a proxy or orchestrator
    that coordinates GPU work but doesn't load models itself).
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
        return {}


# ---------------------------------------------------------------------------
# CPU cores
# ---------------------------------------------------------------------------


def detect_cpu_cores() -> dict[str, int]:
    """Detect available CPU cores.

    Returns ``{"cpu_cores": N}``.  On Linux with cgroups (containers),
    ``os.sched_getaffinity`` respects CPU limits; ``os.cpu_count()`` is
    used as a fallback.
    """
    try:
        count = len(os.sched_getaffinity(0))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    except AttributeError:
        # macOS / systems without sched_getaffinity
        count = os.cpu_count() or 1
    return {CPU_CORES_KEY: count}


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------


def _total_ram_mb() -> int | None:
    """Total physical RAM in MB, honoring cgroup limits inside containers."""
    total_bytes: int | None = None
    if sys.platform == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_bytes = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            pass
        # A cgroup memory limit (container) caps what this consumer can use.
        for limit_path in (
            "/sys/fs/cgroup/memory.max",  # cgroup v2
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
        ):
            try:
                with open(limit_path) as f:
                    raw = f.read().strip()
                if raw != "max":
                    limit = int(raw)
                    if total_bytes is None or limit < total_bytes:
                        total_bytes = limit
                break
            except (OSError, ValueError):
                continue
    elif sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass
    if total_bytes is None:
        return None
    return total_bytes // (1024 * 1024)


def detect_ram_mb(reserve_mb: int = 0) -> dict[str, int]:
    """Detect total system RAM in MB, minus a configurable reserve.

    Args:
        reserve_mb: Headroom to keep out of the pool (OS, page cache,
            non-reslock processes). The registered capacity is
            ``MemTotal - reserve_mb``.

    Returns ``{"ram_mb": N}``, or an empty dict if RAM cannot be detected
    (matching the other ``detect_*`` functions). Inside a Linux container,
    a cgroup memory limit lower than the host's MemTotal wins.

    Raises:
        ValueError: If ``reserve_mb`` is negative or leaves no capacity
            (``reserve_mb >= MemTotal``) — a misconfigured reserve should
            surface loudly rather than silently register nothing.
    """
    if reserve_mb < 0:
        raise ValueError(f"reserve_mb must be non-negative, got {reserve_mb}")
    total = _total_ram_mb()
    if total is None:
        return {}
    if reserve_mb >= total:
        raise ValueError(f"reserve_mb={reserve_mb} leaves no capacity (detected {total} MB total)")
    return {RAM_MB_KEY: total - reserve_mb}


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------


def detect_disk_mb(paths: list[str] | None = None) -> dict[str, int]:
    """Detect total disk space in MB for each path / mount point.

    Args:
        paths: Directories to check.  Defaults to ``["/"]``.
            Pass multiple paths for separate partitions, e.g.
            ``["/", "/data", "/scratch"]``.

    Returns keys like ``disk_root_mb``, ``disk_data_mb``.
    """
    if paths is None:
        paths = ["/"]
    resources: dict[str, int] = {}
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        name = os.path.basename(path) or "root"
        resources[f"disk_{name}_mb"] = usage.total // (1024 * 1024)
    return resources


# ---------------------------------------------------------------------------
# Network bandwidth
# ---------------------------------------------------------------------------


def detect_network_bandwidth() -> dict[str, int]:
    """Detect NIC link speeds in Mbit/s.

    Returns keys like ``net_eth0_mbps``.

    - **Linux**: reads ``/sys/class/net/<iface>/speed``.
    - **macOS**: parses ``networksetup`` output.

    Only physical / active interfaces are included.
    """
    if sys.platform == "linux":
        return _net_bandwidth_linux()
    if sys.platform == "darwin":
        return _net_bandwidth_macos()
    return {}


def _net_bandwidth_linux() -> dict[str, int]:
    resources: dict[str, int] = {}
    try:
        ifaces = os.listdir("/sys/class/net")
    except OSError:
        return {}
    for iface in sorted(ifaces):
        if iface == "lo":
            continue
        try:
            with open(f"/sys/class/net/{iface}/operstate") as f:
                if f.read().strip() != "up":
                    continue
        except OSError:
            continue
        try:
            with open(f"/sys/class/net/{iface}/speed") as f:
                speed = int(f.read().strip())
            if speed > 0:
                resources[f"net_{iface}_mbps"] = speed
        except (OSError, ValueError):
            continue
    return resources


def _net_bandwidth_macos() -> dict[str, int]:
    import re

    resources: dict[str, int] = {}
    try:
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
    except (subprocess.TimeoutExpired, OSError):
        return {}

    # Parse port->device mapping
    ports: list[tuple[str, str]] = []
    port_name = ""
    for line in result.stdout.splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            if port_name and device != "N/A":
                ports.append((port_name, device))

    for port_name, device in ports:
        if device.startswith(("bridge", "utun")):
            continue
        try:
            ifconfig = subprocess.run(
                ["ifconfig", device],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "status: active" not in ifconfig.stdout:
                continue
        except (subprocess.TimeoutExpired, OSError):
            continue
        # Parse link speed from networksetup -getMedia
        try:
            media = subprocess.run(
                ["networksetup", "-getMedia", port_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if media.returncode == 0:
                for mline in media.stdout.splitlines():
                    if "Current:" not in mline:
                        continue
                    m = re.search(r"(\d+)([GM]?)base", mline, re.IGNORECASE)
                    if m:
                        speed = int(m.group(1))
                        if m.group(2).upper() == "G":
                            speed *= 1000
                        if speed > 0:
                            resources[f"net_{device}_mbps"] = speed
                        break
        except (subprocess.TimeoutExpired, OSError):
            continue
    return resources
