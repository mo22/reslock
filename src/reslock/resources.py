"""Detection functions for common system resources.

Each function returns a dict suitable for ``pool.set_resources()``.
Heavy dependencies (torch, subprocess calls) are imported or invoked
lazily inside each function — safe to import this module even when
torch, nvidia-smi, etc. are not available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from reslock.detect import gpu_vram_key

# ---------------------------------------------------------------------------
# GPU VRAM
# ---------------------------------------------------------------------------


def detect_gpu_vram_mb() -> dict[str, int]:
    """Detect per-GPU total VRAM, keyed by host-stable GPU UUID.

    Tries torch first (works inside containers without nvidia-smi),
    falls back to nvidia-smi CLI.

    Returns ``{"gpu_GPU-<uuid>_vram_mb": 24000, ...}``. UUIDs are the
    host-stable identifiers reported by nvidia-smi / CUDA, so two containers
    with partial GPU mappings that share a reslock state file coordinate
    correctly on the same physical card.

    Returns an empty dict if no GPUs are detected.
    """
    result = detect_gpu_vram_mb_torch()
    if result:
        return result
    return detect_gpu_vram_mb_nvidia_smi()


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
    return {"cpu_cores": count}


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
