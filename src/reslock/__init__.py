"""reslock — Resource lock manager for coordinating shared system resources."""

from __future__ import annotations

from reslock.models import Lease, PoolStatus, QueueEntry, State
from reslock.pool import LeaseHandle, ResourcePool
from reslock.resources import (
    detect_cpu_cores,
    detect_disk_mb,
    detect_gpu_vram_mb,
    detect_gpu_vram_mb_nvidia_smi,
    detect_gpu_vram_mb_torch,
    detect_network_bandwidth,
)

__all__ = [
    "Lease",
    "LeaseHandle",
    "PoolStatus",
    "QueueEntry",
    "ResourcePool",
    "State",
    "detect_cpu_cores",
    "detect_disk_mb",
    "detect_gpu_vram_mb",
    "detect_gpu_vram_mb_nvidia_smi",
    "detect_gpu_vram_mb_torch",
    "detect_network_bandwidth",
]
