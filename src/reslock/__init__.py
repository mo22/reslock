"""reslock — Resource lock manager for coordinating shared system resources."""

from __future__ import annotations

from reslock.detect import (
    gpu_resource_key,
    gpu_uuid_for_torch_index,
    gpu_vram_key,
    parse_gpu_vram_key,
)
from reslock.models import SCHEMA_VERSION, Lease, PoolStatus, QueueEntry, State
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
    "SCHEMA_VERSION",
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
    "gpu_resource_key",
    "gpu_uuid_for_torch_index",
    "gpu_vram_key",
    "parse_gpu_vram_key",
]
